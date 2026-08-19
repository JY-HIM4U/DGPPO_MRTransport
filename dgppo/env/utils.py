import numpy as np
import jax.numpy as jnp
import functools as ft
import jax
import jax.random as jr

from scipy.linalg import inv, solve_discrete_are
from typing import Callable, Tuple
from jax.lax import while_loop

from ..utils.typing import Array, Radius, BoolScalar, Pos, State, Action, PRNGKey
from ..utils.utils import merge01
from dgppo.env.obstacle import Obstacle, Rectangle, Cuboid, Sphere, Circle, Polygon


def RK4_step(x_dot_fn: Callable, x: State, u: Action, dt: float) -> Array:
    k1 = x_dot_fn(x, u)
    k2 = x_dot_fn(x + 0.5 * dt * k1, u)
    k3 = x_dot_fn(x + 0.5 * dt * k2, u)
    k4 = x_dot_fn(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u
    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151
    Yields the control law u = -K x
    """

    # first, try to solve the Riccati equation
    X = solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K


def get_lidar(start_point: Pos, obstacles: Obstacle, num_beams: int, sense_range: float, max_returns: int = 32):
    if isinstance(obstacles, (Rectangle, Circle, Polygon)):
        # All 2D obstacles use the same 2D ray pattern
        thetas = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(num_beams, axis=0)
        ends = jnp.stack(
            [starts[..., 0] + jnp.cos(thetas) * sense_range, starts[..., 1] + jnp.sin(thetas) * sense_range],
            axis=-1)
    elif isinstance(obstacles, Cuboid) or isinstance(obstacles, Sphere):
        # 3D obstacles use a spherical pattern
        thetas = jnp.linspace(-np.pi / 2 + 2 * np.pi / num_beams, np.pi / 2 - 2 * np.pi / num_beams, num_beams // 2)
        phis = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(thetas.shape[0] * phis.shape[0] + 2, axis=0)

        def get_end_point(theta, phi):
            return jnp.array([
                start_point[0] + jnp.cos(theta) * jnp.cos(phi) * sense_range,
                start_point[1] + jnp.cos(theta) * jnp.sin(phi) * sense_range,
                start_point[2] + jnp.sin(theta) * sense_range
            ])

        def get_end_point_theta(theta):
            return jax.vmap(lambda phi: get_end_point(theta, phi))(phis)

        ends = merge01(jax.vmap(get_end_point_theta)(thetas))
        ends = jnp.concatenate([ends,
                                start_point[None, :] + jnp.array([[0., 0., sense_range]]),
                                start_point[None, :] + jnp.array([[0., 0., -sense_range]])], axis=0)
    else:
        raise NotImplementedError(f"Obstacle type {type(obstacles)} not supported")
    
    sensor_data = raytracing(starts, ends, obstacles, max_returns)

    return sensor_data


def inside_obstacles(points: Pos, obstacles: Obstacle = None, r: Radius = 0.) -> BoolScalar:
    """
    points: (n, n_dim) or (n_dim, )
    obstacles: tree_stacked obstacles.

    Returns: (n, ) or (,). True if in collision, false otherwise.
    """
    if obstacles is None:
        if points.ndim == 1:
            return jnp.zeros((), dtype=bool)
        return jnp.zeros(points.shape[0], dtype=bool)

    # one point inside one obstacle
    def inside(point: Pos, obstacle: Obstacle):
        return obstacle.inside(point, r)

    # one point inside any obstacle
    def inside_any(point: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(inside, point))(obstacle).max()

    # any point inside any obstacle
    if points.ndim == 1:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros((), dtype=bool)
        is_in = inside_any(points, obstacles)
    else:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros(points.shape[0], dtype=bool)
        is_in = jax.vmap(ft.partial(inside_any, obstacle=obstacles))(points)

    return is_in


def raytracing(starts: Pos, ends: Pos, obstacles: Obstacle, max_returns: int) -> Pos:
    # if the start point if inside the obstacle, return the start point
    is_in = inside_obstacles(starts, obstacles)

    def raytracing_single(start: Pos, end: Pos, obstacle: Obstacle):
        return obstacle.raytracing(start, end)

    def raytracing_any(start: Pos, end: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(raytracing_single, start, end))(obstacle).min()

    if obstacles.center.shape[0] == 0:
        alphas = jnp.ones(starts.shape[0]) * 1e6
    else:
        alphas = jax.vmap(ft.partial(raytracing_any, obstacle=obstacles))(starts, ends)
        alphas *= (1 - is_in)

    # assert max_returns <= alphas.shape[0]
    alphas_return = jnp.argsort(alphas)[:max_returns]

    hitting_points = starts + (ends - starts) * (alphas[..., None])

    return hitting_points[alphas_return]


def get_node_goal_rng(
        key: PRNGKey,
        side_length: float,
        dim: int,
        n: int,
        min_dist: float,
        obstacles: Obstacle = None,
        min_travel: float = None,
        side_length_y: float = None,
        max_travel: float = None,
        side_length_z: float = None
) -> [Pos, Pos]:
    max_iter = 1024  # maximum number of iterations to find a valid initial state/goal
    states = jnp.zeros((n, dim))
    goals = jnp.zeros((n, dim))
    side_length_y = side_length if side_length_y is None else side_length_y
    side_length_z = side_length if side_length_z is None else side_length_z

    if dim == 2:
        max_side = jnp.array([side_length, side_length_y])
    elif dim == 3:
        max_side = jnp.array([side_length, side_length_y, side_length_z])
    else:
        raise NotImplementedError

    def get_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, this_key, _, all_nodes = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        return i_iter, this_key, jr.uniform(use_key, (dim,), minval=min_dist, maxval=max_side-min_dist), all_nodes

    def non_valid_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, _, node, all_nodes = reset_input
        # Only check against nodes that have been placed (non-zero)
        valid_nodes = jnp.any(all_nodes != 0, axis=1)
        # Calculate distances only to valid nodes
        distances = jnp.linalg.norm(all_nodes - node, axis=1)
        # Set distances to invalid nodes to a large value
        distances = jnp.where(valid_nodes, distances, jnp.ones_like(distances) * 1e10)
        dist_min = jnp.min(distances)
        collide = dist_min < min_dist
        inside = inside_obstacles(node, obstacles, r=min_dist)
        valid = ~(collide | inside) | (i_iter >= max_iter)
        return ~valid

    def get_goal(reset_input: Tuple[int, Array, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals, all_states
        i_iter, this_key, _, agent, all_goals, all_states = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        if max_travel is None:
            return (i_iter, this_key,
                    jr.uniform(use_key, (dim,), minval=min_dist, maxval=max_side-min_dist),
                    agent, all_goals, all_states)
        else:
            return (i_iter, this_key, 
                   jr.uniform(use_key, (dim,), minval=min_dist, maxval=max_travel-min_dist) + agent, 
                   agent, all_goals, all_states)

    def non_valid_goal(reset_input: Tuple[int, Array, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals, all_states
        i_iter, _, goal, agent, all_goals, all_states = reset_input
        
        # Check distance to other valid goals (non-zero)
        valid_goals = jnp.any(all_goals != 0, axis=1)
        goal_distances = jnp.linalg.norm(all_goals - goal, axis=1)
        goal_distances = jnp.where(valid_goals, goal_distances, jnp.ones_like(goal_distances) * 1e10)
        dist_min_goals = jnp.min(goal_distances)
        collide_with_goals = dist_min_goals < min_dist
        
        # Check distance to all valid agents (non-zero)
        valid_agents = jnp.any(all_states != 0, axis=1)
        agent_distances = jnp.linalg.norm(all_states - goal, axis=1)
        agent_distances = jnp.where(valid_agents, agent_distances, jnp.ones_like(agent_distances) * 1e10)
        dist_min_agents = jnp.min(agent_distances)
        # min_travel is optional; without it there is no minimum start-to-goal
        # separation to enforce (comparing against None raises under trace)
        if min_travel is None:
            collide_with_agents = jnp.array(False)
        else:
            collide_with_agents = dist_min_agents < min_travel


        # Combine collision checks
        collide = collide_with_goals | collide_with_agents
        
        inside = inside_obstacles(goal, obstacles, r=min_dist)
        outside = jnp.any(goal < min_dist) | jnp.any(goal > max_side-min_dist)
        if max_travel is None:
            too_long = np.array(False, dtype=bool)
        else:
            too_long = jnp.linalg.norm(goal - agent) > max_travel
        
        valid = (~collide & ~inside & ~outside & ~too_long) | (i_iter >= max_iter)
        out = ~valid
        assert out.shape == tuple() and out.dtype == jnp.bool_
        return out

    def reset_body(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        agent_key, goal_key, this_key = jr.split(this_key, 3)
        agent_candidate = jr.uniform(agent_key, (dim,), minval=min_dist, maxval=max_side-min_dist)
        n_iter_agent, _, agent_candidate, _ = while_loop(
            cond_fun=non_valid_node, body_fun=get_node,
            init_val=(0, agent_key, agent_candidate, all_states)
        )
        all_states = all_states.at[agent_id].set(agent_candidate)

        if max_travel is None:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=min_dist, maxval=max_side-min_dist)
        else:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=min_dist, maxval=max_travel-min_dist) + agent_candidate

        n_iter_goal, _, goal_candidate, _, _, _ = while_loop(
            cond_fun=non_valid_goal, body_fun=get_goal,
            init_val=(0, goal_key, goal_candidate, agent_candidate, all_goals, all_states)
        )
        all_goals = all_goals.at[agent_id].set(goal_candidate)
        agent_id += 1

        # if no solution is found, start over
        agent_id = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * agent_id
        all_states = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_states
        all_goals = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_goals

        return agent_id, this_key, all_states, all_goals

    def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        return agent_id < n

    _, _, states, goals = while_loop(
        cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals))

    return states, goals



def get_node_goal_rng_fixedstate(
        key: PRNGKey,
        side_length: float,
        dim: int,
        n: int,
        min_dist: float,
        obstacles: Obstacle = None,
        min_travel: float = None,
        side_length_y: float = None,
        max_travel: float = None,
        side_length_z: float = None,
        state_fixed: Array = None,
        goal_override: Array = None
) -> [Pos, Pos]:
    max_iter = 1024  # maximum number of iterations to find a valid initial state/goal
    states = jnp.zeros((n, dim))
    goals = jnp.zeros((n, dim))
    side_length_y = side_length if side_length_y is None else side_length_y
    side_length_z = side_length if side_length_z is None else side_length_z

    if dim == 2:
        max_side = jnp.array([side_length, side_length_y])
    elif dim == 3:
        max_side = jnp.array([side_length, side_length_y, side_length_z])
    else:
        raise NotImplementedError

    # 1) Fix states to provided value or default [0.2, 0.2, ...]
    if state_fixed is None:
        base = jnp.array([0.2, 0.2] + ([0.0] * max(0, dim - 2)), dtype=jnp.float32)[:dim]
    else:
        base = jnp.array(state_fixed, dtype=jnp.float32)
        base = base[:dim] if base.ndim == 1 else base.reshape(-1)[:dim]
    states = jnp.tile(base[None, :], (n, 1))

    # 2) If user provided explicit goals, use them directly (broadcast if needed)
    if goal_override is not None:
        goals_arr = jnp.array(goal_override, dtype=jnp.float32)
        if goals_arr.ndim == 1:
            goals_arr = jnp.tile(goals_arr[None, :], (n, 1))
        # Optionally clip inside arena margins
        goals_arr = jnp.clip(goals_arr, a_min=min_dist, a_max=max_side - min_dist)
        return states, goals_arr

    def get_goal(reset_input: Tuple[int, Array, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals, all_states
        i_iter, this_key, _, agent, all_goals, all_states = reset_input
        # Sample a random direction and set goal at fixed 2.0m from agent (first two dims)
        ang_key, this_key = jr.split(this_key, 2)
        theta = jr.uniform(ang_key, (), minval=0.0, maxval=2.0 * jnp.pi)
        dir2 = jnp.array([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)
        goal = jnp.zeros_like(agent)
        goal = goal.at[:2].set(agent[:2] + 2.0 * dir2)
        # Keep higher dims (if any) equal to agent to satisfy shape
        if dim > 2:
            goal = goal.at[2:].set(agent[2:])
        i_iter += 1
        return (i_iter, this_key, goal, agent, all_goals, all_states)

    def non_valid_goal(reset_input: Tuple[int, Array, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals, all_states
        i_iter, _, goal, agent, all_goals, all_states = reset_input
        
        # Check distance to other valid goals (non-zero)
        valid_goals = jnp.any(all_goals != 0, axis=1)
        goal_distances = jnp.linalg.norm(all_goals - goal, axis=1)
        goal_distances = jnp.where(valid_goals, goal_distances, jnp.ones_like(goal_distances) * 1e10)
        dist_min_goals = jnp.min(goal_distances)
        collide_with_goals = dist_min_goals < min_dist

        # Enforce fixed distance 2.0 from the agent that owns this goal (first two dims)
        dist_to_agent = jnp.linalg.norm(goal[:2] - agent[:2])
        fixed_dist_ok = jnp.abs(dist_to_agent - 2.0) < 1e-3

        # Combine collision checks
        collide = collide_with_goals
        
        # Require at least 0.3m clearance from obstacles
        inside = inside_obstacles(goal, obstacles, r= min_dist)
        outside = jnp.any(goal < min_dist) | jnp.any(goal > max_side-min_dist)
        too_long = np.array(False, dtype=bool)
        
        valid = (fixed_dist_ok & ~collide & ~inside & ~outside & ~too_long) | (i_iter >= max_iter)
        out = ~valid
        assert out.shape == tuple() and out.dtype == jnp.bool_
        return out

    def reset_body(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        # For fixed state, take the pre-filled state as agent position
        agent_pos = all_states[agent_id]
        _, goal_key = jr.split(this_key, 2)
        # Initial candidate at fixed distance 2.0
        theta0 = jr.uniform(goal_key, (), minval=0.0, maxval=2.0 * jnp.pi)
        dir2_0 = jnp.array([jnp.cos(theta0), jnp.sin(theta0)], dtype=jnp.float32)
        goal_candidate = jnp.zeros((dim,), dtype=jnp.float32).at[:2].set(agent_pos[:2] + 2.0 * dir2_0)
        if dim > 2:
            goal_candidate = goal_candidate.at[2:].set(agent_pos[2:])

        n_iter_goal, _, goal_candidate, _, _, _ = while_loop(
            cond_fun=non_valid_goal, body_fun=get_goal,
            init_val=(0, goal_key, goal_candidate, agent_pos, all_goals, all_states)
        )
        all_goals = all_goals.at[agent_id].set(goal_candidate)
        agent_id += 1

        # if no solution is found, start over
        agent_id = (1 - (n_iter_goal >= max_iter)) * agent_id
        all_states = (1 - (n_iter_goal >= max_iter)) * all_states
        all_goals = (1 - (n_iter_goal >= max_iter)) * all_goals

        return agent_id, this_key, all_states, all_goals

    def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        return agent_id < n

    _, _, states, goals = while_loop(
        cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals))

    return states, goals
