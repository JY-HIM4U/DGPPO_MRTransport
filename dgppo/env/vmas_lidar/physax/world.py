import functools as ft
from typing import Callable, Optional

import ipdb
import jax
import jax.debug as jd
import jax.numpy as jnp
import numpy as np
from typing_extensions import Self
from jax import jit

from .entity import Agent, Entity
from .geometry import get_closest_point_box, get_closest_point_line
from .jax_types import Pos2, Vec1, Vec2
from .shapes import Box, Line, Shape, Sphere, Object
from .vmas_utils import clamp_with_norm, compute_torque


class Default:
    LINE_MIN_DIST = 4 / 6e2

    COLLISION_FORCE = 100
    JOINT_FORCE = 130
    TORQUE_CONSTRAINT_FORCE = 1

    DRAG = 0.0
    LINEAR_FRICTION = 0.0
    ANGULAR_FRICTION = 0.0


class World:

    def __init__(
        self,
        dt: float = 0.1,
        substeps: int = 1,  # if you use joints, higher this value to gain simulation stability
        drag: float = Default.DRAG,
        linear_friction: float = Default.LINEAR_FRICTION,
        angular_friction: float = Default.ANGULAR_FRICTION,
        x_semidim: float = None,
        y_semidim: float = None,
        gravity=np.array([0.0, 0.0]),
        collision_force: float = Default.COLLISION_FORCE,
        torque_constraint_force: float = Default.TORQUE_CONSTRAINT_FORCE,
        contact_margin: float = 1e-3,
        real_num_agents: int = 3,
        num_agents: int = 3,  # Add parameter to pass number of agents to World
        stiffness: float = 0.3
    ):
        # world dims: no boundaries if none
        self._x_semidim = x_semidim
        self._y_semidim = y_semidim

        self._dt = dt
        self.substeps = substeps
        self.sub_dt = self._dt / self.substeps

        # drag coefficient
        self.drag = drag
        # gravity
        self.gravity = gravity
        # friction coefficients
        self.linear_friction = linear_friction
        self.angular_friction = angular_friction

        # constraint response parameters
        self._collision_force = collision_force
        self._contact_margin = contact_margin
        self._torque_constraint_force = torque_constraint_force

        # Pairs of collidable shapes
        self._collidable_pairs = [
            {Sphere, Sphere},
            {Sphere, Box},
            {Sphere, Line},
            {Line, Line},
            {Line, Box},
            {Box, Box},
        ]
        self.real_num_agents = real_num_agents
        self.num_agents = num_agents  # Store number of agents as instance variable
        self.stiffness = stiffness

    @ft.partial(jax.jit, static_argnames=["self"])
    def step(self, entities: list[Entity]):
        info = None

        for substep in range(self.substeps):
            forces_dict = {e: jnp.zeros(2) for e in entities}
            torques_dict = {e: jnp.zeros(1) for e in entities}

            for entity in entities:
                if isinstance(entity, Agent):
                    # apply agent force controls = collect (clamped) forces from agent.state
                    apply_action_force(entity, forces_dict, torques_dict)
                    # apply agent torque controls = collect (clamped) torques from agent.state
                    apply_action_torque(entity, forces_dict, torques_dict)

                # apply friction = collect friction forces from entity.state
                apply_friction_force(
                    entity, self.linear_friction, self.angular_friction, self.sub_dt, forces_dict, torques_dict
                )
                # apply gravity
                apply_gravity(entity, self.gravity, forces_dict, torques_dict)

            info = self._apply_vectorized_enviornment_force(entities, forces_dict, torques_dict)

            # integrate physical state
            entities_new = [self._integrate_state(entity, substep, forces_dict, torques_dict) for entity in entities]
            entities = entities_new

        return entities, info

    # Temporarily disable JIT for debugging
    def _integrate_state_pos(self, entity: Entity, substep: int, forces_dict: dict[Entity, Vec2]) -> Entity:
        # Compute translation
        vel = entity.state.vel
        if substep == 0:
            if entity.drag is not None:
                vel = vel * (1 - entity.drag)
            else:
                vel = vel * (1 - self.drag)
        
        # Calculate acceleration
        force = forces_dict.get(entity, jnp.zeros(2))
        accel = force / entity.mass

        # Debug the velocity update step by step
        vel_update = accel * self.sub_dt
        
        # Store the old velocity for comparison
        old_vel = vel
        vel = vel + vel_update
        
        if entity.max_speed is not None:
            vel = clamp_with_norm(vel, entity.max_speed)
        if entity.v_range is not None:
            vel = vel.clip(-entity.v_range, entity.v_range)
        new_pos = entity.state.pos + vel * self.sub_dt

        new_pos_x = new_pos[..., 0]
        new_pos_y = new_pos[..., 1]

        if self._x_semidim is not None:
            # new_pos_x = new_pos_x.clip(-self._x_semidim, self._x_semidim)
            new_pos_x = new_pos_x.clip(0, self._x_semidim)
        if self._y_semidim is not None:
            # new_pos_y = new_pos_y.clip(-self._y_semidim, self._y_semidim)
            new_pos_y = new_pos_y.clip(0, self._y_semidim)

        new_pos = jnp.stack([new_pos_x, new_pos_y], axis=-1)
        return entity.withstate(pos=new_pos, vel=vel)

    # Remove JIT decorator if present
    # _integrate_state_pos = jit(_integrate_state_pos)  # Comment this out for debugging

    def _integrate_state_rot(self, entity: Entity, substep: int, torques_dict: dict[Entity, Vec2]):
        # Compute rotation
        ang_vel = entity.state.ang_vel

        if substep == 0:
            # This is static.
            if entity.drag is not None:
                ang_vel = ang_vel * (1 - entity.drag)
            else:
                ang_vel = ang_vel * (1 - self.drag)
        ang_vel = ang_vel + (torques_dict[entity] / entity.moment_of_inertia) * self.sub_dt
        if entity.max_angvel is not None:
            ang_vel = clamp_with_norm(ang_vel, entity.max_angvel)
        rot_new = entity.state.rot + ang_vel * self.sub_dt
        # if isinstance(entity.shape, Object):
        #     jax.debug.print("Object force: {}", ang_vel)
        #     jax.debug.print("Object force: {}", torques_dict[entity])
        #     jax.debug.print("Object mass: {}", entity.moment_of_inertia)
        #     jax.debug.print("Object acceleration: {}", rot_new)

        return entity.withstate(rot=rot_new, ang_vel=ang_vel)

    def _integrate_state(
        self, entity: Entity, substep: int, forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec2]
    ):
        if entity.movable:
            entity = self._integrate_state_pos(entity, substep, forces_dict)

        if entity.rotatable:
            entity = self._integrate_state_rot(entity, substep, torques_dict)

        return entity

    def should_check_collision(self, a: Entity, b: Entity) -> bool:
        """Check if we should do collision checking between the two entities. Everything here should be static."""
        if a is b:
            # Static path.
            return False

        # Still static path.
        a_collides_b = a.collides(b)
        b_collides_a = b.collides(a)
        if not a_collides_b or not b_collides_a:
            return False

        a_shape = a.shape
        b_shape = b.shape
        if not a.movable and not a.rotatable and not b.movable and not b.rotatable:
            return False
        if not {a_shape.__class__, b_shape.__class__} in self._collidable_pairs:
            return False

        # In jit, we can't do this performance optimization.
        # if not (
        #     torch.linalg.vector_norm(a.state.pos - b.state.pos, dim=-1)
        #     <= a.shape.circumscribed_radius() + b.shape.circumscribed_radius()
        # ).any():
        #     return False

        return True

    def _apply_vectorized_enviornment_force(
        self, entities: list[Entity], forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]
    ):
        # Hold a list of all the pairs of objects that need checking.
        s_s: list[tuple[Entity, Entity]] = []
        l_s: list[tuple[Entity, Entity]] = []
        b_s: list[tuple[Entity, Entity]] = []
        l_l: list[tuple[Entity, Entity]] = []
        b_l: list[tuple[Entity, Entity]] = []
        b_b: list[tuple[Entity, Entity]] = []
        a_o: list[tuple[Entity, Entity]] = []

        # Setting up the collision pairs. ALL STATIC.
        for a, entity_a in enumerate(entities):
            for b, entity_b in enumerate(entities):
                if b <= a:
                    continue
                # Check if entity_a is collideable with entity_b

                # Add the pair to the correct list.
                if isinstance(entity_a.shape, Sphere) and isinstance(entity_b.shape, Sphere):
                    s_s.append((entity_a, entity_b))
                
                elif isinstance(entity_a.shape, Sphere) and isinstance(entity_b.shape, Object):                    
                    a_o.append((entity_a, entity_b))
                elif isinstance(entity_a.shape, Object) and isinstance(entity_b.shape, Sphere):                    
                    a_o.append((entity_b, entity_a))
                else:
                    raise AssertionError()

        contact_forces_dict = {}
        contact_torques_dict = {}
        
        self._agent_object_interaction(a_o, contact_forces_dict, contact_torques_dict)
        # self._sphere_sphere_collision(s_s, contact_forces_dict, contact_torques_dict)
        # self._sphere_line_collision(l_s, contact_forces_dict, contact_torques_dict)
        # # self._line_line_collision(l_l, forces_dict, torques_dict)
        # self._box_sphere_collision(b_s, contact_forces_dict, contact_torques_dict)
        # self._box_line_collision(b_l, forces_dict, torques_dict)
        # self._box_box_collision(b_b, forces_dict, torques_dict)

        # Add the contact forces and contact torques to the main forces and torques.
        for entity, force in contact_forces_dict.items():
            forces_dict[entity] = forces_dict[entity] + force
        for entity, torque in contact_torques_dict.items():
            torques_dict[entity] = torques_dict[entity] + torque

        # Instead of indexed by entity, index it by the name.
        contact_forces_dict = {entity.name: force for entity, force in contact_forces_dict.items()}
        contact_torques_dict = {entity.name: torque for entity, torque in contact_torques_dict.items()}

        # Return the total contact forces and torques for each agent.
        return {"contact_forces": contact_forces_dict, "contact_torques": contact_torques_dict}
    def _agent_object_interaction(
        self,
        a_o: list[tuple[Entity, Entity]],
        forces_dict: dict[Entity, jnp.ndarray],
        torques_dict: dict[Entity, jnp.ndarray],
    ):
        if len(a_o) == 0:
            return

        # Precompute arrays for all pairs
        n_pairs = len(a_o)
        # Extract agent indices (assumed precomputed from agent names)
        agent_indices = jnp.array([int(agent.name.split('_')[-1]) for agent, _ in a_o], dtype=jnp.int32)
        # (n_pairs, 2)
        agent_pos = jnp.stack([agent.state.pos for agent, _ in a_o], axis=0)
        obj_pos = jnp.stack([obj.state.pos for _, obj in a_o], axis=0)
        obj_angle = jnp.array([obj.state.rot[0] for _, obj in a_o], dtype=jnp.float32)
        obj_length = jnp.array([obj.shape.length for _, obj in a_o], dtype=jnp.float32)

        # Compute vertices for each object.
        # Base angles: fixed shape (self.num_agents,)
        base_angles = jnp.arange(self.num_agents) * (2 * jnp.pi / self.real_num_agents)
        # Broadcast to each pair: shape (n_pairs, self.num_agents)
        base_angles = jnp.broadcast_to(base_angles, (n_pairs, self.num_agents))
        # Expand obj_angle to (n_pairs, 1)
        obj_angle_exp = jnp.expand_dims(obj_angle, -1)
        # Compute vertices for each pair: shape (n_pairs, self.num_agents, 2)
        vertices = obj_pos[:, None, :] + obj_length[:, None, None] * jnp.stack(
            [jnp.cos(obj_angle_exp + base_angles), jnp.sin(obj_angle_exp + base_angles)],
            axis=-1,
        )

        # Gather the vertex corresponding to each agent's index.
        # chosen_vertices: shape (n_pairs, 2)
        chosen_vertices = vertices[jnp.arange(n_pairs), agent_indices]

        # Compute the spring forces and torque vectorized.
        # stiffness = self.stiffness
        delta = agent_pos - chosen_vertices  # (n_pairs, 2)
        dist = jnp.linalg.norm(delta, axis=-1, keepdims=True)  # (n_pairs, 1)
        force_on_vertex = self.stiffness * dist * (delta / (dist + 1e-8))  # (n_pairs, 2)
        force_on_agent = -force_on_vertex

        # Compute torque via 2D cross product: scalar per pair
        r_vector = chosen_vertices - obj_pos  # (n_pairs, 2)
        torque = r_vector[:, 0] * force_on_vertex[:, 1] - r_vector[:, 1] * force_on_vertex[:, 0]  # (n_pairs,)

        # Create a mask: valid if agent index is less than self.real_num_agents.
        valid_mask = (agent_indices < self.real_num_agents).astype(jnp.float32)  # shape (n_pairs,)
        
        # Apply mask to forces and torque.e
        force_on_agent = force_on_agent * valid_mask[:, None]
        force_on_vertex = force_on_vertex * valid_mask[:, None]
        # jax.debug.print("force_on_agent: {x}", x=force_on_agent)
        torque = torque * valid_mask

        # Update forces and torques dictionaries in a final (non-jitted) pass.
        for agent, obj in (a_o):
            # jax.debug.print("agent: {x}", x=agent)
            i = int(agent.name.split('_')[-1])  # Extract index from agent's name
            if agent.movable:
                forces_dict[agent] = forces_dict.get(agent, jnp.zeros(2, dtype=jnp.float32)) + force_on_agent[i]
            if obj.movable:
                forces_dict[obj] = forces_dict.get(obj, jnp.zeros(2, dtype=jnp.float32)) + force_on_vertex[i]
            if obj.rotatable:
                torques_dict[obj] = torques_dict.get(obj, jnp.zeros(1, dtype=jnp.float32)) + jnp.array([torque[i]], dtype=jnp.float32)
        
    # def _agent_object_interaction(
    #     self, a_o: list[tuple[Entity, Entity]], forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]
    # ):
    #     if len(a_o) == 0:
    #         return

    #     for agent, obj in a_o:
    #         agent_pos = agent.state.pos
    #         object_pos = obj.state.pos
    #         object_angle = obj.state.rot[0]
    #         object_length = obj.shape.length

    #         # Calculate the positions of the vertices
    #         base_angles = jnp.array([i * 2 * jnp.pi / self.real_num_agents for i in range(self.num_agents)])
    #         vertices = object_pos + object_length * jnp.stack(
    #             [jnp.cos(object_angle + base_angles), jnp.sin(object_angle + base_angles)], axis=-1
    #         )

    #         # Use the agent's name to determine the index
    #         agent_index = int(agent.name.split('_')[-1])  # Extract index from agent's name
    #         if agent_index >= self.real_num_agents:
    #             continue
    #         vertex_pos = vertices[agent_index]

    #         # Calculate spring force for this agent
    #         stiffness = 10.0
    #         delta_pos = agent_pos - vertex_pos
    #         dist = jnp.linalg.norm(delta_pos)

    #         # Force direction (normalized delta_pos)
    #         force_direction = delta_pos / (dist + 1e-8)

    #         # Calculate force
    #         force_magnitude = stiffness * dist
    #         force_on_vertex = force_magnitude * force_direction
    #         force_on_agent = -force_on_vertex

    #         # Calculate torque (2D cross product)
    #         r_vector = vertex_pos - object_pos
    #         torque = r_vector[0] * force_on_vertex[1] - r_vector[1] * force_on_vertex[0]

    #         # Apply forces and torques
    #         if agent.movable:
    #             if agent not in forces_dict:
    #                 forces_dict[agent] = jnp.zeros(2, dtype=jnp.float32)
    #             forces_dict[agent] = forces_dict[agent] + force_on_agent

    #         if obj.movable:
    #             if obj not in forces_dict:
    #                 forces_dict[obj] = jnp.zeros(2, dtype=jnp.float32)
    #             forces_dict[obj] = forces_dict[obj] + force_on_vertex

    #         if obj.rotatable:
    #             if obj not in torques_dict:
    #                 torques_dict[obj] = jnp.zeros(1, dtype=jnp.float32)
    #             torques_dict[obj] = torques_dict[obj] + jnp.array([torque], dtype=jnp.float32)

    def _sphere_sphere_collision(
        self, s_s: list[tuple[Entity, Entity]], forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]
    ):
        if len(s_s) == 0:
            return

        pos_s_a = []
        pos_s_b = []
        radius_s_a = []
        radius_s_b = []
        for s_a, s_b in s_s:
            a_shape: Sphere = s_a.shape
            b_shape: Sphere = s_b.shape
            assert isinstance(a_shape, Sphere)
            assert isinstance(b_shape, Sphere)

            pos_s_a.append(s_a.state.pos)
            pos_s_b.append(s_b.state.pos)
            radius_s_a.append(a_shape.radius)
            radius_s_b.append(b_shape.radius)

        # (..., n_pairs, 2)
        pos_s_a = jnp.stack(pos_s_a, axis=-2)
        pos_s_b = jnp.stack(pos_s_b, axis=-2)

        # (n_pairs, ). This should just be a vector.
        radius_s_a = np.array(radius_s_a)
        radius_s_b = np.array(radius_s_b)

        force_a, force_b = self._get_constraint_forces(
            pos_s_a,
            pos_s_b,
            dist_min=radius_s_a + radius_s_b,
            force_multiplier=self._collision_force,
        )

        for i, (entity_a, entity_b) in enumerate(s_s):
            self.update_env_forces(entity_a, force_a[:, i], 0, entity_b, force_b[:, i], 0, forces_dict, torques_dict)

    def _sphere_line_collision(
        self, l_s: list[tuple[Entity, Entity]], forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]
    ):
        if len(l_s) == 0:
            return

        pos_l = []
        pos_s = []
        rot_l = []
        radius_s = []
        length_l = []
        for line, sphere in l_s:
            line_shape: Line = line.shape
            sphere_shape = sphere.shape
            assert isinstance(line_shape, Line)
            assert isinstance(sphere_shape, Sphere)

            pos_l.append(line.state.pos)
            pos_s.append(sphere.state.pos)
            rot_l.append(line.state.rot)
            radius_s.append(sphere_shape.radius)
            length_l.append(line_shape.length)

        n_objects = len(pos_l)

        # (..., n_pairs, 2)
        pos_l = jnp.stack(pos_l, axis=-2)
        pos_s = jnp.stack(pos_s, axis=-2)
        rot_l = jnp.stack(rot_l, axis=-2)

        # (n_pairs, ). This should just be a vector.
        radius_s = np.array(radius_s)
        length_l = np.array(length_l)

        closest_point = get_closest_point_line(pos_l, rot_l, length_l, pos_s)
        # (n_objects, 2)
        force_sphere, force_line = self._get_constraint_forces(
            pos_s,
            closest_point,
            dist_min=radius_s + Default.LINE_MIN_DIST,
            force_multiplier=self._collision_force,
        )
        r = closest_point - pos_l
        torque_line = compute_torque(force_line, r)

        assert force_sphere.shape == force_line.shape == (n_objects, 2)
        assert torque_line.shape == (n_objects, 1)

        for ii, (line, sphere) in enumerate(l_s):
            update_forcetorque(line, force_line[..., ii, :], torque_line[..., ii, :], forces_dict, torques_dict)
            update_forcetorque(sphere, force_sphere[..., ii, :], 0, forces_dict, torques_dict)

    def _box_sphere_collision(
        self, b_s: list[tuple[Entity, Entity]], forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]
    ):
        if len(b_s) == 0:
            return

        pos_box = []
        pos_sphere = []
        rot_box = []
        length_box = []
        width_box = []
        not_hollow_box = []
        radius_sphere = []
        for box, sphere in b_s:
            sphere_shape: Sphere = sphere.shape
            box_shape: Box = box.shape

            pos_box.append(box.state.pos)
            pos_sphere.append(sphere.state.pos)
            rot_box.append(box.state.rot)
            length_box.append(box_shape.length)
            width_box.append(box_shape.width)
            not_hollow_box.append(not box_shape.hollow)
            radius_sphere.append(sphere_shape.radius)

        pos_box = jnp.stack(pos_box, axis=-2)
        pos_sphere = jnp.stack(pos_sphere, axis=-2)
        rot_box = jnp.stack(rot_box, axis=-2)
        length_box = np.array(length_box)
        width_box = np.array(width_box)
        not_hollow_box_prior = np.array(not_hollow_box)
        not_hollow_box = not_hollow_box_prior

        radius_sphere = np.array(radius_sphere)

        closest_point_box = get_closest_point_box(
            pos_box,
            rot_box,
            width_box,
            length_box,
            pos_sphere,
        )

        inner_point_box = closest_point_box
        d = jnp.zeros_like(radius_sphere)

        # assert all hollow box.
        assert not not_hollow_box.any()
        # if not_hollow_box_prior.any():
        #     inner_point_box_hollow, d_hollow = get_inner_point_box(
        #         pos_sphere, closest_point_box, pos_box
        #     )
        #     cond = not_hollow_box.unsqueeze(-1).expand(inner_point_box.shape)
        #     inner_point_box = torch.where(
        #         cond, inner_point_box_hollow, inner_point_box
        #     )
        #     d = torch.where(not_hollow_box, d_hollow, d)

        force_sphere, force_box = self._get_constraint_forces(
            pos_sphere,
            inner_point_box,
            dist_min=radius_sphere + Default.LINE_MIN_DIST + d,
            force_multiplier=self._collision_force,
        )
        r = closest_point_box - pos_box
        torque_box = compute_torque(force_box, r)

        for ii, (box, sphere) in enumerate(b_s):
            update_forcetorque(box, force_box[..., ii, :], torque_box[..., ii, :], forces_dict, torques_dict)
            update_forcetorque(sphere, force_sphere[..., ii, :], 0, forces_dict, torques_dict)
            # self.update_env_forces(
            #     entity_a,
            #     force_box[:, i],
            #     torque_box[:, i],
            #     entity_b,
            #     force_sphere[:, i],
            #     0,
            # )

    def _get_constraint_forces(
        self,
        pos_a: Pos2,
        pos_b: Pos2,
        dist_min,
        force_multiplier: float,
        attractive: bool = False,
    ) -> Vec2:
        min_dist = 1e-6
        delta_pos = pos_a - pos_b
        dist = jnp.linalg.vector_norm(delta_pos, axis=-1)
        sign = -1 if attractive else 1

        # softmax penetration
        k = self._contact_margin
        penetration = jnp.logaddexp(0.0, (dist_min - dist) * sign / k) * k
        force = (
            sign * force_multiplier * delta_pos / jnp.where(dist > 0, dist, 1e-8)[..., None] * penetration[..., None]
        )
        force = jnp.where((dist < min_dist)[..., None], 0.0, force)
        if not attractive:
            force = jnp.where((dist > dist_min)[..., None], 0.0, force)
        else:
            force = jnp.where((dist < dist_min)[..., None], 0.0, force)

        n_objects = len(pos_a)
        assert pos_a.shape == pos_b.shape
        assert force.shape == (n_objects, 2)
        return force, -force

    # def update_env_forces(
    #     self,
    #     entity_a: Entity,
    #     f_a,
    #     t_a,
    #     entity_b: Entity,
    #     f_b,
    #     t_b,
    #     forces_dict: dict[Entity, Vec2],
    #     torques_dict: dict[Entity, Vec1],
    # ):
    #     if entity_a.movable:
    #         forces_dict[entity_a] = forces_dict[entity_a] + f_a
    #     if entity_a.rotatable:
    #         torques_dict[entity_a] = torques_dict[entity_a] + t_a
    #     if entity_b.movable:
    #         forces_dict[entity_b] = forces_dict[entity_b] + f_b
    #     if entity_b.rotatable:
    #         torques_dict[entity_b] = torques_dict[entity_b] + t_b
    #


def apply_action_force(agent: Agent, forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]):
    if not agent.movable:
        return

    force = agent.state.force

    if agent.max_f is not None:
        force = clamp_with_norm(force, agent.max_f)

    if agent.f_range is not None:
        force = jnp.clip(force, -agent.f_range, agent.f_range)

    forces_dict[agent] = forces_dict[agent] + force


def apply_action_torque(agent: Agent, forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]):
    if not agent.movable:
        return
    torque = agent.state.torque

    if agent.max_t is not None:
        torque = clamp_with_norm(torque, agent.max_t)

    if agent.t_range is not None:
        torque = jnp.clip(torque, -agent.t_range, agent.t_range)

    torques_dict[agent] = torques_dict[agent] + torque


def apply_gravity(
    entity: Entity, gravity_world: Vec2, forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]
):
    if not entity.movable:
        return

    if not (gravity_world == 0.0).all():
        forces_dict[entity] = forces_dict[entity] + entity.mass * gravity_world
    if entity.gravity is not None:
        forces_dict[entity] = forces_dict[entity] + entity.mass * entity.gravity


def apply_friction_force(
    entity: Entity,
    world_linear_friction: float,
    world_angular_friction: float,
    sub_dt: float,
    forces_dict: dict[Entity, Vec2],
    torques_dict: dict[Entity, Vec1],
):
    def get_friction_force(vel: Vec2, coeff, mass):
        speed = jnp.linalg.vector_norm(vel, axis=-1)
        is_static = speed == 0

        friction_force_constant = coeff * mass

        vel_denom = jnp.where(is_static, 1e-8, speed)
        friction_force = -(vel / vel_denom[:, None]) * jnp.minimum(friction_force_constant, (vel.abs() / sub_dt) * mass)
        friction_force = jnp.where(is_static[:, None], 0.0, friction_force)

        return friction_force

    if entity.linear_friction is not None:
        forces_dict[entity] = forces_dict[entity] + get_friction_force(
            entity.state.vel,
            entity.linear_friction,
            entity.mass,
        )
    elif world_linear_friction > 0:
        forces_dict[entity] = forces_dict[entity] + get_friction_force(
            entity.state.vel,
            world_linear_friction,
            entity.mass,
        )
    if entity.angular_friction is not None:
        torques_dict[entity] = torques_dict[entity] + get_friction_force(
            entity.state.ang_vel,
            entity.angular_friction,
            entity.moment_of_inertia,
        )
    elif world_angular_friction > 0:
        torques_dict[entity] = torques_dict[entity] + get_friction_force(
            entity.state.ang_vel,
            world_angular_friction,
            entity.moment_of_inertia,
        )


def update_forcetorque(entity: Entity, f, t, forces_dict: dict[Entity, Vec2], torques_dict: dict[Entity, Vec1]):
    if entity.movable:
        if entity in forces_dict:
            forces_dict[entity] = forces_dict[entity] + f
        else:
            forces_dict[entity] = f
    if entity.rotatable:
        if entity in torques_dict:
            torques_dict[entity] = torques_dict[entity] + t
        else:
            torques_dict[entity] = t
