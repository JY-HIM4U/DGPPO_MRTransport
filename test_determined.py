import argparse
import datetime
import functools as ft
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.utils import test_rollout
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.utils import jax_jit_np, jax_vmap
from dgppo.utils.typing import Array

def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    with open(os.path.join(args.path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
    )

    # create algorithm
    path = args.path
    model_path = os.path.join(path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
    else:
        step = args.step
    print("step: ", step)

    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.cost_weight,
        actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
        lr_actor=config.lr_actor,
        lr_Vl=config.lr_Vl,
        max_grad_norm=2.0,
        seed=config.seed,
        use_rnn=config.use_rnn,
        rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm,
    )
    algo.load(model_path, step)
    if args.stochastic:
        def act_fn(x, z, rnn_state, key):
            action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
            return action, new_rnn_state
        act_fn = jax.jit(act_fn)
    else:
        act_fn = algo.act
    act_fn = jax.jit(act_fn)
    init_rnn_state = algo.init_rnn_state

    # set up keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    # create rollout function
    rollout_fn = ft.partial(test_rollout,
                            env,
                            act_fn,
                            init_rnn_state,
                            stochastic=args.stochastic)
    rollout_fn = jax_jit_np(rollout_fn)

    def unsafe_mask(graph_: GraphsTuple) -> Array:
        cost = env.get_cost(graph_)
        return jnp.any(cost >= 0.0, axis=-1)

    is_unsafe_fn = jax_jit_np(jax_vmap(unsafe_mask))

    # test results
    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    rates = []

    # test
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        rollout = rollout_fn(key_x0)
        is_unsafes.append(is_unsafe_fn(rollout.graph))

        epi_reward = rollout.rewards.sum()
        epi_cost = rollout.costs.max()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, safe rate: {safe_rate * 100:.3f}%")

        rates.append(np.array(safe_rate))

    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()

    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%"
    )

    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f}\n")
        
        # Save rollout actions as CSV files
        actions_dir = os.path.join(path, "actions")
        if not os.path.exists(actions_dir):
            os.makedirs(actions_dir)
        
        for i_epi, rollout in enumerate(rollouts):
            # Convert rollout.actions from (num_steps, num_agents, action_dim) to DataFrame
            actions_array = np.array(rollout.actions)  # Shape: (num_steps, num_agents, action_dim)
            # print(f"rollout.graph.env_states.a_pos.shape: {rollout.graph.env_states.a_pos.shape}")
            # print(f"rollout.graph.env_states.a_vel.shape: {rollout.graph.env_states.a_vel.shape}")
            agent_position = rollout.graph.env_states.agent[:, :, :2]  # Shape: (num_steps, num_agents, 2)
            agent_velocity = rollout.graph.env_states.agent[:, :, 2:4]  # Shape: (num_steps, num_agents, 2)
            # print(f"rollout.graph.env_states.real_num_agents: {rollout.graph.env_states.real_num_agents}")
            # print(f"rollout.graph.env_states.a_pos[0]: {rollout.graph.env_states.a_pos[0]}")  # First timestep
            # print(f"rollout.graph.env_states.agent[0]: {rollout.graph.env_states.agent[0]}")  # First timestep
            num_steps, num_agents, action_dim = actions_array.shape
            
            # Reshape to (num_steps, num_agents * action_dim)
            actions_reshaped = actions_array.reshape(num_steps, num_agents * action_dim)
            
            # Create column names for actions
            action_column_names = []
            for agent in range(num_agents):
                for dim in range(action_dim):
                    action_column_names.append(f"agent{agent}_action{dim}")
            
            # Create DataFrame for actions
            df_actions = pd.DataFrame(actions_reshaped, columns=action_column_names)
            
            # Save actions as CSV
            csv_filename = f"episode_{i_epi:02d}_actions.csv"
            csv_path = os.path.join(actions_dir, csv_filename)
            df_actions.to_csv(csv_path, index=False)
            print(f"Saved actions to: {csv_path}")
            
            # Save positions and velocities
            # Reshape positions and velocities
            positions_reshaped = agent_position.reshape(num_steps, num_agents * 2)
            velocities_reshaped = agent_velocity.reshape(num_steps, num_agents * 2)
            
            # Create column names for positions and velocities
            pos_column_names = []
            vel_column_names = []
            for agent in range(num_agents):
                pos_column_names.extend([f"agent{agent}_pos_x", f"agent{agent}_pos_y"])
                vel_column_names.extend([f"agent{agent}_vel_x", f"agent{agent}_vel_y"])
            
            # Create DataFrames
            df_positions = pd.DataFrame(positions_reshaped, columns=pos_column_names)
            df_velocities = pd.DataFrame(velocities_reshaped, columns=vel_column_names)
            
            # Save positions and velocities as CSV
            pos_csv_filename = f"episode_{i_epi:02d}_positions.csv"
            vel_csv_filename = f"episode_{i_epi:02d}_velocities.csv"
            pos_csv_path = os.path.join(actions_dir, pos_csv_filename)
            vel_csv_path = os.path.join(actions_dir, vel_csv_filename)
            df_positions.to_csv(pos_csv_path, index=False)
            df_velocities.to_csv(vel_csv_path, index=False)
            print(f"Saved positions to: {pos_csv_path}")
            print(f"Saved velocities to: {vel_csv_path}")
            
            # Create comprehensive CSV with all data (actions, positions, velocities)
            comprehensive_data = np.concatenate([
                actions_reshaped,      # Actions: (num_steps, num_agents * action_dim)
                positions_reshaped,    # Positions: (num_steps, num_agents * 2)
                velocities_reshaped    # Velocities: (num_steps, num_agents * 2)
            ], axis=1)
            
            # Create comprehensive column names
            comprehensive_column_names = []
            comprehensive_column_names.extend(action_column_names)  # Actions
            comprehensive_column_names.extend(pos_column_names)     # Positions  
            comprehensive_column_names.extend(vel_column_names)     # Velocities
            
            # Create comprehensive DataFrame
            df_comprehensive = pd.DataFrame(comprehensive_data, columns=comprehensive_column_names)
            
            # Save comprehensive CSV
            comprehensive_csv_filename = f"episode_{i_epi:02d}_comprehensive.csv"
            comprehensive_csv_path = os.path.join(actions_dir, comprehensive_csv_filename)
            df_comprehensive.to_csv(comprehensive_csv_path, index=False)
            print(f"Saved comprehensive data to: {comprehensive_csv_path}")
            
            # Create and save action plots
            fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3*num_agents))
            if num_agents == 1:
                axes = [axes]  # Make it iterable for single agent
            
            timesteps = np.arange(num_steps)
            
            for agent in range(num_agents):
                ax = axes[agent]
                for dim in range(action_dim):
                    ax.plot(timesteps, actions_array[:, agent, dim], 
                           label=f'Action {dim}', linewidth=1.5)
                
                ax.set_title(f'Agent {agent} Actions')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Action Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save action plot
            plot_filename = f"episode_{i_epi:02d}_actions.png"
            plot_path = os.path.join(actions_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory
            print(f"Saved action plot to: {plot_path}")
            
            # Create and save position plots
            fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3*num_agents))
            if num_agents == 1:
                axes = [axes]  # Make it iterable for single agent
            
            for agent in range(num_agents):
                ax = axes[agent]
                ax.plot(timesteps, agent_position[:, agent, 0], 
                       label='Position X', linewidth=1.5, color='blue')
                ax.plot(timesteps, agent_position[:, agent, 1], 
                       label='Position Y', linewidth=1.5, color='red')
                
                ax.set_title(f'Agent {agent} Positions')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Position')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save position plot
            pos_plot_filename = f"episode_{i_epi:02d}_positions.png"
            pos_plot_path = os.path.join(actions_dir, pos_plot_filename)
            plt.savefig(pos_plot_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory
            print(f"Saved position plot to: {pos_plot_path}")
            
            # Create and save velocity plots
            fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3*num_agents))
            if num_agents == 1:
                axes = [axes]  # Make it iterable for single agent
            
            for agent in range(num_agents):
                ax = axes[agent]
                ax.plot(timesteps, agent_velocity[:, agent, 0], 
                       label='Velocity X', linewidth=1.5, color='green')
                ax.plot(timesteps, agent_velocity[:, agent, 1], 
                       label='Velocity Y', linewidth=1.5, color='orange')
                
                ax.set_title(f'Agent {agent} Velocities')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Velocity')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save velocity plot
            vel_plot_filename = f"episode_{i_epi:02d}_velocities.png"
            vel_plot_path = os.path.join(actions_dir, vel_plot_filename)
            plt.savefig(vel_plot_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory
            print(f"Saved velocity plot to: {vel_plot_path}")
            
            # Create comprehensive plot with subplots arranged in columns (actions, positions, velocities)
            fig, axes = plt.subplots(num_agents, 3, figsize=(18, 3*num_agents))
            if num_agents == 1:
                axes = axes.reshape(1, -1)  # Make it 2D for single agent
            
            timesteps = np.arange(num_steps)
            
            for agent in range(num_agents):
                # Actions subplot (column 0)
                ax_actions = axes[agent, 0]
                for dim in range(action_dim):
                    ax_actions.plot(timesteps, actions_array[:, agent, dim], 
                                   label=f'Action {dim}', linewidth=1.5)
                ax_actions.set_title(f'Agent {agent} Actions')
                ax_actions.set_xlabel('Time Step')
                ax_actions.set_ylabel('Action Value')
                ax_actions.legend()
                ax_actions.grid(True, alpha=0.3)
                
                # Positions subplot (column 1)
                ax_positions = axes[agent, 1]
                ax_positions.plot(timesteps, agent_position[:, agent, 0], 
                                 label='Position X', linewidth=1.5, color='blue')
                ax_positions.plot(timesteps, agent_position[:, agent, 1], 
                                 label='Position Y', linewidth=1.5, color='red')
                ax_positions.set_title(f'Agent {agent} Positions')
                ax_positions.set_xlabel('Time Step')
                ax_positions.set_ylabel('Position')
                ax_positions.legend()
                ax_positions.grid(True, alpha=0.3)
                
                # Velocities subplot (column 2)
                ax_velocities = axes[agent, 2]
                ax_velocities.plot(timesteps, agent_velocity[:, agent, 0], 
                                  label='Velocity X', linewidth=1.5, color='green')
                ax_velocities.plot(timesteps, agent_velocity[:, agent, 1], 
                                  label='Velocity Y', linewidth=1.5, color='orange')
                ax_velocities.set_title(f'Agent {agent} Velocities')
                ax_velocities.set_xlabel('Time Step')
                ax_velocities.set_ylabel('Velocity')
                ax_velocities.legend()
                ax_velocities.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save comprehensive plot
            comprehensive_plot_filename = f"episode_{i_epi:02d}_comprehensive.png"
            comprehensive_plot_path = os.path.join(actions_dir, comprehensive_plot_filename)
            plt.savefig(comprehensive_plot_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory
            print(f"Saved comprehensive plot to: {comprehensive_plot_path}")

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{step}"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
        viz_opts = {}
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--path", type=str, required=True)

    # custom arguments
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--log", action="store_true", default=False)

    # default arguments
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
