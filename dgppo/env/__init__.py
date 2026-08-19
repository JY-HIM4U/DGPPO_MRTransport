import inspect

from typing import Optional

from .base import MultiAgentEnv
from dgppo.env.mpe import MPETarget, MPESpread, MPELine, MPEFormation, MPECorridor, MPEConnectSpread
from dgppo.env.lidar_env import LidarSpread, LidarTarget, LidarLine, LidarBicycleTarget
from dgppo.env.vmas import VMASWheel, VMASReverseTransport, VMASCollaborativeTransport
from dgppo.env.vmas_lidar import VMASCollaborativeTransportLidar, VMASCollaborativeTransportLidar_Determined


ENV = {

    'MPETarget': MPETarget,
    'MPESpread': MPESpread,
    'MPELine': MPELine,
    'MPEFormation': MPEFormation,
    'MPECorridor': MPECorridor,
    'MPEConnectSpread': MPEConnectSpread,
    'LidarSpread': LidarSpread,
    'LidarTarget': LidarTarget,
    'LidarLine': LidarLine,
    'LidarBicycleTarget': LidarBicycleTarget,
    'VMASReverseTransport': VMASReverseTransport,
    'VMASWheel': VMASWheel,
    'VMASCollaborativeTransport': VMASCollaborativeTransport,
    'VMASCollaborativeTransportLidar': VMASCollaborativeTransportLidar,
    'VMASCollaborativeTransportLidar_Determined': VMASCollaborativeTransportLidar_Determined
}


DEFAULT_MAX_STEP = 128


def make_env(
        env_id: str,
        num_agents: int,
        max_step: int = None,
        full_observation: bool = False,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        min_num_agents: int = 3,
        max_num_agents: int = 5,
        reward_dist2goal: float = 0.06,
        reward_dist2goal_theta: float = 0.06,
        reward_dist2goal_threshold: float = 0.001,
        reward_action_norm: float = 0.1,
        reward_agent_vertex_dists: float = 0.1,
        reward_action_diff: float = 0.1,
        agent_vertex_constraint: float = 0.30,
        min_stiffness: float = 0.05,
        max_stiffness: float = 0.15,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = dict(ENV[env_id].PARAMS)
    # envs whose scale differs from the 1.5 m unit box (e.g. a 64 m factory
    # floor) declare their own integration step and horizon
    dt = params.get('default_dt', 0.03)
    if max_step is None:
        max_step = params.get('default_max_step', DEFAULT_MAX_STEP)
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    if full_observation:
        area_size = params['default_area_size']
        params['comm_radius'] = area_size * 10
    env_cls = ENV[env_id]
    kwargs = dict(
        num_agents=num_agents,
        area_size=None,
        max_step=max_step,
        dt=dt,
        params=params,
        min_num_agents=min_num_agents,
        max_num_agents=max_num_agents,
        reward_dist2goal=reward_dist2goal,
        reward_dist2goal_theta=reward_dist2goal_theta,
        reward_dist2goal_threshold=reward_dist2goal_threshold,
        reward_action_norm=reward_action_norm,
        reward_agent_vertex_dists=reward_agent_vertex_dists,
        reward_action_diff=reward_action_diff,
        agent_vertex_constraint=agent_vertex_constraint,
        min_stiffness=min_stiffness,
        max_stiffness=max_stiffness,
    )
    # Only the VMAS envs accept the reward/stiffness kwargs above; passing them
    # to the others (LidarTarget, MPETarget, ...) is a TypeError. Keep whatever
    # the target constructor actually declares, unless it takes **kwargs.
    sig = inspect.signature(env_cls.__init__)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return env_cls(**kwargs)
