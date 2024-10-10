from typing import Union

from gymnasium_robotics import GoalEnv

from zilot.envs.base import EnvMixin
from zilot.envs.gymnasium_mujoco import DSETS as DSETS_MJ
from zilot.envs.gymnasium_mujoco import GOAL_TRANSFORMS as GOAL_TRANSFORMS_MJ
from zilot.envs.gymnasium_mujoco import make_env as make_mujoco_env
from zilot.envs.gymnasium_robotics import DSETS as DSETS_GR
from zilot.envs.gymnasium_robotics import GOAL_TRANSFORMS as GOAL_TRANSFORMS_GR
from zilot.envs.gymnasium_robotics import make_env as make_gr_env

Env = Union[EnvMixin, GoalEnv]


GOAL_TRANSFORMS = {**GOAL_TRANSFORMS_GR, **GOAL_TRANSFORMS_MJ}

DSETS = {**DSETS_GR, **DSETS_MJ}


def make_env(cfg) -> EnvMixin:
    """
    unified interface for making environments
    """
    env = None
    exception = None
    for fn in [make_gr_env, make_mujoco_env]:
        try:
            env: Env = fn(cfg, render_mode="rgb_array", max_episode_steps=10_000_000, autoreset=False)
        except KeyError:
            continue
        except Exception as e:
            exception = e
            break
    if env is None:
        raise ValueError(f'Failed to make environment "{cfg.env}": {exception}')

    cfg.setdefault("obs", "state")
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space["observation"].items()}
    except Exception:  # Box
        cfg.obs_shape = {cfg.obs: env.observation_space["observation"].shape}

    cfg.goal = cfg.get("goal", cfg.obs)  # default: goal is obs which is state
    try:
        cfg.goal_shape = {k: v.shape for k, v in env.observation_space["desired_goal"].items()}
    except Exception:  # Box
        cfg.goal_shape = {cfg.goal: env.observation_space["desired_goal"].shape}

    cfg.action_dim = env.action_space.shape[0]
    cfg.value_scale = cfg.max_episode_length
    print(f"Reward range: [{-cfg.value_scale}, 0]")

    return env
