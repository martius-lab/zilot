from copy import deepcopy
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from torch import TensorType

from zilot.envs.base import EnvMixin
from zilot.utils.gym_util import TensorWrapper, obs_to_tensor


class GCMJWrapper(gym.Wrapper, EnvMixin):
    def __init__(self, env: MujocoEnv, cfg, goal_proj, inverse_proj, goal_space, goal_to_points):
        super().__init__(env)
        self.cfg = cfg
        self._goal_projection = goal_proj
        self._inverse_projection = inverse_proj
        self._goal_space = goal_space
        self._goal_to_points = goal_to_points
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self.observation_space,
                "desired_goal": self._goal_space,
                "achieved_goal": self._goal_space,
            }
        )
        self._current_goal = None
        self._last_obs = None
        self._last_render = None
        self._repeating = None
        self.env: MujocoEnv

    def _generate_target_goal(self) -> np.ndarray:
        return self._goal_space.sample()

    def generate_target_goal(self) -> TensorType:
        return torch.from_numpy(self._generate_target_goal()).float()

    def _to_gc_obs(self, obs):
        return {
            "observation": obs,
            "desired_goal": self._current_goal,
            "achieved_goal": self._goal_projection(obs),
        }

    def compute_success(self, achieved_goal, desired_goal):
        return self.cfg.eval_metric(achieved_goal, desired_goal) <= self.cfg.goal_success_threshold

    def reset(self, *args, **kwargs):
        self._repeating = None
        obs, info = self.env.reset(*args, **kwargs)
        self._current_goal = self._generate_target_goal()
        return self._to_gc_obs(obs), info

    def _reset_to_goal(self, goal: np.ndarray) -> tuple[np.ndarray, dict]:
        qpos_backup = self.env.init_qpos.copy()
        self.env.init_qpos = self._inverse_projection(self.env.init_qpos, goal)
        ret = self.reset()
        self.env.init_qpos = qpos_backup
        return ret

    def reset_to_goal(self, goal: TensorType) -> tuple[TensorType, dict]:
        obs, info = self._reset_to_goal(goal.numpy())
        return obs_to_tensor(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._to_gc_obs(obs)
        success = self.compute_success(obs["achieved_goal"], obs["desired_goal"])
        info["success"] = success
        return obs, 0.0, success, truncated, info

    """ VISUALIZATION """

    def render(self, *args, **kwargs):
        # for debugging, it makes sense to also reflect repetition in rendering
        # (does not work for render_mode="human" though)
        x = self.env.render(*args, **kwargs)
        if self._repeating is not None:
            if self._last_render is None:
                self._last_render = x
            return deepcopy(self._last_render)
        return x

    def draw_goals(self, goals, colors=None):
        points = self._goal_to_points(goals)
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        colors = colors or [[0, 0, 1, 1]] * len(points)
        viewer = self.env.unwrapped.mujoco_renderer.viewer
        for p, c in zip(points, colors):
            viewer.add_marker(pos=p, type=2, size=[0.05, 0.05, 0.05], rgba=c, label="")

    def clear_points(self):
        del self.env.unwrapped.mujoco_renderer.viewer._markers[:]


GOAL_TRANSFORMS = {"halfcheetah": lambda x: x[..., ::2][..., :2]}

DSETS = {"halfcheetah": []}

ENVS = {
    "halfcheetah": {
        "gym_id": "HalfCheetah-v4",
        "env_kwargs": {"exclude_current_positions_from_observation": False},
        "wrapper": partial(
            GCMJWrapper,
            goal_proj=lambda x: np.concatenate([x[..., 0:1], x[..., 2:3]], axis=-1),  # x, z, theta_z, ... -> x, theta_z
            inverse_proj=lambda x, g: np.concatenate([g[..., :1], x[..., 1:2], g[..., 2:3], x[..., 3:]], axis=-1),
            goal_space=gym.spaces.Box(
                np.array([-5, -4 * np.pi], dtype=np.float64),
                np.array([5, 4 * np.pi], dtype=np.float64),
                dtype=np.float64,
            ),
            goal_to_points=lambda x: np.concatenate(
                [x[..., :1], torch.zeros_like(x[..., :1]), 0.1 * torch.ones_like(x[..., :1])],  # [x, 0, .1]
                axis=-1,
            ),
        ),
        "cfg_changes": {
            "goal_success_threshold": 0.5,
            "eval_metric": "${tf:l2_norm}",
            "max_episode_length": 200,
            "discount": 0.99,
            "step_size": 0.05,  # TODO: what is the step size in halfcheetah?
            "horizon": 32,
        },
    },
}


def make_env(cfg, **kwargs) -> gym.Env:
    name = cfg.env
    if name not in ENVS:
        raise KeyError(f"Env {cfg.env} not found.")
    env_cfg = ENVS[name]
    for k, v in env_cfg.get("cfg_changes", {}).items():
        cfg[k] = v
    env = gym.make(env_cfg["gym_id"], **kwargs, **env_cfg.get("env_kwargs", {}))
    env = env_cfg["wrapper"](env, cfg)
    env = TensorWrapper(env)
    return env
