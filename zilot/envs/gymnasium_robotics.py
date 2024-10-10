import os
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from gymnasium_robotics.envs.maze import AntMazeEnv, PointMazeEnv
from gymnasium_robotics.envs.maze.maze import MazeEnv

from zilot.envs.base import EnvMixin
from zilot.utils.gym_util import TensorWrapper, obs_to_tensor
from zilot.utils.plot_util import plot_policy, plot_value_function


class MazeEnvWrapper(EnvMixin, gym.Wrapper):
    def __init__(
        self,
        cfg,
        env: MazeEnv,
        dataset_name: str = "random",
        minari_dset_name: str = "",
    ):
        super().__init__(TensorWrapper(env))
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.minari_dset_name = minari_dset_name
        self.env: MazeEnv
        self.is_ant = hasattr(self.env.unwrapped, "ant_env")

    @property
    def _mujoco_env(self) -> MujocoEnv | PointMazeEnv | AntMazeEnv:
        if hasattr(self.env.unwrapped, "point_env"):
            return self.env.unwrapped.point_env
        elif hasattr(self.env.unwrapped, "ant_env"):
            return self.env.unwrapped.ant_env
        else:
            assert False, "Unknown environment type"

    def reset_to_state(self, state: torch.Tensor) -> torch.Tensor:
        # update internal state of all wrappers
        _ = self.env.reset()
        # update change MujocoEnv state
        state = state.cpu().numpy()
        mje = self._mujoco_env
        mje.set_state(state[: mje.model.nq], state[mje.model.nq :])
        # redo reset of MujocoEnv
        obs, info = mje._get_obs()
        # redo reset of MazeEnv
        obs_dict = self.env.unwrapped._get_obs(obs)
        info["success"] = bool(np.linalg.norm(obs_dict["achieved_goal"] - self.get_wrapper_attr("goal")) <= 0.45)
        # redo reset of TensorWrapper
        obs_dict = obs_to_tensor(obs_dict)
        return obs_dict, info

    def reset_to_goal(self, goal: torch.Tensor) -> torch.Tensor:
        # update internal state of all wrappers
        _ = self.env.reset()
        # update change MujocoEnv state
        goal = goal.cpu().numpy()
        mje = self._mujoco_env
        qpos, qvel = mje.data.qpos, mje.data.qvel
        qpos[0:2] = goal
        mje.set_state(qpos, qvel)
        # redo reset of MujocoEnv
        x = mje._get_obs()
        if isinstance(x, tuple):
            obs, info = x
        else:
            obs, info = x, {}
        # redo reset of MazeEnv
        obs_dict = self.env.unwrapped._get_obs(obs)
        info["success"] = bool(np.linalg.norm(obs_dict["achieved_goal"] - self.get_wrapper_attr("goal")) <= 0.45)
        # redo reset of TensorWrapper
        obs_dict = obs_to_tensor(obs_dict)
        return obs_dict, info

    """ VISUALIZATION """

    def draw_goals(self, points, colors=None):
        h = 0.3 if self.is_ant else 0.0
        points = torch.cat((points, torch.full((points.size(0), 1), h)), dim=1)
        points = points.cpu().numpy()
        colors = colors or [[0, 0, 1, 1]] * len(points)
        viewer = self._mujoco_env.mujoco_renderer.viewer
        sz = 0.10 if self.is_ant else 0.05
        for p, c in zip(points, colors):
            viewer.add_marker(pos=p, type=2, size=[sz, sz, sz], rgba=c, label="")

    def clear_points(self):
        del self._mujoco_env.mujoco_renderer.viewer._markers[:]

    """ UTILS """

    def generate_target_goal(self) -> torch.Tensor:
        return torch.from_numpy(self.env.generate_target_goal()).float()

    """ PLOTTING """

    def plot_policy(self, ax, a, obs, goal):
        _to_np = lambda x: x.cpu().numpy()
        states = _to_np(obs)
        if self.is_ant:
            states = states[..., 1:3]
        else:
            states = states[..., :2]
        goal = _to_np(goal)
        actions = _to_np(a)
        plot_policy(states, goal.squeeze(0), actions, ax=ax, dt=0.01, width=0.01, alpha=0.5)

    def plot_values(self, ax, v, obs, goal):
        _to_np = lambda x: x.cpu().numpy()
        states = _to_np(obs)
        if self.is_ant:
            states = states[..., 1:3]
        else:
            states = states[..., :2]
        goal = _to_np(goal)
        values = _to_np(v)
        plot_value_function(states, goal.squeeze(0), values, ax=ax, alpha=0.5, s=0.5)


# overwrite, see: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/db0baf5742e836df55d24fe5481673bdd1c92d1b/gymnasium_robotics/envs/fetch/fetch_env.py#L375C5-L399C20  # noqa
def _reset_sim(self: MujocoFetchEnv):
    self.data.time = self.initial_time
    self.data.qpos[:] = np.copy(self.initial_qpos)
    self.data.qvel[:] = np.copy(self.initial_qvel)
    if self.model.na != 0:
        self.data.act[:] = None

    # Randomize start position of object.
    if self.has_object:
        if getattr(self, "_object_reset_xpos", None) is None:  # edited
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
        else:  # edited
            object_xpos = self._object_reset_xpos[:2]  # edited
            self._object_reset_xpos = None  # edited
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)

    self._mujoco.mj_forward(self.model, self.data)
    return True


# time to do some scetchy sh*t doo-daa, doo-daa
MujocoFetchEnv._reset_sim = _reset_sim


class FetchEnvWrapper(EnvMixin, gym.Wrapper):
    def __init__(self, cfg, env: MujocoFetchEnv, slide_only: bool = False):
        super().__init__(TensorWrapper(env))
        self.cfg = cfg
        self.env: MujocoFetchEnv
        self.slide_only = slide_only

    def reset(self, *args, **kwargs):
        self._t0 = True
        obs, info = self.env.reset(*args, **kwargs)
        self._target_z_coord = obs["observation"][2] + 1e-6
        self._z_coord = obs["observation"][2]
        return obs, info

    def reset_to_goal(self, goal):
        setattr(self.env.unwrapped, "_object_reset_xpos", goal.cpu().numpy())
        return self.reset()  # this calls the overwritten _reset_sim above

    def step(self, action, *args, **kwargs):
        action = action.clone()  # might be inference mode tensor
        if self.slide_only:  # overwrite z-coord action to move back onto table
            inv_scale = 1 / 0.05
            action[2] = inv_scale * (self._target_z_coord - self._z_coord)  # this will be clipped
        obs, *rest = self.env.step(action, *args, **kwargs)
        self._z_coord = obs["observation"][2]
        # if self._t0:
        #     self._t0 = False
        #     print("obj-z", obs["achieved_goal"][2].item())
        return obs, *rest

    """ UTILS """

    def generate_target_goal(self) -> torch.Tensor:
        return torch.from_numpy(self.unwrapped._sample_goal()).float()

    """ VISUALIZATION """

    def draw_goals(self, points, colors=None):
        points = points.cpu().numpy()
        colors = colors or [[0, 0, 1, 1]] * len(points)
        viewer = self.unwrapped.mujoco_renderer.viewer
        for p, c in zip(points, colors):
            viewer.add_marker(pos=p, type=2, size=[0.02, 0.02, 0.02], rgba=c, label="")

    def clear_points(self):
        viewer = self.unwrapped.mujoco_renderer.viewer
        del viewer._markers[:]

    def set_target(self, target: torch.Tensor):
        self.env.unwrapped.goal = target.cpu().numpy()

    """ PLOTTING """

    def plot_policy(self, ax, a, obs, goal):
        _to_np = lambda x: x.cpu().numpy()
        obs = _to_np(obs)
        goal = _to_np(goal)
        a = _to_np(a)
        if obs.shape[-1] > 3:
            obs = obs[..., 3:6]
            a = a[..., :3] - obs[..., :3]
        plot_policy(obs, goal.squeeze(0), a, ax=ax, dt=0.01, width=0.01, alpha=0.5)

    def plot_values(self, ax, v, obs, goal):
        _to_np = lambda x: x.cpu().numpy()
        if obs.shape[-1] > 3:
            obs = obs[..., 3:6]
        states = _to_np(obs)
        goal = _to_np(goal)
        values = _to_np(v)
        plot_value_function(states, goal.squeeze(0), values, ax=ax, alpha=0.5, s=0.5)


DEFAULT_CAMERA_CONFIG = {
    "distance": 1.85,
    "azimuth": 132.0,
    "elevation": -32.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


class MujocoFetchSlideLargeEnv(MujocoFetchEnv, EzPickle):
    """Fetch Slide Env with large table and goal space of FetchPush."""

    def __init__(self, reward_type="sparse", **kwargs):
        model_path = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "gymnasium_robotics_assets",
                "fetch",
                "slide_large.xml",
            )
        )
        initial_qpos = {
            "robot0:slide0": 0.405,  # from FetchPush
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],  # from FetchPush
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=model_path,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=-0.02,
            target_in_the_air=False,
            target_offset=0.0,  # from FetchPush
            obj_range=0.15,  # from FetchPush
            target_range=0.15,  # from FetchPush
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


gym.register("FetchSlideLarge-v0", entry_point=MujocoFetchSlideLargeEnv, max_episode_steps=50)

fetch_kwargs = {
    "goal_success_threshold": 0.05,
    "eval_metric": "${tf:l2_norm}",
    "max_episode_length": 50,
    "discount": 0.975,
    "step_size": 0.05,
}

ENVS = {
    "pointmaze_medium": {
        "name": "PointMaze_Medium_Diverse_GR-v3",
        "kwargs": {
            "reset_target": True,
            "continuing_task": False,
        },
        "wrapper": MazeEnvWrapper,
        "cfg_changes": {
            "goal_success_threshold": 0.45,  # as used in gymnasium_robotics MazeEnv.compute_terminated
            "eval_metric": "${tf:l2_norm}",
            "max_episode_length": 600,  # default 600
            "discount": 0.995,
            "step_size": 0.05203704163432121,
            "horizon": 64,  # NOTE: since pointmaze has such a small dt, we 4x horizon
        },
    },
    "fetch_reach": {
        "name": "FetchReach-v2",
        "wrapper": FetchEnvWrapper,
        "cfg_changes": {
            **fetch_kwargs,
        },
    },
    "fetch_push": {
        "name": "FetchPush-v2",
        "wrapper": FetchEnvWrapper,
        "cfg_changes": {
            **fetch_kwargs,
        },
    },
    "fetch_slide": {
        "name": "FetchSlide-v2",
        "wrapper": FetchEnvWrapper,
        "cfg_changes": {
            **fetch_kwargs,
        },
    },
    "fetch_pick_and_place": {
        "name": "FetchPickAndPlace-v2",
        "wrapper": FetchEnvWrapper,
        "cfg_changes": {
            **fetch_kwargs,
        },
    },
    "fetch_slide_large_2D": {
        "name": "FetchSlideLarge-v0",
        "wrapper": partial(FetchEnvWrapper, slide_only=True),
        "cfg_changes": {
            **fetch_kwargs,
        },
    },
}


GOAL_TRANSFORMS = {
    "pointmaze_medium": lambda x: x[..., :2],
    "fetch_reach": lambda x: x[..., :3],
    "fetch_push": lambda x: x[..., 3:6],
    "fetch_slide": lambda x: x[..., 3:6],
    "fetch_pick_and_place": lambda x: x[..., 3:6],
    "fetch_slide_large_2D": lambda x: x[..., 3:6],
}


DSETS = {
    "pointmaze_medium": {"minari": ("pointmaze-medium-v2",)},
    "fetch_push": {
        "awgcsl-all": ("FetchPush", "all"),
        "awgcsl-random": ("FetchPush", "random"),
        "awgcsl-expert": ("FetchPush", "expert"),
    },
    "fetch_slide": {
        "awgcsl-all": ("FetchSlide", "all"),
        "awgcsl-random": ("FetchSlide", "random"),
        "awgcsl-expert": ("FetchSlide", "expert"),
    },
    "fetch_pick_and_place": {
        "awgcsl-all": ("FetchPick", "all"),
        "awgcsl-random": ("FetchPick", "random"),
        "awgcsl-expert": ("FetchPick", "expert"),
    },
    "fetch_side_large_2D": {},
}


def make_env(cfg, **kwargs) -> gym.Env:
    name = cfg.env
    if name not in ENVS:
        raise KeyError(f"Env {cfg.env} not found.")
    env_cfg = ENVS[name]
    env = gym.make(
        env_cfg["name"],
        **env_cfg.get("kwargs", {}),
        **kwargs,
    )
    env_cfg["wrapper"] = partial(env_cfg["wrapper"])
    env = env_cfg["wrapper"](cfg, env)
    for k, v in env_cfg.get("cfg_changes", {}).items():
        cfg[k] = v
    return env
