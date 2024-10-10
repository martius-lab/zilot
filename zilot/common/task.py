import abc
import math
import warnings
from enum import Enum

import gymnasium as gym
import numpy as np
import ot
import torch
from omegaconf import Container, DictConfig

import zilot.types as ty
from zilot.envs import Env
from zilot.utils.color_util import GOAL_COLOR
from zilot.utils.lis_util import lis_update

GOAL_CLR = (*GOAL_COLOR, 0.8)


class TaskSpec(Enum):
    SINGLE = 1
    MULTI = 2


class Task(abc.ABC, gym.Wrapper):
    _task_spec: TaskSpec | None = None

    def __init__(self, cfg: Container, env: Env):
        if self.task_spec is None:
            raise ValueError("Task must have a task_spec attribute")
        super().__init__(env)
        self.cfg = cfg

    @property
    def task_spec(self) -> TaskSpec:
        return self._task_spec

    @property
    @abc.abstractmethod
    def max_episode_length(self) -> int:
        pass

    @abc.abstractmethod
    def reset_task(self) -> None:
        """
        Resets random state so that the same goals are sampled again.
        """
        pass


class GoalTask(Task):
    _task_spec = TaskSpec.SINGLE

    def __init__(self, cfg: Container, env: Env):
        super().__init__(cfg, gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_length))
        self.env: Env
        self.seed = False

    @property
    def max_episode_length(self) -> int:
        return self.cfg.max_episode_length

    def reset_task(self):
        self.seed = True

    def reset(self):
        s = self.cfg.seed if self.seed else None
        self.seed = False
        obs, info = self.env.reset(seed=s)
        obs["desired_goal"] = obs["desired_goal"].unsqueeze(0)
        return obs, info

    def step(self, action):
        self._stats = {}
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = info.get("success", False) or info.get("is_success", False)
        done = success or truncated or terminated
        info["metrics"] = {
            "success": success,
            "goal_frac": float(success),
            "W1": self.cfg.eval_metric(obs["achieved_goal"], obs["desired_goal"]),
            "last": success,
            "gidx": int(success),
        }
        return obs, reward, terminated, done, info


class TrajectoryTask(Task):
    _task_spec = TaskSpec.SINGLE

    def __init__(
        self,
        cfg: Container,
        env: Env,
        expert_trajectories: list[ty.Goal],
        steps_per_goal: int = 1,
    ):
        super().__init__(cfg, env)
        self.env: Env
        self.expert_traj = None
        self.expert_trajectories = expert_trajectories
        self.steps_per_goal = steps_per_goal
        self.demo_idx = 0

    @property
    def max_episode_length(self) -> int:
        return math.ceil(max(len(ex) * self.steps_per_goal for ex in self.expert_trajectories))

    def reset_task(self):
        _ = self.env.reset(seed=self.cfg.seed)
        self.demo_idx = 0
        self.expert_traj = None
        self.agent_history = None

    def reset(self):
        if self.demo_idx >= len(self.expert_trajectories):
            raise ValueError("Expert trajectories exhausted")
        self.expert_traj = self.expert_trajectories[self.demo_idx]
        self.demo_idx += 1
        obs0, info = self.env.reset_to_goal(self.expert_traj[0])
        obs0["desired_goal"] = self.expert_traj.clone()
        self.agent_history = [obs0["achieved_goal"].clone()]
        self.ep_len = 0
        self.lis_dp = []
        return obs0, info

    def step(self, action):
        stats = {}
        obs, reward, _, _, info = self.env.step(action)
        self.ep_len += 1
        obs_as_goal = obs["achieved_goal"]

        achieved_this_step_mask = self.cfg.eval_metric(obs_as_goal, self.expert_traj) <= self.cfg.goal_success_threshold
        achieved_this_step = torch.where(achieved_this_step_mask)[0]
        assert achieved_this_step.dim() == 1
        achieved_this_step = achieved_this_step.sort().values  # should be sorted, but who knows
        for gid in achieved_this_step:
            self.lis_dp = lis_update(gid.item(), self.lis_dp)
        n_goals = len(self.lis_dp)

        success = n_goals >= self.expert_traj.size(0)
        stats["success"] = success
        stats["goal_frac"] = n_goals / self.expert_traj.size(0)
        stats["last"] = achieved_this_step_mask[-1].item()
        stats["gidx"] = achieved_this_step[-1].item() if achieved_this_step.numel() > 0 else -1
        stats.update(self._compute_stats())
        info["metrics"] = stats
        done = self.ep_len >= self.steps_per_goal * self.expert_traj.size(0)
        self.agent_history.append(obs_as_goal.clone())
        del obs["desired_goal"]
        try:
            self.env.clear_points()
            colors = [GOAL_CLR for _ in range(self.expert_traj.size(0))]
            if self.cfg.env != "halfcheetah":
                self.env.draw_goals(self.expert_traj.clone(), colors=colors)
        except Exception as e:
            warnings.warn(f"Failed to draw points: {e}")
        try:  # move actual single goal out of the way if it exists
            p = self.expert_traj[-1].clone()
            p[:] = -99
            self.env.get_wrapper_attr("set_target")(p)
        except Exception as e:
            warnings.warn(f"Failed to move target out of the way: {e}")
        # NOTE: we do not terminate early, since W1 might decrease for next few steps if goal_success_threshold is high
        return obs, reward, done, done, info

    def _compute_stats(self) -> float:
        a = self.expert_traj
        b = torch.stack(self.agent_history)
        M = self.cfg.eval_metric(a[:, None, :], b[None, :, :])
        W1 = ot.emd2([], [], M.cpu().numpy())
        return dict(W1=W1)


class FixedTrajectoryTask(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        trajectory: str = "L-sparse",
        n: int = 512,
        **kwargs,
    ):
        is_dense = trajectory.endswith("-dense")
        shape = trajectory.split("-")[0]
        step_size = cfg.step_size if is_dense else 1e8

        # goal space
        # z is the exact z that the object has after one step in the environment (env.reset(); env.step())
        z = {
            "fetch_pick_and_place": 0.42473605275154114,
            "fetch_push": 0.4247373938560486,
            "fetch_slide_large_2D": 0.4249346852302551,
        }[cfg.env]
        y_lo, y_hi = {
            "fetch_pick_and_place": (0.45, 1.05),
            "fetch_push": (0.45, 1.05),
            "fetch_slide_large_2D": (0.4, 1.1),
        }[cfg.env]
        x_lo, x_hi = {
            "fetch_pick_and_place": (1.15, 1.4),
            "fetch_push": (1.15, 1.4),
            "fetch_slide_large_2D": (1.1, 1.4),
        }[cfg.env]
        marg = 0.05
        margin = torch.tensor([marg, marg, 0.0])
        min_goal = torch.tensor([x_lo, y_lo, z]) + margin
        max_goal = torch.tensor([x_hi, y_hi, z]) - margin
        middle = (min_goal + max_goal) / 2

        def S(i):
            p1 = torch.tensor([middle[0], min_goal[1], middle[2]])
            p2 = torch.tensor([middle[0], middle[1], middle[2]])
            p3 = torch.tensor([middle[0], max_goal[1], middle[2]])

            def circle(p1, p2, right=True):
                c = (p1 + p2) / 2
                r = torch.norm(p2 - p1) / 2
                length = max(math.ceil(math.pi * r / step_size), 2) + 1
                t = torch.linspace(0, math.pi, length)
                if right:
                    t = t - math.pi / 2
                else:
                    t = t.flip(0) + math.pi / 2
                x = c[0] + r * torch.cos(t)
                y = c[1] + r * torch.sin(t)
                z = middle[2].expand_as(x)
                return torch.stack([x, y, z], dim=1)

            ori = (i % 4) < 2
            curve = torch.cat([circle(p1, p2, ori)[:-1], circle(p2, p3, not ori)], dim=0)
            flip = i % 2
            if flip:
                curve = curve.flip(0)
            return curve

        def L(i):
            p1, p2, p3, p4 = [
                min_goal,
                torch.tensor([max_goal[0], min_goal[1], middle[2]]),
                max_goal,
                torch.tensor([min_goal[0], max_goal[1], middle[2]]),
            ]
            variants = [
                [p1, p2, p3],
                [p2, p1, p4],
                [p3, p4, p1],
                [p4, p3, p2],
            ]
            p1, p2, p3 = variants[i % 4]
            t1 = torch.linspace(0, 1, max(torch.norm(p2 - p1).div(step_size).ceil().int(), 2))
            t2 = torch.linspace(0, 1, max(torch.norm(p3 - p2).div(step_size).ceil().int(), 2))
            ll1 = (1 - t1[:-1, None]) * p1 + t1[:-1, None] * p2
            ll2 = (1 - t2[:, None]) * p2 + t2[:, None] * p3
            return torch.cat([ll1, ll2], dim=0)

        def U(i):
            p1, p2, p3, p4 = [
                min_goal,
                torch.tensor([max_goal[0], min_goal[1], middle[2]]),
                max_goal,
                torch.tensor([min_goal[0], max_goal[1], middle[2]]),
            ]
            variants = [
                [p1, p4, p3, p2],
                [p2, p3, p4, p1],
                [p3, p2, p1, p4],
                [p4, p1, p2, p3],
            ]
            p1, p2, p3, p4 = variants[i % 4]
            t1 = torch.linspace(0, 1, max(torch.norm(p2 - p1).div(step_size).ceil().int(), 2))
            t2 = torch.linspace(0, 1, max(torch.norm(p3 - p2).div(step_size).ceil().int(), 2))
            t3 = torch.linspace(0, 1, max(torch.norm(p4 - p3).div(step_size).ceil().int(), 2))
            ll1 = (1 - t1[:-1, None]) * p1 + t1[:-1, None] * p2
            ll2 = (1 - t2[:-1, None]) * p2 + t2[:-1, None] * p3
            ll3 = (1 - t3[:, None]) * p3 + t3[:, None] * p4
            return torch.cat([ll1, ll2, ll3], dim=0)

        if shape == "S":
            expert_trajectories = [S(i) for i in range(n)]
        elif shape == "L":
            expert_trajectories = [L(i) for i in range(n)]
        elif shape == "U":
            expert_trajectories = [U(i) for i in range(n)]
        else:
            raise ValueError(f"Invalid trajectory: {trajectory}")

        n_steps = cfg.eval_metric(expert_trajectories[0][:-1], expert_trajectories[0][1:]).sum().item() / cfg.step_size
        n_goals = len(expert_trajectories[0])
        slack_per_step = 8.0
        kwargs.setdefault("steps_per_goal", slack_per_step * n_steps / n_goals)

        super().__init__(cfg, env, expert_trajectories, **kwargs)


class MazeCircleTask(TrajectoryTask):
    circle = np.array(
        [
            [-2.5, -2.5],
            [-0.5, -2.5],
            [-0.5, -1.5],
            [0.5, -1.5],
            [0.5, 0.5],
            [-1.5, 0.5],
            [-1.5, -0.5],
            [-2.5, -0.5],
            [-2.5, -2.5],
        ]
    )
    path = np.array(
        [
            [-2.5, 2.5],
            [-1.5, 1.5],
            [-1.5, 0.5],
            [0.5, 0.5],
            [0.5, -0.5],
            [2.5, -0.5],
            [2.5, -2.5],
            [1.5, -2.5],
        ]
    )

    def __init__(self, cfg: Container, env: Env, n: int = 512, trajectory: str = "path-sparse", **kwargs):
        traj, kind = trajectory.split("-")

        if traj == "circle":
            c = self.circle.copy()
        elif traj == "path":
            c = self.path.copy()
        else:
            raise ValueError(f"Invalid trajectory: {trajectory}")

        if kind == "dense":
            stride = 12
            ii = np.ceil(cfg.eval_metric(c[:-1], c[1:]) / cfg.step_size / stride).astype(int)
            t = np.concatenate(
                [
                    np.linspace(1, 0, ni + 1)[:, None] * c1 + np.linspace(0, 1, ni + 1)[:, None] * c2
                    for c1, c2, ni in zip(c[:-1], c[1:], ii)
                ]
            )
            expert_trajectories = [torch.from_numpy(t.copy()).float() for _ in range(n)]
        elif kind == "sparse":
            expert_trajectories = [torch.from_numpy(c.copy()).float() for _ in range(n)]
        else:
            raise ValueError(f"Invalid trajectory: {trajectory}")

        n_steps = cfg.eval_metric(expert_trajectories[0][:-1], expert_trajectories[0][1:]).sum().item() / cfg.step_size
        n_goals = len(expert_trajectories[0])
        slack_per_step = 2.0
        kwargs.setdefault("steps_per_goal", slack_per_step * n_steps / n_goals)
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahFrontFlip(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        # t_max = 21  # HalfCheetah-v4 runs at 20Hz (frameskip=5, sim timestep=0.01), flips take ~1s
        start = torch.tensor([1, np.deg2rad(180)], dtype=torch.float32)
        end = torch.tensor([2, np.deg2rad(360)], dtype=torch.float32)

        flip = torch.stack([start, end], dim=0)

        expert_trajectories = [flip.clone() for _ in range(n)]

        kwargs.setdefault("steps_per_goal", 80 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahBackFlip(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        # t_max = 21  # HalfCheetah-v4 runs at 20Hz (frameskip=5, sim timestep=0.01), flips take ~1s
        start = torch.tensor([0, np.deg2rad(-180)], dtype=torch.float32)
        end = torch.tensor([0, np.deg2rad(-360)], dtype=torch.float32)

        flip = torch.stack([start, end], dim=0)

        expert_trajectories = [flip.clone() for _ in range(n)]

        kwargs.setdefault("steps_per_goal", 80 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahRunningFrontFlip(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        # t_max = 21  # HalfCheetah-v4 runs at 20Hz (frameskip=5, sim timestep=0.01), flips take ~1s

        flip = torch.tensor(
            [
                [4, np.deg2rad(0)],
                [5, np.deg2rad(180)],
                [6, np.deg2rad(360)],
            ],
            dtype=torch.float32,
        )

        expert_trajectories = [flip.clone() for _ in range(n)]

        kwargs.setdefault("steps_per_goal", 150 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahRunningBackFlip(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        # t_max = 21  # HalfCheetah-v4 runs at 20Hz (frameskip=5, sim timestep=0.01), flips take ~1s

        flip = torch.tensor(
            [
                [4, np.deg2rad(0)],
                [4, np.deg2rad(-180)],
                [4, np.deg2rad(-360)],
            ],
            dtype=torch.float32,
        )

        expert_trajectories = [flip.clone() for _ in range(n)]

        kwargs.setdefault("steps_per_goal", 150 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahRunForward(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        start = torch.tensor([1.0, np.deg2rad(0)], dtype=torch.float32)
        end = torch.tensor([5.0, np.deg2rad(0)], dtype=torch.float32)
        t = torch.linspace(0, 1, 10)
        run = (1 - t[:, None]) * start + t[:, None] * end
        expert_trajectories = [run.clone() for _ in range(n)]
        kwargs.setdefault("steps_per_goal", 120 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahRunBackward(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        start = torch.tensor([-1.0, np.deg2rad(0)], dtype=torch.float32)
        end = torch.tensor([-5.0, np.deg2rad(0)], dtype=torch.float32)
        t = torch.linspace(0, 1, 10)
        run = (1 - t[:, None]) * start + t[:, None] * end
        expert_trajectories = [run.clone() for _ in range(n)]
        kwargs.setdefault("steps_per_goal", 120 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahHopForward(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        start = torch.tensor([1.0, np.deg2rad(-66)], dtype=torch.float32)
        end = torch.tensor([5.0, np.deg2rad(-66)], dtype=torch.float32)
        t = torch.linspace(0, 1, 10)
        run = (1 - t[:, None]) * start + t[:, None] * end
        expert_trajectories = [run.clone() for _ in range(n)]
        kwargs.setdefault("steps_per_goal", 300 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


class CheetahHopBackward(TrajectoryTask):
    def __init__(
        self,
        cfg: Container,
        env: Env,
        n: int = 512,
        **kwargs,
    ):
        start = torch.tensor([-1.0, np.deg2rad(66)], dtype=torch.float32)
        end = torch.tensor([-5.0, np.deg2rad(66)], dtype=torch.float32)
        t = torch.linspace(0, 1, 10)
        run = (1 - t[:, None]) * start + t[:, None] * end
        expert_trajectories = [run.clone() for _ in range(n)]
        kwargs.setdefault("steps_per_goal", 300 / expert_trajectories[0].size(0))
        super().__init__(cfg, env, expert_trajectories, **kwargs)


def make_task_from_env(name: str, cfg: DictConfig, env: Env) -> Task:
    if name == "goal":
        return GoalTask(cfg=cfg, env=env)
    if "fetch" in cfg.env.lower():
        if name in ["S-sparse", "L-sparse", "U-sparse", "S-dense", "L-dense", "U-dense"]:
            return FixedTrajectoryTask(cfg=cfg, env=env, trajectory=name)
    if "maze" in cfg.env.lower():
        if name in ["circle-sparse", "circle-dense", "path-sparse", "path-dense"]:
            return MazeCircleTask(cfg=cfg, env=env, trajectory=name)
    if "cheetah" in cfg.env.lower():
        if name == "frontflip":
            return CheetahFrontFlip(cfg=cfg, env=env)
        elif name == "frontflip-running":
            return CheetahRunningFrontFlip(cfg=cfg, env=env)
        elif name == "backflip":
            return CheetahBackFlip(cfg=cfg, env=env)
        elif name == "backflip-running":
            return CheetahRunningBackFlip(cfg=cfg, env=env)
        elif name == "hop-forward":
            return CheetahHopForward(cfg=cfg, env=env)
        elif name == "hop-backward":
            return CheetahHopBackward(cfg=cfg, env=env)
        elif name == "run-forward":
            return CheetahRunForward(cfg=cfg, env=env)
        elif name == "run-backward":
            return CheetahRunBackward(cfg=cfg, env=env)

    raise ValueError(f"Unknown task {name} for env {cfg.env}")


def make_tasks_from_env(cfg: DictConfig, env: Env) -> dict[str, Task]:
    return {k: make_task_from_env(name=k, cfg=cfg, env=env) for k in cfg.tasks.keys()}


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from zilot.envs import make_env
    from zilot.parse import parse_cfg

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    cfg = OmegaConf.load("cfg/config.yaml")
    cfg = parse_cfg(cfg)
    env = make_env(cfg)

    def imitate_task_usage(cfg, env):
        task = GoalTask(cfg, env)
        task.reset_task()
        seq1 = [task.reset() for _ in range(10)]
        task.reset_task()
        seq2 = [task.reset() for _ in range(10)]
        return seq1, seq2

    seq1, seq2 = imitate_task_usage(cfg, env)
    seq3, seq4 = imitate_task_usage(cfg, env)
    for s1, s2, s3, s4 in zip(seq1, seq2, seq3, seq4):
        assert (s1[0] == s2[0]).all() and (s1[0] == s3[0]).all() and (s1[0] == s4[0]).all()
        assert (s1[1] == s2[1]).all() and (s1[1] == s3[1]).all() and (s1[1] == s4[1]).all()
    print("Success!")
