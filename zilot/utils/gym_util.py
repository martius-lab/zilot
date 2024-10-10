from collections import defaultdict
from typing import List

import gymnasium as gym
import numpy as np
import torch
from gymnasium_robotics import GoalEnv
from tensordict import TensorDict
from torch import TensorType
from tqdm.auto import tqdm

from zilot.third_party.mbrl.math import powerlaw_psd_gaussian


def obs_to_tensor(obs):
    if isinstance(obs, dict):
        return TensorDict({k: obs_to_tensor(v) for k, v in obs.items()}, batch_size=[])
    return torch.from_numpy(obs).float()


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env):
        super().__init__(env)

    def rand_act(self):
        return torch.from_numpy(self.env.action_space.sample()).float()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs_to_tensor(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.detach().cpu().numpy())
        info = defaultdict(float, info)
        info["success"] = float(info["success"])
        return (
            obs_to_tensor(obs),
            torch.tensor(reward, dtype=torch.float32),
            terminated,
            truncated,
            info,
        )


def generate_random_exploration_dataset(
    env: GoalEnv,
    seed: int = 0,
    episode_lengths: list | None = None,
) -> list[TensorDict]:
    """
    Generate a random exploration dataset for a given environment.

    Args:
        env: The environment to generate the dataset for.
        episode_length: The length of each episode in the dataset.
        num_episodes: The number of episodes in the dataset.
        seed: The random seed to use for generating the dataset.

    Returns:
        dict with keys "observations", "actions", and "timeouts" as numpy arrays.
    """

    # generate actions with seed but without changing the global RNG state
    torch_cpu_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    actions = [
        powerlaw_psd_gaussian(1.0, (*env.action_space.shape, le), "cpu").moveaxis(-1, 0).numpy()
        for le in episode_lengths
    ]
    actions = np.concatenate(actions)
    torch.set_rng_state(torch_cpu_rng_state)  # restore the original RNG state

    env.action_space.seed(seed)
    env_samples = np.stack([env.action_space.sample() for _ in range(10000)])
    mean = np.mean(env_samples, axis=0)
    std = np.std(env_samples, axis=0)
    actions = actions * std + mean
    if hasattr(env.action_space, "bounded_below"):
        actions[:, env.action_space.bounded_below] = np.clip(actions, env.action_space.low, None)[
            :, env.action_space.bounded_below
        ]
    if hasattr(env.action_space, "bounded_above"):
        actions[:, env.action_space.bounded_above] = np.clip(actions, None, env.action_space.high)[
            :, env.action_space.bounded_above
        ]

    tds = []
    actions = torch.as_tensor(actions, dtype=torch.float32)

    total = np.sum(episode_lengths)
    with tqdm(total=total, desc="Generating Random Dataset") as pbar:
        idx = 0
        for ep_len in episode_lengths:
            observations = []
            goals = []
            obs, _ = env.reset(seed=seed if idx == 0 else None)
            for _ in range(ep_len):
                observations.append(obs["observation"])
                goals.append(obs["achieved_goal"])
                obs, _, _, _, _ = env.step(actions[idx])
                idx += 1
                pbar.update(1)
            observations = torch.stack(observations)
            goals = torch.stack(goals)
            done = torch.zeros(ep_len, dtype=torch.bool)
            tds.append(
                TensorDict(
                    dict(obs=observations, action=actions[idx - ep_len : idx], achieved_goal=goals, done=done),
                    batch_size=ep_len,
                )
            )

    return tds


def random_subsample(
    expert_trajectories: List[TensorType], stride: float, offset_frac: float | None = None, seed: int = 0
) -> List[TensorType]:
    """
    Randomly subsample expert trajectories.

    Args:
        expert_trajectories: The expert trajectories to subsample.
        stride: The stride to use for subsampling.
        offset_frac: if not None, sample offset~Unif[-offset_frac*stride, offset_frac*stride) for each index.

    Returns:
        The subsampled expert trajectories.
    """
    rng = np.random.RandomState(seed)
    r = []
    for traj in tqdm(expert_trajectories, desc="subsampling trajectories"):
        indices = np.arange(0, len(traj), stride).astype(int)
        if offset_frac is not None:
            offsets = rng.rand(len(indices))
            offsets = (offsets - 0.5) * 2 * offset_frac * stride
            offsets[0] = 0
            indices = np.clip((indices + offsets).round().astype(int), 0, len(traj) - 1)
        indices = torch.from_numpy(indices)
        r.append(traj[indices])
    return r
