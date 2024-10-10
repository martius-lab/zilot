import torch
from omegaconf import Container
from tensordict import TensorDict, TensorDictBase
from torchrl.data.replay_buffers import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


def add_done_masks(r: TensorDict) -> TensorDict:
    """Add `done` and `next_done` masks to the batch.
    r: TensorDict [H+1, B, keys...]
    """

    def numerical_equality(x: torch.Tensor, y: torch.Tensor):
        return (x - y).abs().max(-1).values <= 1e-6

    achieved_goal = r["achieved_goal"]
    goal = r["goal"]
    r["done"] = numerical_equality(achieved_goal.flatten(2), goal.flatten(2))  # [H+1, B]
    r["next_done"] = torch.zeros_like(r["done"])
    r["next_done"][:-1] = numerical_equality(achieved_goal[1:].flatten(2), goal[:-1].flatten(2))
    # we can leave the last next_done as 0, as it is not used
    return r


class EpisodicTensorDictReplayBuffer(TensorDictReplayBuffer):
    def __init__(
        self,
        *args,
        cfg: Container,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self._last_end_excl = 0
        self._this_end = 0
        self.episode = 0

    def add(self, data: TensorDictBase):
        if "ep_end" in data.keys():
            raise ValueError("ep_end key is reserved")
        data["ep_end"] = torch.tensor(-1, device=data.device, dtype=torch.long)
        data["episode"] = torch.tensor(self.episode, device=data.device, dtype=torch.long)
        idx = super().add(data)
        self[idx]["ep_end"] = idx
        self._this_end = idx
        return idx

    def extend(self, data: TensorDictBase):
        if "ep_end" in data.keys():
            raise ValueError("ep_end key is reserved")
        data["ep_end"] = torch.full((len(data),), -1, device=data.device, dtype=torch.long)
        data["episode"] = torch.full((len(data),), self.episode, device=data.device, dtype=torch.long)
        idx = super().extend(data)
        self[idx]["ep_end"] = idx
        self._this_end = idx[-1]
        self.end_episode()
        return idx

    def end_episode(self):
        if self._this_end < self._last_end_excl:  # handle wrap-around
            self["ep_end"][self._last_end_excl :] = self._this_end
            self["ep_end"][: self._this_end + 1] = self._this_end
        else:
            self["ep_end"][self._last_end_excl : self._this_end + 1] = self._this_end
        self._last_end_excl = self._this_end + 1
        self.episode += 1

    def _sample_goal_indices(self, indices: torch.Tensor):
        episode_ends = self["ep_end"][indices]
        episode_ends = torch.where(indices > episode_ends, episode_ends + len(self), episode_ends)

        # see Eysenbach et al: https://arxiv.org/pdf/2206.07568.pdf
        # note that torch.Tensor.geometric_ samples from [1, inf) which is what we want
        # https://pytorch.org/docs/stable/generated/torch.Tensor.geometric_.html
        geom = torch.empty_like(indices).geometric_(1 - self.cfg.discount)
        geom_indices = (indices + geom).clamp_max_(episode_ends) % len(self)
        random_indices = torch.randint_like(indices, len(self))

        rng = torch.rand_like(indices, dtype=torch.float32)
        rand_mask = rng < self.cfg.p_rand
        curr_mask = torch.logical_and(self.cfg.p_rand <= rng, rng < (self.cfg.p_rand + self.cfg.p_curr))
        future_mask = (self.cfg.p_rand + self.cfg.p_curr) <= rng

        goal_indices = torch.empty_like(indices).fill_(-1)
        goal_indices = torch.where(curr_mask, indices + 1, goal_indices)
        goal_indices = torch.where(future_mask, geom_indices, goal_indices)
        goal_indices = goal_indices.clamp_max_(episode_ends)  # the two above should be sampled within the episode
        # random indices are negative examples -> no need to clamp
        goal_indices = torch.where(rand_mask, random_indices, goal_indices)
        goal_indices = torch.where(goal_indices >= len(self), goal_indices - len(self), goal_indices)
        assert goal_indices.min() >= 0 and goal_indices.max() < len(self)
        return goal_indices

    def sample(self, batch_size: int | None = None, return_info: bool = None) -> TensorDictBase:
        if return_info is not None and return_info:
            raise NotImplementedError("return_info=True not implemented")
        r: TensorDict = super().sample(batch_size)  # [(B H+1), keys...]
        ep_len = self.cfg.n_steps + 1  # H + 1
        batch_size = batch_size or self._batch_size // ep_len
        r = r.reshape(batch_size, ep_len)  # [B, H+1,  keys...]
        r = r.transpose(0, 1)  # [H+1, B, keys...]
        goal_indices = self._sample_goal_indices(r["index"].flatten())
        r["goal"] = self["achieved_goal"][goal_indices].view(ep_len, batch_size, *self["achieved_goal"].shape[1:])
        assert r["goal"].size() == r["achieved_goal"].size()
        r = add_done_masks(r)
        return r


class GoalCondBuffer(EpisodicTensorDictReplayBuffer):
    def __init__(
        self,
        max_size: int,
        cfg: Container,
        **kwargs,
    ):
        print(f"Allocating buffer on {cfg.device} of size {max_size / 1024**3:.3f} GB")
        device = torch.device(cfg.device)
        super().__init__(
            storage=LazyTensorStorage(max_size, device),
            sampler=SliceSampler(
                slice_len=(cfg.n_steps + 1), traj_key="episode", end_key=None, truncated_key=None, strict_length=True
            ),
            writer=None,
            pin_memory=False,  # TODO: we currently assume that the buffer fits in target device memory
            batch_size=cfg.batch_size * (cfg.n_steps + 1),
            cfg=cfg,
            **kwargs,
        )
