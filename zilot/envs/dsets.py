import os

import numpy as np
import torch
from tensordict import TensorDict
from tqdm.auto import tqdm

from zilot.common.buffer import EpisodicTensorDictReplayBuffer as Buffer
from zilot.common.logger import Logger
from zilot.envs import DSETS, GOAL_TRANSFORMS
from zilot.utils import awgcsl_data_util, minari_dataset_util


def set_data_src(data_src: str) -> None:
    awgcsl_data_util.set_dataset_path(os.path.join(data_src, "awgcsl"))
    minari_dataset_util.set_dataset_path(os.path.join(data_src, "minari"))


def log_custom_dset(logger: Logger, buffer: Buffer, n: int) -> None:
    n = min(len(buffer), n)
    data = {
        "obs": buffer["obs"][:n],
        "achieved_goal": buffer["achieved_goal"][:n],
        "action": buffer["action"][:n],
        "done": buffer["done"][:n],
        "episode": buffer["episode"][:n],
    }
    data = {k: v.cpu().numpy() for k, v in data.items()}
    logger.log_dset(data)


def _load_custom_dset(cfg, logger: Logger, name: str) -> list[TensorDict]:
    tds = []
    dset = logger.load_dset(name)
    sz = dset["obs"].shape[0]
    beg = 0
    gtf = GOAL_TRANSFORMS.get(cfg.env, None)
    with tqdm(total=sz, desc="converting custom dataset") as pbar:
        while beg < sz:
            end = np.where(dset["episode"] == dset["episode"][beg])[0].max() + 1
            td = TensorDict(
                {
                    "obs": torch.from_numpy(dset["obs"][beg:end]),
                    "achieved_goal": torch.from_numpy(dset["achieved_goal"][beg:end]),
                    "action": torch.from_numpy(dset["action"][beg:end]),
                    "done": torch.from_numpy(dset["done"][beg:end]),
                },
                batch_size=end - beg,
            )
            if gtf is not None:  # allows rewriting the goal
                td["achieved_goal"] = gtf(td["obs"])
            tds.append(td)
            pbar.update(end - beg)
            beg = end
    return tds


def get_dataset(dset: str | None, cfg, logger: Logger | None = None) -> list[TensorDict]:
    if dset is None:
        raise ValueError("no dataset specified")
    data_src = dset.split("-")[0]
    if data_src == "awgcsl":
        dset_args = DSETS[cfg.env][dset]
        return awgcsl_data_util.get_dataset(*dset_args)
    elif data_src == "minari":
        dset_args = DSETS[cfg.env][dset]
        return minari_dataset_util.get_dataset(*dset_args)
    elif data_src == "custom":
        name = cfg.env + "-" + "-".join(dset.split("-")[1:])
        if logger is None:
            raise ValueError("need to set `logger` to load custom dataset")
        return _load_custom_dset(cfg, logger, name)
    else:
        raise ValueError(f"Unknown data source: {data_src}")
