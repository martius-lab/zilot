import os
import pickle

import torch
from tensordict import TensorDict
from tqdm.auto import tqdm

DIR = os.path.join(os.getcwd(), "data", "awgcsl")


def set_dataset_path(cache_dir):
    global DIR
    os.makedirs(cache_dir, exist_ok=True)
    DIR = cache_dir


def convert_buffer(path):
    tds = []
    with open(path, "rb") as f:
        buffer = pickle.load(f)
        N = buffer["u"].shape[0]

        for i in tqdm(range(N), desc="Converting AWGCSL buffer"):
            obs = torch.from_numpy(buffer["o"][i][:-1]).float()
            achieved_goal = torch.from_numpy(buffer["ag"][i][:-1]).float()
            action = torch.from_numpy(buffer["u"][i]).float()

            tds.append(
                TensorDict(dict(obs=obs, action=action, achieved_goal=achieved_goal), batch_size=len(obs), device="cpu")
            )

    return tds


def get_dataset(env_name, kind=None) -> list[TensorDict]:
    kinds = ["random", "expert"]
    if kind is None or kind == "all":
        kind = kinds
    elif kind not in kinds:
        raise ValueError(f"Kind {kind} not supported")
    else:
        kind = [kind]
    paths = [os.path.join(DIR, k, env_name, "buffer.pkl") for k in kind]
    datasets = [convert_buffer(p) for p in paths]
    return [td for x in zip(*datasets) for td in x]
