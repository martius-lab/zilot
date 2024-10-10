import os

import minari
import torch
from tensordict import TensorDict
from tqdm.auto import tqdm

os.environ["MINARI_DATASETS_PATH"] = os.path.join(os.getcwd(), "data", "minari")


def set_dataset_path(path: str):
    os.makedirs(path, exist_ok=True)
    os.environ["MINARI_DATASETS_PATH"] = path


def _cvt_to_td_dset(dset: minari.MinariDataset) -> list[TensorDict]:
    tds = []
    for ep in tqdm(dset.iterate_episodes(), desc="Converting Minari Dataset", total=dset.total_episodes, unit="ep"):
        tds.append(
            TensorDict(
                dict(
                    obs=torch.from_numpy(ep.observations["observation"][:-1]).float(),
                    achieved_goal=torch.from_numpy(ep.observations["achieved_goal"][:-1]).float(),
                    action=torch.from_numpy(ep.actions).float(),
                    done=torch.from_numpy(ep.terminations).bool(),
                ),
                batch_size=ep.total_timesteps,
            )
        )
    return tds


def get_dataset(minari_dset_name: str):
    minari_dset = minari.load_dataset(minari_dset_name, download=True)
    return _cvt_to_td_dset(minari_dset)


if __name__ == "__main__":
    import gymnasium as gym
    import gymnasium_robotics  # noqa F401
    import matplotlib.pyplot as plt

    from zilot.utils.gym_util import TensorWrapper, generate_random_exploration_dataset

    env = TensorWrapper(gym.make("AntMaze_Medium_Diverse_GR-v4"))
    dset = generate_random_exploration_dataset(env, episode_lengths=[1000] * 100)
    states_gt = torch.cat([d["achieved_goal"] for d in dset], dim=0)

    set_dataset_path(os.path.join(os.getcwd(), "data", "minari"))
    data = get_dataset("antmaze-medium-diverse-v1")
    states = torch.cat([d["achieved_goal"] for d in data], dim=0)

    plt.figure()
    plt.title("Minari vs. Random Rollouts in AntMaze_Medium_Diverse_GR-v4")
    plt.scatter(states[:, 0], states[:, 1], label="minari", alpha=0.1, c="red", s=0.1)
    plt.scatter(states_gt[:, 0], states_gt[:, 1], label="random gen.", alpha=0.1, c="blue", s=0.1)
    plt.legend()
    path = "/tmp/minari_dataset.png"
    plt.savefig(path)
    print(f"Saved dataset visualization to {path}")
