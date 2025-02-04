import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("tf", lambda x: hydra.utils.get_method("zilot.utils.tf_util." + x))


def l2_norm(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return torch.norm(x - y, dim=-1, p=2)
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.linalg.norm(x - y, axis=-1, ord=2)
    else:
        raise ValueError(f"Unsupported types {type(x)} and {type(y)}")


def yaw_diff(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x_v = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        y_v = torch.cat([torch.cos(y), torch.sin(y)], dim=-1)
        d_v = torch.acos(torch.sum(x_v * y_v, dim=-1))
        return d_v.abs()
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        x_v = np.concatenate([np.cos(x), np.sin(x)], axis=-1)
        y_v = np.concatenate([np.cos(y), np.sin(y)], axis=-1)
        d_v = np.arccos(np.sum(x_v * y_v, axis=-1))
        return np.abs(d_v)
    else:
        raise ValueError(f"Unsupported types {type(x)} and {type(y)}")


if __name__ == "__main__":
    x = torch.linspace(-4 * torch.pi, 4 * torch.pi, 1000).unsqueeze(-1)
    y = torch.zeros_like(x)
    d = yaw_diff(x, y)

    import matplotlib.pyplot as plt

    t = torch.arange(1000)
    plt.plot(t, x, label="yaw1")
    plt.plot(t, y, label="yaw2")
    plt.plot(t, d, label="yaw_diff")
    plt.hlines(torch.pi, 0, 1000, color="b", label="pi")
    plt.legend()
    plt.savefig("/tmp/yaw_diff.png")
    print("/tmp/yaw_diff.png")
