import abc
import math
import os
import warnings
from typing import Any, Optional

import h5py
import hydra.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from omegaconf import Container, OmegaConf

import wandb
import zilot.utils.dict_util as du
from zilot.model import Model


class Logger(abc.ABC):
    def __init__(self, cfg: Container, dir: os.PathLike):
        self.cfg = cfg
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

    @abc.abstractmethod
    def init(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def finish(self, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def logdir(self) -> os.PathLike:
        pass

    @property
    @abc.abstractmethod
    def step(self) -> int:
        pass

    @abc.abstractmethod
    def log(self, metrics, step: Optional[int] = None, commit: bool = True):
        pass

    @abc.abstractmethod
    def log_model(self, model: Model, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def load_model(self, model: Model, name: str = None, tag: str = None):
        pass

    @abc.abstractmethod
    def log_dset(self, dset: dict[str, np.ndarray]) -> str:
        pass

    @abc.abstractmethod
    def load_dset(self, name: str, tag: str = None) -> dict[str, np.ndarray]:
        pass


def make_logger(cfg: Container) -> Logger:
    os.makedirs(cfg.metadata.dir, exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg, dir=cfg.metadata.dir, _recursive_=False)
    return logger


class WandbLogger(Logger):
    def __init__(self, cfg: Container, dir: os.PathLike):
        super().__init__(cfg, dir)
        wandb.login()

    def init(self, *args, **kwargs):
        wandb.init(
            *args,
            **kwargs,
            config=OmegaConf.to_container(self.cfg, resolve=False, throw_on_missing=False),
            settings=wandb.Settings(start_method="thread"),  # potentially needed for hydra multiruns
        )

    def finish(self, *args, **kwargs):
        wandb.finish(*args, **kwargs)

    @property
    def logdir(self) -> os.PathLike:
        x = os.path.join(wandb.run.dir, "custom")
        os.makedirs(x, exist_ok=True)
        return x

    @property
    def step(self) -> int:
        return wandb.run.step

    def _cwt_leaf(self, x) -> Any:
        if isinstance(x, Figure):
            r = wandb.Image(x)
            plt.close(x)
            return r
        elif isinstance(x, str) and os.path.isfile(x):
            name, ext = os.path.splitext(x)
            ext = ext[1:]
            name = os.path.basename(name)
            if ext in ["png", "jpg", "jpeg"]:
                return wandb.Image(x, caption=name)
            elif ext in ["mp4", "gif"]:
                return wandb.Video(x, caption=name, fps=10, format=ext)
            else:
                return x
        elif isinstance(x, pd.DataFrame):
            return wandb.Table(dataframe=x)
        elif isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[-1] in [1, 3, 4]:
                return wandb.Image(x)
            elif x.ndim == 4:
                # dir = os.path.join(self.logdir, "videos", str(self.step))
                # os.makedirs(dir, exist_ok=True)
                # fd, name = tempfile.mkstemp(dir=dir, suffix=".mp4")
                # os.close(fd)
                # imageio.mimwrite(name, x, fps=fps)
                # return wandb.Video(name, fps=fps, format="mp4")
                x = x.transpose(0, 3, 1, 2)
                n_frames = x.shape[0]
                max_vid_len = 30  # seconds
                fps = max(10, n_frames // max_vid_len)
                fps = min(fps, 60)  # clamp max
                return wandb.Video(x, fps=fps, format="mp4")
            else:
                warnings.warn(f"Unsupported ndarray shape {x.shape}")
        else:
            return x

    def log(self, metrics, step: Optional[int] = None, commit: bool = True):
        du.apply_(self._cwt_leaf, metrics)
        wandb.run.log(metrics, step=step, commit=commit)

    @property
    def _wandb_model_name(self) -> str:
        s = f"{self.cfg.env}-{self.cfg.seed}"
        if not OmegaConf.is_missing(self.cfg, "model_name") and self.cfg.model_name is not None:
            s = f"{self.cfg.model_name}-" + s
        if self.cfg.model == "fb":
            s = f"fb-{s}"
        return s

    def log_model(self, model: Model, **kwargs) -> str:
        state_dict = model.state_dict()
        kwargs.setdefault("step", self.step)
        name = "-".join([f"{k}={v}" for k, v in kwargs.items()])
        fp = os.path.join(self.logdir, "models", name + ".pth")
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        torch.save(state_dict, fp)
        wandb.log_model(fp, name=self._wandb_model_name)
        print(f"Logged model {self._wandb_model_name} ({name})")
        return self._wandb_model_name

    def load_model(self, model: Model, name: str = None, tag: str = None):
        name = name or self._wandb_model_name
        tag = tag or "latest"
        fp = wandb.use_model(f"{name}:{tag}")
        state_dict = torch.load(fp, map_location=model.device)
        model.load_state_dict(state_dict)
        print(f"Loaded model {name}:{tag} ({os.path.basename(fp)})")

    @property
    def _wandb_dset_name(self) -> str:
        kw = [] if self.cfg.dset is None else self.cfg.dset.split("-")[1]
        return "-".join([self.cfg.env, str(self.cfg.seed), *kw])

    def log_dset(self, dset: dict[str, np.ndarray]) -> str:
        dset_size = math.ceil(dset[next(iter(dset))].shape[0] / 1000)
        name = self._wandb_dset_name + "-" + str(dset_size) + "k"
        d = os.path.join(self.logdir, "dsets")
        os.makedirs(d, exist_ok=True)
        file = os.path.join(d, name + ".hdf5")
        with h5py.File(file, "w") as f:
            for k, v in dset.items():
                f.create_dataset(k, data=v, compression="gzip")
        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_file(file)
        wandb.log_artifact(artifact)

    def load_dset(self, name: str, tag: str = None) -> dict[str, np.ndarray]:
        name = name or self._wandb_dset_name
        tag = tag or "latest"
        artf: wandb.Artifact = wandb.use_artifact(f"{name}:{tag}")
        dir = artf.download()
        if dir == "":
            raise FileNotFoundError(f"Artifact {name}:{tag} directory is empty!")
        file = os.path.join(dir, name + ".hdf5")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Artifact {name}:{tag} not found (should be at {file})")
        with h5py.File(file, "r") as f:
            dset = {k: np.array(v) for k, v in f.items()}
        return dset
