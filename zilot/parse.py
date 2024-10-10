import re

import omegaconf
import torch
from omegaconf import Container, OmegaConf


def _set_debug_defaults(cfg: Container) -> Container:
    cfg.steps = 10_000
    cfg.seed_steps = 5
    cfg.log_freq = 1
    cfg.val_n = 64
    cfg.val_n_steps = cfg.n_steps
    cfg.val_freq = 1250 if cfg.val_freq is not None else None
    cfg.rollout_freq = 2500 if cfg.rollout_freq is not None else None
    cfg.num_rollouts = 2
    cfg.log_model_freq = None  # do not create model checkpoints
    cfg.log_dset_freq = None  # do not log dataset


def parse_cfg(cfg: Container) -> Container:
    # debug
    if cfg.debug:
        _set_debug_defaults(cfg)

    # fill in stuff
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    if OmegaConf.is_missing(cfg, "p_rand"):
        cfg.p_rand = 1.0 - cfg.p_curr - cfg.p_future
    if OmegaConf.is_missing(cfg, "p_curr"):
        cfg.p_curr = 1.0 - cfg.p_rand - cfg.p_future
    if OmegaConf.is_missing(cfg, "p_future"):
        cfg.p_future = 1.0 - cfg.p_rand - cfg.p_curr
    assert cfg.p_rand + cfg.p_curr + cfg.p_future == 1.0
    try:
        if OmegaConf.is_missing(cfg, "elite_ratio"):
            cfg.setdefault("elite_ratio", cfg.num_elites / cfg.population_size)
        elif OmegaConf.is_missing(cfg, "num_elites"):
            cfg.setdefault("num_elites", int(round(cfg.population_size * cfg.elite_ratio)))
    except Exception as e:
        raise omegaconf.errors.MissingMandatoryValue("Either num_elites or elite_ratio must be provided") from e

    # env/task naming convenience
    if not OmegaConf.is_missing(cfg, "env_task"):
        cfg.env, *task_parts = cfg.env_task.split("-")
        cfg.task = "-".join(task_parts)

    # defaults
    if cfg.job == "dset":
        cfg.setdefault("planner", ["gt_gt_mpc"])
        cfg.setdefault("train", "online")
        if cfg.log_dset_freq is None:
            cfg.log_dset_freq = cfg.steps
        cfg.setdefault("num_rollouts", 5)
        cfg.setdefault("render", True)
    if cfg.job == "train":
        cfg.setdefault("planner", ["pi", "mpc", "gt_gt_mpc"])
        cfg.setdefault("train", "offline")
        if cfg.log_model_freq is None:
            cfg.log_model_freq = cfg.steps
        cfg.setdefault("num_rollouts", 5)
        cfg.setdefault("render", True)
    if cfg.job == "eval":
        if OmegaConf.is_missing(cfg, "planner"):
            raise omegaconf.errors.MissingMandatoryValue("`planner` must be provided for `job` == 'eval'")

    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {cfg.device}")
    cfg.setdefault("val_n_steps", cfg.n_steps)
    cfg.setdefault("val_n", cfg.batch_size)
    cfg.val_n = max(int(cfg.val_n / (cfg.val_n_steps + 1)), 1) * (cfg.val_n_steps + 1)

    # Metadata
    extra_tags = ["debug"] if cfg.debug else []
    extra_tags.append(f"seed={cfg.seed}")
    extra_tags.append(cfg.env)
    extra_tags.extend([cfg.planner] if isinstance(cfg.planner, str) else [])
    extra_tags.extend([cfg.task] if isinstance(cfg.task, str) else [])
    extra_tags.extend(cfg.name.split("-"))
    extra_tags = [re.sub("[^0-9a-zA-Z]+", "-", tag) for tag in extra_tags if isinstance(tag, str) and len(tag) > 0]
    cfg.metadata.tags.extend(extra_tags)

    return cfg
