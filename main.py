import os
import sys

os.environ["HYDRA_FULL_ERROR"] = "1"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", 1)

import hydra
import seaborn as sns
import torch
from omegaconf import Container

sns.set_theme(style="white", font_scale=1.5)

import zilot.utils.tf_util  # noqa: F401 -> register ${tf:<>} resolver
from zilot.common.logger import Logger, make_logger
from zilot.common.training import train
from zilot.envs import Env, make_env
from zilot.envs.dsets import set_data_src
from zilot.evaluation import evaluate
from zilot.model import Model, make_model
from zilot.parse import parse_cfg
from zilot.utils.seed_util import set_seed

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="cfg", config_name="config", version_base="1.3.2")
def main(cfg: Container) -> None:
    parse_cfg(cfg)

    project_name = "zero-shot-il-ot"
    if "WORK" in os.environ.keys():
        data_src = os.path.join(os.environ["WORK"], project_name, "data")
    else:
        data_src = os.path.join(hydra.utils.get_original_cwd(), "data")
    set_data_src(data_src)

    # cfg.metadata.dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg.metadata.dir = os.path.join(hydra.utils.get_original_cwd(), "outputs")

    logger: Logger = make_logger(cfg)
    logger.init(**cfg.metadata)

    set_seed(cfg.seed)

    env: Env = make_env(cfg)

    model: Model = make_model(cfg)

    if cfg.job == "eval":
        logger.load_model(model, tag=cfg.load_tag)
        metrics = evaluate(cfg, model, env, log_all=True)
        logger.log({"eval": metrics})
    else:
        train(cfg, model, env, logger)

    logger.finish()


if __name__ == "__main__":
    main()
