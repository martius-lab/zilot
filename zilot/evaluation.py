import os
import tempfile
from typing import Any

import numpy as np
import torch
from numcodecs import Blosc
from omegaconf import Container, ListConfig
from tqdm.auto import tqdm
from zarr.storage import DirectoryStore

import wandb
import zilot.utils.dict_util as du
import zilot.utils.stat_util as su
from zilot.common.planning import Planner, make_planner_from_model
from zilot.common.task import Task, make_task_from_env
from zilot.envs import Env
from zilot.model import Model
from zilot.utils.collector_util import EpCollector, collect, zarr_to_dict
from zilot.utils.img_util import cat_videos
from zilot.utils.seed_util import set_seed

# from zilot.utils.plan_vis_util import render_plans


def _merge(*dicts: list[dict[str, Any]]) -> dict[str, Any]:
    res = {}
    for d in dicts:
        if d is None:
            continue
        shared_keys = set(res.keys()) & set(d.keys())
        if shared_keys:
            raise ValueError(f"Duplicate keys in dicts: {shared_keys}")
        res.update(d)
    return res


def eval_task_planner(
    col,
    cfg: Container,
    task: Task,
    planner: Planner,
    num_episodes: int,
    video_tag: str = "",
) -> None:
    # Register non-strict keys for plans
    for plan_key in ["points_a", "points_b", "coupling", "weights"]:
        col.register_key("plan/" + plan_key, strict=False)
    for i in range(cfg.num_iterations):
        col.register_key(f"traj/{i}", strict=False)
        col.register_key(f"scores/{i}", strict=False)

    set_seed(cfg.seed)

    task.reset_task()

    with tqdm(total=num_episodes * task.max_episode_length, desc=f"eval: {video_tag}", unit="step") as pbar:
        for n in range(num_episodes):
            done = False
            obs, info = task.reset()
            planner.reset(obs["desired_goal"])  # only pass goal to planner in reset

            col.reset()
            col.add({"goal": obs["desired_goal"].clone().cpu().numpy()})

            step = 0
            while not done:
                step_metrics = dict(step=step, ep=n)

                action, planner_metrics = planner.plan(obs)

                step_metrics = _merge(step_metrics, planner_metrics)

                obs, r, terminated, truncated, info = task.step(action)
                done = terminated or truncated

                step_metrics = _merge(
                    step_metrics,
                    dict(reward=r, obs=obs["observation"].clone()),
                )
                step_metrics = _merge(step_metrics, info.get("metrics", dict()))
                if cfg.render and task.render_mode == "rgb_array":
                    step_metrics["render"] = task.render()  # cv2.resize( (256, 256), cv2.INTER_LINEAR)
                col.add(du.apply(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, step_metrics))

                step += 1
                pbar.update(1)


def _cat_vids(**kwargs):
    vids = [list(v) for v in kwargs.values()]
    n = min(map(len, vids))
    return cat_videos(*[np.concatenate(v[:n]) for v in vids])


ENV_TASK = {
    "pointmaze_medium": ["goal", "path-sparse", "path-dense", "circle-sparse", "circle-dense"],
    "halfcheetah": [
        "frontflip",
        "frontflip-running",
        "hop-forward",
        "hop-backward",
        "backflip",
        "backflip-running",
        "run-forward",
        "run-backward",
        "goal",
    ],
    "fetch_slide_large_2D": ["S-dense", "S-sparse", "L-dense", "L-sparse", "U-dense", "U-sparse", "goal"],
    "fetch_push": ["S-dense", "S-sparse", "L-dense", "L-sparse", "U-dense", "U-sparse", "goal"],
    "fetch_pick_and_place": ["S-dense", "S-sparse", "L-dense", "L-sparse", "U-dense", "U-sparse", "goal"],
}


def evaluate(cfg: Container, model: Model, env: Env, log_all: bool = False, step=None) -> dict[str, Any]:
    planners = cfg.planner if isinstance(cfg.planner, ListConfig) else [cfg.planner]
    tasks = cfg.task if isinstance(cfg.task, ListConfig) else ([cfg.task] if cfg.task != "all" else ENV_TASK[cfg.env])
    evaluations = [(t, p) for t in tasks for p in planners]

    is_multi_eval = len(evaluations) > 1

    results = {}

    for t, p in evaluations:
        scratch_dir = os.environ.get("LOCAL_SCRATCH", "/tmp")
        print(f"Using scratch dir: {scratch_dir}")
        path = tempfile.mkdtemp(suffix=".zarr", dir=scratch_dir)

        task = make_task_from_env(t, cfg, env)
        planner = make_planner_from_model(p, cfg, model)
        with collect(path, EpCollector, DirectoryStore, max_ram="1GB", compressor=Blosc(cname="zstd", clevel=7)) as col:
            eval_task_planner(col, cfg, task, planner, cfg.num_rollouts, video_tag=f"{t}-{p}")
        del task
        del planner

        metrics = zarr_to_dict(path)
        metrics = {
            k: su.aggregate(
                d,
                success=su.Agg("any", "success"),
                last=su.Agg("any", "last"),
                W1=su.Agg("min", "W1"),
                gidx=su.Agg("max", "gidx"),
                goal_frac=su.Agg("max", "goal_frac"),
                ep_reward=su.Agg("sum", "reward"),
                video=su.Agg("stack", "render"),
                # plan=su.Agg(render_plans, "plan"),
            )
            for k, d in metrics.items()
        }
        metrics = su.stack_nested_dict(metrics)
        metrics = su.aggregate(
            metrics,
            sr=su.Agg("mean", "success"),
            last=su.Agg("mean", "last"),
            W1=su.Agg("mean", "W1"),
            gidx=su.Agg("mean", "gidx"),
            goal_frac=su.Agg("mean", "goal_frac"),
            ep_reward=su.Agg("mean", "ep_reward"),
            video=su.Agg("cat", "video"),
            # plan=su.Agg(_cat_vids, ("plan", "video")),
        )

        if log_all and not is_multi_eval:
            if cfg.zip_logs:  # no need to compress, just archive
                fd, arch_loc = tempfile.mkstemp(suffix=".tar", dir=scratch_dir)
                os.close(fd)
                os.system(f"tar -cf {arch_loc} -C {path} .")
                os.system(f"rm -rf {path}")
                art = wandb.Artifact(name="eval_results_arch", type="results")
                art.add_file(arch_loc)
                wandb.log_artifact(art)

            else:
                art = wandb.Artifact(name="eval_results", type="results")
                art.add_dir(path)
                wandb.log_artifact(art)
        else:
            os.system(f"rm -rf {path}")

        if is_multi_eval:
            results.setdefault(t, {})[p] = metrics
        else:
            results = metrics

    fd, table = tempfile.mkstemp(suffix=".csv", dir=scratch_dir)
    os.close(fd)
    with open(table, "w") as f:
        f.write("env,task,planner,seed,sr,last,W1,gidx,goal_frac,ep_reward\n")
        if is_multi_eval:
            for t, d in results.items():
                for p, m in d.items():
                    f.write(
                        f"{cfg.env},{t},{p},{cfg.seed},{m['sr']},{m['last']},{m['W1']},{m['gidx']},{m['goal_frac']},{m['ep_reward']}\n"  # noqa
                    )
        else:
            f.write(
                f"{cfg.env},{cfg.task},{cfg.planner},{cfg.seed},{results['sr']},{results['last']},{results['W1']},{results['gidx']},{results['goal_frac']},{results['ep_reward']}\n"  # noqa
            )
    table_name = "eval_table" + (f"_{step}" if step is not None else "")
    art = wandb.Artifact(name=table_name, type="eval_table")
    art.add_file(table)
    wandb.log_artifact(art)

    return results
