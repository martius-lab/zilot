import warnings

import cv2
import gymnasium as gym
import numpy as np
import torch
from omegaconf import Container
from tensordict import TensorDict
from tqdm.auto import tqdm

from zilot.common.buffer import GoalCondBuffer, add_done_masks
from zilot.common.logger import Logger
from zilot.envs import Env
from zilot.envs.dsets import get_dataset, log_custom_dset
from zilot.evaluation import evaluate
from zilot.model import Model
from zilot.utils.gym_util import generate_random_exploration_dataset
from zilot.validation import validate


def train(cfg: Container, model: Model, env: Env, logger: Logger) -> None:
    if cfg.train == "offline":
        offline(cfg, model, env, logger)
    elif cfg.train == "online":
        online(cfg, model, env, logger)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


def _is_step(i: int, freq: int | None, total: int) -> bool:
    return freq is not None and (i % freq == freq - 1 or i == total - 1)


def _make_buffer_from_dset(cfg: Container, dataset: list[TensorDict]) -> GoalCondBuffer:
    # filter out episodes that are too short
    dataset = [x for x in dataset if len(x) > cfg.n_steps]
    n_transitions = sum(len(x) for x in dataset)
    print(f"Total transitions: {n_transitions}")

    buffer = GoalCondBuffer(max_size=n_transitions, cfg=cfg)
    for td in tqdm(dataset, desc="Filling Buffer", total=len(dataset)):
        buffer.extend(td)
    assert len(buffer) == n_transitions, f"Buffer size mismatch: {len(buffer)} != {n_transitions}"

    return buffer


def _make_val_batch(cfg: Container, env: Env) -> TensorDict:
    ep_len = cfg.val_n_steps + 1
    assert cfg.val_n % (cfg.val_n_steps + 1) == 0
    b = cfg.val_n // (cfg.val_n_steps + 1) if not cfg.debug else 1
    rng = np.random.default_rng(cfg.seed)
    ep_lens = np.array([ep_len + rng.integers(0, cfg.max_episode_length)] * b, dtype=int)
    ep_batches = generate_random_exploration_dataset(env, seed=cfg.seed, episode_lengths=ep_lens)
    val_batch = torch.stack([b[-ep_len:] for b in ep_batches], dim=1)  # Float[Tensor, "ep_len b *"]
    # Take the last achieved goal as the goal
    val_batch["goal"] = val_batch["achieved_goal"][-1].clone().expand_as(val_batch["achieved_goal"])
    val_batch = add_done_masks(val_batch)
    return val_batch


def offline(cfg: Container, model: Model, env: Env, logger: Logger) -> None:
    dataset = get_dataset(cfg.dset, cfg, logger)
    buffer = _make_buffer_from_dset(cfg, dataset)
    val_batch = _make_val_batch(cfg, env)

    with tqdm(total=cfg.steps, desc="training") as pbar:
        for i in range(cfg.steps):
            metrics = {}
            log_model = _is_step(i, cfg.log_model_freq, cfg.steps)
            rollout = _is_step(i, cfg.rollout_freq, cfg.steps) or log_model
            val = _is_step(i, cfg.val_freq, cfg.steps) or rollout
            log = _is_step(i, cfg.log_freq, cfg.steps) or val

            batch = buffer.sample()

            nbatch = model.preproc(batch)
            train_metrics = model.update(nbatch)
            metrics["train"] = train_metrics

            if val:
                train_batch = torch.cat(
                    [buffer.sample() for _ in range(int(cfg.val_n / cfg.batch_size / cfg.n_steps) + 1)], dim=1
                )
                val_metrics = validate(cfg, model, env, train_batch)
                val_metrics_random = validate(cfg, model, env, val_batch.clone())
                metrics["val"] = {"dset": val_metrics, "random": val_metrics_random}

            if rollout:
                eval_metrics = evaluate(cfg, model, env, step=i)
                metrics["eval"] = eval_metrics

            if log:
                logger.log(metrics, step=i, commit=True)
            else:
                del metrics

            if log_model:
                logger.log_model(model)

            pbar.update(1)


def _env_obs_to_buffer_transition(obs: TensorDict, action: torch.Tensor, terminated: bool) -> TensorDict:
    obs["obs"] = obs["observation"]
    obs["action"] = action
    obs["done"] = torch.tensor(terminated, dtype=torch.bool)
    del obs["observation"]
    del obs["desired_goal"]
    return obs


def online(cfg: Container, model: Model, env: Env, logger: Logger) -> None:
    train_env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_length)

    buffer = GoalCondBuffer(max_size=cfg.steps + cfg.max_episode_length, cfg=cfg)
    val_batch = _make_val_batch(cfg, env)

    @torch.no_grad()
    def model_act(obs, goal):
        nobs, ngoal = model.preproc_obs(obs), model.preproc_goal(goal)
        z, zg = model.Enc(nobs), model.EncG(ngoal)
        nact = model.TrainPi(z, zg)
        if isinstance(nact, tuple):
            nact, log = nact
        else:
            log = {}
        return model.postproc_action(nact), log

    def rand_act(obs, goal):
        return train_env.get_wrapper_attr("rand_act")(), {}

    @torch.no_grad()
    def rollout_ep(max_n, act_fn, render=False):
        obs, _ = train_env.reset()
        terminated, truncated = False, False
        n = 0
        transitions = []
        logs = []
        imgs = []
        while not truncated:
            o, g = obs["observation"], obs["desired_goal"]
            action, log = act_fn(o, g)
            transitions.append(_env_obs_to_buffer_transition(obs, action, terminated))
            logs.append(log)
            n += 1
            obs, _, terminated, truncated, _ = train_env.step(action)
            truncated = truncated or n >= max_n
            if render:
                imgs.append(cv2.resize(train_env.render(), (256, 256), interpolation=cv2.INTER_LINEAR))
        if render:
            vid = np.stack(imgs, axis=0)
        return n, torch.stack(transitions), logs, vid if render else None

    assert (
        hasattr(env.action_space, "low")
        and (env.action_space.low == -1.0).all()
        and hasattr(env.action_space, "high")
        and (env.action_space.high == 1.0).all()
    ), "Action is assumed [-1, 1] in mbrl MPC planners"

    i = 0
    i_update = 0
    render = False
    with tqdm(desc="training", total=cfg.steps) as pbar:
        while i < cfg.steps:
            # ROLLOUT
            model.reset()
            n, transitions, roll_logs, vid = rollout_ep(
                cfg.steps - i, model_act if i >= cfg.seed_steps else rand_act, render
            )
            i += n
            if n > cfg.n_steps:
                buffer.extend(transitions)
            else:
                # NOTE: we could handle this case but it would include a non-strict sampler and then some ugly code
                warnings.warn(f"Rollout was too short: {n} steps (dropping)")
            pbar.update(n)

            # MODEL UPDATE
            if i < cfg.seed_steps:
                continue  # do not update model yet

            render = False
            while i_update <= min(i, cfg.steps - 1):
                metrics = {}

                render = render or _is_step(i_update, cfg.log_rollout_video_freq, cfg.steps)

                log_dset = _is_step(i_update, cfg.log_dset_freq, cfg.steps)
                log_model = _is_step(i_update, cfg.log_model_freq, cfg.steps)
                roll = _is_step(i_update, cfg.rollout_freq, cfg.steps) or log_model
                val = _is_step(i_update, cfg.val_freq, cfg.steps) or roll
                log = _is_step(i_update, cfg.log_freq, cfg.steps) or val or roll or log_model or log_dset

                batch = buffer.sample()
                nbatch = model.preproc(batch)
                train_metrics = model.update(nbatch)

                if i_update == cfg.freeze_scale_at:
                    model.freeze_scales()
                    print(f"freezing model scales at step={i_update}")

                metrics["train"] = train_metrics
                if len(roll_logs) > 0:
                    metrics["rollout"] = roll_logs.pop(0)
                if vid is not None and log:
                    metrics["rollout"]["video"] = vid
                    vid = None

                if val:
                    train_batch = torch.cat(
                        [buffer.sample() for _ in range(int(cfg.val_n / cfg.batch_size / cfg.n_steps) + 1)], dim=1
                    )
                    val_metrics = validate(cfg, model, env, train_batch)
                    val_metrics_random = validate(cfg, model, env, val_batch.clone())
                    metrics["val"] = {"dset": val_metrics, "random": val_metrics_random}

                if roll:
                    eval_metrics = evaluate(cfg, model, env, step=i_update)
                    metrics["eval"] = eval_metrics

                if log:
                    logger.log(metrics, step=i_update, commit=True)
                else:
                    del metrics

                # if log_model:
                #     logger.log_model(model)

                if log_dset:
                    log_custom_dset(logger, buffer, n=i_update + 1)

                i_update += 1
