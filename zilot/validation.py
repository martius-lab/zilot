from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from omegaconf import Container
from tensordict import TensorDict

from zilot.envs import Env
from zilot.model import Model
from zilot.model.util import gc_value_to_steps, rollout_fwd
from zilot.utils.plot_util import fig2data


@torch.no_grad()
def _plot_validation(cfg: Container, model: Model, env: Env, batch: TensorDict, n_goals: int = 1) -> None:
    obs = batch["obs"].flatten(0, -2)
    goals = batch["goal"][-1, :n_goals]
    nobs = model.preproc_obs(obs)
    ngoals = model.preproc_goal(goals)
    zs = model.Enc(nobs)
    zgs = model.EncG(ngoals)
    B, n_goal = zs.size(0), zgs.size(0)
    zs = zs.expand(n_goal, B, -1)
    zgs = zgs.unsqueeze(1).expand(n_goal, B, -1)
    fig, axs = plt.subplots(n_goal, 3, figsize=(15, 5 * n_goal + 1))
    if n_goal == 1:
        axs = [axs]
    n_fails = 0
    if cfg.available.V:
        v = model.V(zs, zgs)
        try:
            for v, ax, goal in zip(v, axs, goals):
                env.get_wrapper_attr("plot_values")(ax[1], v, obs, goal.unsqueeze(0))
        except NotImplementedError:
            n_fails += 1
            pass
    if cfg.available.Vg:
        vg = model.Vg(zs, zgs)
        try:
            for vg, ax, goal in zip(vg, axs, goals):
                env.get_wrapper_attr("plot_values")(ax[2], vg, obs, goal.unsqueeze(0))
        except NotImplementedError:
            n_fails += 1
            pass
    if cfg.available.Pi:
        mu, _, _, _ = model.Pi(zs, zgs)
        try:
            for a, ax, goal in zip(mu, axs, goals):
                env.get_wrapper_attr("plot_policy")(ax[0], a, obs, goal.unsqueeze(0))
        except NotImplementedError:
            n_fails += 1
            pass
    for ax in axs:
        ax[0].set_title("Policy")
        ax[1].set_title("Value")
        ax[2].set_title("Goal Value")
    fig.tight_layout()
    if n_fails == 3:
        return None
    return fig2data(fig)


@torch.no_grad()
def _val_v_mrr(V, zs, zs_as_zgs) -> float:
    NUM = min(2**8, zs.size(0))
    CHUNK = min(2**6, zs.size(0))
    zs = zs.clone().view(-1, zs.size(-1))  # [n, z_dim]
    zs_as_zgs = zs_as_zgs.clone().view(-1, zs_as_zgs.size(-1))  # [n, z_dim]
    idx = torch.randint(zs.size(0), (NUM,))
    zg = zs_as_zgs[idx]  # [NUM, z_dim]
    v = torch.empty(zs.size(0), NUM, device=zs.device)
    for i in range(0, zs.size(0), CHUNK):
        v[i : min(i + CHUNK, zs.size(0))] = V(zs[i : min(i + CHUNK, zs.size(0)), None], zg)
    ranks = 1.0 + (v[idx, torch.arange(idx.size(0), device=idx.device, dtype=idx.dtype)] < v).sum(dim=-1).float()
    mrr = (1.0 / ranks).mean().item()
    return mrr


@torch.no_grad()
def validate(cfg: Container, model: Model, env: Env, batch: TensorDict) -> Dict[str, Any]:
    val_metrics = {}

    val = _plot_validation(cfg, model, env, batch.clone(), n_goals=4)
    if val is not None:
        val_metrics["fig"] = val

    batch = batch.to(model.device)

    eval_metric_dist_to_goal = cfg.eval_metric(batch["achieved_goal"], batch["goal"])
    dist_0_under_eval_metric = eval_metric_dist_to_goal <= cfg.goal_success_threshold
    invalid_under_eval = torch.logical_and(  # once we are inside threshold, going away from goal is invalid state
        eval_metric_dist_to_goal >= eval_metric_dist_to_goal.roll(1, 0), dist_0_under_eval_metric.cummax(0).values
    )
    invalid_under_eval[0] = False
    invalid_under_eval = invalid_under_eval
    dist_to_goal = (~dist_0_under_eval_metric).float()
    dist_to_goal = dist_to_goal.cummin(0).values  # stay 0 if 0
    dist_to_goal = dist_to_goal.flip(0).cumsum(0).flip(0)  # sum from the right
    dist_n_masks = [torch.logical_and(~invalid_under_eval, dist_to_goal == n) for n in range(len(batch["obs"]))]
    n = 0
    while n < len(dist_n_masks) and dist_n_masks[n].any():
        n += 1
    dist_n_masks = dist_n_masks[:n]  # only keep the ones with pos entries

    nbatch = model.preproc(batch)
    nacts = nbatch["action"]
    zs = model.Enc(nbatch["obs"])
    zs_as_zgs = model.EncG(nbatch["achieved_goal"])
    zgs = model.EncG(nbatch["goal"])

    x_min = eval_metric_dist_to_goal.min().item()
    x_max = eval_metric_dist_to_goal.max().item()
    x = torch.linspace(x_min, x_max, 100)
    y1 = x / cfg.step_size
    y2 = (x - cfg.goal_success_threshold) / cfg.step_size
    avg_z_diff = torch.norm(zs[:-1] - zs[1:], dim=-1, p=2).mean()

    if cfg.available.Fwd:
        z_next = rollout_fwd(model.Fwd, zs[0], nacts[:-1].transpose(0, 1)).transpose(0, 1)
        z_next_gt = zs[1:]
        d = torch.norm(z_next - z_next_gt, dim=-1, p=2) / avg_z_diff
        rrnorm = (d.mean(-1) / torch.arange(1, d.size(0) + 1, device=d.device, dtype=d.dtype)).sum()
        val_metrics["fwd.rrnorm"] = rrnorm.item()
        d_mean = d.mean(-1)
        d_std = d.std(-1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.errorbar(
            range(1, len(d_mean) + 1), d_mean.cpu().numpy(), yerr=d_std.nan_to_num(1e-20).clamp_min(1e-20).cpu().numpy()
        )
        ax.set_title("Fwd relative MSE")
        ax.set_xlabel("# Steps")
        ax.set_ylabel("rel MSE")
        ax.grid()
        fig.tight_layout()
        val_metrics["fwd.rel_mse"] = fig2data(fig)

    if cfg.available.V:
        val_metrics["V.mrr"] = _val_v_mrr(model.V, zs[0], zs_as_zgs[0])
        v_same = model.V(zs, zs_as_zgs)
        val_metrics["V.@same.mean"] = v_same.mean().item()
        val_metrics["V.@same.std"] = v_same.std().item()
        v_dist_n = [model.V(zs[mask], zgs[mask]) for mask in dist_n_masks]
        for i in range(min(3, len(v_dist_n))):
            val_metrics[f"V.@{i}"] = v_dist_n[i].mean().item()
        v_dist_n = [gc_value_to_steps(v) for v in v_dist_n]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax.errorbar(
            range(len(v_dist_n)),
            [v.mean().item() for v in v_dist_n],
            [v.nan_to_num(1e-20).clamp_min(1e-20).std().item() for v in v_dist_n],
        )
        ax.set_title("V values")
        ax.set_xlabel("Steps to goal")
        ax.set_ylabel("Value")
        ax.grid()
        fig.tight_layout()
        vs = model.V(zs, zgs)
        vs = gc_value_to_steps(vs)
        ax2.scatter(
            eval_metric_dist_to_goal[~invalid_under_eval].cpu().numpy(),
            vs[~invalid_under_eval].cpu().numpy(),
            s=1,
            color="b",
        )
        ax2.plot(x.cpu().numpy(), y1.cpu().numpy(), color="red")
        ax2.plot(x.cpu().numpy(), y2.cpu().numpy(), color="orange")
        ax2.set_title(f"Steps vs. Eval Metric (env_step={cfg.step_size})")
        ax2.set_xlabel("Distance to goal")
        ax2.set_ylabel("Predicted Steps")
        val_metrics["V.pred"] = fig2data(fig)

    if cfg.available.Vg:
        val_metrics["Vg.mrr"] = _val_v_mrr(model.Vg, zs_as_zgs[0], zs_as_zgs[0])
        vg_same = model.Vg(zs_as_zgs, zs_as_zgs)
        val_metrics["Vg.@same.mean"] = vg_same.mean().item()
        val_metrics["Vg.@same.std"] = vg_same.std().item()
        vg_dist_n = [model.Vg(zs_as_zgs[mask], zgs[mask]) for mask in dist_n_masks]
        for i in range(min(3, len(vg_dist_n))):
            val_metrics[f"Vg.@{i}"] = vg_dist_n[i].mean().item()
        vg_dist_n = [gc_value_to_steps(vg) for vg in vg_dist_n]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax.errorbar(
            range(len(vg_dist_n)),
            [vg.mean().item() for vg in vg_dist_n],
            [vg.std().nan_to_num(1e-20).clamp_min(1e-20).item() for vg in vg_dist_n],
        )
        ax.set_title("Vg values")
        ax.set_xlabel("Steps to goal")
        ax.set_ylabel("Value")
        ax.grid()
        fig.tight_layout()
        vgs = model.Vg(zs_as_zgs, zgs)
        vgs = gc_value_to_steps(vgs)
        ax2.scatter(
            eval_metric_dist_to_goal[~invalid_under_eval].cpu().numpy(),
            vgs[~invalid_under_eval].cpu().numpy(),
            s=1,
            color="b",
        )
        ax2.plot(x.cpu().numpy(), y1.cpu().numpy(), color="red")
        ax2.plot(x.cpu().numpy(), y2.cpu().numpy(), color="orange")
        ax2.set_title(f"Steps vs. Eval Metric (env_step={cfg.step_size})")
        ax2.set_xlabel("Distance to goal")
        ax2.set_ylabel("Predicted Steps")
        val_metrics["Vg.pred"] = fig2data(fig)

    return val_metrics
