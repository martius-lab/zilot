import abc
import math

import hydra
import torch
from omegaconf import Container

import zilot.types as ty
import zilot.utils.ot_util as ot_util
from zilot.envs import GOAL_TRANSFORMS
from zilot.model import Model
from zilot.model.util import rollout_fwd, rollout_fwd_z0
from zilot.utils.torch_util import append_single, prepend, prepend_single

# =====================================================================================================================
# Goal-reach classifiers
# =====================================================================================================================


class Classifier(torch.nn.Module):
    _needs = []

    def __init__(self, cfg: Container, model: Model, threshold: float | None, method: str = "Cls"):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.method = method
        if method == "Cls":
            self._needs.append("Cls")
            self.goal_threshold = threshold
        elif method == "V":
            self._needs.append("V")
            self.goal_threshold = threshold
        else:
            raise ValueError(f"Invalid method: {method}")

    def forward(self, z: ty.Latent, zg: ty.GLatent) -> ty.Value:
        if self.method == "Cls":
            return self.model.Cls(z, zg) >= self.goal_threshold
        elif self.method == "V":
            return -self.model.V(z, zg) <= self.goal_threshold
        else:
            raise ValueError(f"Invalid method: {self.method}")


# =====================================================================================================================
# MPC Objectives
# =====================================================================================================================


class Objective(abc.ABC):
    _needs = []

    def reset(self, zg: ty.Latent):
        pass

    def step(self, z: ty.Latent):
        pass

    @abc.abstractmethod
    def __call__(self, a: ty.Action):
        pass


# ===== Classic MPC ===================================================================================================

best_mpc_cls_cfg_thresholds = {
    "fetch_pick_and_place": 5,
    "fetch_push": 5,
    "fetch_slide_large_2D": 5,
    "halfcheetah": 5,
    "pointmaze_medium": 5,
}


class SequentialMyopicMPC(Objective):
    _needs = ["V", "Fwd"]

    def __init__(self, cfg: Container, model: Model, cls_cfg: Container, kind: str):
        self.cfg = cfg
        self.model = model
        if cfg.use_best_threshold:
            threshold = best_mpc_cls_cfg_thresholds[cfg.env]
            cls_cfg.threshold = threshold
            print(f"Using best threshold for {cfg.env}: {threshold}")
        self.cls = hydra.utils.instantiate(cls_cfg, cfg=cfg, model=model, _recursive_=False)
        self._needs.extend(self.cls._needs)
        if kind == "default_cls":
            self._needs.extend("Cls")
        self.kind = kind
        self.zg = None
        self.z = None
        self.idx = 0

    def reset(self, zg: ty.Latent):
        self.zg = zg
        self.idx = 0

    def step(self, z: ty.Latent):
        self.z = z
        while self.idx < self.zg.size(0) - 1 and self.cls(z, self.zg[self.idx]).item():
            self.idx += 1

    def _default_cls(self, zs: ty.Latent, zg: ty.GLatent):
        B, H = zs.size(0), zs.size(1)
        bootstrap = self.model.V(zs[..., -1, :], zg)
        immediate_rewards = torch.full((B, H - 1), -1.0, device=bootstrap.device, dtype=bootstrap.dtype)
        p_done_here = self.model.Cls(zs, zg)
        p_not_done_to_here = torch.cumprod(1 - p_done_here, dim=-1)
        p_not_done_to_here = p_not_done_to_here.roll(1, -1)
        p_not_done_to_here[..., 0] = 1.0
        v = p_not_done_to_here * torch.cat([immediate_rewards, bootstrap.unsqueeze(-1)], dim=-1)
        value = v.sum(dim=-1)
        log = dict()
        if self.cfg.draw_plans:
            log["plan"] = dict(
                zs=zs, zgs=zg.expand(B, 1, -1), coupling=torch.ones((B, H, 1), dtype=zs.dtype), weights=v.unsqueeze(-1)
            )
        return value, log

    def _default(self, zs: ty.Latent, zg: ty.GLatent):
        B, H = zs.size(0), zs.size(1)
        bootstrap = self.model.V(zs[..., -1, :], zg)
        immediate_rewards = torch.full((B, H - 1), -1.0, device=bootstrap.device, dtype=bootstrap.dtype)
        mask = ~self.cls(zs, zg).cummax(dim=-1).values
        v = mask.float() * torch.cat([immediate_rewards, bootstrap.unsqueeze(-1)], dim=-1)
        value = v.sum(dim=-1)
        log = dict()
        if self.cfg.draw_plans:
            log["plan"] = dict(
                zs=zs, zgs=zg.expand(B, 1, -1), coupling=torch.ones((B, H, 1), dtype=zs.dtype), weights=v.unsqueeze(-1)
            )
        return value, log

    def _uniform(self, zs: ty.Latent, zg: ty.GLatent):
        B = zs.size(0)
        v = self.model.V(zs, zg)
        mask = (~self.cls(zs, zg).cummax(dim=-1).values).float()
        weights = mask / mask.sum(dim=-1, keepdim=True)  # coupling is product measure of uniform
        v = v * weights
        value = v.sum(dim=-1)
        log = dict()
        if self.cfg.draw_plans:
            log["plan"] = dict(zs=zs, zgs=zg.expand(B, 1, -1), coupling=weights.unsqueeze(-1), weights=v.unsqueeze(-1))
        return value, log

    def __call__(self, a: ty.Action):
        zs = rollout_fwd_z0(self.model.Fwd, self.z, a)
        zg = self.zg[self.idx]
        if self.kind == "default":
            return self._default(zs, zg)
        elif self.kind == "default_cls":
            return self._default_cls(zs, zg)
        elif self.kind == "uniform":
            return self._uniform(zs, zg)
        else:
            raise ValueError(f"Invalid kind: {self.kind}")


# ===== ZILOT =========================================================================================================


class ZILOT(Objective):
    _needs = ["Fwd", "Vg", "V"]

    def __init__(
        self,
        cfg: Container,
        model: Model,
        **ot_kwargs,
    ):
        self.cfg = cfg
        self.model = model
        self.ot_kwargs = ot_kwargs

    def reset(self, g: ty.GLatent):
        assert g.dim() == 2
        self.g = g
        t = self.model.Vg(g[:-1], g[1:]).neg_().clamp_min_(0.0)
        self.g_time = torch.cat([torch.tensor([0.0], device=t.device, dtype=t.dtype), t]).cumsum(0)
        self.s = torch.empty((0, g.shape[1]), device=g.device, dtype=g.dtype)
        self.v = torch.empty((0, g.shape[0]), device=g.device, dtype=g.dtype)

    def step(self, s: ty.Latent):
        self.s = append_single(self.s, s)
        self.v = append_single(self.v, self.model.V(s, self.g))
        if self.v.size(0) == 1:
            self.g_time = self.g_time + self.v[0, 0].neg().clamp_min(0.0)  # add -V(s_0, g_1)

    def __call__(self, act: ty.Action) -> tuple[torch.Tensor, dict]:
        H = act.size(1)
        k = self.v.size(0) - 1
        reach = k + H
        g_mask = (self.g_time < reach).roll(1, 0)
        g_mask[0] = True
        g = self.g[g_mask]
        f = rollout_fwd(self.model.Fwd, self.s[-1], act)
        v_fg = self.model.V(f[..., None, :], g)  # [B H T]
        c = prepend(self.v[:, g_mask], v_fg, dim=1).neg().clamp_min(0.0)  # [B reach T]
        # make invariant to T_max and remove outliers that slow convergence
        c.mul_((1.0 / self.cfg.value_scale)).clamp_(0.0, 1.0)
        H_trunc = math.floor(max(1, min(H, self.g_time[-1].item() - k)))
        a = torch.zeros_like(c[:, :, 0])
        a[..., : k + H_trunc] = 1.0 / (k + H_trunc)
        b = torch.ones_like(c[:, 0, :]) / c.size(2)
        cost, pi = ot_util.sinkhorn_log_unbalanced(a, b, c, **self.ot_kwargs)
        log = dict()
        if self.cfg.draw_plans:
            log["plan"] = dict(
                zs=prepend(self.s, f, dim=1), zgs=self.g[g_mask].expand(c.size(0), -1, -1), coupling=pi, weights=c
            )
        if self.cfg.record_traj:
            log["traj"] = f
        return cost.neg(), log


# ===== ZILOT ablations ===============================================================================================


class ZILOTh(Objective):
    _needs = ["Fwd", "Vg", "V", "DecG", "Dec"]

    def __init__(
        self,
        cfg: Container,
        model: Model,
        **ot_kwargs,
    ):
        self.cfg = cfg
        self.model = model
        self.ot_kwargs = ot_kwargs

        self.gtf = GOAL_TRANSFORMS[cfg.env]

    def reset(self, g: ty.GLatent):
        assert g.dim() == 2
        self.g = g
        t = self.model.Vg(g[:-1], g[1:]).neg_().clamp_min_(0.0)
        self.g_time = torch.cat([torch.tensor([0.0], device=t.device, dtype=t.dtype), t]).cumsum(0)
        self.s = torch.empty((0, g.shape[1]), device=g.device, dtype=g.dtype)
        self.vh = torch.empty((0, g.shape[0]), device=g.device, dtype=g.dtype)

    def _h(self, s, g):
        g = self.model.DecG(g)
        s = self.gtf(self.model.Dec(s))
        # NOTE: rescale by step size so that the scale of the cost is similar to ZILOT ==> can reuse OT hparams
        return self.cfg.eval_metric(s, g) / self.cfg.step_size

    def step(self, s: ty.Latent):
        self.s = append_single(self.s, s)
        self.vh = append_single(self.vh, self._h(s, self.g))
        if self.vh.size(0) == 1:
            self.g_time = self.g_time + self.model.V(s, self.g).neg().clamp_min(0.0)  # add -V(s_0, g_1)

    def __call__(self, act: ty.Action) -> tuple[torch.Tensor, dict]:
        H = act.size(1)
        k = self.vh.size(0) - 1
        reach = k + H
        g_mask = (self.g_time < reach).roll(1, 0)
        g_mask[0] = True
        g = self.g[g_mask]
        f = rollout_fwd(self.model.Fwd, self.s[-1], act)
        vh_fg = self._h(f[..., None, :], g)  # [B H T]
        c = prepend(self.vh[:, g_mask], vh_fg, dim=1)  # [B reach T]
        # make invariant to T_max and remove outliers that slow convergence
        c.mul_((1.0 / self.cfg.value_scale)).clamp_(0.0, 1.0)
        H_trunc = math.floor(max(1, min(H, self.g_time[-1].item() - k)))
        a = torch.zeros_like(c[:, :, 0])
        a[..., : k + H_trunc] = 1.0 / (k + H_trunc)
        b = torch.ones_like(c[:, 0, :]) / c.size(2)
        cost, pi = ot_util.sinkhorn_log_unbalanced(a, b, c, **self.ot_kwargs)
        log = dict()
        if self.cfg.draw_plans:
            log["plan"] = dict(
                zs=prepend(self.s, f, dim=1), zgs=self.g[g_mask].expand(c.size(0), -1, -1), coupling=pi, weights=c
            )
        if self.cfg.record_traj:
            log["traj"] = f
        return cost.neg(), log


class ZILOTCls(Objective):
    _needs = ["Fwd", "Vg", "V"]

    def __init__(
        self,
        cfg: Container,
        model: Model,
        cls_cfg: Container,
        **ot_kwargs,
    ):
        self.cfg = cfg
        self.model = model
        self.cls = hydra.utils.instantiate(cls_cfg, cfg=cfg, model=model, _recursive_=False)
        self._needs.extend(self.cls._needs)
        self.ot_kwargs = ot_kwargs

    def reset(self, g: ty.GLatent):
        assert g.dim() == 2
        self.g_all = g
        self.g_idx = 0
        self.s = None

    def step(self, s: ty.Latent):
        self.s = s
        while self.g_idx < self.g_all.size(0) - 1 and self.cls(s, self.g_all[self.g_idx]).item():
            self.g_idx += 1
        # treat as ZILOT from here
        self.g = self.g_all[self.g_idx :]
        t = self.model.Vg(self.g[:-1], self.g[1:]).neg_().clamp_min_(0.0)
        self.g_time = prepend_single(self.model.V(s, self.g[0]).neg().clamp_min(0.0), t).cumsum(0)

    def __call__(self, act: ty.Action) -> tuple[torch.Tensor, dict]:
        H = act.size(1)
        k = 0  # NOTE: k = 0 always
        reach = k + H
        g_mask = (self.g_time < reach).roll(1, 0)
        g_mask[0] = True
        g = self.g[g_mask]
        f = rollout_fwd(self.model.Fwd, self.s, act)
        v_fg = self.model.V(f[..., None, :], g)  # [B H T]
        # NOTE: no history
        c = v_fg.neg().clamp_min(0.0)
        # make invariant to T_max and remove outliers that slow convergence
        c.mul_((1.0 / self.cfg.value_scale)).clamp_(0.0, 1.0)
        H_trunc = math.floor(max(1, min(H, self.g_time[-1].item() - k)))
        a = torch.zeros_like(c[:, :, 0])
        a[..., : k + H_trunc] = 1.0 / (k + H_trunc)
        b = torch.ones_like(c[:, 0, :]) / c.size(2)
        cost, pi = ot_util.sinkhorn_log_unbalanced_fixed(a, b, c, **self.ot_kwargs)
        log = dict()
        if self.cfg.draw_plans:
            log["plan"] = dict(
                zs=prepend(self.s, f, dim=1), zgs=self.g[g_mask].expand(c.size(0), -1, -1), coupling=pi, weights=c
            )
        if self.cfg.record_traj:
            log["traj"] = f
        return cost.neg(), log


class ZILOTbasic(Objective):
    _needs = ["Fwd", "V"]

    def __init__(
        self,
        cfg: Container,
        model: Model,
        **ot_kwargs,
    ):
        self.cfg = cfg
        self.model = model
        self.ot_kwargs = ot_kwargs

    def reset(self, g: ty.GLatent):
        assert g.dim() == 2
        self.g = g
        self.s = torch.empty((0, g.shape[1]), device=g.device, dtype=g.dtype)
        self.v = torch.empty((0, g.shape[0]), device=g.device, dtype=g.dtype)

    def step(self, s: ty.Latent):
        self.s = append_single(self.s, s)
        self.v = append_single(self.v, self.model.V(s, self.g))

    def __call__(self, act: ty.Action) -> tuple[torch.Tensor, dict]:
        f = rollout_fwd(self.model.Fwd, self.s[-1], act)
        v_fg = self.model.V(f[..., None, :], self.g)  # [B H T]
        c = prepend(self.v, v_fg, dim=1).neg().clamp_min(0.0)  # [B reach T]
        # make invariant to T_max and remove outliers that slow convergence
        c.mul_((1.0 / self.cfg.value_scale)).clamp_(0.0, 1.0)
        a = torch.ones_like(c[:, :, 0]) / c.size(1)
        b = torch.ones_like(c[:, 0, :]) / c.size(2)
        cost, pi = ot_util.sinkhorn_log_unbalanced(a, b, c, **self.ot_kwargs)
        log = dict()
        if self.cfg.draw_plans:
            log["plan"] = dict(
                zs=prepend(self.s, f, dim=1), zgs=self.g.expand(c.size(0), -1, -1), coupling=pi, weights=c
            )
        if self.cfg.record_traj:
            log["traj"] = f
        return cost.neg(), log
