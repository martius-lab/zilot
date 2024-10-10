import einops
import numpy as np
import torch
import torch.nn.functional as F
from functorch import combine_state_for_ensemble
from torch import nn

import zilot.types as ty
from zilot.model import Model
from zilot.model.util import rollout_fwd_ensemble
from zilot.third_party.mbrl.trajectory_opt import ICEMOptimizer


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness="different", **kwargs)
        self.vmap_ens = torch.vmap(fn, in_dims=(0, 0, 0), randomness="same", **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def forward_ensemble(self, *args, **kwargs):
        return self.vmap_ens([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr


class RunningScale(torch.nn.Module):
    """Online Normalizer"""

    def __init__(self, dim, tau=0.01, eps=1e-6):
        super().__init__()
        self.register_buffer("_min", -torch.ones(dim))
        self.register_buffer("_max", torch.ones(dim))
        self.register_buffer("_min_target", torch.full((dim,), torch.inf))
        self.register_buffer("_max_target", torch.full((dim,), -torch.inf))
        self.tau = tau
        self.eps = eps
        self._frozen = False

    def _stats(self, x: torch.Tensor):
        assert x.shape[-1] == self._min.shape[-1]
        x = x.reshape(-1, x.shape[-1])
        return x.min(0).values, x.max(0).values

    def update(self, x):
        if not self._frozen:
            new_min, new_max = self._stats(x)
            self._min_target = torch.minimum(self._min_target, new_min)
            self._max_target = torch.maximum(self._max_target, new_max)
        self._min.lerp_(self._min_target, self.tau)
        self._max.lerp_(self._max_target, self.tau)

    @property
    def mid(self):
        mid = (self._max + self._min) / 2.0
        return mid

    @property
    def range(self):
        return ((self._max - self._min) / 2.0).clamp_min(self.eps)

    def normalize(self, x):
        return (x - self.mid) / self.range

    def denormalize(self, x):
        return x * self.range + self.mid

    def forward(self, x):
        return self.normalize(x)


def mlp(in_dim, mlp_dims, out_dim, dropout=0.01):
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims
    mlp = []
    for d1, d2 in zip(dims, dims[1:]):
        mlp.append(nn.Sequential(nn.Linear(d1, d2), nn.Mish(), nn.Dropout(dropout)))
    mlp.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*mlp)


class DynamicsModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        state_dim = cfg.obs_shape[cfg.obs][0]
        self._dynamics = Ensemble(
            mlp(
                state_dim + cfg.action_dim,
                3 * [512],
                state_dim,
                dropout=0.05,
            )
            for _ in range(cfg.num_fwd)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def delta(self, z, a) -> torch.Tensor:
        """
        Predicts the next latent state given the current latent state and action.
        """
        B = torch.broadcast_shapes(z.shape[:-1], a.shape[:-1])
        z, a = z.expand(*B, -1), a.expand(*B, -1)
        x = torch.cat([z, a], dim=-1)
        return self._dynamics.forward(x)

    def delta_ensemble(self, z, a) -> torch.Tensor:
        """
        Predicts the next latent state given the current latent state and action.
        """
        B = torch.broadcast_shapes(z.shape[:-1], a.shape[:-1])
        z, a = z.expand(*B, -1), a.expand(*B, -1)
        x = torch.cat([z, a], dim=-1)
        return self._dynamics.forward_ensemble(x)


class Dynamics(Model):
    _provides = {
        "Fwd": True,
        "FwdEnsemble": True,
        "Enc": True,
        "EncG": True,
        "Dec": True,
        "DecG": True,
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device)

        horizon = 20
        self.curious_optimizer: ICEMOptimizer = ICEMOptimizer(
            num_iterations=3,
            elite_ratio=0.02,
            population_size=512,
            population_decay_factor=0.5,
            colored_noise_exponent=2.0,
            lower_bound=np.tile(np.full((cfg.action_dim,), -1.0), (horizon, 1)).tolist(),
            upper_bound=np.tile(np.full((cfg.action_dim,), 1.0), (horizon, 1)).tolist(),
            keep_elite_frac=1.0,
            alpha=0.1,
            device=cfg.device,
            return_mean_elites=False,
            population_size_module=None,
        )

        self.model = DynamicsModel(cfg)
        self.model.to(self.device)
        self.scale = RunningScale(cfg.obs_shape[cfg.obs][0])
        self.scale.to(self.device)
        self.delta_scale = RunningScale(cfg.obs_shape[cfg.obs][0])
        self.delta_scale.to(self.device)
        self.disagreement_scale = RunningScale(cfg.obs_shape[cfg.obs][0])  # not frozen, only for measuring disagreement
        self.disagreement_scale.to(self.device)

        self.optim = torch.optim.RAdam(self.model.parameters(), lr=1e-3, weight_decay=1e-4, decoupled_weight_decay=True)

    """ TRAINING """

    def update(self, batch: ty.Batch) -> dict:
        batch = self.preproc(batch)

        self.model.train()

        # NOTE: the z, zs, zs_gt here are all the actual states because this model does not use Enc/Dec
        zs_gt = batch["obs"]
        delta_gt = zs_gt[1:] - zs_gt[:-1]
        action = batch["action"]

        # update model scale
        self.scale.update(zs_gt)
        self.disagreement_scale.update(zs_gt)
        self.delta_scale.update(delta_gt)

        # Latent rollout
        z = zs_gt[0].expand(self.cfg.num_fwd, *zs_gt[0].shape)
        delta_n = torch.empty(self.cfg.n_steps, *z.shape, device=self.device)
        for t in range(self.cfg.n_steps):
            d_n = self.model.delta_ensemble(self.scale(z), action[t])  # [num_fwd, B, dim_z]
            delta_n[t] = d_n
            z = z + self.delta_scale.denormalize(d_n)

        delta_gt_n = self.delta_scale.normalize(delta_gt)

        # compute L2 error of 1-step prediction
        L2_err = torch.norm(delta_n.detach() - delta_gt_n.detach().unsqueeze(1).expand_as(delta_n), p=2, dim=-1)
        L2_err = L2_err[0].mean()

        rho = torch.pow(self.cfg.rho, torch.arange(self.cfg.n_steps, device=self.device))
        consistency_loss = F.mse_loss(delta_n, delta_gt_n.unsqueeze(1).expand_as(delta_n), reduction="none")
        consistency_loss = einops.reduce(consistency_loss, "T num_fwd B ... -> T B", "mean")
        consistency_loss = einops.reduce(consistency_loss, "T B -> T", "mean")
        consistency_loss = consistency_loss * rho
        consistency_loss = consistency_loss.mean()
        loss = consistency_loss * self.cfg.consistency_coef

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()

        self.model.eval()

        scale_r = self.scale.range
        delta_scale_r = self.delta_scale.range

        return {
            "consistency_loss": loss.item(),
            "L2_err": L2_err.item(),
            "grad_norm": grad_norm.item(),
            "scale.mean": scale_r.mean().item(),
            "scale.min": scale_r.min().item(),
            "scale.max": scale_r.max().item(),
            "scale_delta.mean": delta_scale_r.mean().item(),
            "scale_delta.min": delta_scale_r.min().item(),
            "scale_delta.max": delta_scale_r.max().item(),
        }

    def reset(self) -> None:
        self.curious_optimizer.reset()

    """ INFERENCE """

    def Enc(self, obs: ty.Obs) -> ty.Latent:
        return obs

    def EncG(self, task: ty.Goal) -> ty.GLatent:
        return task

    def Dec(self, z: ty.Latent) -> ty.Obs:
        return z

    def DecG(self, zg: ty.GLatent) -> ty.Goal:
        return zg

    @torch.compile(mode="max-autotune")
    def ensemble_disagreement(self, z: ty.Latent, a: ty.Action) -> ty.Value:
        zs = rollout_fwd_ensemble(self.FwdEnsemble, z, a, self.cfg.num_fwd)
        return self.disagreement_scale(zs).std(dim=0).nan_to_num_(0).clamp_(0, 10).mean(dim=(-1, -2))

    def TrainPi(self, z: ty.Latent, zg: ty.Latent) -> tuple[ty.Action, dict]:
        log = {}

        def _cb(_, disagreements: torch.Tensor, i: int):
            if i == self.curious_optimizer.num_iterations - 1:
                log["ensemble_disagreement"] = disagreements.max().item()

        a = self.curious_optimizer.optimize(lambda a: self.ensemble_disagreement(z, a), callback=_cb)[0]
        return a, log

    def Fwd(self, z: ty.Latent, a: ty.Action) -> ty.Latent:
        delta_n = self.model.delta(self.scale(z), a).mean(dim=0)
        return z + self.delta_scale.denormalize(delta_n)

    def FwdEnsemble(self, z: ty.Latent, a: ty.Action) -> ty.Latent:
        delta_n = self.model.delta_ensemble(self.scale(z), a)
        return z + self.delta_scale.denormalize(delta_n)

    """ UTILS"""

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.model.load_state_dict(state_dict["model"])

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["model"] = self.model.state_dict()
        return d

    def freeze_scales(self):
        self.scale._frozen = True
        self.delta_scale._frozen = True
        # NOTE: not freezing scale for disagreements
