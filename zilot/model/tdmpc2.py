"""
Contains the contents of
    - tdmpc2.py
    - layers.py
    - math.py
    - scale.py
    - world_model.py
from https://github.com/nicklashansen/tdmpc2
with minor changes
"""

import math
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble

import zilot.types as ty
from zilot.model import Model


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.trunc_normal_(p, std=0.02)  # Weight
                nn.init.constant_(m[i + 1], 0)  # Bias


def zero_(params):
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)


def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


@torch.jit.script
def comp_log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std


@torch.jit.script
def _gaussian_logprob(residual):
    return residual - 0.5 * math.log(2 * math.pi)


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
    return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdim=True)
    return mu, pi, log_pi


@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


DREG_BINS = None


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    global DREG_BINS
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    if DREG_BINS is None:
        DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
    return symexp(x)


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness="different", **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div_(255.0).sub_(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, shape):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    out = {}
    for k in shape:
        if k == "state":
            out[k] = mlp(
                shape[k][0],
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
        elif k == "rgb":
            out[k] = conv(shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out)


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Modifications:
    - removed task embeddings
    - added goal conditioning
    - added extra MLPs for value functions
    - added decoders for visualization
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = enc(cfg, cfg.obs_shape)
        self._goal_encoder = enc(cfg, cfg.goal_shape)
        self._dynamics = mlp(cfg.latent_dim + cfg.action_dim, 2 * [cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg))
        self._pi = mlp(cfg.latent_dim + cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
        self._reward = mlp(cfg.latent_dim + cfg.action_dim + cfg.latent_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))
        self._Qs = Ensemble(
            [
                mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.latent_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )

        # extra MLPs
        self._Vg = mlp(cfg.latent_dim + cfg.latent_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))
        self._V_phi = mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 512)
        self._V_psi = mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 512)

        # decoder for visualization
        self.register_module("_decoder", None)
        self.register_module("_goal_decoder", None)
        if cfg.obs == "state":
            self._decoder = mlp(
                cfg.latent_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.obs_shape[self.cfg.obs][0],
            )
        if cfg.goal == "state":
            self._goal_decoder = mlp(
                cfg.latent_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.goal_shape[self.cfg.goal][0],
            )

        self.apply(weight_init)
        zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)

    def encode(self, obs):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def encode_goal(self, goal):
        """
        Encodes a goal observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.goal == "rgb" and goal.ndim == 5:
            return torch.stack([self._goal_encoder[self.cfg.goal](o) for o in goal])
        return self._goal_encoder[self.cfg.goal](goal)

    def decode(self, z):
        """
        Decodes a latent representation into an observation.
        This implementation assumes a single state-based observation.
        """
        assert self._decoder is not None
        return self._decoder(z)

    def decode_goal(self, zg):
        """
        Decodes a goal latent representation into an observation.
        This implementation assumes a single state-based observation.
        """
        assert self._goal_decoder is not None
        return self._goal_decoder(zg)

    def next(self, z, a):
        """
        Predicts the next latent state given the current latent state and action.
        """
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def pi(self, z, zg):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """

        # Gaussian policy prior
        z = torch.cat(torch.broadcast_tensors(z, zg), dim=-1)
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = comp_log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        log_pi = gaussian_logprob(eps, log_std, size=None)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def reward(self, z, a, zg):
        """
        Predicts instantaneous (single-step) reward.
        """
        B = torch.broadcast_shapes(z.shape[:-1], a.shape[:-1], zg.shape[:-1])
        z = torch.cat([x.expand(B + (-1,)) for x in [z, a, zg]], dim=-1)
        return self._reward(z)

    def Q(self, z, a, zg, return_type="min", target=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        B = torch.broadcast_shapes(z.shape[:-1], a.shape[:-1], zg.shape[:-1])
        z = torch.cat([x.expand(B + (-1,)) for x in [z, a, zg]], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == "all":
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = two_hot_inv(Q1, self.cfg), two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2

    def Vg(self, z1, z2):
        """
        Predicts the value function with goal-conditioned architecture.
        """
        z1, z2 = torch.broadcast_tensors(z1, z2)
        z = torch.cat([z1, z2], dim=-1)
        return self._Vg(z)

    def V(self, z, zg):
        """
        Predicts the value function with 2-stream architecture.
        """
        z_phi = self._V_phi(z)
        zg_psi = self._V_psi(zg)
        return torch.einsum("... d, ... d -> ...", z_phi, zg_psi)


class RunningScale:
    """Running trimmed scale estimator."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._value = torch.ones(1, dtype=torch.float32, device=torch.device(cfg.device))
        self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=torch.device(cfg.device))

    def state_dict(self):
        return dict(value=self._value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict["value"])
        self._percentiles.data.copy_(state_dict["percentiles"])

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self._value.data.lerp_(value, self.cfg.tau)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        return f"RunningScale(S: {self.value})"


class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {"params": self.model._encoder.parameters(), "lr": self.cfg.lr * self.cfg.enc_lr_scale},
                {"params": self.model._goal_encoder.parameters(), "lr": self.cfg.lr * self.cfg.enc_lr_scale},
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
            ],
            lr=self.cfg.lr,
        )
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.vg_optim = torch.optim.Adam(self.model._Vg.parameters(), lr=self.cfg.lr)
        self.v_optim = torch.optim.Adam(
            [{"params": self.model._V_phi.parameters()}, {"params": self.model._V_psi.parameters()}], lr=self.cfg.lr
        )
        self.dec_optim = torch.optim.Adam(
            [{"params": x.parameters()} for x in [self.model._decoder, self.model._goal_decoder] if x is not None],
            lr=self.cfg.lr,
        )
        self.model.eval()
        self.scale = RunningScale(cfg)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def update_pi(self, zs, zg):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.
                zg (torch.Tensor): Goal latent state.

        Returns:
                float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, zg)
        qs = self.model.Q(zs, pis, zg, return_type="avg")
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    def update_vg(self, z1, z2, target):
        self.vg_optim.zero_grad(set_to_none=True)
        v = self.model.Vg(z1, z2)
        loss = 0.0
        for t in range(self.cfg.n_steps):
            loss += soft_ce(v[t], target[t], self.cfg).mean() * self.cfg.rho**t
        loss *= 1 / self.cfg.n_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._Vg.parameters(), self.cfg.grad_clip_norm)
        self.vg_optim.step()
        return loss.item()

    def update_v(self, z, zg, target):
        self.v_optim.zero_grad(set_to_none=True)
        scale = float(self.cfg.value_scale)
        v = self.model.V(z, zg)
        loss = 0.0
        for t in range(self.cfg.n_steps):
            loss += F.mse_loss(v[t] / scale, target[t].squeeze(-1) / scale) * self.cfg.rho**t
        loss *= 1 / self.cfg.n_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*self.model._V_phi.parameters(), *self.model._V_psi.parameters()], self.cfg.grad_clip_norm
        )
        self.v_optim.step()
        return loss.item()

    def update_dec(self, z, obs, zg, goal):
        self.dec_optim.zero_grad(set_to_none=True)
        dec_loss = 0.0
        if self.model._decoder is not None:
            dec_obs = self.model.decode(z)
            dec_loss += F.mse_loss(dec_obs, obs)
        if self.model._goal_decoder is not None:
            dec_goal = self.model.decode_goal(zg)
            dec_loss += F.mse_loss(dec_goal, goal)
        dec_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for x in [self.model._decoder, self.model._goal_decoder] if x is not None for p in x.parameters()],
            self.cfg.grad_clip_norm,
        )
        self.dec_optim.step()
        return dec_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, zg, done, next_done):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                zg (torch.Tensor): Goal latent states.
                done (torch.Tensor): z is terminal state (td-target is 0).
                next_done (torch.Tensor): z_next is terminal state (td-target is -1).

        Returns:
                torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, zg)[1]
        v = self.model.Q(next_z, pi, zg, return_type="min", target=True)
        r = -1.0
        td_target = r + v
        td_target[next_done] = r  # No bootstrapping if next_z state is terminal
        td_target[done] = 0.0  # NOTE: we explicitly drive Q-values on terminal states to 0.
        return td_target.clamp(-self.cfg.value_scale, 0.0)  # NOTE: since we don't discount we have to clamp

    def update(self, td):
        obs = td["obs"]
        action = td["action"][:-1]
        goal = td["goal"]
        achieved_goal = td["achieved_goal"]
        done = td["done"]
        next_done = td["next_done"]
        reward = (-1.0 + done.float()).unsqueeze(-1)

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:])
            all_zg = self.model.encode_goal(goal)
            zs_as_zg = self.model.encode_goal(achieved_goal[:-1])
            td_targets = self._td_target(next_z, all_zg[:-1], done[:-1], next_done[:-1])

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.n_steps + 1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0])
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.n_steps):
            z = self.model.next(z, action[t])
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t + 1] = z

        # Predictions
        zg = self.model.encode_goal(goal[:-1])
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, zg, return_type="all")
        reward_preds = self.model.reward(_zs, action, zg)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.n_steps):
            reward_loss += soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= 1 / self.cfg.n_steps
        reward_loss *= 1 / self.cfg.n_steps
        value_loss *= 1 / (self.cfg.n_steps * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), all_zg)

        # Update value functions
        vg_loss = self.update_vg(zs_as_zg, zg.detach(), td_targets)
        v_loss = self.update_v(_zs.detach(), zg.detach(), td_targets)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Update decoders
        dec_loss = self.update_dec(_zs.detach(), obs[:-1], zg.detach(), goal[:-1])

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "total_loss": float(total_loss.mean().item()),
            "pi_loss": pi_loss,
            "vg_loss": vg_loss,
            "v_loss": v_loss,
            "dec_loss": dec_loss,
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }


class TDMPC2Model(Model):
    _provides = {
        "Pi": True,
        "R": True,
        "Q": True,
        "V": True,
        "Vg": True,
        "Fwd": True,
        "Dec": None,
        "DecG": None,
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device)
        self.tdmpc2 = TDMPC2(cfg)

        self._provides["Dec"] = self.tdmpc2.model._decoder is not None
        self._provides["DecG"] = self.tdmpc2.model._goal_decoder is not None

    """ TRAINING """

    def update(self, batch: ty.Batch) -> Dict[str, Any]:
        return self.tdmpc2.update(batch)

    """ INFERENCE """

    def Enc(self, obs: ty.Obs) -> ty.Latent:
        return self.tdmpc2.model.encode(obs)

    def EncG(self, task: ty.Goal) -> ty.GLatent:
        return self.tdmpc2.model.encode_goal(task)

    def Pi(self, z: ty.Latent, zg: ty.Latent) -> Tuple[ty.Action, ty.Action, ty.Value, ty.Action]:
        return self.tdmpc2.model.pi(z, zg)

    def Q(self, z: ty.Latent, a: ty.Action, zg: ty.GLatent) -> ty.Value:
        return self.tdmpc2.model.Q(z, a, zg, return_type="avg").squeeze(-1)

    def V(self, z: ty.Latent, zg: ty.GLatent) -> ty.Value:
        if self.cfg.use_gt_V:
            return self.GTV(z, zg)
        elif self.cfg.use_V:
            return self.tdmpc2.model.V(z, zg)
        else:
            return self.Q(z, self.Pi(z, zg)[1], zg)

    def Vg(self, z1: ty.GLatent, z2: ty.GLatent) -> ty.Value:
        return two_hot_inv(self.tdmpc2.model.Vg(z1, z2), self.cfg).squeeze(-1)

    def Fwd(self, z: ty.Latent, a: ty.Action) -> ty.Latent:
        return self.tdmpc2.model.next(z, a)

    def Dec(self, z: ty.Latent) -> ty.Obs:
        return self.tdmpc2.model.decode(z)

    def DecG(self, zg: ty.GLatent) -> ty.Obs:
        return self.tdmpc2.model.decode_goal(zg)

    """ UTILS"""

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.tdmpc2.load_state_dict(state_dict["tdmpc2"])

    def state_dict(self) -> Dict[str, Any]:
        d = super().state_dict()
        d["tdmpc2"] = self.tdmpc2.state_dict()
        return d
