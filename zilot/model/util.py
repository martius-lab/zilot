from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

import zilot.types as ty


def gc_value_to_steps(value: Tensor) -> Tensor:
    return (-value).clamp_min_(0.0)


def rollout_fwd(
    fwd: Callable[[ty.Latent, ty.Action], ty.Latent],
    z0: Float[Tensor, "*#B dim_z"],
    a: Float[Tensor, "*#B H dim_a"],
) -> Float[Tensor, "*B H dim_z"]:
    """
    Rollout a forward model.
    args:
        fwd: the forward model.
        z0: the initial latent state.
        a: the actions.
    returns:
        the latents z_0:H-1
    """
    *Ba, H, _ = a.size()
    *Bz, dim_z = z0.size()
    B = torch.broadcast_shapes(Ba, Bz)
    a = a.expand(*B, H, -1)
    z = z0.expand(*B, dim_z)
    zs = torch.empty((*B, H, dim_z), device=z0.device, dtype=z0.dtype)
    zs = zs.moveaxis(len(B), 0)
    a = a.moveaxis(len(B), 0)
    for i in range(H):
        z = fwd(z, a[i])
        zs[i] = z
    return zs.moveaxis(0, len(B))


def rollout_fwd_with_V(
    fwd: Callable[[ty.Latent, ty.Action], ty.Latent],
    V: Callable[[ty.Latent, ty.GLatent], ty.Value],
    z0: Float[Tensor, "*#B dim_z"],
    a: Float[Tensor, "*#B H dim_a"],
    zgs: Float[Tensor, "T dim_z"],
) -> tuple[Float[Tensor, "*B H dim_z"], Float[Tensor, "*B H T"]]:
    """
    Rollout a forward model and compute V values.
    args:
        fwd: the forward model.
        V: the V function.
        z0: the initial latent state.
        a: the actions.
        zg: the goal latent states.
    returns:
        the latents z_0:H-1, and all point-wise V values of (z_0:H-1) x zg.
    """
    *Ba, H, _ = a.size()
    *Bz, dim_z = z0.size()
    T, _ = zgs.size()
    B = torch.broadcast_shapes(Ba, Bz)
    a = a.expand(*B, H, -1)
    z = z0.expand(*B, dim_z)
    zs = torch.empty((*B, H, dim_z), device=z0.device, dtype=z0.dtype)
    vs = torch.empty((*B, H, T), device=z0.device, dtype=z0.dtype)
    zs = zs.moveaxis(len(B), 0)
    a = a.moveaxis(len(B), 0)
    vs = vs.moveaxis(len(B), 0)
    for i in range(H):
        z = fwd(z, a[i])
        zs[i] = z
        vs[i] = V(z.unsqueeze(-2), zgs)
    return zs.moveaxis(0, len(B)), vs.moveaxis(0, len(B))


def rollout_fwd_z0(
    fwd: Callable[[ty.Latent, ty.Action], ty.Latent],
    z0: Float[Tensor, "*#B dim_z"],
    a: Float[Tensor, "*#B H dim_a"],
) -> Float[Tensor, "*B H+1 dim_z"]:
    """
    Rollout a forward model.
    args:
        fwd: the forward model.
        z0 the initial latent state.
        a the actions.
    returns:
        the latent states including the initial state.
    """
    zs = rollout_fwd(fwd, z0, a)
    z0 = z0.unsqueeze(-2).expand_as(zs[..., :1, :])
    return torch.cat([z0, zs], dim=-2)


def rollout_fwd_ensemble(
    fwd: Callable[[ty.Latent, ty.Action], ty.Latent],
    z0: Float[Tensor, "*#B dim_z"],
    a: Float[Tensor, "*#B H dim_a"],
    ensemble_size: int,
) -> Float[Tensor, "ensemble_size *B H dim_z"]:
    """
    Rollout a forward model ensemble from the same initial state with the same actions.
    args:
        fwd: the forward model.
        z0: the initial latent state.
        a: the actions.
        ensemble_size: the size of the ensemble.
    returns:
        the latents z_0:H-1 for each ensemble member.
    """
    *Ba, H, _ = a.size()
    *Bz, dim_z = z0.size()
    B = torch.broadcast_shapes(Ba, Bz)
    a = a.expand(*B, H, -1)
    z = z0.expand(ensemble_size, *B, dim_z)
    zs = torch.empty((ensemble_size, *B, H, dim_z), device=z0.device, dtype=z0.dtype)
    zs = zs.moveaxis(len(B) + 1, 0)
    a = a.moveaxis(len(B), 0)
    for i in range(H):
        z = fwd(z, a[i])
        zs[i] = z
    return zs.moveaxis(0, len(B) + 1)


def rollout_fwd_ensemble_with_V(
    fwd: Callable[[ty.Latent, ty.Action], ty.Latent],
    V: Callable[[ty.Latent, ty.GLatent], ty.Value],
    z0: Float[Tensor, "*#B dim_z"],
    a: Float[Tensor, "*#B H dim_a"],
    zgs: Float[Tensor, "T dim_z"],
    ensemble_size: int,
) -> tuple[Float[Tensor, "ensemble_size *B H dim_z"], Float[Tensor, "ensemble_size *B H T"]]:
    """
    Rollout a forward model ensemble from the same initial state with the same actions.
    Additionally, compute pairwise V values between all produced latents and goal latents.

    args:
        fwd: the forward model.
        V: the Value function.
        z0: the initial latent state.
        a: the actions.
        zgs: the goal latent states.
        ensemble_size: the size of the ensemble.

    returns:
        the latents z_0:H-1 for each ensemble member.
        the values of each of those latents with respect to each goal latent.
    """
    *Ba, H, _ = a.size()
    *Bz, dim_z = z0.size()
    B = torch.broadcast_shapes(Ba, Bz)
    a = a.expand(*B, H, -1)
    z = z0.expand(ensemble_size, *B, dim_z)
    zs = torch.empty((ensemble_size, *B, H, dim_z), device=z0.device, dtype=z0.dtype)
    vs = torch.empty((ensemble_size, *B, H, zgs.size(0)), device=z0.device, dtype=z0.dtype)
    zs = zs.moveaxis(len(B) + 1, 0)
    vs = vs.moveaxis(len(B) + 1, 0)
    a = a.moveaxis(len(B), 0)
    for i in range(H):
        z = fwd(z, a[i])
        assert z.size(0) == ensemble_size, f"Expected ensemble size {ensemble_size}, got {z.size(0)}"
        zs[i] = z
        vs[i] = V(z.unsqueeze(-2), zgs)
    return zs.moveaxis(0, len(B) + 1), vs.moveaxis(0, len(B) + 1)


def rollout_fwd_ensemble_z0(
    fwd: Callable[[ty.Latent, ty.Action], ty.Latent],
    z0: Float[Tensor, "*#B dim_z"],
    a: Float[Tensor, "*#B H dim_a"],
    ensemble_size: int,
) -> Float[Tensor, "ensemble_size *B H+1 dim_z"]:
    """
    Rollout a forward model ensemble from the same initial state with the same actions.
    args:
        fwd: the forward model.
        z0 the initial latent state.
        a the actions.
        ensemble_size: the size of the ensemble.
    returns:
        the latent states including the initial state for each ensemble member.
    """
    zs = rollout_fwd_ensemble(fwd, z0, a, ensemble_size)
    z0 = z0.unsqueeze(-2).expand_as(zs[..., :1, :])
    return torch.cat([z0, zs], dim=-2)
