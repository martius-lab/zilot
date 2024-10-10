import torch
import torch.nn as nn


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


def dim_from_back(shape: torch.Size, dim: int) -> int:
    """Get the dimension index from the back."""
    return dim if dim < 0 else dim - len(shape)


def prepend_single(x: torch.Tensor, xs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """torch.cat with appropriate expansion."""
    dim = dim_from_back(xs.shape, dim)
    x_shape = list(xs.shape)
    x_shape[dim] = 1
    return torch.cat([x.unsqueeze(dim).expand(x_shape), xs], dim=dim)


def prepend(x: torch.Tensor, xs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """torch.cat with appropriate expansion."""
    dim = dim_from_back(xs.shape, dim)
    x_shape = list(xs.shape)
    x_shape[dim] = x.shape[dim]
    return torch.cat([x.expand(x_shape), xs], dim=dim)


def append_single(xs: torch.Tensor, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """torch.cat with appropriate expansion."""
    dim = dim_from_back(xs.shape, dim)
    x_shape = list(xs.shape)
    x_shape[dim] = 1
    return torch.cat([xs, x.unsqueeze(dim).expand(x_shape)], dim=dim)


def append(xs: torch.Tensor, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """torch.cat with appropriate expansion."""
    dim = dim_from_back(xs.shape, dim)
    x_shape = list(xs.shape)
    x_shape[dim] = x.shape[dim]
    return torch.cat([xs, x.expand(x_shape)], dim=dim)
