import abc
import warnings
from typing import Any, Dict, Mapping

import torch
from tensordict import TensorDictBase
from torch import TensorType

import zilot.utils.dict_util as du


def _fit(x: torch.Tensor, from_dim: int):
    from_dim += 1  # include batch dimension
    x = x.view(-1, *x.shape[from_dim:])
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    min = x.min(dim=0).values
    max = x.max(dim=0).values
    return mean, std, min, max


class Normalizer(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def normalize(self, x: TensorType) -> TensorType:
        pass

    @abc.abstractmethod
    def denormalize(self, x: TensorType) -> TensorType:
        pass

    @abc.abstractmethod
    def fit(self, x: TensorType) -> None:
        pass

    @abc.abstractmethod
    def soft_update(self, other: "Normalizer", tau=0.1) -> None:
        pass

    def forward(self, x: TensorType) -> TensorType:
        raise NotImplementedError("Use normalize or denormalize instead.")

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        pass

    @abc.abstractmethod
    def state_dict(self, *args, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_state_dict(state_dict: Mapping[str, Any], *args, **kwargs):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class SingleNormalizer(Normalizer):
    def __init__(self, mode: str = "minmax", from_dim: int = 0, eps: float = 1e-6):
        """
        Args:
            mode: normalization mode
        """
        super().__init__()
        self.mode = mode
        if mode not in ["gaussian", "minmax", "maxabs", "none"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.from_dim = from_dim
        self.register_buffer("shift", None)
        self.register_buffer("scale", None)
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

    def fit(self, x: torch.Tensor) -> None:
        mean, std, min, max = _fit(x, self.from_dim)
        if self.mode == "gaussian":
            self.shift = mean
            self.scale = std
        elif self.mode == "minmax":
            self.shift = (min + max) / 2.0
            self.scale = (max - min) / 2.0
        elif self.mode == "maxabs":
            self.shift = torch.zeros_like(mean)
            self.scale = torch.max(torch.abs(min), torch.abs(max))
        elif self.mode == "none":
            self.shift = torch.zeros_like(mean)
            self.scale = torch.ones_like(mean)

        assert torch.all(self.scale >= 0.0)
        self.scale = torch.where(self.scale < self.eps, self.eps, self.scale)

    def soft_update(self, other: "SingleNormalizer", tau=0.1) -> None:
        if self.mode != other.mode:
            raise ValueError(f"Mode mismatch: {self.mode} != {other.mode}")
        if self.shift is None and self.scale is None:
            self = self.from_state_dict(other.state_dict())
            return

        if self.mode == "gaussian":
            self.shift.lerp_(other.shift, tau)
            self.scale.lerp_(other.scale, tau)
        elif self.mode == "minmax":
            other_min = other.shift - other.scale
            other_max = other.shift + other.scale
            self_min = self.shift - self.scale
            self_max = self.shift + self.scale
            min = torch.minimum(self_min, other_min)
            max = torch.maximum(self_max, other_max)
            shift = (min + max) / 2.0
            scale = (max - min) / 2.0
            self.shift.lerp_(shift, tau)
            self.scale.lerp_(scale, tau)
        elif self.mode == "maxabs":
            max = torch.maximum(torch.abs(self.scale), torch.abs(other.scale))
            self.scale.lerp_(max, tau)
        elif self.mode == "none":
            pass  # do nothing

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        self.shift = state_dict["shift"]
        self.scale = state_dict["scale"]
        self.eps = state_dict["eps"]
        self.mode = state_dict["mode"]
        self.from_dim = state_dict["from_dim"]

    def state_dict(self, *args, **kwargs):
        sd = {
            "shift": self.shift,
            "scale": self.scale,
            "eps": self.eps,
            "mode": self.mode,
            "from_dim": self.from_dim,
        }
        return sd

    @staticmethod
    def from_state_dict(state_dict: Mapping[str, Any], *args, **kwargs):
        x = SingleNormalizer()
        x.load_state_dict(state_dict, *args, **kwargs)
        return x

    def __repr__(self):
        return f"SingleNormalizer(mode={self.mode}, from_dim={self.from_dim}, eps={self.eps}, shape={self.shift.shape})"

    def __str__(self):
        return self.__repr__()


class MultiNormalizer(Normalizer):
    """
    A multi normalizer.
    """

    def __init__(self, normalizers: Dict[str, Normalizer] = {}):
        """
        Args:
            normalizers: Dict[str, Normalizers]
        """
        super().__init__()
        self.ns = torch.nn.ModuleDict(normalizers)

    def normalize(self, x: TensorDictBase) -> TensorDictBase:
        return du.map(du.apply(lambda n: n.normalize, self.ns), x)

    def denormalize(self, x: TensorDictBase) -> TensorDictBase:
        return du.map(du.apply(lambda n: n.denormalize, self.ns), x)

    def __getitem__(self, key) -> Normalizer:
        return self.ns[key]

    def fit(self, x: TensorDictBase) -> None:
        for k, v in x.items():
            if k in self.ns:
                self.ns[k].fit(v)

    def soft_update(self, other: "MultiNormalizer", tau=0.1) -> None:
        for k, v in self.ns.items():
            v.soft_update(other.ns[k], tau)

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        if not state_dict.get("_is_multi", False):
            warnings.warn("Loading from malformed statedict.")
        for k, v in state_dict.items():
            if k == "_is_multi":
                continue
            if k in self.ns:
                self.ns[k].load_state_dict(v, *args, **kwargs)
            else:
                if v.get("_is_multi", False):
                    self.ns[k] = MultiNormalizer.from_state_dict(v, *args, **kwargs)
                else:
                    self.ns[k] = SingleNormalizer.from_state_dict(v, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        sd = {k: v.state_dict() for k, v in self.ns.items()}
        sd["_is_multi"] = True
        return sd

    @staticmethod
    def from_state_dict(state_dict: Mapping[str, Any], *args, **kwargs):
        x = MultiNormalizer()
        x.load_state_dict(state_dict, *args, **kwargs)
        return x

    def __repr__(self):
        return f"MultiNormalizer({self.ns})"

    def __str__(self):
        return self.__repr__()
