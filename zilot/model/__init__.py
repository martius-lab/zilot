import abc
from typing import Any, Dict, Tuple

import torch
from omegaconf import Container

import zilot.types as ty


class Model(abc.ABC):
    _provides: Dict[str, bool] = {"Pi": False, "R": False, "V": False, "Vg": False, "Fwd": False, "Cls": False}
    _device: torch.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    """ TRAINING """

    @abc.abstractmethod
    def update(self, batch: ty.Batch) -> Dict[str, Any]:
        pass

    def reset(self):
        """resets any internal state of the model"""
        pass

    """ INFERENCE """

    def Enc(self, obs: ty.Obs) -> ty.Latent:
        return obs  # Default identity

    def EncG(self, goal: ty.Obs) -> ty.Latent:
        return goal  # Default identity

    def Dec(self, z: ty.Latent) -> ty.Obs:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `Dec`")

    def DecG(self, zg: ty.Latent) -> ty.Obs:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `DecG`")

    def Pi(self, z: ty.Latent, zg: ty.Latent) -> Tuple[ty.Action, ty.Action, ty.Value, ty.Action]:
        """Pi(z, zg) -> (mean, sample, log_prob, log_std)"""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `Pi`")

    def TrainPi(self, z: ty.Latent, zg: ty.Latent) -> ty.Action:
        """Pi(z, zg) -> sample for training"""
        return self.Pi(z, zg)[1]  # Default to sample

    def R(self, z: ty.Latent, a: ty.Action, zg: ty.GLatent) -> ty.Value:
        """R(z, a, zg) -> reward"""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `R`")

    def Q(self, z: ty.Latent, a: ty.Action, zg: ty.GLatent) -> ty.Value:
        """Q(z, a, zg)"""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `R`")

    def V(self, z: ty.Latent, zg: ty.GLatent) -> ty.Value:
        """V(z, zg)"""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `V`")

    def Vg(self, zg1: ty.GLatent, zg2: ty.GLatent) -> ty.Value:
        """Vg(zg1, zg2)"""
        raise NotImplementedError(f"{self.__class__} does not implement `Vg`")

    def Vs(self, zg1: ty.Latent, zg2: ty.Latent) -> ty.Value:
        """Vs(zs1, zs2)"""
        raise NotImplementedError(f"{self.__class__} does not implement `Vs`")

    def Fwd(self, z: ty.Latent, a: ty.Action) -> ty.Latent:
        """Fwd(z, a) -> z' (forward dynamics)"""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `Fwd`")

    def FwdEnsemble(self, z: ty.Latent, a: ty.Action) -> ty.Latent:
        """FwdEnsemble( num_ensemble x z, a) -> num_ensemble x z' (forward dynamics)"""
        return self.Fwd(z, a)  # Default to single ensemble

    def Cls(self, z: ty.Latent, zg: ty.GLatent) -> ty.Value:
        """Cls(z, zg) = probability that z is same state as zg"""
        raise NotImplementedError(f"{self.__class__} does not implement `Cls`")

    """ UTILS """

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass  # legacy

    def state_dict(self) -> Dict[str, Any]:
        return {}

    @property
    def device(self) -> torch.device:
        return self._device

    def preproc(self, batch: ty.Batch) -> ty.Batch:
        batch = batch.to(self.device, non_blocking=True)
        return batch

    def preproc_obs(self, obs: ty.Obs) -> ty.Obs:
        obs = obs.to(self.device, non_blocking=True)
        return obs

    def preproc_goal(self, goal: ty.Obs) -> ty.Obs:
        goal = goal.to(self.device, non_blocking=True)
        return goal

    def postproc_action(self, action: ty.Action) -> ty.Action:
        return action

    def freeze_scales(self):
        pass


# setup model
from zilot.model.curious import Dynamics
from zilot.model.tdmpc2 import TDMPC2Model

JOB_TO_MODEL = {"train": TDMPC2Model, "eval": TDMPC2Model, "dset": Dynamics}


def make_model(cfg: Container) -> Model:
    model: Model = JOB_TO_MODEL[cfg.job](cfg)
    # set availability in config
    for k in cfg.available.keys():
        cfg.available[k] = model._provides.get(k, False)
    return model
