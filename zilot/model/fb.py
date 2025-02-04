import torch

from zilot.model import Model
from zilot.third_party.fb.fb_ddpg import FBDDPGAgent


class FBWrapper(Model):
    def __init__(self, cfg) -> None:
        bw_act = "batchnorm" if "pointmaze" in cfg.env else ("none" if "halfcheetah" in cfg.env else "L2")
        self.agent = FBDDPGAgent(
            obs_shape=cfg.obs_shape["state"],
            goal_shape=cfg.goal_shape["state"],
            action_dim=cfg.action_dim,
            device=cfg.device,
            # special hparams for some of the environments
            z_dim=100 if "pointmaze" in cfg.env else 50,
            lr_coef=1e-2 if "pointmaze" in cfg.env else 1.0,
            backward_last_act=bw_act,
            gamma=cfg.discount,
        )
        self.agent.train(False)
        self.step = 0

    def update(self, td) -> None:
        self.agent.train(True)
        x = self.agent.update(td, self.step)
        self.step += 1
        self.agent.train(False)
        return x

    def _get_all_torch_modules(self):
        for k, v in self.agent.__dict__.items():
            if isinstance(v, torch.nn.Module):
                yield k, v

    def load_state_dict(self, state_dict) -> None:
        for k, v in self._get_all_torch_modules():
            v.load_state_dict(state_dict[k])

    def state_dict(self):
        sd = {k: v.state_dict() for k, v in self._get_all_torch_modules()}
        print("CHECK: FBWrapper.state_dict", sd.keys())
        return sd
