# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
import typing as tp

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .fb_modules import Actor, BackwardMap, ForwardMap, OnlineCov

# from .url_benchmark import goals as _goals
from .url_benchmark import utils


@dataclasses.dataclass
class FBDDPGAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.fb_ddpg.FBDDPGAgent"
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING
    goal_shape: tp.Tuple[int, ...] = omegaconf.MISSING
    action_dim: int = omegaconf.MISSING
    device: str = omegaconf.MISSING
    gamma: float = omegaconf.MISSING
    lr: float = 1e-4
    lr_coef: float = 1.0
    fb_target_tau: float = 0.01  # 0.001-0.01
    hidden_dim: int = 1024  # 128, 2048
    backward_hidden_dim: int = 526  # 512
    feature_dim: int = 512  # 128, 1024
    z_dim: int = 50  # 100
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    ortho_coef: float = 1.0  # 0.01-10
    mix_ratio: float = 0.5
    future_ratio: float = 0.0
    preprocess: bool = True
    norm_z: bool = True
    add_trunk: bool = True
    backward_last_act: str = "L2"


class FBDDPGAgent:
    def __init__(self, **kwargs):
        cfg = FBDDPGAgentConfig(**kwargs)
        print(cfg)
        self.cfg = cfg
        self.action_dim = cfg.action_dim

        # models
        self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_dim < self.obs_dim:
            print(f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        goal_dim = self.obs_dim
        goal_dim = cfg.goal_shape[0]
        if cfg.z_dim < goal_dim:
            print(f"z_dim {cfg.z_dim} should not be smaller that goal_dim {goal_dim}")
        # create the network
        self.actor = Actor(
            self.obs_dim,
            cfg.z_dim,
            self.action_dim,
            cfg.feature_dim,
            cfg.hidden_dim,
            preprocess=cfg.preprocess,
            add_trunk=self.cfg.add_trunk,
        ).to(cfg.device)
        self.forward_net = ForwardMap(
            self.obs_dim,
            cfg.z_dim,
            self.action_dim,
            cfg.feature_dim,
            cfg.hidden_dim,
            preprocess=cfg.preprocess,
            add_trunk=self.cfg.add_trunk,
        ).to(cfg.device)
        self.backward_net = BackwardMap(goal_dim, cfg.z_dim, cfg.backward_hidden_dim).to(cfg.device)
        # build up the target network
        self.backward_target_net = BackwardMap(goal_dim, cfg.z_dim, cfg.backward_hidden_dim).to(cfg.device)
        self.forward_target_net = ForwardMap(
            self.obs_dim,
            cfg.z_dim,
            self.action_dim,
            cfg.feature_dim,
            cfg.hidden_dim,
            preprocess=cfg.preprocess,
            add_trunk=self.cfg.add_trunk,
        ).to(cfg.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_coef * cfg.lr)
        self.fb_opt = torch.optim.Adam(
            [
                {"params": self.forward_net.parameters()},  # type: ignore
                {"params": self.backward_net.parameters(), "lr": cfg.lr_coef * cfg.lr},
            ],
            lr=cfg.lr,
        )

        self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)

        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.actor, self.forward_net, self.backward_net, self.online_cov]:
            net.train(training)

    def sample_z(self, size, device: str = "cpu"):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand((size, self.cfg.z_dim), dtype=torch.float32, device=device)
            z = np.sqrt(self.cfg.z_dim) * uniform_rdv * gaussian_rdv
        return z

    def act(self, obs, z, eval_mode=True, step: int = 10_000_000) -> tp.Any:
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        return action

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        z: torch.Tensor,
        step: int,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
            target_B = self.backward_target_net(next_goal)  # batch x z_dim
            target_M1 = torch.einsum("sd, td -> st", target_F1, target_B)  # batch x batch
            target_M2 = torch.einsum("sd, td -> st", target_F2, target_B)  # batch x batch
            target_M = torch.min(target_M1, target_M2)

        # compute FB loss
        F1, F2 = self.forward_net(obs, z, action)
        B = self.backward_net(next_goal)
        M1 = torch.einsum("sd, td -> st", F1, B)  # batch x batch
        M2 = torch.einsum("sd, td -> st", F2, B)  # batch x batch
        II = torch.eye(*M1.size(), device=M1.device)
        off_diag = ~II.bool()
        fb_offdiag: tp.Any = 0.5 * sum((M - self.cfg.gamma * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
        fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.ortho_coef * orth_loss

        # optimize FB
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()

        metrics["target_M"] = target_M.mean().item()
        metrics["M1"] = M1.mean().item()
        metrics["F1"] = F1.mean().item()
        metrics["B"] = B.mean().item()
        metrics["B_norm"] = torch.norm(B, dim=-1).mean().item()
        metrics["z_norm"] = torch.norm(z, dim=-1).mean().item()
        metrics["fb_loss"] = fb_loss.item()
        metrics["fb_diag"] = fb_diag.item()
        metrics["fb_offdiag"] = fb_offdiag.item()
        metrics["orth_loss"] = orth_loss.item()
        metrics["orth_loss_diag"] = orth_loss_diag.item()
        metrics["orth_loss_offdiag"] = orth_loss_offdiag.item()
        eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
        metrics["orth_linf"] = torch.max(torch.abs(eye_diff)).item()
        metrics["orth_l2"] = eye_diff.norm().item() / math.sqrt(B.shape[1])

        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, z, stddev)
        action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.forward_net(obs, z, action)
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["q"] = Q.mean().item()
        metrics["actor_logprob"] = log_prob.mean().item()

        return metrics

    def update(self, td: TensorDict, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        # if step % self.cfg.update_every_steps != 0:
        #     return metrics

        obs = td["obs"][0]
        action = td["action"][0]
        next_obs = td["obs"][1]
        next_goal = td["achieved_goal"][1]

        bsz = obs.shape[0]
        z = self.sample_z(bsz, device=self.cfg.device)

        backward_input = td["achieved_goal"][1]
        future_goal = td["goal"][1]

        with torch.no_grad():
            B = self.backward_net(backward_input)

        # update covariance
        if step % 1000 == 0:
            with torch.no_grad():
                self.online_cov(B)

        if self.cfg.mix_ratio > 0:
            perm = torch.randperm(bsz, device=z.device)
            mix_z = B[perm]
            z = torch.where((torch.rand((bsz, 1), device=z.device) < self.cfg.mix_ratio), mix_z, z)

        # hindsight replay
        if self.cfg.future_ratio > 0:
            with torch.no_grad():
                future_z = self.backward_net(future_goal)
            z = torch.where((torch.rand((bsz, 1), device=z.device) < self.cfg.future_ratio), future_z, z)

        metrics.update(self.update_fb(obs=obs, action=action, next_obs=next_obs, next_goal=next_goal, z=z, step=step))

        # update actor
        metrics.update(self.update_actor(obs, z, step))

        # update critic target
        utils.soft_update_params(self.forward_net, self.forward_target_net, self.cfg.fb_target_tau)
        utils.soft_update_params(self.backward_net, self.backward_target_net, self.cfg.fb_target_tau)

        return metrics
