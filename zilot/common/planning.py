import abc
import math

import hydra
import torch
from omegaconf import Container

import zilot.types as ty
import zilot.utils.dict_util as du
from zilot.common.objectives import Objective
from zilot.common.task import TaskSpec
from zilot.envs import GOAL_TRANSFORMS
from zilot.model import Model
from zilot.model.util import rollout_fwd
from zilot.third_party.mbrl.trajectory_opt import TrajectoryOptimizer, make_optimizer


class LatentPlanner(abc.ABC):
    compatible_tasks: list[TaskSpec] = []
    _needs: list[str] = []

    def __init__(self, cfg: Container, model: Model):
        self.cfg = cfg
        self.model = model
        for x in self._needs:
            self._raise_not_available(x)

    @abc.abstractmethod
    def reset(self, zg: ty.GLatent) -> None:
        pass

    @abc.abstractmethod
    def plan(self, z: ty.Latent) -> ty.Action:
        pass

    """ UTILS """

    def _raise_not_available(self, x):
        if not self.cfg.available.get(x, False):
            raise ValueError(f"{x} must be available to instantiate {self.__class__.__name__}")


# =====================================================================================================================
# Sequential
# =====================================================================================================================


def _prepare_plan(cfg, model, plan):
    gtf = GOAL_TRANSFORMS.get(cfg.env, None)
    if plan and gtf and cfg.available.Dec and cfg.available.DecG:
        points_a = model.Dec(plan["zs"])
        points_a = gtf(points_a)[..., :2].view(-1, 2).cpu().numpy()
        points_b = model.DecG(plan["zgs"])
        points_b = points_b[..., :2].view(-1, 2).cpu().numpy()
        coupling = plan["coupling"].cpu().to_dense().numpy()  # TODO: make plan handle sparse coupling
        weights = plan["weights"].cpu().numpy()
        return dict(points_a=points_a, points_b=points_b, coupling=coupling, weights=weights)
    return None


class MPC(LatentPlanner):
    compatible_tasks: list[TaskSpec] = [TaskSpec.SINGLE]

    def __init__(
        self,
        cfg: Container,
        model: Model,
        optimizer_cfg: Container,
        objective_cfg: Container,
    ):
        self.optimizer: TrajectoryOptimizer = make_optimizer(optimizer_cfg, cfg=cfg)
        self.objective: Objective = hydra.utils.instantiate(objective_cfg, cfg=cfg, model=model, _recursive_=False)
        self._needs.extend(self.objective._needs)
        super().__init__(cfg, model)

    """ Plotting Callbacks """

    def _cb(self, x: ty.Action, score_logs: tuple, it: int) -> None:
        scores, logs = score_logs
        if self.cfg.record_traj and self.cfg.available.Dec and "traj" in logs:
            traj = logs["traj"]
            self._step_log.setdefault("traj", []).append(self.model.Dec(traj).cpu().numpy())
            self._step_log.setdefault("scores", []).append(scores.cpu().numpy())
            del logs["traj"]
        if it == self.optimizer.optimizer.num_iterations - 1:
            best_idx = scores.argmax()
            logs = du.apply_(lambda x: x[best_idx] if isinstance(x, torch.Tensor) else x, logs)
            self._step_log.update(logs)

    def _prepare_stats(self) -> dict[str, float]:
        if self.cfg.draw_plans:
            plan = _prepare_plan(self.cfg, self.model, self._step_log.get("plan", None))
            if plan is not None:
                self._step_log["plan"] = plan
        if self.cfg.record_traj and "traj" in self._step_log and "scores" in self._step_log:
            self._step_log["traj"] = {str(i): x for i, x in enumerate(self._step_log["traj"])}
            self._step_log["scores"] = {str(i): x for i, x in enumerate(self._step_log["scores"])}
        return self._step_log

    """ Interface """

    def reset(self, zg: ty.GLatent) -> None:
        self.optimizer.reset()
        self.objective.reset(zg)

    def plan(self, z: ty.Latent) -> ty.Action:
        self._step_log = {}  # reset step log
        self.objective.step(z)
        a = self.optimizer.optimize(self.objective, callback=self._cb)
        return a[0], self._prepare_stats()


# =====================================================================================================================
# Policy
# =====================================================================================================================


# ===== Myopic Pi =====================================================================================================

best_pi_cls_cfg_thresholds = {
    "fetch_pick_and_place": 5,
    "fetch_push": 4,
    "fetch_slide_large_2D": 5,
    "halfcheetah": 4,
    "pointmaze_medium": 5,
}


class Pi(LatentPlanner):
    compatible_tasks: list[TaskSpec] = [TaskSpec.SINGLE]
    _needs = ["Pi"]

    def __init__(self, cfg: Container, model: Model, cls_cfg: Container):
        if cfg.use_best_threshold:
            threshold = best_pi_cls_cfg_thresholds[cfg.env]
            cls_cfg.threshold = threshold
            print(f"Using best threshold for {cfg.env}: {threshold}")
        self.cls: torch.nn.Module = hydra.utils.instantiate(cls_cfg, cfg=cfg, model=model, _recursive_=False)
        self._needs.extend(self.cls._needs)
        super().__init__(cfg, model)

    def reset(self, zg: ty.GLatent) -> None:
        self.zgs = zg
        self.gidx = 0

    def plan(self, z: ty.Latent) -> ty.Action:
        while self.gidx + 1 < len(self.zgs) and self.cls(z, self.zgs[self.gidx]).item():
            self.gidx += 1
        zg = self.zgs[self.gidx]
        a = self.model.Pi(z, zg)[0]
        log = {}
        if self.cfg.draw_plans:
            plan = dict(zs=z.unsqueeze(0), zgs=zg.unsqueeze(0), coupling=torch.ones((1, 1)), weights=torch.ones((1, 1)))
            log["plan"] = _prepare_plan(self.cfg, self.model, plan)
        return a, log


# =====================================================================================================================
# Base Planner
# =====================================================================================================================


class Planner:
    """
    Base planner, wrapps latent planner and provides encoding and normalization
    """

    def __init__(self, model: Model, planner: LatentPlanner):
        self.model = model
        self.planner = planner

    @torch.inference_mode()
    def reset(self, task: ty.Goal) -> None:
        self.model.reset()
        ntask = self.model.preproc_goal(task)
        zg = self.model.EncG(ntask)
        self.planner.reset(zg)

    @torch.inference_mode()
    def plan(self, obs: dict[str, ty.Obs]) -> ty.Action:
        obs = obs["observation"]
        device_in = obs.device
        nobs = self.model.preproc_obs(obs)
        z = self.model.Enc(nobs)
        x = self.planner.plan(z)
        if isinstance(x, tuple):
            x, aux = x
        else:
            aux = dict()
        x = self.model.postproc_action(x)
        return x.to(device_in), aux

    def is_compatible(self, task_spec: TaskSpec):
        return task_spec in self.planner.compatible_tasks


class GTPi:
    """
    Myopic policy with ground-truth success metric
    """

    def __init__(self, cfg: Container, model: Model, threshold: float = None):
        self.cfg = cfg
        self.model = model
        self.threshold = float(threshold or cfg.goal_success_threshold)

    @torch.inference_mode()
    def reset(self, task: ty.Goal) -> None:
        self.goals = task
        self.goal_idx = 0

    def gt_cls(self, g1, g2):
        return self.cfg.eval_metric(g1, g2) <= self.threshold

    @torch.inference_mode()
    def plan(self, obs: dict[str, ty.Obs]) -> ty.Action:
        obs_as_goal = obs["achieved_goal"]
        obs = obs["observation"]
        while self.goal_idx + 1 < len(self.goals) and self.gt_cls(obs_as_goal, self.goals[self.goal_idx]):
            self.goal_idx += 1
        goal = self.goals[self.goal_idx]
        device_in = obs.device
        nobs = self.model.preproc_obs(obs)
        ngoal = self.model.preproc_goal(goal)
        z = self.model.Enc(nobs)
        zg = self.model.EncG(ngoal)
        x = self.model.Pi(z, zg)[0]
        x = self.model.postproc_action(x)
        return x.to(device_in), dict()

    def is_compatible(self, task_spec: TaskSpec):
        return task_spec == TaskSpec.SINGLE


class GTMPC:
    """
    Myopic MPC with ground-truth success metric for classification AND as objective
    """

    def __init__(self, cfg: Container, model: Model, threshold: float = None):
        self.cfg = cfg
        self.model = model
        self.optimizer: TrajectoryOptimizer = make_optimizer(cfg.optimizer_cfg, cfg=cfg)

        # NOTE: we assume the below info is not available to our model, but use it here as a baseline
        self.threshold = threshold or cfg.goal_success_threshold
        self.goal_tf = GOAL_TRANSFORMS[self.cfg.env]

    @torch.inference_mode()
    def reset(self, task: ty.Goal) -> None:
        self.goals = task
        self.goal_idx = 0
        self.optimizer.reset()
        self.model.reset()

    def gt_cls(self, g1, g2):
        return self.cfg.eval_metric(g1, g2) <= self.threshold

    @torch.inference_mode()
    def plan(self, obs: dict[str, ty.Obs]) -> ty.Action:
        obs_as_goal = obs["achieved_goal"]
        obs = obs["observation"]
        while self.goal_idx + 1 < len(self.goals) and self.gt_cls(obs_as_goal, self.goals[self.goal_idx]):
            self.goal_idx += 1
        goal = self.goals[self.goal_idx]
        device_in = obs.device
        nobs = self.model.preproc_obs(obs)
        ngoal = self.model.preproc_goal(goal)
        z = self.model.Enc(nobs)
        zg = self.model.EncG(ngoal)

        def objective(a):
            zs = rollout_fwd(self.model.Fwd, z.clone(), a)
            B, H, _ = zs.size()
            s = self.model.Dec(zs)
            s_as_goal = self.goal_tf(s)
            v = -torch.ones(B, H, device=zs.device)
            v[:, -1] = self.model.V(z, zg)
            mask = ~self.gt_cls(s_as_goal, ngoal).cummax(dim=-1).values  # once done stay done
            return (v * mask.float()).sum(dim=-1)

        a = self.optimizer.optimize(objective)
        x = a[0]
        x = self.model.postproc_action(x)
        return x.to(device_in), dict()

    def is_compatible(self, task_spec: TaskSpec):
        return task_spec == TaskSpec.SINGLE


class GTGTMPC:
    """
    Myopic MPC with ground-truth success metric for classification AND as objective
    """

    def __init__(self, cfg: Container, model: Model, threshold: float = None):
        self.cfg = cfg
        self.model = model
        self.optimizer: TrajectoryOptimizer = make_optimizer(cfg.optimizer_cfg, cfg=cfg)

        # NOTE: we assume the below info is not available to our model, but use it here as a baseline
        self.threshold = threshold or cfg.goal_success_threshold
        self.goal_tf = GOAL_TRANSFORMS[self.cfg.env]

    @torch.inference_mode()
    def reset(self, task: ty.Goal) -> None:
        self.goals = task
        self.goal_idx = 0
        self.optimizer.reset()
        self.model.reset()

    def gt_cls(self, g1, g2):
        return self.cfg.eval_metric(g1, g2) <= self.threshold

    @torch.inference_mode()
    def plan(self, obs: dict[str, ty.Obs]) -> ty.Action:
        obs_as_goal = obs["achieved_goal"]
        obs = obs["observation"]
        while self.goal_idx + 1 < len(self.goals) and self.gt_cls(obs_as_goal, self.goals[self.goal_idx]):
            self.goal_idx += 1
        goal = self.goals[self.goal_idx]
        device_in = obs.device
        nobs = self.model.preproc_obs(obs)
        ngoal = self.model.preproc_goal(goal)
        z = self.model.Enc(nobs)

        def objective(a):
            zs = rollout_fwd(self.model.Fwd, z.clone(), a)
            B, H, _ = zs.size()
            s = self.model.Dec(zs)
            s_as_goal = self.goal_tf(s)
            v = -self.cfg.eval_metric(s_as_goal[:, -1], ngoal) / self.cfg.step_size
            rewards = torch.cat([-torch.ones(B, H - 1, device=zs.device), v.unsqueeze(-1)], dim=-1)
            mask = ~self.gt_cls(s_as_goal, ngoal).cummax(dim=-1).values  # once done stay done
            return (rewards * mask).sum(dim=-1)

        a = self.optimizer.optimize(objective)
        x = a[0]
        x = self.model.postproc_action(x)
        return x.to(device_in), dict()

    def is_compatible(self, task_spec: TaskSpec):
        return task_spec == TaskSpec.SINGLE


# =====================================================================================================================
# FB-IL
# =====================================================================================================================


class FB_ER:
    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.agent = model.agent

    @torch.inference_mode()
    def _get_z_from_goals(self, goals: torch.Tensor):
        goals = goals.to(self.cfg.device)
        assert goals.ndim == 2, "Expected goals to be 2D"
        # https://openreview.net/pdf?id=SHNjk4h0jn eq. 8
        # 1/l * sum_{t >= 0} B(goal_{t+1}) * r(goal_{t+1}), where r(goal) = 1
        # in maze and in fetch, the very first state is a goal,
        # so we have to exclude it since eq. 8 only summs from s_1, ...
        # for cheetah we don't have this problem since the initial state is not a goal
        if len(goals) > 1 and ("maze" in self.cfg.env or "fetch" in self.cfg.env):
            goals = goals[1:]
        zs = self.agent.backward_net(goals)
        z = zs.mean(dim=0)  # 1/l * sum ...
        if self.agent.cfg.norm_z:
            z = math.sqrt(z.size(-1)) * torch.nn.functional.normalize(z, dim=-1)
        return z

    @torch.inference_mode()
    def reset(self, task: ty.Goal) -> None:
        self._z = self._get_z_from_goals(task.to(self.cfg.device))

    @torch.inference_mode()
    def plan(self, obs: dict[str, ty.Obs]) -> ty.Action:
        obs = obs["observation"].to(self.cfg.device)
        a = self.agent.act(obs.unsqueeze(0), self._z.unsqueeze(0), eval_mode=True).squeeze(0)
        return a, dict()

    def is_compatible(self, task_spec):
        return task_spec == TaskSpec.SINGLE


class FB_RER:
    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.agent = model.agent

    @torch.inference_mode()
    def _get_z_from_goals(self, goals: torch.Tensor):
        goals = goals.to(self.cfg.device)
        assert goals.ndim == 2, "Expected goals to be 2D"
        # https://openreview.net/pdf?id=SHNjk4h0jn eq. 8
        # 1/l * sum_{t >= 0} B(goal_{t+1}) * r(goal_{t+1}), where r(goal) = 1
        # in maze and in fetch, the very first state is a goal,
        # so we have to exclude it since eq. 8 only summs from s_1, ...
        # for cheetah we don't have this problem since the initial state is not a goal
        if len(goals) > 1 and ("maze" in self.cfg.env or "fetch" in self.cfg.env):
            goals = goals[1:]
        covB = self.agent.online_cov(None)
        B = self.agent.backward_net(goals)
        EBBT = torch.einsum("ni,nj->nij", B, B).mean(dim=0)
        EB = B.mean(dim=0)  # 1/l * sum ...
        z = torch.einsum("ki,ij,j->k", covB, (covB + EBBT).inverse(), EB)
        if self.agent.cfg.norm_z:
            z = math.sqrt(z.size(-1)) * torch.nn.functional.normalize(z, dim=-1)
        return z

    @torch.inference_mode()
    def reset(self, task: ty.Goal) -> None:
        self._z = self._get_z_from_goals(task.to(self.cfg.device))

    @torch.inference_mode()
    def plan(self, obs: dict[str, ty.Obs]) -> ty.Action:
        obs = obs["observation"].to(self.cfg.device)
        a = self.agent.act(obs.unsqueeze(0), self._z.unsqueeze(0), eval_mode=True).squeeze(0)
        return a, dict()

    def is_compatible(self, task_spec):
        return task_spec == TaskSpec.SINGLE


class FB_Goal:
    def __init__(self, cfg: Container, model: Model, threshold: float = None) -> None:
        self.cfg = cfg
        self.agent = model.agent
        self.threshold = float(threshold or cfg.goal_success_threshold)

    @torch.inference_mode()
    def reset(self, task: ty.Goal) -> None:
        self.goals = task.to(self.cfg.device)
        self.goal_idx = 0

    def gt_cls(self, g1, g2):
        return self.cfg.eval_metric(g1, g2) <= self.threshold

    @torch.inference_mode()
    def plan(self, obs: dict[str, ty.Obs]) -> ty.Action:
        obs = obs.to(self.cfg.device)
        obs_as_goal = obs["achieved_goal"]
        while self.goal_idx + 1 < len(self.goals) and self.gt_cls(obs_as_goal, self.goals[self.goal_idx]):
            self.goal_idx += 1
        goal = self.goals[self.goal_idx]

        z = self.agent.backward_net(goal.unsqueeze(0).to(self.cfg.device))

        obs = obs["observation"]
        a = self.agent.act(obs.unsqueeze(0), z, eval_mode=True).squeeze(0)
        return a, dict()

    def is_compatible(self, task_spec):
        return task_spec == TaskSpec.SINGLE


# =====================================================================================================================
# Utils
# =====================================================================================================================

NO_WRAP = ["gt_pi", "gt_mpc", "gt_gt_mpc", "fb_goal", "fb_er", "fb_rer"]


def make_planner_from_model(name, cfg: Container, model: Model) -> Planner:
    spec = cfg.planners[name]
    if name in NO_WRAP:
        return hydra.utils.instantiate(spec, cfg=cfg, model=model, _recursive_=False)
    planner: LatentPlanner = hydra.utils.instantiate(spec, cfg=cfg, model=model, _recursive_=False)
    return Planner(model, planner)


def make_planners_from_model(cfg: Container, model: Model) -> dict[str, Planner]:
    return {k: make_planner_from_model(k, cfg, model) for k in cfg.planners.keys()}
