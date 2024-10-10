import abc

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


class Pi(LatentPlanner):
    compatible_tasks: list[TaskSpec] = [TaskSpec.SINGLE]
    _needs = ["Pi"]

    def __init__(self, cfg: Container, model: Model, cls_cfg: Container):
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


# =====================================================================================================================
# Utils
# =====================================================================================================================


def make_planner_from_model(name, cfg: Container, model: Model) -> Planner:
    spec = cfg.planners[name]
    planner: LatentPlanner = hydra.utils.instantiate(spec, cfg=cfg, model=model, _recursive_=False)
    return Planner(model, planner)


def make_planners_from_model(cfg: Container, model: Model) -> dict[str, Planner]:
    return {k: make_planner_from_model(k, cfg, model) for k in cfg.planners.keys()}
