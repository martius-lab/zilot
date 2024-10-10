# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Sequence, Tuple

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import zilot.third_party.mbrl.math as math


class Optimizer:
    num_iterations: int | None = None

    def __init__(self):
        pass

    def reset(self):
        pass

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs optimization.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial solution, if necessary.
            samples (tensor, optional): [b, *xdims] samples to include in the initial population.

        Returns:
            (torch.Tensor): the best solution found.
        """
        pass


class CEMOptimizer(Optimizer):
    """Implements the Cross-Entropy Method optimization algorithm.

    A good description of CEM [1] can be found at https://arxiv.org/pdf/2008.06389.pdf. This
    code implements the version described in Section 2.1, labeled CEM_PETS
    (but note that the shift-initialization between planning time steps is handled outside of
    this class by TrajectoryOptimizer).

    This implementation also returns the best solution found as opposed
    to the mean of the last generation.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        elite_ratio (float): the proportion of the population that will be kept as
            elite (rounds up).
        population_size (int): the size of the population.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        alpha (float): momentum term.
        device (torch.device): device where computations will be performed.
        return_mean_elites (bool): if ``True`` returns the mean of the elites of the last
            iteration. Otherwise, it returns the max solution found over all iterations.
        clipped_normal (bool); if ``True`` samples are drawn from a normal distribution
            and clipped to the bounds. If ``False``, sampling uses a truncated normal
            distribution up to the bounds. Defaults to ``False``.

    [1] R. Rubinstein and W. Davidson. "The cross-entropy method for combinatorial and continuous
    optimization". Methodology and Computing in Applied Probability, 1999.
    """

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        alpha: float,
        device: torch.device,
        return_mean_elites: bool = False,
        clipped_normal: bool = False,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(np.int32)
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device

        self._clipped_normal = clipped_normal

    def _init_population_params(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x0.clone()
        if self._clipped_normal:
            dispersion = torch.ones_like(mean)
        else:
            dispersion = ((self.upper_bound - self.lower_bound) ** 2) / 16
        return mean, dispersion

    def _sample_population(
        self, mean: torch.Tensor, dispersion: torch.Tensor, population: torch.Tensor
    ) -> torch.Tensor:
        # fills population with random samples
        # for truncated normal, dispersion should be the variance
        # for clipped normal, dispersion should be the standard deviation
        if self._clipped_normal:
            pop = mean + dispersion * torch.randn_like(population)
            pop = torch.where(pop > self.lower_bound, pop, self.lower_bound)
            population = torch.where(pop < self.upper_bound, pop, self.upper_bound)
            return population
        else:
            lb_dist = mean - self.lower_bound
            ub_dist = self.upper_bound - mean
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.min(mv, dispersion)

            population = math.truncated_normal_(population)
            return population * torch.sqrt(constrained_var) + mean

    def _update_population_params(
        self, elite: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_mu = torch.mean(elite, dim=0)
        if self._clipped_normal:
            new_dispersion = torch.std(elite, dim=0)
        else:
            new_dispersion = torch.var(elite, dim=0)
        mu = self.alpha * mu + (1 - self.alpha) * new_mu
        dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion
        return mu, dispersion

    @torch.inference_mode()
    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs the optimization using CEM.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.

        Returns:
            (torch.Tensor): the best solution found.
        """
        mu, dispersion = self._init_population_params(x0)
        best_solution = torch.empty_like(mu)
        best_value = -np.inf
        population = torch.zeros((self.population_size,) + x0.shape).to(device=self.device)
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)

            x = obj_fun(population)

            if callback is not None:
                callback(population, x, i)

            if isinstance(x, torch.Tensor):  # TODO: do this via check for tuples?
                values = x
            else:
                values = x[0]

            # filter out NaN values
            values[values.isnan()] = -1e-10
            best_values, elite_idx = values.topk(self.elite_num)
            elite = population[elite_idx]

            mu, dispersion = self._update_population_params(elite, mu, dispersion)

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()

        return mu if self.return_mean_elites else best_solution


class MPPIOptimizer(Optimizer):
    """Implements the Model Predictive Path Integral optimization algorithm.

    A derivation of MPPI can be found at https://arxiv.org/abs/2102.09027
    This version is closely related to the original TF implementation used in PDDM with
    some noise sampling modifications and the addition of refinement steps.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        population_size (int): the size of the population.
        gamma (float): reward scaling term.
        sigma (float): noise scaling term used in action sampling.
        beta (float): correlation term between time steps.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        device (torch.device): device where computations will be performed.
    """

    def __init__(
        self,
        num_iterations: int,
        population_size: int,
        gamma: float,
        sigma: float,
        beta: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        device: torch.device,
    ):
        super().__init__()
        self.planning_horizon = len(lower_bound)
        self.population_size = population_size
        self.action_dimension = len(lower_bound[0])
        self.mean = torch.zeros(
            (self.planning_horizon, self.action_dimension),
            device=device,
            dtype=torch.float32,
        )

        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.var = sigma**2 * torch.ones_like(self.lower_bound)
        self.beta = beta
        self.gamma = gamma
        self.refinements = num_iterations
        self.num_iterations = num_iterations
        self.device = device

    def reset(self):
        self.mean = torch.zeros(
            (self.planning_horizon, self.action_dimension),
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of MPPI planner.
        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): Not required
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
        Returns:
            (torch.Tensor): the best solution found.
        """
        past_action = self.mean[0]
        self.mean[:-1] = self.mean[1:].clone()

        for k in range(self.refinements):
            # sample noise and update constrained variances
            noise = torch.empty(
                size=(
                    self.population_size,
                    self.planning_horizon,
                    self.action_dimension,
                ),
                device=self.device,
            )
            noise = math.truncated_normal_(noise)

            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            mv = torch.minimum(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.minimum(mv, self.var)
            population = noise.clone() * torch.sqrt(constrained_var)

            # smoothed actions with noise
            population[:, 0, :] = self.beta * (self.mean[0, :] + noise[:, 0, :]) + (1 - self.beta) * past_action
            for i in range(max(self.planning_horizon - 1, 0)):
                population[:, i + 1, :] = (
                    self.beta * (self.mean[i + 1] + noise[:, i + 1, :]) + (1 - self.beta) * population[:, i, :]
                )
            # clipping actions
            # This should still work if the bounds between dimensions are different.
            population = torch.where(population > self.upper_bound, self.upper_bound, population)
            population = torch.where(population < self.lower_bound, self.lower_bound, population)

            x = obj_fun(population)

            if callback is not None:
                callback(population, x, i)

            if isinstance(x, torch.Tensor):
                values = x
            else:
                values = x[0]

            values[values.isnan()] = -1e-10

            # weight actions
            weights = torch.reshape(
                torch.exp(self.gamma * (values - values.max())),
                (self.population_size, 1, 1),
            )
            norm = torch.sum(weights) + 1e-10
            weighted_actions = population * weights
            self.mean = torch.sum(weighted_actions, dim=0) / norm

        return self.mean.clone()


class ICEMOptimizer(Optimizer):
    """Implements the Improved Cross-Entropy Method (iCEM) optimization algorithm.

    iCEM improves the sample efficiency over standard CEM and was introduced by
    [2] for real-time planning.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        elite_ratio (float): the proportion of the population that will be kept as
            elite (rounds up).
        population_size (int): the size of the population.
        population_decay_factor (float): fixed factor for exponential decrease in population size
        colored_noise_exponent (float): colored-noise scaling exponent for generating correlated
            action sequences.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        keep_elite_frac (float): the fraction of elites to keep (or shift) during CEM iterations
        alpha (float): momentum term.
        device (torch.device): device where computations will be performed.
        return_mean_elites (bool): if ``True`` returns the mean of the elites of the last
            iteration. Otherwise, it returns the max solution found over all iterations.
        population_size_module (int, optional): if specified, the population is rounded to be
            a multiple of this number. Defaults to ``None``.

    [2] C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek and
    G, Martius, Georg. "Sample-efficient Cross-Entropy Method for Real-time Planning".
    Conference on Robot Learning, 2020.
    """

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        population_decay_factor: float,
        colored_noise_exponent: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        keep_elite_frac: float,
        alpha: float,
        device: torch.device,
        return_mean_elites: bool = False,
        population_size_module: Optional[int] = None,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.population_decay_factor = population_decay_factor
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(np.int32)
        self.colored_noise_exponent = colored_noise_exponent
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.initial_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
        self.keep_elite_frac = keep_elite_frac
        self.keep_elite_size = np.ceil(keep_elite_frac * self.elite_num).astype(np.int32)
        self.elite = None
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.population_size_module = population_size_module
        self.device = device

        if self.population_size_module:
            self.keep_elite_size = self._round_up_to_module(self.keep_elite_size, self.population_size_module)

    def reset(self):
        self.elite = None

    @staticmethod
    def _round_up_to_module(value: int, module: int) -> int:
        if value % module == 0:
            return value
        return value + (module - value % module)

    @torch.inference_mode()
    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs the optimization using iCEM.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
        Returns:
            (torch.Tensor): the best solution found.
        """
        mu = x0.clone() if x0 is not None else (self.upper_bound + self.lower_bound) / 2
        var = self.initial_var.clone()

        best_solution = torch.empty_like(mu)
        best_value = -np.inf

        for i in range(self.num_iterations):
            decay_population_size = np.ceil(
                np.max(
                    (
                        self.population_size * self.population_decay_factor**-i,
                        2 * self.elite_num,
                    )
                )
            ).astype(np.int32)

            if self.population_size_module:
                decay_population_size = self._round_up_to_module(decay_population_size, self.population_size_module)
            # the last dimension is used for temporal correlations
            population = math.powerlaw_psd_gaussian(
                self.colored_noise_exponent,
                size=(decay_population_size, mu.shape[1], mu.shape[0]),
                device=self.device,
            ).transpose(1, 2)
            population = torch.minimum(population * torch.sqrt(var) + mu, self.upper_bound)
            population = torch.maximum(population, self.lower_bound)

            if self.elite is not None:
                kept_elites = torch.index_select(
                    self.elite,
                    dim=0,
                    index=torch.randperm(self.elite_num, device=self.device)[: self.keep_elite_size],
                )
                if i == 0:
                    population[: kept_elites.size(0), :-1] = kept_elites[:, 1:]
                elif i == self.num_iterations - 1:
                    population[0] = mu
                else:
                    population[: kept_elites.size(0)] = kept_elites

            x = obj_fun(population)

            if callback is not None:
                callback(population, x, i)

            if isinstance(x, torch.Tensor):
                values = x
            else:
                values = x[0]

            # filter out NaN values
            values.nan_to_num_(-torch.inf)
            best_values, elite_idx = values.topk(self.elite_num)
            self.elite = population[elite_idx]

            new_mu = torch.mean(self.elite, dim=0)
            new_var = torch.var(self.elite, unbiased=False, dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()

        return mu if self.return_mean_elites else best_solution


# TODO: if this is necessary, adjust torch.inference_mode() in common.planning.py
# class GDOptimizer(Optimizer):
#     """Uses GD to optimize the objective function."""

#     def __init__(
#         self,
#         lower_bound: Sequence[Sequence[float]],
#         upper_bound: Sequence[Sequence[float]],
#         device: torch.device,
#         population_size: int = 8,
#         learning_rate: float = 0.1,
#         max_iter: int = 100,
#     ):
#         super().__init__()
#         self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
#         self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
#         self._device = device
#         self._learning_rate = learning_rate
#         self._batch_size = population_size
#         self._max_iter = max_iter
#         self.num_iterations = max_iter

#     def optimize(
#         self,
#         obj_fun: Callable[[torch.Tensor], torch.Tensor],
#         x0: Optional[torch.Tensor] = None,
#         callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
#         **kwargs,
#     ):
#         """Runs the optimization using GD.
#         Args:
#             obj_fun (callable(tensor) -> tensor): objective function to maximize. MUST be differentiable.
#             x0 (tensor, optional): initial solution. Must be consistent with lower/upper bounds.
#             callback (callable(tensor, tensor, int) -> any, optional): if given, this
#                 function will be called after every iteration, passing it as input the full
#                 population tensor, its corresponding objective function values, and
#                 the index of the current iteration. This can be used for logging and plotting
#                 purposes.
#         """
#         population = torch.rand((self._batch_size, *self.lower_bound.shape), device=self._device)
#         population = self.lower_bound + population * (self.upper_bound - self.lower_bound)
#         if x0 is not None:
#             population[0] = x0
#         population.requires_grad = True
#         optimizer = torch.optim.RMSprop([population], lr=self._learning_rate, weight_decay=0)
#         for i in range(self._max_iter):
#             optimizer.zero_grad()
#             x = obj_fun(population)

#             if callback is not None:
#                 callback(population, x, i)

#             if isinstance(x, torch.Tensor):
#                 loss = -x
#             else:
#                 loss = -x[0]

#             loss.sum().backward()
#             optimizer.step()
#             population.data.clip_(self.lower_bound, self.upper_bound)  # reproject to the feasible set
#         best = torch.argmin(loss)
#         return population[best].detach()


class TrajectoryOptimizer:
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions. It also
    takes care of shifting previous solution for the next optimization call, if the user desires.

    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        planning_horizon: int,
        replan_freq: int = 1,
        keep_last_solution: bool = True,
    ):
        optimizer_cfg.lower_bound = np.tile(action_lb, (planning_horizon, 1)).tolist()
        optimizer_cfg.upper_bound = np.tile(action_ub, (planning_horizon, 1)).tolist()
        self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
        self.initial_solution = (
            ((torch.tensor(action_lb) + torch.tensor(action_ub)) / 2).float().to(optimizer_cfg.device)
        )
        self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.previous_solution = self.initial_solution.clone()
        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.horizon = planning_horizon

    def optimize(
        self,
        trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor],
        callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """Runs the trajectory optimization.

        Args:
            trajectory_eval_fn (callable(tensor) -> tensor): A function that receives a batch
                of action sequences and returns a batch of objective function values (e.g.,
                accumulated reward for each sequence). The shape of the action sequence tensor
                will be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size,
                planning horizon, and action dimension, respectively.
            callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (torch.Tensor): the best solution found by the optimizer (with shape ``H x A``).
        """
        best_solution = self.optimizer.optimize(
            trajectory_eval_fn,
            x0=self.previous_solution,
            callback=callback,
        )
        if self.keep_last_solution:
            self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
            # Note that initial_solution[i] is the same for all values of [i],
            # so just pick i = 0
            self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        return best_solution

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.optimizer.reset()
        self.previous_solution = self.initial_solution.clone()


def make_optimizer(optimizer_cfg: omegaconf.Container, cfg: omegaconf.Container, **kwargs) -> TrajectoryOptimizer:
    """Instantiates an optimizer from the configuration.

    Args:
        optimizer_cfg (omegaconf.Container): the configuration of the optimizer to use.
        cfg (omegaconf.Container): the global configuration.

    Returns:
        (TrajectoryOptimizer): the instantiated optimizer.
    """
    optimizer_cfg.setdefault("population_size", cfg.population_size)
    optimizer_cfg.setdefault("num_iterations", cfg.num_iterations)
    optimizer_cfg.setdefault("elite_ratio", cfg.elite_ratio)
    optimizer_cfg.setdefault("device", cfg.device)
    kwargs.setdefault("planning_horizon", cfg.horizon)
    kwargs.setdefault("replan_freq", 1)
    kwargs.setdefault("keep_last_solution", False)
    return TrajectoryOptimizer(
        optimizer_cfg=optimizer_cfg,
        action_lb=np.full((cfg.action_dim,), -1.0),
        action_ub=np.full((cfg.action_dim,), 1.0),
        **kwargs,
    )
