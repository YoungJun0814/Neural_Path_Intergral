"""Finite-grid minimum-action tilt with a defensive exact likelihood."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    FrozenBaselineProposal,
    freeze_baseline_proposal,
)
from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.physics_engine import TwoDriverRBergomiPaths

from .rbergomi_common import RBergomiBaselineProblem


@dataclass(frozen=True)
class LargeDeviationTrainingConfig:
    optimizer_steps: int = 300
    learning_rate: float = 0.05
    penalty: float = 100.0
    feasibility_margin: float = 0.0
    restarts: int = 4
    defensive_weight: float = 0.1
    occupation_temperature: float = 0.05

    def __post_init__(self) -> None:
        if (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps < 1
        ):
            raise ValueError("optimizer_steps must be a positive integer")
        if (
            isinstance(self.restarts, bool)
            or not isinstance(self.restarts, int)
            or self.restarts < 1
        ):
            raise ValueError("restarts must be a positive integer")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("penalty", self.penalty),
            ("occupation_temperature", self.occupation_temperature),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.feasibility_margin) or self.feasibility_margin < 0.0:
            raise ValueError("feasibility_margin must be finite and nonnegative")
        if not 0.0 < self.defensive_weight < 1.0:
            raise ValueError("defensive_weight must lie in (0, 1)")


def _optimization_score(
    problem: RBergomiBaselineProblem,
    paths: TwoDriverRBergomiPaths,
    *,
    occupation_temperature: float,
) -> torch.Tensor:
    """Differentiable training surrogate; final feasibility remains exact."""

    task = problem.task
    if not isinstance(task, DownsideExcursionTask):
        return problem.score(paths)
    hit_margin = math.log(task.hit_barrier) - torch.amin(paths.log_spot, dim=1)
    soft_occupation = (
        torch.sum(
            torch.sigmoid(
                (math.log(task.stress_level) - paths.log_spot[:, 1:]) / occupation_temperature
            ),
            dim=1,
        )
        * paths.step_dt
    )
    return torch.minimum(hit_margin, soft_occupation - task.minimum_occupation)


def _first_feasible_on_ray(
    problem: RBergomiBaselineProblem,
    direction: torch.Tensor,
    *,
    initial_radius: float,
    maximum_expansions: int = 24,
    bisection_steps: int = 40,
) -> tuple[torch.Tensor | None, int]:
    """Bracket and refine the first exact hard-event point on a fixed ray."""

    norm = float(torch.linalg.vector_norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        return None, 0
    unit = direction / norm
    lower = 0.0
    upper = max(initial_radius, 1e-6)
    evaluations = 0
    with torch.no_grad():
        for _ in range(maximum_expansions):
            candidate = unit * upper
            event = bool(problem.hard_event(problem.simulate_latent(candidate.unsqueeze(0)))[0])
            evaluations += 1
            if event:
                break
            lower = upper
            upper *= 2.0
        else:
            return None, evaluations
        for _ in range(bisection_steps):
            midpoint = 0.5 * (lower + upper)
            candidate = unit * midpoint
            event = bool(problem.hard_event(problem.simulate_latent(candidate.unsqueeze(0)))[0])
            evaluations += 1
            if event:
                upper = midpoint
            else:
                lower = midpoint
    return unit * upper, evaluations


def train_large_deviation_proposal(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
    config: LargeDeviationTrainingConfig | None = None,
) -> FrozenBaselineProposal:
    """Find a low-action event point and freeze a defensive one-axis tilt.

    Optimization may use a smooth occupation proxy, but a candidate is accepted
    only if the exact finite-grid hard event is true.  The zero-mean component
    guarantees full support and bounds ``dP/dQ`` by ``1/defensive_weight``.
    """

    if config is None:
        config = LargeDeviationTrainingConfig()
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("training_seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(training_seed)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    candidates: list[torch.Tensor] = []
    failed_restarts = 0
    screening_samples = 0
    for restart in range(config.restarts):
        if restart == 0:
            initial = torch.zeros(problem.latent_dimension, dtype=torch.float64)
        else:
            initial = 0.05 * torch.randn(
                problem.latent_dimension, generator=generator, dtype=torch.float64
            )
        latent = initial.requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=config.learning_rate)
        for _ in range(config.optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            paths = problem.simulate_latent(latent.unsqueeze(0))
            score = _optimization_score(
                problem,
                paths,
                occupation_temperature=config.occupation_temperature,
            )[0]
            objective = (
                0.5 * torch.sum(latent.square())
                + config.penalty * torch.relu(config.feasibility_margin - score).square()
            )
            if not torch.isfinite(objective):
                raise FloatingPointError("large-deviation objective became nonfinite")
            objective.backward()
            optimizer.step()
        frozen = latent.detach().clone()
        feasible, evaluations = _first_feasible_on_ray(
            problem,
            frozen,
            initial_radius=float(torch.linalg.vector_norm(frozen)),
        )
        screening_samples += evaluations
        if feasible is not None:
            candidates.append(feasible)
        else:
            failed_restarts += 1
    if not candidates:
        fallback = torch.zeros(problem.latent_dimension, dtype=torch.float64)
        fallback[problem.local_dimension :] = -1.0 / math.sqrt(problem.steps)
        feasible, evaluations = _first_feasible_on_ray(
            problem,
            fallback,
            initial_radius=1.0,
        )
        screening_samples += evaluations
        if feasible is None:
            raise RuntimeError("large-deviation training found no exact event-feasible action")
        candidates.append(feasible)
    action = min(candidates, key=lambda item: float(torch.sum(item.square())))
    zero = tuple(0.0 for _ in range(problem.latent_dimension))
    learned = tuple(float(value) for value in action)
    total_steps = config.optimizer_steps * config.restarts
    total_work = 3 * total_steps * (problem.latent_dimension + problem.steps)
    total_work += screening_samples * (problem.latent_dimension + problem.steps)
    cost = BaselineCostLedger(
        optimizer_steps=total_steps,
        hyperparameter_trials=1,
        failed_restarts=failed_restarts,
        screening_samples=screening_samples,
        algorithmic_work_units=float(total_work),
        wall_seconds=time.perf_counter() - wall_started,
        cpu_seconds=time.process_time() - cpu_started,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return freeze_baseline_proposal(
        method="ld_subspace_is",
        task_id=problem.task_id,
        dimension=problem.latent_dimension,
        training_seed=training_seed,
        training_cost=cost,
        training_budget_work_units=float(total_work),
        component_means=(zero, learned),
        component_weights=(config.defensive_weight, 1.0 - config.defensive_weight),
    )
