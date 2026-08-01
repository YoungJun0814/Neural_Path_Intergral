"""Cross-entropy Gaussian tilts with exact ordinary importance sampling."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    FrozenBaselineProposal,
    freeze_baseline_proposal,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes

from .rbergomi_common import RBergomiBaselineProblem


@dataclass(frozen=True)
class CEMTrainingConfig:
    iterations: int = 8
    samples_per_iteration: int = 2048
    elite_fraction: float = 0.1
    smoothing: float = 0.7
    defensive_weight: float = 0.1
    max_mean_norm: float = 20.0

    def __post_init__(self) -> None:
        for name, value in (
            ("iterations", self.iterations),
            ("samples_per_iteration", self.samples_per_iteration),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError("elite_fraction must lie in (0, 1)")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must lie in (0, 1]")
        if not 0.0 < self.defensive_weight < 1.0:
            raise ValueError("defensive_weight must lie in (0, 1)")
        if not math.isfinite(self.max_mean_norm) or self.max_mean_norm <= 0.0:
            raise ValueError("max_mean_norm must be finite and positive")


def train_cem_proposal(
    problem: RBergomiBaselineProblem,
    *,
    method: Literal["pure_cem", "defensive_cem"],
    training_seed: int,
    config: CEMTrainingConfig | None = None,
) -> FrozenBaselineProposal:
    """Fit an adaptive mean shift, then freeze its exact Gaussian density.

    The score is used only during training.  The later estimate remains the
    ordinary mean of hard-event indicators multiplied by the exact density
    ratio, so elite selection cannot bias the reported probability.
    """

    if config is None:
        config = CEMTrainingConfig()
    if method not in {"pure_cem", "defensive_cem"}:
        raise ValueError("CEM method must be pure_cem or defensive_cem")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("training_seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(training_seed)
    dimension = problem.latent_dimension
    mean = torch.zeros(dimension, dtype=torch.float64)
    elite_count = max(1, math.ceil(config.elite_fraction * config.samples_per_iteration))
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for _ in range(config.iterations):
        noise = torch.randn(
            (config.samples_per_iteration, dimension),
            generator=generator,
            dtype=torch.float64,
        )
        latent = noise + mean.unsqueeze(0)
        score = problem.score(problem.simulate_latent(latent))
        if score.ndim != 1 or not torch.isfinite(score).all():
            raise FloatingPointError("CEM training score is invalid")
        elite_indices = torch.topk(score, k=elite_count, largest=True).indices
        fitted = torch.mean(latent[elite_indices], dim=0)
        mean = (1.0 - config.smoothing) * mean + config.smoothing * fitted
        norm = torch.linalg.vector_norm(mean)
        if not torch.isfinite(norm):
            raise FloatingPointError("CEM mean became nonfinite")
        if float(norm) > config.max_mean_norm:
            mean = mean * (config.max_mean_norm / float(norm))

    training_samples = config.iterations * config.samples_per_iteration
    work = training_samples * (2 * dimension + problem.steps)
    cost = BaselineCostLedger(
        training_samples=training_samples,
        optimizer_steps=config.iterations,
        hyperparameter_trials=1,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - wall_started,
        cpu_seconds=time.process_time() - cpu_started,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    learned = tuple(float(value) for value in mean)
    if method == "pure_cem":
        return freeze_baseline_proposal(
            method=method,
            task_id=problem.task_id,
            dimension=dimension,
            training_seed=training_seed,
            training_cost=cost,
            training_budget_work_units=float(work),
            location=learned,
        )
    zero = tuple(0.0 for _ in range(dimension))
    return freeze_baseline_proposal(
        method=method,
        task_id=problem.task_id,
        dimension=dimension,
        training_seed=training_seed,
        training_cost=cost,
        training_budget_work_units=float(work),
        component_means=(zero, learned),
        component_weights=(config.defensive_weight, 1.0 - config.defensive_weight),
    )
