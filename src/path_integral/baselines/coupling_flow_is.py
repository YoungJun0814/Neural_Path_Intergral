"""One-layer triangular affine coupling baseline with exact density."""

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
from src.path_integral.provenance import process_peak_resident_memory_bytes

from .rbergomi_common import RBergomiBaselineProblem


@dataclass(frozen=True)
class FlowTrainingConfig:
    screening_samples: int = 8192
    elite_fraction: float = 0.1
    max_log_scale: float = 1.5
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.screening_samples, bool)
            or not isinstance(self.screening_samples, int)
            or self.screening_samples < 2
        ):
            raise ValueError("screening_samples must be an integer of at least two")
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError("elite_fraction must lie in (0, 1)")
        if not math.isfinite(self.max_log_scale) or self.max_log_scale <= 0.0:
            raise ValueError("max_log_scale must be finite and positive")
        if not math.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")


def train_coupling_flow_proposal(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
    config: FlowTrainingConfig | None = None,
) -> FrozenBaselineProposal:
    """Moment-fit an invertible triangular flow to high-score target paths.

    This is intentionally a transparent, reproducible strong baseline rather
    than a claim of a universal flow.  Bounded log-scales make the map globally
    invertible, and the framework evaluates its analytic Jacobian exactly.
    """

    if config is None:
        config = FlowTrainingConfig()
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("training_seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(training_seed)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    latent = torch.randn(
        (config.screening_samples, problem.latent_dimension),
        generator=generator,
        dtype=torch.float64,
    )
    score = problem.score(problem.simulate_latent(latent))
    if score.ndim != 1 or not torch.isfinite(score).all():
        raise FloatingPointError("flow screening score is invalid")
    elite_count = max(2, math.ceil(config.elite_fraction * config.screening_samples))
    elite = latent[torch.topk(score, k=elite_count, largest=True).indices]
    split = problem.latent_dimension // 2
    location = torch.mean(elite, dim=0)
    first = elite[:, :split] - location[:split]
    second = elite[:, split:] - location[split:]
    identity = torch.eye(split, dtype=torch.float64)
    regression = torch.linalg.solve(
        first.T @ first + config.ridge * identity,
        first.T @ second,
    )
    residual = second - first @ regression
    residual_scale = torch.sqrt(torch.mean(residual.square(), dim=0) + config.ridge)
    desired_log_scale = torch.clamp(
        torch.log(residual_scale),
        min=-0.95 * config.max_log_scale,
        max=0.95 * config.max_log_scale,
    )
    scale_bias = torch.atanh(desired_log_scale / config.max_log_scale)
    transformed = problem.latent_dimension - split
    scale_matrix = torch.zeros((transformed, split), dtype=torch.float64)
    shift_matrix = regression.T.contiguous()
    shift_bias = torch.zeros(transformed, dtype=torch.float64)
    tensors = (location, scale_bias, scale_matrix, shift_matrix, shift_bias)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("fitted flow parameter became nonfinite")
    regression_work = elite_count * split * transformed
    work = config.screening_samples * (problem.latent_dimension + problem.steps)
    work += regression_work
    cost = BaselineCostLedger(
        optimizer_steps=1,
        hyperparameter_trials=1,
        screening_samples=config.screening_samples,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - wall_started,
        cpu_seconds=time.process_time() - cpu_started,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return freeze_baseline_proposal(
        method="flow_is",
        task_id=problem.task_id,
        dimension=problem.latent_dimension,
        training_seed=training_seed,
        training_cost=cost,
        training_budget_work_units=float(work),
        location=tuple(float(value) for value in location),
        flow_split=split,
        flow_scale_matrix=tuple(tuple(float(value) for value in row) for row in scale_matrix),
        flow_scale_bias=tuple(float(value) for value in scale_bias),
        flow_shift_matrix=tuple(tuple(float(value) for value in row) for row in shift_matrix),
        flow_shift_bias=tuple(float(value) for value in shift_bias),
        flow_max_log_scale=config.max_log_scale,
        conditional_integral="baseline_only",
    )
