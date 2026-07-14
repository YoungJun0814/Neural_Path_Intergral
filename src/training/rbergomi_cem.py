"""Two-driver cross-entropy baseline for terminal rBergomi tail events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.controllers import ConstantTwoDriverControl
from src.physics_engine import RBergomiSimulator

TailMode = Literal["left", "right"]


@dataclass(frozen=True)
class TwoDriverCEMIteration:
    iteration: int
    control_before: tuple[float, float]
    control_candidate: tuple[float, float]
    control_after: tuple[float, float]
    level: float
    elite_fraction: float
    elite_weight_ess: float
    hard_event_fraction: float
    hard_probability_estimate: float


@dataclass(frozen=True)
class TwoDriverCEMResult:
    control: tuple[float, float]
    converged: bool
    history: tuple[TwoDriverCEMIteration, ...]


def fit_rbergomi_two_driver_cem(
    simulator: RBergomiSimulator,
    *,
    spot: float,
    maturity: float,
    dt: float,
    threshold: float,
    mode: TailMode,
    initial_control: tuple[float, float],
    num_paths: int,
    seed: int,
    max_iterations: int = 8,
    elite_quantile: float = 0.90,
    smoothing: float = 0.70,
    min_elite_paths: int = 32,
    control_bound: float = 8.0,
    target_level_repetitions: int = 2,
) -> TwoDriverCEMResult:
    """Fit a constant independent-basis drift by likelihood-weighted Gaussian MLE."""
    if mode not in ("left", "right"):
        raise ValueError("mode must be 'left' or 'right'")
    if not all(math.isfinite(value) for value in initial_control):
        raise ValueError("initial_control must be finite")
    if num_paths <= 0 or max_iterations <= 0:
        raise ValueError("num_paths and max_iterations must be positive")
    if not 0.0 < elite_quantile < 1.0 or not 0.0 < smoothing <= 1.0:
        raise ValueError("elite_quantile and smoothing are outside their valid ranges")
    if min_elite_paths <= 0 or min_elite_paths > num_paths:
        raise ValueError("min_elite_paths is outside its valid range")
    if control_bound <= 0.0 or target_level_repetitions <= 0:
        raise ValueError("control_bound and target repetitions must be positive")
    score_sign = -1.0 if mode == "left" else 1.0
    target_score = score_sign * threshold
    control = torch.tensor(initial_control, dtype=torch.float64)
    history: list[TwoDriverCEMIteration] = []
    target_hits = 0
    torch.manual_seed(seed)
    for iteration in range(max_iterations):
        proposal = ConstantTwoDriverControl(float(control[0]), float(control[1]))
        paths = simulator.simulate_controlled_two_driver(
            S0=spot,
            T=maturity,
            dt=dt,
            num_paths=num_paths,
            control_fn=proposal,
            record_augmented=True,
            dtype=torch.float64,
        )
        assert paths.target_brownian_increments is not None
        score = score_sign * paths.spot[:, -1]
        quantile_level = float(torch.quantile(score, elite_quantile))
        level = min(target_score, quantile_level)
        elite = score >= level
        if int(elite.sum()) < min_elite_paths:
            indices = torch.topk(score, k=min_elite_paths).indices
            elite = torch.zeros_like(score, dtype=torch.bool)
            elite[indices] = True
        normalized = torch.softmax(paths.log_likelihood[elite], dim=0)
        sufficient = paths.target_brownian_increments.sum(dim=1) / maturity
        candidate = torch.sum(normalized[:, None] * sufficient[elite], dim=0)
        updated = (1.0 - smoothing) * control + smoothing * candidate
        updated = torch.clamp(updated, min=-control_bound, max=control_bound)
        event = score >= target_score
        probability = torch.mean(event.double() * torch.exp(paths.log_likelihood))
        history.append(
            TwoDriverCEMIteration(
                iteration=iteration,
                control_before=(float(control[0]), float(control[1])),
                control_candidate=(float(candidate[0]), float(candidate[1])),
                control_after=(float(updated[0]), float(updated[1])),
                level=level,
                elite_fraction=float(elite.double().mean()),
                elite_weight_ess=float(
                    torch.reciprocal(torch.sum(normalized.square()))
                ),
                hard_event_fraction=float(event.double().mean()),
                hard_probability_estimate=float(probability),
            )
        )
        control = updated
        if level >= target_score:
            target_hits += 1
            if target_hits >= target_level_repetitions:
                return TwoDriverCEMResult(
                    control=(float(control[0]), float(control[1])),
                    converged=True,
                    history=tuple(history),
                )
        else:
            target_hits = 0
    return TwoDriverCEMResult(
        control=(float(control[0]), float(control[1])),
        converged=False,
        history=tuple(history),
    )
