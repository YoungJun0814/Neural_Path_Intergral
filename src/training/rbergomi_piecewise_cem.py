"""Likelihood-weighted constant and time-piecewise CEM for path functionals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from src.path_integral import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
)
from src.path_integral.rbergomi_fft import simulate_rbergomi_fft
from src.physics_engine import RBergomiSimulator

PiecewiseValues = tuple[tuple[float, float], ...]
PiecewiseCEMTask: TypeAlias = TerminalThresholdTask | DiscreteBarrierHitTask | DownsideExcursionTask
PriceDriverSign: TypeAlias = Literal["negative", "positive"] | None


@dataclass(frozen=True)
class PiecewiseCEMIteration:
    iteration: int
    control_before: PiecewiseValues
    control_candidate: PiecewiseValues
    control_after: PiecewiseValues
    level: float
    elite_fraction: float
    elite_weight_ess: float
    hard_event_fraction: float
    hard_probability_estimate: float


@dataclass(frozen=True)
class PiecewiseCEMResult:
    control: PiecewiseValues
    converged: bool
    history: tuple[PiecewiseCEMIteration, ...]


def _as_values(control: torch.Tensor) -> PiecewiseValues:
    return tuple((float(row[0]), float(row[1])) for row in control.detach().cpu())


def _segment_sufficient_statistics(
    target_increments: torch.Tensor,
    *,
    segments: int,
    step_dt: float,
) -> torch.Tensor:
    if target_increments.ndim != 3 or target_increments.shape[2] != 2:
        raise ValueError("target_increments must have shape (paths, steps, 2)")
    steps = target_increments.shape[1]
    step_index = torch.arange(steps, device=target_increments.device)
    segment_index = torch.clamp(step_index * segments // steps, max=segments - 1)
    statistics: list[torch.Tensor] = []
    for segment in range(segments):
        selected = segment_index == segment
        duration = float(selected.sum()) * step_dt
        if duration <= 0.0:
            raise ValueError("every CEM segment must contain at least one time step")
        statistics.append(target_increments[:, selected].sum(dim=1) / duration)
    return torch.stack(statistics, dim=1)


def _project_control(
    control: torch.Tensor,
    *,
    control_bound: float,
    price_driver_sign: PriceDriverSign,
    minimum_price_driver_magnitude: float,
) -> torch.Tensor:
    """Project a CEM update onto its frozen box and price-sign constraints."""

    projected = torch.clamp(control, min=-control_bound, max=control_bound)
    if price_driver_sign is None:
        return projected
    if minimum_price_driver_magnitude <= 0.0:
        raise ValueError(
            "a constrained price driver requires a positive minimum magnitude"
        )
    result = projected.clone()
    if price_driver_sign == "negative":
        result[:, 1] = torch.minimum(
            result[:, 1],
            torch.full_like(result[:, 1], -minimum_price_driver_magnitude),
        )
    elif price_driver_sign == "positive":
        result[:, 1] = torch.maximum(
            result[:, 1],
            torch.full_like(result[:, 1], minimum_price_driver_magnitude),
        )
    else:
        raise ValueError("price_driver_sign must be negative, positive, or None")
    return result


def fit_rbergomi_piecewise_cem(
    simulator: RBergomiSimulator,
    task: PiecewiseCEMTask,
    *,
    spot: float,
    maturity: float,
    dt: float,
    initial_control: PiecewiseValues,
    num_paths: int,
    seed: int,
    max_iterations: int = 10,
    elite_quantile: float = 0.90,
    smoothing: float = 0.70,
    min_elite_paths: int = 64,
    control_bound: float = 8.0,
    target_level_repetitions: int = 2,
    price_driver_sign: PriceDriverSign = None,
    minimum_price_driver_magnitude: float = 0.0,
) -> PiecewiseCEMResult:
    """Fit equal-duration Gaussian mean shifts by target-coordinate weighted MLE."""
    if not initial_control or any(len(value) != 2 for value in initial_control):
        raise ValueError("initial_control must contain two-driver segments")
    if not all(math.isfinite(entry) for value in initial_control for entry in value):
        raise ValueError("initial controls must be finite")
    if num_paths <= 0 or max_iterations <= 0:
        raise ValueError("num_paths and max_iterations must be positive")
    if not 0.0 < elite_quantile < 1.0 or not 0.0 < smoothing <= 1.0:
        raise ValueError("elite_quantile and smoothing are outside their valid ranges")
    if min_elite_paths <= 0 or min_elite_paths > num_paths:
        raise ValueError("min_elite_paths is outside its valid range")
    if control_bound <= 0.0 or target_level_repetitions <= 0:
        raise ValueError("control bound and target repetitions must be positive")
    if price_driver_sign is None and minimum_price_driver_magnitude != 0.0:
        raise ValueError(
            "minimum price-driver magnitude requires an explicit sign constraint"
        )
    if price_driver_sign is not None and (
        minimum_price_driver_magnitude <= 0.0
        or minimum_price_driver_magnitude > control_bound
    ):
        raise ValueError(
            "constrained price-driver magnitude must lie in (0, control_bound]"
        )

    segments = len(initial_control)
    control = _project_control(
        torch.tensor(initial_control, dtype=torch.float64),
        control_bound=control_bound,
        price_driver_sign=price_driver_sign,
        minimum_price_driver_magnitude=minimum_price_driver_magnitude,
    )
    history: list[PiecewiseCEMIteration] = []
    target_hits = 0
    torch.manual_seed(seed)
    for iteration in range(max_iterations):
        proposal = TimePiecewiseTwoDriverControl(_as_values(control), maturity=maturity)
        paths = simulate_rbergomi_fft(
            simulator,
            S0=spot,
            T=maturity,
            dt=dt,
            num_paths=num_paths,
            control_fn=proposal,
            dtype=torch.float64,
        )
        assert paths.target_brownian_increments is not None
        score = task.score(paths.spot, paths.step_dt)
        quantile_level = float(torch.quantile(score, elite_quantile))
        level = min(0.0, quantile_level)
        elite = score >= level
        if int(elite.sum()) < min_elite_paths:
            indices = torch.topk(score, k=min_elite_paths).indices
            elite = torch.zeros_like(score, dtype=torch.bool)
            elite[indices] = True

        normalized = torch.softmax(paths.log_likelihood[elite], dim=0)
        sufficient = _segment_sufficient_statistics(
            paths.target_brownian_increments,
            segments=segments,
            step_dt=paths.step_dt,
        )
        candidate = torch.sum(normalized[:, None, None] * sufficient[elite], dim=0)
        updated = _project_control(
            (1.0 - smoothing) * control + smoothing * candidate,
            control_bound=control_bound,
            price_driver_sign=price_driver_sign,
            minimum_price_driver_magnitude=minimum_price_driver_magnitude,
        )
        event = task.hard_event(paths.spot, paths.step_dt)
        probability = torch.mean(event.double() * torch.exp(paths.log_likelihood))
        history.append(
            PiecewiseCEMIteration(
                iteration=iteration,
                control_before=_as_values(control),
                control_candidate=_as_values(candidate),
                control_after=_as_values(updated),
                level=level,
                elite_fraction=float(elite.double().mean()),
                elite_weight_ess=float(torch.reciprocal(torch.sum(normalized.square()))),
                hard_event_fraction=float(event.double().mean()),
                hard_probability_estimate=float(probability),
            )
        )
        control = updated
        if level >= 0.0:
            target_hits += 1
            if target_hits >= target_level_repetitions:
                return PiecewiseCEMResult(
                    control=_as_values(control),
                    converged=True,
                    history=tuple(history),
                )
        else:
            target_hits = 0
    return PiecewiseCEMResult(
        control=_as_values(control),
        converged=False,
        history=tuple(history),
    )
