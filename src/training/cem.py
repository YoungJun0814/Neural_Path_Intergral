"""Trajectory-likelihood cross-entropy method for constant Brownian controls.

Unlike the removed state-classification helper, this module keeps each elite
label, likelihood ratio, and sufficient statistic on the *same trajectory*.
The update is the weighted maximum-likelihood projection of the intermediate
rare-event distribution onto a constant-drift Gaussian family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from src.physics_engine import MarketSimulator


@dataclass(frozen=True)
class CEMBatch:
    """One proposal batch used by a cross-entropy update."""

    score: torch.Tensor
    log_base_over_proposal: torch.Tensor
    base_sufficient_statistic: torch.Tensor

    def validate(self) -> None:
        if self.score.ndim != 1:
            raise ValueError("score must be one-dimensional")
        if self.log_base_over_proposal.shape != self.score.shape:
            raise ValueError("log likelihood ratios must match score shape")
        if self.base_sufficient_statistic.shape != self.score.shape:
            raise ValueError("sufficient statistics must match score shape")
        if not all(
            torch.isfinite(value).all()
            for value in (
                self.score,
                self.log_base_over_proposal,
                self.base_sufficient_statistic,
            )
        ):
            raise ValueError("CEM batch contains non-finite values")


class ConstantControlSampler(Protocol):
    """Sampler for the one-parameter constant Brownian-drift family."""

    def __call__(self, control: float, num_paths: int) -> CEMBatch: ...


@dataclass(frozen=True)
class CEMIteration:
    iteration: int
    control_before: float
    control_candidate: float
    control_after: float
    level: float
    elite_fraction: float
    elite_weight_ess: float
    target_event_fraction_under_proposal: float
    target_probability_estimate: float


@dataclass(frozen=True)
class CEMResult:
    control: float
    converged: bool
    history: tuple[CEMIteration, ...]


def fit_constant_control_cem(
    sampler: ConstantControlSampler,
    *,
    initial_control: float,
    target_score: float,
    num_paths: int,
    max_iterations: int = 8,
    elite_quantile: float = 0.90,
    smoothing: float = 0.70,
    min_elite_paths: int = 32,
    control_bounds: tuple[float, float] = (-10.0, 10.0),
    target_level_repetitions: int = 2,
) -> CEMResult:
    """Fit a constant control for the event ``score >= target_score``.

    For batch trajectories drawn from ``Q_u``, the update computes

    ``u_new = E_Q[1{score>=gamma} (dP/dQ) T(X)] / E_Q[1{...} (dP/dQ)]``,

    where ``T(X)`` is the MLE sufficient statistic for the Gaussian drift.
    ``gamma`` advances adaptively by an elite quantile until it reaches the
    requested event threshold.
    """
    if initial_control == 0.0:
        raise ValueError("initial_control must be nonzero for likelihood reconstruction")
    if num_paths <= 0 or max_iterations <= 0:
        raise ValueError("num_paths and max_iterations must be positive")
    if not 0.0 < elite_quantile < 1.0:
        raise ValueError("elite_quantile must lie in (0, 1)")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    if min_elite_paths <= 0 or min_elite_paths > num_paths:
        raise ValueError("min_elite_paths must lie in [1, num_paths]")
    lower, upper = control_bounds
    if lower >= upper:
        raise ValueError("control_bounds must be increasing")
    if target_level_repetitions <= 0:
        raise ValueError("target_level_repetitions must be positive")

    control = float(initial_control)
    history: list[CEMIteration] = []
    target_hits = 0

    for iteration in range(max_iterations):
        batch = sampler(control, num_paths)
        batch.validate()

        quantile_level = float(torch.quantile(batch.score, elite_quantile))
        level = min(float(target_score), quantile_level)
        elite = batch.score >= level
        elite_count = int(elite.sum())
        if elite_count < min_elite_paths:
            # Ties or a discrete score may make a quantile mask unexpectedly
            # small. Select the top paths deterministically in that case.
            top_indices = torch.topk(batch.score, k=min_elite_paths).indices
            elite = torch.zeros_like(batch.score, dtype=torch.bool)
            elite[top_indices] = True
            elite_count = min_elite_paths

        elite_log_weights = batch.log_base_over_proposal[elite]
        normalized_weights = torch.softmax(elite_log_weights, dim=0)
        candidate = float(torch.sum(normalized_weights * batch.base_sufficient_statistic[elite]))
        updated = (1.0 - smoothing) * control + smoothing * candidate
        updated = min(max(updated, lower), upper)

        elite_ess = float(1.0 / torch.sum(normalized_weights**2))
        event = batch.score >= target_score
        target_probability = float(
            torch.mean(
                event.to(batch.log_base_over_proposal.dtype)
                * torch.exp(batch.log_base_over_proposal)
            )
        )
        history.append(
            CEMIteration(
                iteration=iteration,
                control_before=control,
                control_candidate=candidate,
                control_after=updated,
                level=level,
                elite_fraction=elite_count / num_paths,
                elite_weight_ess=elite_ess,
                target_event_fraction_under_proposal=float(event.float().mean()),
                target_probability_estimate=target_probability,
            )
        )
        control = updated

        if level >= target_score:
            target_hits += 1
            if target_hits >= target_level_repetitions:
                return CEMResult(control=control, converged=True, history=tuple(history))
        else:
            target_hits = 0

    return CEMResult(control=control, converged=False, history=tuple(history))


class HestonTerminalLossSampler:
    """Adapt ``MarketSimulator`` to constant-control trajectory CEM.

    The score defaults to ``-S_T`` so a terminal downside event ``S_T <= K``
    is represented by ``score >= -K``.  For constant nonzero control ``u``,
    the returned likelihood obeys

    ``log(dP/dQ_u) = -u W_T^Q - 0.5 u^2 T``.

    Hence ``W_T^P/T = -log(dP/dQ_u)/(uT) + 0.5u`` is the sufficient statistic
    used in the weighted Gaussian MLE update.
    """

    def __init__(
        self,
        simulator: MarketSimulator,
        *,
        spot: float,
        variance: float,
        maturity: float,
        dt: float,
        model_type: str = "heston",
        score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if spot <= 0.0 or variance < 0.0 or maturity <= 0.0 or dt <= 0.0:
            raise ValueError("spot, maturity and dt must be positive; variance must be nonnegative")
        self.simulator = simulator
        self.spot = float(spot)
        self.variance = float(variance)
        self.maturity = float(maturity)
        self.dt = float(dt)
        self.model_type = model_type
        self.score_fn = score_fn or (lambda terminal_spot: -terminal_spot)

    def __call__(self, control: float, num_paths: int) -> CEMBatch:
        if abs(control) < 1e-8:
            raise ValueError("constant control is too close to zero for stable reconstruction")

        def constant_control(
            _time: float,
            spot: torch.Tensor,
            _variance: torch.Tensor,
            _average_spot: torch.Tensor,
        ) -> torch.Tensor:
            return torch.full_like(spot, control)

        paths, _variance, log_weight, _barrier, _running_average = (
            self.simulator.simulate_controlled(
                S0=self.spot,
                v0=self.variance,
                T=self.maturity,
                dt=self.dt,
                num_paths=num_paths,
                control_fn=constant_control,
                model_type=self.model_type,
            )
        )
        sufficient_statistic = -log_weight / (control * self.maturity) + 0.5 * control
        return CEMBatch(
            score=self.score_fn(paths[:, -1]),
            log_base_over_proposal=log_weight,
            base_sufficient_statistic=sufficient_statistic,
        )
