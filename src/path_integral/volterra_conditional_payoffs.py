"""Stable terminal conditional payoffs for lognormal Gaussian-Volterra models.

The implementation conditions on every local Volterra coordinate and integrates the
complete independent price driver.  ``epsilon=1`` is the declared finite-grid rBergomi
law; ``epsilon<1`` is the small-noise family frozen in the V15 scaling contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask


def _require_cpu_float64(name: str, value: torch.Tensor) -> None:
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError(f"{name} must be CPU float64")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _logdiffexp(log_large: torch.Tensor, log_small: torch.Tensor) -> torch.Tensor:
    """Return ``log(exp(log_large)-exp(log_small))`` without cancellation."""

    _require_cpu_float64("log_large", log_large)
    _require_cpu_float64("log_small", log_small)
    if log_large.shape != log_small.shape:
        raise ValueError("log-difference inputs must have the same shape")
    tolerance = 64.0 * torch.finfo(torch.float64).eps * torch.maximum(
        torch.ones_like(log_large), torch.abs(log_large)
    )
    if bool((log_small > log_large + tolerance).any()):
        raise FloatingPointError("log-difference terms have invalid order")
    ratio_log = torch.minimum(log_small - log_large, torch.zeros_like(log_large))
    # log1p(-exp(x)) is unstable for x close to zero; expm1 retains the gap.
    gap = -torch.expm1(ratio_log)
    result = log_large + torch.log(gap)
    return torch.where(gap > 0.0, result, torch.full_like(result, -torch.inf))


@dataclass(frozen=True)
class LognormalTerminalPayoffBatch:
    standardized_left_threshold: torch.Tensor
    log_left_probability: torch.Tensor
    left_probability: torch.Tensor
    log_right_probability: torch.Tensor
    right_probability: torch.Tensor
    log_put_value: torch.Tensor
    put_value: torch.Tensor
    log_call_value: torch.Tensor
    call_value: torch.Tensor
    conditional_forward: torch.Tensor


def evaluate_lognormal_terminal_payoffs(
    log_mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    strike: float,
) -> LognormalTerminalPayoffBatch:
    """Evaluate digital, put and call values for ``log(S_T) ~ N(m, variance)``."""

    _require_cpu_float64("log_mean", log_mean)
    _require_cpu_float64("variance", variance)
    if log_mean.ndim != 1 or variance.shape != log_mean.shape:
        raise ValueError("lognormal parameters must be one-dimensional and aligned")
    if bool((variance <= 0.0).any()):
        raise ValueError("lognormal variance must be strictly positive")
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")

    log_strike = math.log(strike)
    standard_deviation = torch.sqrt(variance)
    threshold = (log_strike - log_mean) / standard_deviation
    log_left = torch.special.log_ndtr(threshold)
    log_right = torch.special.log_ndtr(-threshold)
    left = torch.exp(log_left)
    right = torch.exp(log_right)
    log_forward = log_mean + 0.5 * variance
    forward = torch.exp(log_forward)

    put_large = log_strike + log_left
    put_small = log_forward + torch.special.log_ndtr(threshold - standard_deviation)
    log_put = _logdiffexp(put_large, put_small)
    put = torch.exp(log_put)

    call_large = log_forward + torch.special.log_ndtr(-threshold + standard_deviation)
    call_small = log_strike + log_right
    log_call = _logdiffexp(call_large, call_small)
    call = torch.exp(log_call)

    values = (log_left, log_right, left, right, put, call, forward)
    if any(bool(torch.isnan(value).any()) for value in values):
        raise FloatingPointError("conditional lognormal payoff produced NaN")
    if any(not torch.isfinite(value).all() for value in (left, right, put, call, forward)):
        raise FloatingPointError("conditional lognormal payoff is not finite")
    return LognormalTerminalPayoffBatch(
        standardized_left_threshold=threshold,
        log_left_probability=log_left,
        left_probability=left,
        log_right_probability=log_right,
        right_probability=right,
        log_put_value=log_put,
        put_value=put,
        log_call_value=log_call,
        call_value=call,
        conditional_forward=forward,
    )


@dataclass(frozen=True)
class ConditionalVolterraTerminalBatch:
    epsilon: float
    integrated_variance: torch.Tensor
    terminal_log_mean: torch.Tensor
    terminal_log_variance: torch.Tensor
    payoffs: LognormalTerminalPayoffBatch


def evaluate_rbergomi_conditional_terminal(
    problem: RBergomiBaselineProblem,
    local_standard_normal: torch.Tensor,
    *,
    epsilon: float = 1.0,
    strike: float | None = None,
) -> ConditionalVolterraTerminalBatch:
    """Evaluate the V15 conditional terminal law under the BLP finite-grid model.

    The input remains standard normal under the reference law.  Internally the local
    innovations are multiplied by ``sqrt(epsilon)``.  The simulator's unit-noise Itô
    drift is then corrected from ``-I/2`` to ``-epsilon*I/2`` and the independent
    price-driver variance is multiplied by ``epsilon``.
    """

    if strike is None:
        if not isinstance(problem.task, TerminalThresholdTask):
            raise ValueError("a strike is required when the problem task is not terminal")
        resolved_strike = float(problem.task.level)
    else:
        resolved_strike = float(strike)
    if local_standard_normal.ndim != 2 or (
        local_standard_normal.shape[1] != problem.local_dimension
    ):
        raise ValueError("local Gaussian sample has the wrong shape")
    _require_cpu_float64("local_standard_normal", local_standard_normal)
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must lie in (0, 1]")
    if not math.isfinite(resolved_strike) or resolved_strike <= 0.0:
        raise ValueError("strike must be finite and positive")

    scaled_local = math.sqrt(epsilon) * local_standard_normal
    # Both parts of the Wick correction scale with the variance of the Volterra
    # driver.  Scaling only the innovations while retaining the unit-noise
    # compensator would converge to the wrong deterministic variance curve.
    paths = problem.simulate_local(
        scaled_local,
        variance_compensator_scale=epsilon,
    )
    integrated_variance = problem.step_dt * torch.sum(paths.variance[:, :-1], dim=1)
    if not torch.isfinite(integrated_variance).all() or bool(
        (integrated_variance <= 0.0).any()
    ):
        raise FloatingPointError("integrated variance is invalid")

    # simulate_local uses unit diffusion and therefore subtracts I/2.  Add back the
    # unused fraction so the small-noise family has the required -epsilon*I/2 drift.
    log_mean = paths.log_spot[:, -1] + 0.5 * (1.0 - epsilon) * integrated_variance
    log_variance = epsilon * (1.0 - problem.rho**2) * integrated_variance
    payoffs = evaluate_lognormal_terminal_payoffs(
        log_mean,
        log_variance,
        strike=resolved_strike,
    )
    return ConditionalVolterraTerminalBatch(
        epsilon=float(epsilon),
        integrated_variance=integrated_variance,
        terminal_log_mean=log_mean,
        terminal_log_variance=log_variance,
        payoffs=payoffs,
    )
