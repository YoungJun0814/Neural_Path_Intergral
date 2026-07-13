"""Closed-form Brownian path-integral oracles used as correctness gates."""

from __future__ import annotations

import math

from scipy.special import log_ndtr, ndtr


def _validate_horizon(horizon: float) -> None:
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")


def gaussian_exponential_tilt_log_normalizer(tilt: float, horizon: float) -> float:
    r"""Return ``log E[exp(tilt * B_T)] = 0.5 * tilt^2 * T``."""
    _validate_horizon(horizon)
    if not math.isfinite(tilt):
        raise ValueError("tilt must be finite")
    return 0.5 * tilt * tilt * horizon


def gaussian_exponential_tilt_optimal_control(tilt: float, horizon: float) -> float:
    """Return the Föllmer drift for ``g(B_T)=exp(tilt*B_T)``."""
    _validate_horizon(horizon)
    if not math.isfinite(tilt):
        raise ValueError("tilt must be finite")
    return tilt


def gaussian_exponential_tilt_pi_objective(
    tilt: float,
    control: float,
    horizon: float,
) -> float:
    r"""Return the constant-control PI objective ``-a*u*T + u^2*T/2``."""
    _validate_horizon(horizon)
    if not math.isfinite(tilt) or not math.isfinite(control):
        raise ValueError("tilt and control must be finite")
    return (-tilt * control + 0.5 * control * control) * horizon


def gaussian_exponential_tilt_pi_gap(
    tilt: float,
    control: float,
    horizon: float,
) -> float:
    """Return the PI optimum gap, equal to the forward KL in this toy model."""
    _validate_horizon(horizon)
    if not math.isfinite(tilt) or not math.isfinite(control):
        raise ValueError("tilt and control must be finite")
    return 0.5 * (control - tilt) ** 2 * horizon


def gaussian_exponential_tilt_relative_variance(
    tilt: float,
    control: float,
    horizon: float,
) -> float:
    r"""Return ``Var_Q(gL)/Z^2 = exp((tilt-control)^2*T)-1``."""
    _validate_horizon(horizon)
    if not math.isfinite(tilt) or not math.isfinite(control):
        raise ValueError("tilt and control must be finite")
    return math.expm1((tilt - control) ** 2 * horizon)


def gaussian_left_tail_probability(
    current: float,
    barrier: float,
    remaining_time: float,
) -> float:
    r"""Return ``P(B_T <= barrier | B_t=current)`` for Brownian motion."""
    _validate_horizon(remaining_time)
    if not math.isfinite(current) or not math.isfinite(barrier):
        raise ValueError("current and barrier must be finite")
    z_score = (barrier - current) / math.sqrt(remaining_time)
    return float(ndtr(z_score))


def gaussian_left_tail_doob_drift(
    current: float,
    barrier: float,
    remaining_time: float,
) -> float:
    r"""Return ``partial_x log P(B_T<=barrier | B_t=x)``.

    The drift is negative, as required to steer Brownian paths toward a left
    tail.  ``log_ndtr`` keeps the inverse Mills ratio stable in rare tails.
    This is the continuous hard-conditional oracle; it is a verification
    target, not a claim that a bounded finite-step control represents the
    singular conditional law exactly.
    """
    _validate_horizon(remaining_time)
    if not math.isfinite(current) or not math.isfinite(barrier):
        raise ValueError("current and barrier must be finite")
    root_time = math.sqrt(remaining_time)
    z_score = (barrier - current) / root_time
    log_density = -0.5 * z_score * z_score - 0.5 * math.log(2.0 * math.pi)
    inverse_mills_ratio = math.exp(log_density - float(log_ndtr(z_score)))
    return -inverse_mills_ratio / root_time
