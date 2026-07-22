"""Independent Black--Scholes probability oracles for eta-zero checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.stats


def _validate_inputs(spot: float, volatility: float, maturity: float) -> None:
    if any(not math.isfinite(value) for value in (spot, volatility, maturity)):
        raise ValueError("Black--Scholes inputs must be finite")
    if spot <= 0.0 or volatility <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, volatility, and maturity must be positive")


def black_scholes_left_digital_probability(
    *, spot: float, level: float, volatility: float, maturity: float
) -> float:
    """Return ``P(S_T <= level)`` under zero rates and risk-neutral GBM."""

    _validate_inputs(spot, volatility, maturity)
    if not math.isfinite(level) or level <= 0.0:
        raise ValueError("level must be finite and positive")
    standardized = (math.log(level / spot) + 0.5 * volatility**2 * maturity) / (
        volatility * math.sqrt(maturity)
    )
    return float(scipy.stats.norm.cdf(standardized))


def black_scholes_continuous_lower_barrier_probability(
    *, spot: float, barrier: float, volatility: float, maturity: float
) -> float:
    """Return ``P(inf_{t<=T} S_t <= barrier)`` by reflection with drift."""

    _validate_inputs(spot, volatility, maturity)
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("barrier must be finite and positive")
    if barrier >= spot:
        return 1.0
    drift = -0.5 * volatility**2
    boundary = math.log(barrier / spot)
    scale = volatility * math.sqrt(maturity)
    probability = scipy.stats.norm.cdf((boundary - drift * maturity) / scale)
    probability += math.exp(2.0 * drift * boundary / volatility**2) * scipy.stats.norm.cdf(
        (boundary + drift * maturity) / scale
    )
    return float(min(1.0, max(0.0, probability)))


@dataclass(frozen=True)
class BlackScholesDiscreteBarrierOracle:
    probability: float
    steps: int
    state_points: int
    upper_standard_deviations: float
    one_step_closed_form: bool


def black_scholes_discrete_lower_barrier_probability(
    *,
    spot: float,
    barrier: float,
    volatility: float,
    maturity: float,
    steps: int,
    state_points: int = 1201,
    upper_standard_deviations: float = 8.0,
) -> BlackScholesDiscreteBarrierOracle:
    """Deterministically integrate the killed one-dimensional Markov density.

    The spatial grid covers the lower barrier through a declared upper Gaussian-tail
    cutoff.  This is a numerical finite-grid oracle, not a closed form.  Convergence
    in ``state_points`` and ``upper_standard_deviations`` must accompany reported
    values with more than one monitoring time.
    """

    _validate_inputs(spot, volatility, maturity)
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("barrier must be finite and positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if state_points < 101 or state_points % 2 == 0:
        raise ValueError("state_points must be an odd integer of at least 101")
    if not math.isfinite(upper_standard_deviations) or upper_standard_deviations < 5.0:
        raise ValueError("upper_standard_deviations must be finite and at least five")
    if barrier >= spot:
        return BlackScholesDiscreteBarrierOracle(
            1.0,
            steps,
            state_points,
            upper_standard_deviations,
            steps == 1,
        )
    if steps == 1:
        probability = black_scholes_left_digital_probability(
            spot=spot,
            level=barrier,
            volatility=volatility,
            maturity=maturity,
        )
        return BlackScholesDiscreteBarrierOracle(
            probability, steps, state_points, upper_standard_deviations, True
        )

    step_dt = maturity / steps
    step_scale = volatility * math.sqrt(step_dt)
    drift_step = -0.5 * volatility**2 * step_dt
    lower = math.log(barrier / spot)
    upper = max(
        upper_standard_deviations * volatility * math.sqrt(maturity),
        -0.5 * volatility**2 * maturity
        + upper_standard_deviations * volatility * math.sqrt(maturity),
    )
    grid = np.linspace(lower, upper, state_points, dtype=np.float64)
    spacing = float(grid[1] - grid[0])
    weights = np.full(state_points, spacing, dtype=np.float64)
    weights[[0, -1]] *= 0.5
    inv_scale = 1.0 / step_scale
    normalizer = inv_scale / math.sqrt(2.0 * math.pi)
    density = normalizer * np.exp(-0.5 * np.square((grid - drift_step) * inv_scale))
    transition = normalizer * np.exp(
        -0.5 * np.square((grid[:, None] - grid[None, :] - drift_step) * inv_scale)
    )
    for _step in range(1, steps):
        density = transition @ (density * weights)
    survival = float(np.dot(density, weights))
    probability = float(min(1.0, max(0.0, 1.0 - survival)))
    return BlackScholesDiscreteBarrierOracle(
        probability,
        steps,
        state_points,
        upper_standard_deviations,
        False,
    )
