"""Analytic and quadrature oracles for a symmetric Gaussian two-tail event."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def _validate(horizon: float, threshold: float) -> None:
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")


def gaussian_two_tail_probability(horizon: float, threshold: float) -> float:
    """Return ``P(|W_T| >= threshold)``."""
    _validate(horizon, threshold)
    return float(2.0 * ndtr(-threshold / math.sqrt(horizon)))


def gaussian_single_drift_second_moment(
    drift: float,
    *,
    horizon: float,
    threshold: float,
) -> float:
    """Return the exact second moment for one constant Brownian drift."""
    _validate(horizon, threshold)
    if not math.isfinite(drift):
        raise ValueError("drift must be finite")
    root = math.sqrt(horizon)
    left = ndtr((-threshold + drift * horizon) / root)
    right = ndtr((-threshold - drift * horizon) / root)
    return float(math.exp(drift * drift * horizon) * (left + right))


def gaussian_symmetric_mixture_log_q_over_p(
    terminal_brownian: np.ndarray | float,
    *,
    drift: float,
    horizon: float,
) -> np.ndarray:
    """Return analytic ``log(dQ_mix/dP)`` for equal ``+/- drift`` experts."""
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(drift):
        raise ValueError("drift must be finite")
    terminal = np.asarray(terminal_brownian, dtype=np.float64)
    absolute = np.abs(drift * terminal)
    log_cosh = absolute + np.log1p(np.exp(-2.0 * absolute)) - math.log(2.0)
    return -0.5 * drift * drift * horizon + log_cosh


def gaussian_symmetric_mixture_second_moment(
    drift: float,
    *,
    horizon: float,
    threshold: float,
    quadrature_tolerance: float = 1e-11,
) -> float:
    """Return the balance-mixture second moment by one-dimensional quadrature."""
    _validate(horizon, threshold)
    if not math.isfinite(drift):
        raise ValueError("drift must be finite")
    if not math.isfinite(quadrature_tolerance) or quadrature_tolerance <= 0.0:
        raise ValueError("quadrature_tolerance must be finite and positive")
    log_normalizer = -0.5 * math.log(2.0 * math.pi * horizon)

    def integrand(value: float) -> float:
        log_density = log_normalizer - 0.5 * value * value / horizon
        log_ratio = float(
            gaussian_symmetric_mixture_log_q_over_p(
                value, drift=drift, horizon=horizon
            )
        )
        return math.exp(log_density - log_ratio)

    left = quad(
        integrand,
        -math.inf,
        -threshold,
        epsabs=quadrature_tolerance,
        epsrel=quadrature_tolerance,
        limit=300,
    )[0]
    right = quad(
        integrand,
        threshold,
        math.inf,
        epsabs=quadrature_tolerance,
        epsrel=quadrature_tolerance,
        limit=300,
    )[0]
    return float(left + right)
