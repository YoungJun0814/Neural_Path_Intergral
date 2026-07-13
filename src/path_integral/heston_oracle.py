"""Deterministic soft-desirability and two-driver Heston control oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, roots_legendre

from src.evaluation.heston_reference import (
    HestonReferenceParams,
    heston_terminal_cdf_state_derivatives_vectorized,
    heston_terminal_cdf_vectorized,
)


@dataclass(frozen=True)
class HestonOracleNumerics:
    """Numerical contract for the deterministic Heston oracle.

    Gradients use second-order differences at two resolutions followed by
    Richardson extrapolation.  ``minimum_desirability`` is a fail-fast gate,
    not a floor silently inserted into the oracle value.
    """

    quadrature_order: int = 96
    integration_limit: float = 180.0
    maximum_integration_limit: float = 1440.0
    integration_limit_growth: float = 2.0
    cdf_epsabs: float = 1e-9
    cdf_epsrel: float = 1e-8
    log_spot_step: float = 0.004
    variance_relative_step: float = 0.02
    minimum_variance_step: float = 1e-4
    minimum_desirability: float = 1e-14

    def validate(self) -> None:
        if self.quadrature_order < 8:
            raise ValueError("quadrature_order must be at least 8")
        positive_values = (
            self.integration_limit,
            self.maximum_integration_limit,
            self.integration_limit_growth,
            self.cdf_epsabs,
            self.cdf_epsrel,
            self.log_spot_step,
            self.variance_relative_step,
            self.minimum_variance_step,
            self.minimum_desirability,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("all Heston oracle numerical tolerances must be finite and positive")
        if self.log_spot_step >= 0.2:
            raise ValueError("log_spot_step is too large for a local derivative")
        if self.maximum_integration_limit < self.integration_limit:
            raise ValueError("maximum_integration_limit must not be below integration_limit")
        if self.integration_limit_growth <= 1.0:
            raise ValueError("integration_limit_growth must be greater than one")
        if self.minimum_desirability >= 0.5:
            raise ValueError("minimum_desirability must be smaller than 0.5")


@dataclass(frozen=True)
class HestonLogDesirabilityGradient:
    """Analytic Fourier derivatives cross-checked by Richardson differences."""

    desirability: float
    log_desirability: float
    d_log_h_d_log_spot: float
    d_log_h_d_variance: float
    finite_difference_d_log_h_d_log_spot: float
    finite_difference_d_log_h_d_variance: float
    log_spot_step: float
    variance_step: float
    variance_scheme: str
    log_spot_error_estimate: float
    variance_error_estimate: float
    integration_limit_used: float


@dataclass(frozen=True)
class HestonOracleControl:
    """Independent-basis Föllmer/Doob drift and its gradient diagnostics."""

    control_1: float
    control_2: float
    gradient: HestonLogDesirabilityGradient


@lru_cache(maxsize=16)
def _uniform_gauss_legendre(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # ``numpy.leggauss`` calls a dense eigenvalue routine which can abort in
    # some Windows Torch/MKL environments.  SciPy's root generator avoids that
    # native-library conflict while producing the same quadrature rule.
    nodes, weights = roots_legendre(order)
    uniforms = np.asarray(0.5 * (nodes + 1.0), dtype=np.float64)
    uniform_weights = np.asarray(0.5 * weights, dtype=np.float64)
    return uniforms, uniform_weights


def _validate_state_and_target(
    *,
    spot: float,
    variance: float,
    remaining_time: float,
    barrier: float,
    temperature: float,
    params: HestonReferenceParams,
    numerics: HestonOracleNumerics,
) -> None:
    conditional_params = replace(params, v0=variance)
    conditional_params.validate()
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("variance must be finite and nonnegative")
    if not math.isfinite(remaining_time) or remaining_time < 0.0:
        raise ValueError("remaining_time must be finite and nonnegative")
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("barrier must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not all(math.isfinite(value) for value in (params.r, params.q)):
        raise ValueError("rate and dividend yield must be finite")
    numerics.validate()


def _logistic_mixture_grid(
    barrier: float,
    temperature: float,
    quadrature_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    uniforms, weights = _uniform_gauss_legendre(quadrature_order)
    logistic_nodes = np.log(uniforms) - np.log1p(-uniforms)
    thresholds = barrier * (1.0 + temperature * logistic_nodes)
    return thresholds, weights, thresholds > 0.0


def _next_integration_limit(current: float, numerics: HestonOracleNumerics) -> float:
    return min(
        numerics.maximum_integration_limit,
        current * numerics.integration_limit_growth,
    )


def heston_soft_left_tail_desirability(
    *,
    spot: float,
    variance: float,
    remaining_time: float,
    barrier: float,
    temperature: float,
    params: HestonReferenceParams,
    numerics: HestonOracleNumerics | None = None,
) -> float:
    r"""Return ``E[sigmoid((K-S_T)/(tau*K)) | S_t=spot,v_t=variance]``.

    If ``Y`` has the standard logistic distribution, integration by parts gives

    ``h = E_Y[F_{S_T|state}(K + tau*K*Y)]``.

    Gauss--Legendre nodes on a uniform variable and the vectorized Heston CDF
    evaluate this identity deterministically.  At zero remaining time the
    terminal sigmoid is returned directly.
    """
    resolved = numerics if numerics is not None else HestonOracleNumerics()
    _validate_state_and_target(
        spot=spot,
        variance=variance,
        remaining_time=remaining_time,
        barrier=barrier,
        temperature=temperature,
        params=params,
        numerics=resolved,
    )
    if remaining_time == 0.0:
        return float(expit((barrier - spot) / (temperature * barrier)))

    thresholds, weights, positive = _logistic_mixture_grid(
        barrier, temperature, resolved.quadrature_order
    )
    conditional_params = replace(params, v0=variance)
    integration_limit = resolved.integration_limit
    while True:
        cdf_values = np.zeros_like(thresholds)
        if np.any(positive):
            cdf_values[positive] = heston_terminal_cdf_vectorized(
                thresholds[positive],
                spot=spot,
                maturity=remaining_time,
                params=conditional_params,
                integration_limit=integration_limit,
                epsabs=resolved.cdf_epsabs,
                epsrel=resolved.cdf_epsrel,
            )
        if not np.any(np.diff(cdf_values) < -1e-7):
            break
        next_limit = _next_integration_limit(integration_limit, resolved)
        if next_limit <= integration_limit:
            raise RuntimeError(
                "vectorized Heston CDF remains nonmonotone at maximum_integration_limit"
            )
        integration_limit = next_limit
    desirability = float(np.dot(weights, cdf_values))
    if not math.isfinite(desirability) or not 0.0 <= desirability <= 1.0:
        raise FloatingPointError("soft Heston desirability is outside [0, 1]")
    return desirability


def _checked_log_desirability(
    *,
    spot: float,
    variance: float,
    remaining_time: float,
    barrier: float,
    temperature: float,
    params: HestonReferenceParams,
    numerics: HestonOracleNumerics,
) -> float:
    value = heston_soft_left_tail_desirability(
        spot=spot,
        variance=variance,
        remaining_time=remaining_time,
        barrier=barrier,
        temperature=temperature,
        params=params,
        numerics=numerics,
    )
    if value <= numerics.minimum_desirability:
        raise FloatingPointError(
            "soft desirability is below the configured reliability gate; "
            "increase numerical precision or use a less extreme soft target"
        )
    return math.log(value)


def _analytic_log_desirability_gradient(
    *,
    spot: float,
    variance: float,
    remaining_time: float,
    barrier: float,
    temperature: float,
    params: HestonReferenceParams,
    numerics: HestonOracleNumerics,
) -> tuple[float, float, float, float]:
    """Return ``(h, d_log_h/d_log_spot, d_log_h/d_variance)`` by Fourier calculus."""
    thresholds, weights, positive = _logistic_mixture_grid(
        barrier, temperature, numerics.quadrature_order
    )
    conditional_params = replace(params, v0=variance)
    integration_limit = numerics.integration_limit
    while True:
        cdf_values = np.zeros_like(thresholds)
        cdf_log_spot_derivatives = np.zeros_like(thresholds)
        cdf_variance_derivatives = np.zeros_like(thresholds)
        if np.any(positive):
            derivatives = heston_terminal_cdf_state_derivatives_vectorized(
                thresholds[positive],
                spot=spot,
                maturity=remaining_time,
                params=conditional_params,
                integration_limit=integration_limit,
                epsabs=numerics.cdf_epsabs,
                epsrel=numerics.cdf_epsrel,
            )
            cdf_values[positive] = derivatives.cdf
            cdf_log_spot_derivatives[positive] = derivatives.d_cdf_d_log_spot
            cdf_variance_derivatives[positive] = derivatives.d_cdf_d_variance
        if not np.any(np.diff(cdf_values) < -1e-7):
            break
        next_limit = _next_integration_limit(integration_limit, numerics)
        if next_limit <= integration_limit:
            raise RuntimeError(
                "vectorized Heston CDF remains nonmonotone at maximum_integration_limit"
            )
        integration_limit = next_limit
    desirability = float(np.dot(weights, cdf_values))
    if desirability <= numerics.minimum_desirability:
        raise FloatingPointError(
            "soft desirability is below the configured reliability gate; "
            "increase numerical precision or use a less extreme soft target"
        )
    derivative_log_spot = float(np.dot(weights, cdf_log_spot_derivatives)) / desirability
    derivative_variance = float(np.dot(weights, cdf_variance_derivatives)) / desirability
    if not math.isfinite(derivative_log_spot) or not math.isfinite(derivative_variance):
        raise FloatingPointError("analytic Heston log-desirability gradient is nonfinite")
    return desirability, derivative_log_spot, derivative_variance, integration_limit


def heston_log_desirability_gradient(
    *,
    spot: float,
    variance: float,
    remaining_time: float,
    barrier: float,
    temperature: float,
    params: HestonReferenceParams,
    numerics: HestonOracleNumerics | None = None,
) -> HestonLogDesirabilityGradient:
    r"""Return Richardson-verified derivatives of ``log h`` in ``(log S,v)``.

    Variance uses central differences away from zero and a second-order forward
    formula near the degenerate boundary.  The reported error estimates are
    analytic-Fourier versus Richardson discrepancies, not full certified
    Fourier/quadrature error bounds.
    """
    resolved = numerics if numerics is not None else HestonOracleNumerics()
    _validate_state_and_target(
        spot=spot,
        variance=variance,
        remaining_time=remaining_time,
        barrier=barrier,
        temperature=temperature,
        params=params,
        numerics=resolved,
    )
    if remaining_time <= 0.0:
        raise ValueError("remaining_time must be positive for an oracle gradient")

    def log_h(candidate_spot: float, candidate_variance: float) -> float:
        return _checked_log_desirability(
            spot=candidate_spot,
            variance=candidate_variance,
            remaining_time=remaining_time,
            barrier=barrier,
            temperature=temperature,
            params=params,
            numerics=resolved,
        )

    base_value, analytic_log_spot, analytic_variance, integration_limit_used = (
        _analytic_log_desirability_gradient(
            spot=spot,
            variance=variance,
            remaining_time=remaining_time,
            barrier=barrier,
            temperature=temperature,
            params=params,
            numerics=resolved,
        )
    )
    base_log_h = math.log(base_value)
    log_step = resolved.log_spot_step
    log_half_step = 0.5 * log_step
    log_plus = log_h(spot * math.exp(log_step), variance)
    log_minus = log_h(spot * math.exp(-log_step), variance)
    log_plus_half = log_h(spot * math.exp(log_half_step), variance)
    log_minus_half = log_h(spot * math.exp(-log_half_step), variance)
    derivative_log_step = (log_plus - log_minus) / (2.0 * log_step)
    derivative_log_half = (log_plus_half - log_minus_half) / log_step
    finite_difference_log_spot = (4.0 * derivative_log_half - derivative_log_step) / 3.0

    variance_step = max(
        resolved.minimum_variance_step,
        resolved.variance_relative_step * max(variance, params.theta),
    )
    variance_half_step = 0.5 * variance_step
    if variance >= variance_step:
        variance_scheme = "central"
        variance_plus = log_h(spot, variance + variance_step)
        variance_minus = log_h(spot, variance - variance_step)
        variance_plus_half = log_h(spot, variance + variance_half_step)
        variance_minus_half = log_h(spot, variance - variance_half_step)
        derivative_variance_step = (variance_plus - variance_minus) / (2.0 * variance_step)
        derivative_variance_half = (
            variance_plus_half - variance_minus_half
        ) / variance_step
    else:
        variance_scheme = "forward"
        variance_plus_half = log_h(spot, variance + variance_half_step)
        variance_plus = log_h(spot, variance + variance_step)
        variance_plus_two = log_h(spot, variance + 2.0 * variance_step)
        derivative_variance_step = (
            -3.0 * base_log_h + 4.0 * variance_plus - variance_plus_two
        ) / (2.0 * variance_step)
        derivative_variance_half = (
            -3.0 * base_log_h + 4.0 * variance_plus_half - variance_plus
        ) / variance_step
    finite_difference_variance = (
        4.0 * derivative_variance_half - derivative_variance_step
    ) / 3.0
    log_spot_error = abs(analytic_log_spot - finite_difference_log_spot)
    variance_error = abs(analytic_variance - finite_difference_variance)

    return HestonLogDesirabilityGradient(
        desirability=base_value,
        log_desirability=base_log_h,
        d_log_h_d_log_spot=analytic_log_spot,
        d_log_h_d_variance=analytic_variance,
        finite_difference_d_log_h_d_log_spot=finite_difference_log_spot,
        finite_difference_d_log_h_d_variance=finite_difference_variance,
        log_spot_step=log_step,
        variance_step=variance_step,
        variance_scheme=variance_scheme,
        log_spot_error_estimate=log_spot_error,
        variance_error_estimate=variance_error,
        integration_limit_used=integration_limit_used,
    )


def heston_soft_oracle_control(
    *,
    spot: float,
    variance: float,
    remaining_time: float,
    barrier: float,
    temperature: float,
    params: HestonReferenceParams,
    numerics: HestonOracleNumerics | None = None,
) -> HestonOracleControl:
    r"""Return the independent-basis soft Heston Föllmer drift.

    With ``x=log S`` the controls are

    ``u1 = sqrt(v) * (d_x log h + rho*xi*d_v log h)`` and
    ``u2 = sqrt(v) * xi*sqrt(1-rho^2)*d_v log h``.
    """
    gradient = heston_log_desirability_gradient(
        spot=spot,
        variance=variance,
        remaining_time=remaining_time,
        barrier=barrier,
        temperature=temperature,
        params=params,
        numerics=numerics,
    )
    root_variance = math.sqrt(variance)
    correlation_perp = math.sqrt(max(1.0 - params.rho * params.rho, 0.0))
    control_1 = root_variance * (
        gradient.d_log_h_d_log_spot
        + params.rho * params.xi * gradient.d_log_h_d_variance
    )
    control_2 = (
        root_variance
        * params.xi
        * correlation_perp
        * gradient.d_log_h_d_variance
    )
    if not math.isfinite(control_1) or not math.isfinite(control_2):
        raise FloatingPointError("Heston oracle control is nonfinite")
    return HestonOracleControl(control_1=control_1, control_2=control_2, gradient=gradient)
