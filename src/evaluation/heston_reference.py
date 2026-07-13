"""Independent semi-analytic reference prices for the risk-neutral Heston model.

The implementation uses the stable ("little Heston trap") representation of
the characteristic function and one-dimensional quadrature.  It is deliberately
kept separate from the Monte Carlo simulator so the two implementations can be
used as independent numerical cross-checks.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq


@dataclass(frozen=True)
class HestonReferenceParams:
    """Risk-neutral Heston parameters.

    The variance dynamics are
    ``dv = kappa * (theta - v) dt + xi * sqrt(v) dW_v`` and the spot drift is
    ``(r - q) S dt``.  The Feller condition is not required by the formula.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    r: float = 0.0
    q: float = 0.0

    def validate(self) -> None:
        if self.v0 < 0.0 or self.theta < 0.0:
            raise ValueError("v0 and theta must be nonnegative")
        if self.kappa <= 0.0 or self.xi <= 0.0:
            raise ValueError("kappa and xi must be positive")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")


@dataclass(frozen=True)
class HestonTerminalCDFStateDerivatives:
    """Vectorized CDF and derivatives with respect to current ``(log S, v)``."""

    cdf: NDArray[np.float64]
    d_cdf_d_log_spot: NDArray[np.float64]
    d_cdf_d_variance: NDArray[np.float64]


def _heston_characteristic_components(
    u: complex,
    *,
    spot: float,
    maturity: float,
    params: HestonReferenceParams,
) -> tuple[complex, complex]:
    """Return the characteristic function and its log-linear ``v0`` coefficient."""
    params.validate()
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if maturity < 0.0:
        raise ValueError("maturity must be nonnegative")
    if maturity == 0.0:
        return cmath.exp(1j * u * math.log(spot)), 0.0j

    iu = 1j * u
    a = params.kappa - params.rho * params.xi * iu
    d = cmath.sqrt(a * a + params.xi**2 * (u * u + iu))
    if d.real < 0.0:
        d = -d

    g = (a - d) / (a + d)
    exp_minus_dt = cmath.exp(-d * maturity)
    log_ratio = cmath.log((1.0 - g * exp_minus_dt) / (1.0 - g))
    c = iu * (math.log(spot) + (params.r - params.q) * maturity)
    c += (params.kappa * params.theta / params.xi**2) * ((a - d) * maturity - 2.0 * log_ratio)
    variance_coefficient = ((a - d) / params.xi**2) * (
        (1.0 - exp_minus_dt) / (1.0 - g * exp_minus_dt)
    )
    characteristic = cmath.exp(c + variance_coefficient * params.v0)
    return characteristic, variance_coefficient


def heston_characteristic_function(
    u: complex,
    *,
    spot: float,
    maturity: float,
    params: HestonReferenceParams,
) -> complex:
    """Return ``E[exp(i u log(S_T))]`` under the risk-neutral measure."""
    characteristic, _variance_coefficient = _heston_characteristic_components(
        u, spot=spot, maturity=maturity, params=params
    )
    return characteristic


def heston_call_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    params: HestonReferenceParams,
    integration_limit: float = 150.0,
    epsabs: float = 1e-9,
    epsrel: float = 1e-8,
) -> float:
    """Price a European call using Heston's ``P1/P2`` Fourier integrals."""
    params.validate()
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0.0:
        raise ValueError("maturity must be nonnegative")
    if integration_limit <= 0.0:
        raise ValueError("integration_limit must be positive")
    if maturity == 0.0:
        return max(spot - strike, 0.0)

    p1 = _heston_exercise_probability(
        probability=1,
        spot=spot,
        strike=strike,
        maturity=maturity,
        params=params,
        integration_limit=integration_limit,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    p2 = _heston_exercise_probability(
        probability=2,
        spot=spot,
        strike=strike,
        maturity=maturity,
        params=params,
        integration_limit=integration_limit,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    price = (
        spot * math.exp(-params.q * maturity) * p1 - strike * math.exp(-params.r * maturity) * p2
    )
    # Suppress only integration-scale negative noise; a materially negative
    # value indicates invalid parameters or insufficient quadrature accuracy.
    if price < -1e-7:
        raise RuntimeError(f"Heston quadrature produced a negative call price: {price}")
    return max(float(price), 0.0)


def _heston_exercise_probability(
    *,
    probability: int,
    spot: float,
    strike: float,
    maturity: float,
    params: HestonReferenceParams,
    integration_limit: float,
    epsabs: float,
    epsrel: float,
) -> float:
    if probability not in (1, 2):
        raise ValueError("probability index must be 1 or 2")
    log_strike = math.log(strike)
    phi_minus_i = heston_characteristic_function(-1j, spot=spot, maturity=maturity, params=params)

    def integrand(u: float) -> float:
        u_safe = max(u, 1e-12)
        argument = complex(u_safe, -1.0) if probability == 1 else complex(u_safe)
        phi = heston_characteristic_function(argument, spot=spot, maturity=maturity, params=params)
        denominator = 1j * u_safe * (phi_minus_i if probability == 1 else 1.0)
        return (cmath.exp(-1j * u_safe * log_strike) * phi / denominator).real

    integral, _ = quad(
        integrand,
        0.0,
        integration_limit,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=500,
    )
    return float(0.5 + integral / math.pi)


def heston_terminal_cdf(
    *,
    terminal_spot: float,
    spot: float,
    maturity: float,
    params: HestonReferenceParams,
    integration_limit: float = 200.0,
    epsabs: float = 1e-10,
    epsrel: float = 1e-9,
) -> float:
    """Return ``P(S_T <= terminal_spot)`` by Gil-Pelaez inversion."""
    params.validate()
    if terminal_spot <= 0.0 or spot <= 0.0:
        raise ValueError("terminal_spot and spot must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    upper_tail = _heston_exercise_probability(
        probability=2,
        spot=spot,
        strike=terminal_spot,
        maturity=maturity,
        params=params,
        integration_limit=integration_limit,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return min(max(1.0 - upper_tail, 0.0), 1.0)


def heston_terminal_cdf_vectorized(
    terminal_spots: ArrayLike,
    *,
    spot: float,
    maturity: float,
    params: HestonReferenceParams,
    integration_limit: float = 200.0,
    epsabs: float = 1e-10,
    epsrel: float = 1e-9,
) -> NDArray[np.float64]:
    """Return Heston terminal CDF values from one vector-valued quadrature.

    This is algebraically identical to repeated Gil--Pelaez inversions in
    :func:`heston_terminal_cdf`, but evaluates the characteristic function only
    once per integration node for all thresholds.  It is used by the soft
    path-integral oracle, which needs a deterministic CDF mixture at many
    nearby terminal levels.
    """
    params.validate()
    thresholds = np.asarray(terminal_spots, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("terminal_spots must be a nonempty one-dimensional array")
    if not np.isfinite(thresholds).all() or np.any(thresholds <= 0.0):
        raise ValueError("all terminal_spots must be finite and positive")
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    if not math.isfinite(integration_limit) or integration_limit <= 0.0:
        raise ValueError("integration_limit must be finite and positive")
    if not math.isfinite(epsabs) or not math.isfinite(epsrel) or epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("epsabs and epsrel must be finite and positive")

    log_thresholds = np.log(thresholds)

    def integrand(u: float) -> NDArray[np.float64]:
        u_safe = max(float(u), 1e-12)
        characteristic = heston_characteristic_function(
            complex(u_safe), spot=spot, maturity=maturity, params=params
        )
        oscillatory = np.exp(-1j * u_safe * log_thresholds) * characteristic
        return np.asarray(np.imag(oscillatory) / u_safe, dtype=np.float64)

    integral, _error = quad_vec(
        integrand,
        0.0,
        integration_limit,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=500,
    )
    cdf = 0.5 - np.asarray(integral, dtype=np.float64) / math.pi
    return np.asarray(np.clip(cdf, 0.0, 1.0), dtype=np.float64)


def heston_terminal_cdf_state_derivatives_vectorized(
    terminal_spots: ArrayLike,
    *,
    spot: float,
    maturity: float,
    params: HestonReferenceParams,
    integration_limit: float = 200.0,
    epsabs: float = 1e-10,
    epsrel: float = 1e-9,
) -> HestonTerminalCDFStateDerivatives:
    r"""Return CDF values and analytic Fourier state derivatives.

    For real Fourier frequency ``u``, ``partial_x phi = i*u*phi`` and
    ``partial_v phi = D(u)*phi``, where ``x=log(spot)`` and ``D`` is the
    affine Heston variance coefficient.  Differentiating the Gil--Pelaez
    integral gives a gradient independent of the finite-difference oracle
    check.
    """
    params.validate()
    thresholds = np.asarray(terminal_spots, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("terminal_spots must be a nonempty one-dimensional array")
    if not np.isfinite(thresholds).all() or np.any(thresholds <= 0.0):
        raise ValueError("all terminal_spots must be finite and positive")
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    if not math.isfinite(integration_limit) or integration_limit <= 0.0:
        raise ValueError("integration_limit must be finite and positive")
    if not math.isfinite(epsabs) or not math.isfinite(epsrel) or epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("epsabs and epsrel must be finite and positive")

    log_thresholds = np.log(thresholds)
    count = thresholds.size

    def integrand(u: float) -> NDArray[np.float64]:
        u_safe = max(float(u), 1e-12)
        characteristic, variance_coefficient = _heston_characteristic_components(
            complex(u_safe), spot=spot, maturity=maturity, params=params
        )
        oscillatory = np.exp(-1j * u_safe * log_thresholds) * characteristic
        cdf_integrand = np.imag(oscillatory) / u_safe
        log_spot_integrand = np.imag(1j * u_safe * oscillatory) / u_safe
        variance_integrand = np.imag(variance_coefficient * oscillatory) / u_safe
        return np.asarray(
            np.concatenate((cdf_integrand, log_spot_integrand, variance_integrand)),
            dtype=np.float64,
        )

    integral, _error = quad_vec(
        integrand,
        0.0,
        integration_limit,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=500,
    )
    integral_array = np.asarray(integral, dtype=np.float64)
    raw_cdf = 0.5 - integral_array[:count] / math.pi
    derivative_log_spot = -integral_array[count : 2 * count] / math.pi
    derivative_variance = -integral_array[2 * count :] / math.pi
    clipped = (raw_cdf <= 0.0) | (raw_cdf >= 1.0)
    derivative_log_spot = np.where(clipped, 0.0, derivative_log_spot)
    derivative_variance = np.where(clipped, 0.0, derivative_variance)
    return HestonTerminalCDFStateDerivatives(
        cdf=np.asarray(np.clip(raw_cdf, 0.0, 1.0), dtype=np.float64),
        d_cdf_d_log_spot=np.asarray(derivative_log_spot, dtype=np.float64),
        d_cdf_d_variance=np.asarray(derivative_variance, dtype=np.float64),
    )


def heston_left_tail_quantile(
    probability: float,
    *,
    spot: float,
    maturity: float,
    params: HestonReferenceParams,
    lower_spot: float | None = None,
    upper_spot: float | None = None,
    integration_limit: float = 200.0,
) -> float:
    """Invert the terminal Heston CDF for a left-tail probability."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    lower = lower_spot if lower_spot is not None else spot * math.exp(-10.0)
    upper = upper_spot if upper_spot is not None else spot * math.exp(5.0)
    if not 0.0 < lower < upper:
        raise ValueError("quantile bounds must satisfy 0 < lower_spot < upper_spot")

    def root(terminal_spot: float) -> float:
        return (
            heston_terminal_cdf(
                terminal_spot=terminal_spot,
                spot=spot,
                maturity=maturity,
                params=params,
                integration_limit=integration_limit,
            )
            - probability
        )

    if root(lower) >= 0.0 or root(upper) <= 0.0:
        raise ValueError("quantile bounds do not bracket the requested probability")
    return float(brentq(root, lower, upper, xtol=1e-9, rtol=1e-11, maxiter=100))


def heston_put_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    params: HestonReferenceParams,
    **quadrature_options: float,
) -> float:
    """Price a European put by put-call parity."""
    call = heston_call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        params=params,
        **quadrature_options,
    )
    return float(
        call - spot * math.exp(-params.q * maturity) + strike * math.exp(-params.r * maturity)
    )
