"""Cross-checks for the independent Heston Fourier reference implementation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.evaluation.heston_reference import (
    HestonReferenceParams,
    heston_call_price,
    heston_characteristic_function,
    heston_left_tail_quantile,
    heston_put_price,
    heston_terminal_cdf,
    heston_terminal_cdf_state_derivatives_vectorized,
    heston_terminal_cdf_vectorized,
)
from src.physics_engine import MarketSimulator


@pytest.fixture
def params() -> HestonReferenceParams:
    return HestonReferenceParams(
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.30,
        rho=-0.70,
        r=0.02,
        q=0.01,
    )


def test_characteristic_function_normalization_and_first_moment(
    params: HestonReferenceParams,
) -> None:
    spot = 100.0
    maturity = 1.25
    assert heston_characteristic_function(
        0.0, spot=spot, maturity=maturity, params=params
    ) == pytest.approx(1.0)
    first_moment = heston_characteristic_function(-1j, spot=spot, maturity=maturity, params=params)
    expected = spot * math.exp((params.r - params.q) * maturity)
    assert first_moment.real == pytest.approx(expected, rel=1e-11)
    assert abs(first_moment.imag) < 1e-11


def test_call_monotonicity_and_put_call_parity(params: HestonReferenceParams) -> None:
    spot = 100.0
    maturity = 1.0
    calls = [
        heston_call_price(spot=spot, strike=strike, maturity=maturity, params=params)
        for strike in (80.0, 100.0, 120.0)
    ]
    assert calls[0] > calls[1] > calls[2] > 0.0

    strike = 105.0
    call = heston_call_price(spot=spot, strike=strike, maturity=maturity, params=params)
    put = heston_put_price(spot=spot, strike=strike, maturity=maturity, params=params)
    parity_rhs = spot * math.exp(-params.q * maturity) - strike * math.exp(-params.r * maturity)
    assert call - put == pytest.approx(parity_rhs, abs=1e-10)


def test_terminal_cdf_and_quantile_inversion(params: HestonReferenceParams) -> None:
    spot = 100.0
    maturity = 1.0
    lower = heston_terminal_cdf(terminal_spot=75.0, spot=spot, maturity=maturity, params=params)
    upper = heston_terminal_cdf(terminal_spot=100.0, spot=spot, maturity=maturity, params=params)
    assert 0.0 < lower < upper < 1.0

    target = 0.01
    quantile = heston_left_tail_quantile(target, spot=spot, maturity=maturity, params=params)
    recovered = heston_terminal_cdf(
        terminal_spot=quantile, spot=spot, maturity=maturity, params=params
    )
    assert recovered == pytest.approx(target, abs=2e-8)


def test_vectorized_terminal_cdf_matches_scalar_inversion(
    params: HestonReferenceParams,
) -> None:
    thresholds = np.array([70.0, 85.0, 100.0, 125.0])
    vectorized = heston_terminal_cdf_vectorized(
        thresholds,
        spot=100.0,
        maturity=0.7,
        params=params,
        integration_limit=180.0,
    )
    scalar = np.array(
        [
            heston_terminal_cdf(
                terminal_spot=float(threshold),
                spot=100.0,
                maturity=0.7,
                params=params,
                integration_limit=180.0,
            )
            for threshold in thresholds
        ]
    )

    assert np.all(np.diff(vectorized) > 0.0)
    assert np.allclose(vectorized, scalar, atol=2e-12, rtol=2e-12)


def test_vectorized_cdf_analytic_state_derivatives_match_finite_differences(
    params: HestonReferenceParams,
) -> None:
    thresholds = np.array([80.0, 95.0, 110.0])
    spot = 100.0
    maturity = 0.7
    derivatives = heston_terminal_cdf_state_derivatives_vectorized(
        thresholds,
        spot=spot,
        maturity=maturity,
        params=params,
        integration_limit=180.0,
    )
    log_spot_step = 1e-4
    cdf_spot_plus = heston_terminal_cdf_vectorized(
        thresholds,
        spot=spot * math.exp(log_spot_step),
        maturity=maturity,
        params=params,
        integration_limit=180.0,
    )
    cdf_spot_minus = heston_terminal_cdf_vectorized(
        thresholds,
        spot=spot * math.exp(-log_spot_step),
        maturity=maturity,
        params=params,
        integration_limit=180.0,
    )
    finite_log_spot = (cdf_spot_plus - cdf_spot_minus) / (2.0 * log_spot_step)

    variance_step = 1e-5
    cdf_variance_plus = heston_terminal_cdf_vectorized(
        thresholds,
        spot=spot,
        maturity=maturity,
        params=replace(params, v0=params.v0 + variance_step),
        integration_limit=180.0,
    )
    cdf_variance_minus = heston_terminal_cdf_vectorized(
        thresholds,
        spot=spot,
        maturity=maturity,
        params=replace(params, v0=params.v0 - variance_step),
        integration_limit=180.0,
    )
    finite_variance = (cdf_variance_plus - cdf_variance_minus) / (2.0 * variance_step)

    assert np.allclose(
        derivatives.d_cdf_d_log_spot, finite_log_spot, atol=2e-7, rtol=2e-7
    )
    assert np.allclose(
        derivatives.d_cdf_d_variance, finite_variance, atol=2e-6, rtol=2e-6
    )


def test_monte_carlo_matches_fourier_reference(params: HestonReferenceParams) -> None:
    """Full-truncation MC should agree within sampling plus discretization error."""
    torch.manual_seed(20260713)
    spot = 100.0
    strike = 100.0
    maturity = 0.5
    n_paths = 40_000

    simulator = MarketSimulator(
        mu=params.r - params.q,
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=params.rho,
        device="cpu",
    )
    paths, _ = simulator.simulate(
        S0=spot,
        v0=params.v0,
        T=maturity,
        dt=1.0 / 512.0,
        num_paths=n_paths,
    )
    discounted_payoff = math.exp(-params.r * maturity) * torch.clamp(paths[:, -1] - strike, min=0.0)
    mc_price = float(discounted_payoff.mean())
    standard_error = float(discounted_payoff.std(unbiased=True) / math.sqrt(n_paths))
    reference = heston_call_price(spot=spot, strike=strike, maturity=maturity, params=params)

    # Four MC standard errors plus a small Euler-discretization allowance.
    assert np.isfinite(reference)
    assert abs(mc_price - reference) < 4.0 * standard_error + 0.08


@pytest.mark.parametrize(
    ("field", "value"),
    [("v0", -0.1), ("kappa", 0.0), ("xi", 0.0), ("rho", 1.1)],
)
def test_invalid_parameters_rejected(field: str, value: float) -> None:
    kwargs = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)
    kwargs[field] = value
    invalid = HestonReferenceParams(**kwargs)
    with pytest.raises(ValueError):
        heston_call_price(spot=100.0, strike=100.0, maturity=1.0, params=invalid)
