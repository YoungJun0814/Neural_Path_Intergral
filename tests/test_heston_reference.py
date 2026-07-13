"""Cross-checks for the independent Heston Fourier reference implementation."""

from __future__ import annotations

import math

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
