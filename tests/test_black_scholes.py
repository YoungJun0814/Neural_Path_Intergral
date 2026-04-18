"""Analytic checks against Black–Scholes closed form."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from src.ml_models import BlackScholesModel


def bs_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


@pytest.mark.parametrize(
    "S0,K,T,r,sigma",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (100.0, 90.0, 0.5, 0.03, 0.25),
        (50.0, 55.0, 0.25, 0.01, 0.30),
        (100.0, 100.0, 2.0, 0.04, 0.10),
    ],
)
def test_bs_model_matches_closed_form(S0, K, T, r, sigma):
    bs = BlackScholesModel(sigma=sigma)
    price = bs.price(S0=S0, K=K, T=T, r=r)
    ref = bs_call(S0, K, T, r, sigma)
    assert math.isclose(price, ref, rel_tol=1e-6, abs_tol=1e-6)


def test_bs_put_call_parity():
    """C − P = S − K·exp(−rT). Use BS call and synth put via parity."""
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
    C = bs_call(S0, K, T, r, sigma)
    # Put via parity
    P = C - S0 + K * math.exp(-r * T)
    # P should equal BS put closed form
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    P_bs = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    assert math.isclose(P, P_bs, rel_tol=1e-6, abs_tol=1e-6)


def test_bs_intrinsic_at_expiry():
    bs = BlackScholesModel(sigma=0.3)
    assert bs.price(100.0, 90.0, 0.0, 0.05) == 10.0
    assert bs.price(100.0, 120.0, 0.0, 0.05) == 0.0


def test_bs_calibrate_recovers_sigma():
    """Given BS-generated prices, calibrate must recover the true sigma."""
    true_sigma = 0.23
    S0, T, r = 100.0, 1.0, 0.04
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    prices = np.array([bs_call(S0, K, T, r, true_sigma) for K in strikes])
    bs = BlackScholesModel(sigma=0.5)  # bad init
    recovered = bs.calibrate(prices, S0, strikes, T, r)
    assert math.isclose(recovered, true_sigma, abs_tol=1e-3)
