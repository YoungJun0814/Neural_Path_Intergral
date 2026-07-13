from __future__ import annotations

import math

import pytest
import torch

from src.physics_engine import MarketSimulator


def test_log_euler_is_exact_for_constant_variance_gbm() -> None:
    torch.manual_seed(1234)
    paths = 20_000
    spot = 100.0
    variance = 0.04
    drift = 0.03
    maturity = 0.75
    simulator = MarketSimulator(
        mu=drift,
        kappa=1.0,
        theta=variance,
        xi=0.0,
        rho=0.0,
        device="cpu",
    )
    prices, _ = simulator.simulate(
        S0=spot,
        v0=variance,
        T=maturity,
        dt=0.07,
        num_paths=paths,
    )
    assert torch.all(prices > 0.0)
    expected_mean = spot * math.exp(drift * maturity)
    standard_error = float(prices[:, -1].std(unbiased=True) / math.sqrt(paths))
    assert float(prices[:, -1].mean()) == pytest.approx(expected_mean, abs=4.0 * standard_error)


def test_strong_negative_control_remains_strictly_positive() -> None:
    simulator = MarketSimulator(
        mu=0.0,
        kappa=1.5,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        device="cpu",
    )

    def negative_control(_time, spot, _variance, _average):
        return torch.full_like(spot, -8.0)

    prices, _variance, log_weight, _barrier, _average = simulator.simulate_controlled(
        S0=100.0,
        v0=0.04,
        T=1.0,
        dt=1.0 / 64.0,
        num_paths=2_000,
        control_fn=negative_control,
    )
    assert torch.all(prices > 0.0)
    assert torch.isfinite(log_weight).all()
