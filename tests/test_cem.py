"""Validation tests for trajectory-likelihood cross-entropy updates."""

from __future__ import annotations

import math

import pytest
import torch

from src.physics_engine import MarketSimulator
from src.training.cem import (
    CEMBatch,
    HestonTerminalLossSampler,
    fit_constant_control_cem,
)


class GaussianLeftTailSampler:
    """Exactly tractable N(0, 1) left-tail benchmark."""

    def __call__(self, control: float, num_paths: int) -> CEMBatch:
        w_q = torch.randn(num_paths, dtype=torch.float64)
        x_under_base_coordinates = w_q + control
        log_weight = -control * w_q - 0.5 * control**2
        return CEMBatch(
            score=-x_under_base_coordinates,
            log_base_over_proposal=log_weight,
            base_sufficient_statistic=x_under_base_coordinates,
        )


def test_gaussian_cem_recovers_left_tail_and_reduces_variance() -> None:
    torch.manual_seed(20260713)
    result = fit_constant_control_cem(
        GaussianLeftTailSampler(),
        initial_control=-1.0,
        target_score=3.0,
        num_paths=30_000,
        max_iterations=7,
        elite_quantile=0.90,
        smoothing=0.8,
    )
    assert result.converged
    assert -4.0 < result.control < -2.5
    assert result.history[-1].target_event_fraction_under_proposal > 0.25

    n_eval = 100_000
    base_x = torch.randn(n_eval, dtype=torch.float64)
    base_estimator = (base_x <= -3.0).double()

    control = result.control
    w_q = torch.randn(n_eval, dtype=torch.float64)
    x_q = w_q + control
    likelihood = torch.exp(-control * w_q - 0.5 * control**2)
    is_estimator = (x_q <= -3.0).double() * likelihood

    analytic_probability = 0.5 * math.erfc(3.0 / math.sqrt(2.0))
    assert float(is_estimator.mean()) == pytest.approx(analytic_probability, rel=0.04)
    assert float(is_estimator.var()) < 0.05 * float(base_estimator.var())


def test_heston_adapter_keeps_trajectory_likelihood_and_moves_downside() -> None:
    torch.manual_seed(8912)
    simulator = MarketSimulator(
        mu=0.0,
        kappa=1.5,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        device="cpu",
    )
    sampler = HestonTerminalLossSampler(
        simulator,
        spot=100.0,
        variance=0.04,
        maturity=0.5,
        dt=1.0 / 128.0,
    )
    base_batch = sampler(-0.25, 12_000)
    result = fit_constant_control_cem(
        sampler,
        initial_control=-0.75,
        target_score=-80.0,
        num_paths=12_000,
        max_iterations=6,
        elite_quantile=0.85,
        smoothing=0.7,
    )
    fitted_batch = sampler(result.control, 12_000)

    assert result.control < -0.75
    assert float((fitted_batch.score >= -80.0).float().mean()) > float(
        (base_batch.score >= -80.0).float().mean()
    )
    # A likelihood ratio must integrate to one; this catches a sign error in
    # either the simulator or CEM adapter without relying on the event itself.
    mean_likelihood = float(torch.exp(fitted_batch.log_base_over_proposal).mean())
    assert mean_likelihood == pytest.approx(1.0, abs=0.08)


def test_cem_rejects_zero_initial_control() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        fit_constant_control_cem(
            GaussianLeftTailSampler(),
            initial_control=0.0,
            target_score=3.0,
            num_paths=100,
        )
