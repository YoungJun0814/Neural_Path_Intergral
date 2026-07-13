"""Verification of the soft Heston desirability and two-driver oracle."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.special import expit

from src.evaluation.heston_reference import HestonReferenceParams, heston_terminal_cdf
from src.path_integral import (
    HestonOracleNumerics,
    heston_log_desirability_gradient,
    heston_soft_left_tail_desirability,
    heston_soft_oracle_control,
)
from src.physics_engine import MarketSimulator


@pytest.fixture()
def params() -> HestonReferenceParams:
    return HestonReferenceParams(
        v0=0.04,
        kappa=1.8,
        theta=0.04,
        xi=0.45,
        rho=-0.65,
        r=0.03,
        q=0.0,
    )


def _numerics(**overrides: float | int) -> HestonOracleNumerics:
    values: dict[str, float | int] = {
        "quadrature_order": 48,
        "integration_limit": 180.0,
        "cdf_epsabs": 1e-9,
        "cdf_epsrel": 1e-8,
        "log_spot_step": 0.004,
        "variance_relative_step": 0.02,
        "minimum_variance_step": 1e-4,
        "minimum_desirability": 1e-14,
    }
    values.update(overrides)
    return HestonOracleNumerics(**values)


def test_zero_maturity_desirability_is_terminal_sigmoid(
    params: HestonReferenceParams,
) -> None:
    spot = 103.0
    barrier = 85.0
    temperature = 0.06
    value = heston_soft_left_tail_desirability(
        spot=spot,
        variance=0.04,
        remaining_time=0.0,
        barrier=barrier,
        temperature=temperature,
        params=params,
    )

    assert value == pytest.approx(float(expit((barrier - spot) / (temperature * barrier))))


def test_logistic_mixture_quadrature_converges_and_approaches_hard_cdf(
    params: HestonReferenceParams,
) -> None:
    common = dict(
        spot=100.0,
        variance=0.04,
        remaining_time=0.5,
        barrier=85.0,
        params=params,
    )
    order_32 = heston_soft_left_tail_desirability(
        temperature=0.05, numerics=_numerics(quadrature_order=32), **common
    )
    order_64 = heston_soft_left_tail_desirability(
        temperature=0.05, numerics=_numerics(quadrature_order=64), **common
    )
    near_hard = heston_soft_left_tail_desirability(
        temperature=0.005, numerics=_numerics(quadrature_order=64), **common
    )
    hard = heston_terminal_cdf(
        terminal_spot=85.0,
        spot=100.0,
        maturity=0.5,
        params=params,
        integration_limit=180.0,
    )

    assert order_32 == pytest.approx(order_64, abs=2e-5)
    assert near_hard == pytest.approx(hard, abs=4e-4)
    assert 0.0 < hard < order_64 < 1.0


def test_soft_desirability_matches_independent_heston_monte_carlo(
    params: HestonReferenceParams,
) -> None:
    oracle = heston_soft_left_tail_desirability(
        spot=100.0,
        variance=params.v0,
        remaining_time=0.5,
        barrier=85.0,
        temperature=0.05,
        params=params,
    )
    paths = 30_000
    torch.manual_seed(20260713)
    simulator = MarketSimulator(
        mu=params.r - params.q,
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=params.rho,
        device="cpu",
    )
    spot_paths, _variance_paths = simulator.simulate(
        S0=100.0,
        v0=params.v0,
        T=0.5,
        dt=1.0 / 512.0,
        num_paths=paths,
    )
    soft_payoff = torch.sigmoid((85.0 - spot_paths[:, -1]) / (0.05 * 85.0)).double()
    monte_carlo = float(soft_payoff.mean())
    standard_error = float(soft_payoff.std(unbiased=True)) / math.sqrt(paths)

    assert abs(monte_carlo - oracle) < 4.0 * standard_error + 0.002


def test_richardson_gradients_converge_and_have_left_tail_signs(
    params: HestonReferenceParams,
) -> None:
    common = dict(
        spot=100.0,
        variance=0.04,
        remaining_time=0.5,
        barrier=85.0,
        temperature=0.05,
        params=params,
    )
    coarse = heston_log_desirability_gradient(
        numerics=_numerics(log_spot_step=0.008, variance_relative_step=0.04), **common
    )
    fine = heston_log_desirability_gradient(
        numerics=_numerics(log_spot_step=0.004, variance_relative_step=0.02), **common
    )

    assert fine.variance_scheme == "central"
    assert fine.d_log_h_d_log_spot < 0.0
    assert fine.d_log_h_d_variance > 0.0
    assert coarse.d_log_h_d_log_spot == pytest.approx(
        fine.d_log_h_d_log_spot, abs=2e-4
    )
    assert coarse.d_log_h_d_variance == pytest.approx(
        fine.d_log_h_d_variance, abs=2e-3
    )
    assert fine.log_spot_error_estimate < 1e-4
    assert fine.variance_error_estimate < 1e-3
    assert fine.finite_difference_d_log_h_d_log_spot == pytest.approx(
        fine.d_log_h_d_log_spot, abs=1e-7
    )
    assert fine.finite_difference_d_log_h_d_variance == pytest.approx(
        fine.d_log_h_d_variance, abs=1e-7
    )


def test_oracle_gradient_converges_in_logistic_quadrature_order(
    params: HestonReferenceParams,
) -> None:
    common = dict(
        spot=100.0,
        variance=0.04,
        remaining_time=0.5,
        barrier=85.0,
        temperature=0.05,
        params=params,
    )
    order_96 = heston_log_desirability_gradient(
        numerics=_numerics(quadrature_order=96), **common
    )
    order_128 = heston_log_desirability_gradient(
        numerics=_numerics(quadrature_order=128), **common
    )

    assert order_96.d_log_h_d_log_spot == pytest.approx(
        order_128.d_log_h_d_log_spot, abs=3e-5
    )
    assert order_96.d_log_h_d_variance == pytest.approx(
        order_128.d_log_h_d_variance, abs=7e-5
    )


def test_two_driver_oracle_matches_diffusion_transpose_formula(
    params: HestonReferenceParams,
) -> None:
    variance = 0.04
    oracle = heston_soft_oracle_control(
        spot=100.0,
        variance=variance,
        remaining_time=0.5,
        barrier=85.0,
        temperature=0.05,
        params=params,
        numerics=_numerics(),
    )
    gradient = oracle.gradient
    expected_control_1 = math.sqrt(variance) * (
        gradient.d_log_h_d_log_spot
        + params.rho * params.xi * gradient.d_log_h_d_variance
    )
    expected_control_2 = (
        math.sqrt(variance)
        * params.xi
        * math.sqrt(1.0 - params.rho**2)
        * gradient.d_log_h_d_variance
    )

    assert oracle.control_1 == pytest.approx(expected_control_1, abs=1e-13)
    assert oracle.control_2 == pytest.approx(expected_control_2, abs=1e-13)
    assert oracle.control_1 < 0.0
    assert oracle.control_2 > 0.0


def test_variance_boundary_uses_finite_one_sided_gradient(
    params: HestonReferenceParams,
) -> None:
    gradient = heston_log_desirability_gradient(
        spot=100.0,
        variance=1e-5,
        remaining_time=0.25,
        barrier=90.0,
        temperature=0.08,
        params=params,
        numerics=_numerics(
            minimum_variance_step=2e-4,
            integration_limit=500.0,
            cdf_epsabs=1e-10,
            cdf_epsrel=1e-9,
        ),
    )

    assert gradient.variance_scheme == "forward"
    assert math.isfinite(gradient.d_log_h_d_log_spot)
    assert math.isfinite(gradient.d_log_h_d_variance)
    assert gradient.desirability > 0.0


def test_near_maturity_oracle_adapts_fourier_cutoff(
    params: HestonReferenceParams,
) -> None:
    oracle = heston_soft_oracle_control(
        spot=100.0,
        variance=0.04,
        remaining_time=0.01,
        barrier=85.0,
        temperature=0.05,
        params=params,
        numerics=_numerics(
            quadrature_order=64,
            integration_limit=180.0,
            maximum_integration_limit=1440.0,
        ),
    )

    assert oracle.gradient.integration_limit_used > 180.0
    assert math.isfinite(oracle.control_1)
    assert math.isfinite(oracle.control_2)


def test_oracle_rejects_terminal_gradient_and_unreliable_probability(
    params: HestonReferenceParams,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        heston_log_desirability_gradient(
            spot=100.0,
            variance=0.04,
            remaining_time=0.0,
            barrier=85.0,
            temperature=0.05,
            params=params,
        )

    with pytest.raises(FloatingPointError, match="reliability gate"):
        heston_log_desirability_gradient(
            spot=100.0,
            variance=0.04,
            remaining_time=0.5,
            barrier=85.0,
            temperature=0.05,
            params=params,
            numerics=_numerics(minimum_desirability=0.2),
        )
