"""Closed-form and finite-dimensional eta-zero oracle checks."""

from __future__ import annotations

import pytest

from src.path_integral import (
    black_scholes_continuous_lower_barrier_probability,
    black_scholes_discrete_lower_barrier_probability,
    black_scholes_left_digital_probability,
)


def test_one_monitoring_time_barrier_equals_terminal_digital() -> None:
    digital = black_scholes_left_digital_probability(
        spot=100.0, level=90.0, volatility=0.2, maturity=0.25
    )
    barrier = black_scholes_discrete_lower_barrier_probability(
        spot=100.0,
        barrier=90.0,
        volatility=0.2,
        maturity=0.25,
        steps=1,
    )
    assert barrier.probability == pytest.approx(digital, abs=2e-12)


def test_discrete_monitoring_is_bounded_by_continuous_monitoring() -> None:
    discrete = black_scholes_discrete_lower_barrier_probability(
        spot=100.0,
        barrier=90.0,
        volatility=0.2,
        maturity=0.25,
        steps=4,
        state_points=1201,
    )
    continuous = black_scholes_continuous_lower_barrier_probability(
        spot=100.0, barrier=90.0, volatility=0.2, maturity=0.25
    )
    assert 0.0 < discrete.probability <= continuous + 2e-6 < 1.0


def test_discrete_quadrature_is_stable_under_spatial_refinement() -> None:
    coarse = black_scholes_discrete_lower_barrier_probability(
        spot=100.0,
        barrier=90.0,
        volatility=0.2,
        maturity=0.25,
        steps=4,
        state_points=601,
    )
    fine = black_scholes_discrete_lower_barrier_probability(
        spot=100.0,
        barrier=90.0,
        volatility=0.2,
        maturity=0.25,
        steps=4,
        state_points=1201,
    )
    assert abs(coarse.probability - fine.probability) < 3e-5


def test_initially_hit_barrier_has_probability_one() -> None:
    assert (
        black_scholes_continuous_lower_barrier_probability(
            spot=100.0, barrier=100.0, volatility=0.2, maturity=1.0
        )
        == 1.0
    )
    assert (
        black_scholes_discrete_lower_barrier_probability(
            spot=100.0, barrier=101.0, volatility=0.2, maturity=1.0, steps=4
        ).probability
        == 1.0
    )
