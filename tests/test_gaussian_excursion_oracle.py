"""M0 tests for the finite-grid Gaussian path-functional oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.path_integral import (
    GaussianExcursionSpec,
    build_gaussian_excursion_oracle,
    simulate_gaussian_excursion,
)
from src.path_integral.gaussian_excursion_oracle import _normal_transition_matrices


def _spec() -> GaussianExcursionSpec:
    return GaussianExcursionSpec(
        steps=12,
        maturity=1.0,
        hit_barrier=-1.3,
        stress_level=-0.6,
        minimum_occupation=0.25,
    )


def test_transition_probability_and_brownian_moment_normalize() -> None:
    grid = np.linspace(-4.0, 4.0, 101)
    probability, moment = _normal_transition_matrices(grid, step_dt=0.1)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=2e-15)
    assert np.allclose(moment.sum(axis=1), 0.0, atol=2e-15)
    assert np.all(probability >= 0.0)


def test_event_aligned_dynamic_program_converges_under_grid_refinement() -> None:
    coarse = build_gaussian_excursion_oracle(
        _spec(), state_minimum=-5.0, state_maximum=5.0, state_points=101
    )
    medium = build_gaussian_excursion_oracle(
        _spec(), state_minimum=-5.0, state_maximum=5.0, state_points=201
    )
    fine = build_gaussian_excursion_oracle(
        _spec(), state_minimum=-5.0, state_maximum=5.0, state_points=401
    )
    assert abs(fine.reference_probability - medium.reference_probability) < abs(
        medium.reference_probability - coarse.reference_probability
    )
    assert abs(fine.reference_probability - medium.reference_probability) < 3e-4
    assert np.max(np.abs(fine.projected_control)) <= fine.control_bound


def test_constant_proposal_uses_exact_discrete_gaussian_likelihood() -> None:
    control = -0.4
    sample = simulate_gaussian_excursion(
        _spec(), num_paths=20_000, seed=2701, constant_control=control
    )
    expected_log_likelihood = (
        -control * sample.terminal_state + 0.5 * control * control * _spec().maturity
    )
    assert np.allclose(sample.likelihood, np.exp(expected_log_likelihood), atol=2e-14)
    assert float(np.mean(sample.likelihood)) == pytest.approx(1.0, abs=0.015)


def test_projected_oracle_is_unbiased_with_lower_second_moment() -> None:
    oracle = build_gaussian_excursion_oracle(
        _spec(), state_minimum=-5.0, state_maximum=5.0, state_points=301
    )
    natural = simulate_gaussian_excursion(_spec(), num_paths=80_000, seed=2702)
    projected = simulate_gaussian_excursion(
        _spec(), num_paths=80_000, seed=2703, oracle=oracle
    )
    combined_error = math.sqrt(natural.standard_error**2 + projected.standard_error**2)
    assert abs(projected.estimate - natural.estimate) < 4.0 * combined_error + 5e-4
    assert abs(oracle.reference_probability - natural.estimate) < 4.0 * natural.standard_error + 1e-3
    assert natural.second_moment / projected.second_moment > 3.0
