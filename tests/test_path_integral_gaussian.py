"""Analytic Gaussian gates for path-integral control and PICE."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    brownian_log_likelihood,
    fit_constant_pice,
    gaussian_exponential_tilt_log_normalizer,
    gaussian_exponential_tilt_optimal_control,
    gaussian_exponential_tilt_pi_gap,
    gaussian_exponential_tilt_pi_objective,
    gaussian_exponential_tilt_relative_variance,
    gaussian_left_tail_doob_drift,
    gaussian_left_tail_probability,
    reconstruct_candidate_increments,
    tilted_divergence_diagnostics,
)


def test_exponential_tilt_pi_oracle_and_zero_variance_control() -> None:
    tilt = -1.3
    horizon = 0.7
    optimal = gaussian_exponential_tilt_optimal_control(tilt, horizon)
    log_normalizer = gaussian_exponential_tilt_log_normalizer(tilt, horizon)

    assert optimal == tilt
    assert gaussian_exponential_tilt_pi_objective(tilt, optimal, horizon) == pytest.approx(
        -log_normalizer
    )
    assert gaussian_exponential_tilt_pi_gap(tilt, optimal, horizon) == 0.0
    assert gaussian_exponential_tilt_relative_variance(tilt, optimal, horizon) == 0.0

    displaced = 0.4
    expected_gap = 0.5 * (displaced - tilt) ** 2 * horizon
    assert gaussian_exponential_tilt_pi_gap(tilt, displaced, horizon) == pytest.approx(
        expected_gap
    )


def test_relative_variance_chi_square_and_renyi_identity() -> None:
    log_contributions = torch.tensor([-2.0, -0.3, 0.1, 1.2, 2.0], dtype=torch.float64)
    diagnostics = tilted_divergence_diagnostics(log_contributions)
    contributions = torch.exp(log_contributions)
    empirical_relative_variance = contributions.square().mean() / contributions.mean() ** 2 - 1.0

    assert diagnostics.relative_variance == pytest.approx(empirical_relative_variance)
    assert diagnostics.chi_square == diagnostics.relative_variance
    assert torch.expm1(diagnostics.renyi2) == pytest.approx(empirical_relative_variance)
    assert diagnostics.contribution_ess_fraction == pytest.approx(
        1.0 / (1.0 + empirical_relative_variance)
    )


def test_optimal_exponential_tilt_has_constant_path_contribution() -> None:
    torch.manual_seed(71)
    tilt = -0.9
    horizon = 1.4
    proposal_noise = torch.randn(4096, dtype=torch.float64) * math.sqrt(horizon)
    target_terminal = proposal_noise + tilt * horizon
    log_likelihood = -tilt * proposal_noise - 0.5 * tilt * tilt * horizon
    log_contribution = tilt * target_terminal + log_likelihood
    diagnostics = tilted_divergence_diagnostics(log_contribution)

    assert torch.max(log_contribution) - torch.min(log_contribution) < 2e-15
    assert diagnostics.log_normalizer == pytest.approx(
        gaussian_exponential_tilt_log_normalizer(tilt, horizon), abs=2e-15
    )
    assert diagnostics.relative_variance == pytest.approx(0.0, abs=2e-15)


def test_nonoptimal_control_matches_analytic_relative_variance() -> None:
    torch.manual_seed(23)
    paths = 250_000
    tilt = -0.75
    behavior = -0.35
    horizon = 0.8
    proposal_noise = torch.randn(paths, dtype=torch.float64) * math.sqrt(horizon)
    target_terminal = proposal_noise + behavior * horizon
    log_likelihood = -behavior * proposal_noise - 0.5 * behavior * behavior * horizon
    log_contribution = tilt * target_terminal + log_likelihood
    diagnostics = tilted_divergence_diagnostics(log_contribution)
    expected = gaussian_exponential_tilt_relative_variance(tilt, behavior, horizon)

    assert torch.exp(log_likelihood).mean() == pytest.approx(1.0, abs=0.004)
    assert diagnostics.relative_variance == pytest.approx(expected, abs=0.003)


def test_pice_recovers_constant_exponential_tilt_off_policy() -> None:
    torch.manual_seed(31)
    paths = 250_000
    tilt = -0.8
    behavior = -0.3
    horizon = 0.7
    behavior_noise = torch.randn(paths, dtype=torch.float64) * math.sqrt(horizon)
    target_terminal = behavior_noise + behavior * horizon
    behavior_log_likelihood = (
        -behavior * behavior_noise - 0.5 * behavior * behavior * horizon
    )
    log_target_over_behavior = tilt * target_terminal + behavior_log_likelihood

    fit = fit_constant_pice(target_terminal, log_target_over_behavior, horizon)

    assert float(fit.control) == pytest.approx(tilt, abs=0.01)
    assert 0.0 < float(fit.effective_sample_fraction) <= 1.0


def test_off_policy_candidate_residual_matches_target_coordinate_density() -> None:
    target_increments = torch.tensor(
        [[[0.1, -0.2], [0.05, 0.03]], [[-0.04, 0.07], [0.02, -0.09]]],
        dtype=torch.float64,
    )
    candidate_controls = torch.tensor(
        [[[0.3, -0.1], [0.2, 0.4]], [[-0.2, 0.1], [0.5, -0.3]]],
        dtype=torch.float64,
    )
    dt = 0.25
    candidate_increments = reconstruct_candidate_increments(
        target_increments, candidate_controls, dt
    )
    log_likelihood = brownian_log_likelihood(candidate_controls, candidate_increments, dt)
    target_coordinate_formula = -torch.sum(
        candidate_controls * target_increments, dim=(1, 2)
    ) + 0.5 * dt * torch.sum(candidate_controls.square(), dim=(1, 2))

    assert torch.allclose(log_likelihood, target_coordinate_formula)


def test_gaussian_left_tail_doob_drift_has_correct_sign_and_gradient() -> None:
    current = 0.4
    barrier = -1.2
    remaining_time = 0.6
    epsilon = 1e-5
    drift = gaussian_left_tail_doob_drift(current, barrier, remaining_time)
    finite_difference = (
        math.log(
            gaussian_left_tail_probability(current + epsilon, barrier, remaining_time)
        )
        - math.log(
            gaussian_left_tail_probability(current - epsilon, barrier, remaining_time)
        )
    ) / (2.0 * epsilon)

    assert drift < 0.0
    assert drift == pytest.approx(finite_difference, rel=1e-9)
    assert math.isfinite(gaussian_left_tail_doob_drift(10.0, -3.0, 0.5))
