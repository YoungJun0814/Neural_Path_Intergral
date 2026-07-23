"""Theorem-to-code falsification tests for V6 rBergomi diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    DiscreteBarrierHitTask,
    barrier_obligation_diagnostics,
    coefficient_moment_diagnostics,
    direction_regularity_diagnostics,
    evaluate_rbergomi_threshold_coupling,
    slope_lower_tail_diagnostics,
    terminal_rate_contract,
    terminal_slope_inverse_moment_bound,
)


def test_direction_diagnostic_matches_the_implemented_pair_sum_convention() -> None:
    fine = torch.full((8,), 1.0 / math.sqrt(8), dtype=torch.float64)
    coarse = fine.reshape(-1, 2).sum(dim=1)
    diagnostics = direction_regularity_diagnostics(
        fine, declared_coarse_weights=coarse
    )
    assert diagnostics.positive
    assert diagnostics.unit_normalized
    assert diagnostics.coarse_consistent
    assert diagnostics.sqrt_steps_maximum_weight == pytest.approx(1.0)
    assert diagnostics.inverse_sqrt_steps_l1_mass == pytest.approx(1.0)


def test_direction_diagnostic_detects_but_does_not_repair_a_coupling_error() -> None:
    fine = torch.full((4,), 0.5, dtype=torch.float64)
    diagnostics = direction_regularity_diagnostics(
        fine, declared_coarse_weights=torch.tensor([1.0, 0.9])
    )
    assert diagnostics.coarse_consistent is False
    assert diagnostics.coarse_aggregation_error == pytest.approx(0.1)


def test_slope_tail_diagnostic_reports_empirical_inverse_moments_only() -> None:
    slopes = torch.tensor([0.1, 0.2, 0.4, 0.8], dtype=torch.float64)
    diagnostics = slope_lower_tail_diagnostics(
        slopes, inverse_orders=(1.0, 2.0), lower_tail_floors=(0.15,)
    )
    assert diagnostics.sample_count == 4
    assert diagnostics.minimum_slope == 0.1
    assert dict(diagnostics.inverse_moments)[1.0] == pytest.approx(
        float(torch.mean(1.0 / slopes))
    )
    assert dict(diagnostics.lower_tail_probabilities)[0.15] == 0.25


def test_terminal_slope_inverse_moment_bound_is_exact_for_constant_volatility() -> None:
    steps = 16
    direction = torch.full((steps,), 1.0 / math.sqrt(steps), dtype=torch.float64)
    result = terminal_slope_inverse_moment_bound(
        direction,
        step_dt=1.0 / steps,
        maturity=1.0,
        hurst=0.1,
        xi=0.04,
        eta=0.0,
        rho=-0.7,
        order=2.0,
        minimum_grid_scaled_l1_mass=1.0,
    )
    exact_slope = math.sqrt(1.0 - 0.7**2) * math.sqrt(0.04)
    assert result.grid_scaled_l1_mass == pytest.approx(1.0)
    assert result.upper_bound == pytest.approx(exact_slope**-2.0)


def test_piecewise_positive_direction_has_a_grid_uniform_scaled_l1_mass() -> None:
    masses = []
    for steps in (16, 64):
        schedule = torch.cat(
            (torch.full((steps // 2,), 2.0), torch.ones(steps // 2))
        ).to(torch.float64)
        direction = schedule / torch.linalg.vector_norm(schedule)
        result = terminal_slope_inverse_moment_bound(
            direction,
            step_dt=1.0 / steps,
            maturity=1.0,
            hurst=0.05,
            xi=0.04,
            eta=1.5,
            rho=-0.7,
            order=4.0,
        )
        masses.append(result.grid_scaled_l1_mass)
        assert math.isfinite(result.upper_bound)
    assert masses[0] == pytest.approx(masses[1])


def test_terminal_rate_contract_separates_candidate_rate_from_journal_claim_and_barrier() -> None:
    contract = terminal_rate_contract(0.12, epsilon_margin=0.01)
    assert contract.coefficient_lp_exponent == pytest.approx(0.11)
    assert contract.threshold_lp_exponent == pytest.approx(0.11)
    assert contract.dcs_second_moment_exponent == pytest.approx(0.22)
    assert contract.weak_bias_exponent == pytest.approx(0.11)
    assert contract.fft_has_log_factor
    assert contract.mlmc_epsilon_polynomial_exponent == pytest.approx(1.0 / 0.11)
    assert not contract.barrier_included
    assert "independent mathematical review" in contract.proof_status
    assert not contract.journal_claim_ready

    with pytest.raises(ValueError, match="epsilon"):
        terminal_rate_contract(0.12, epsilon_margin=0.12)


def test_coefficient_diagnostic_returns_raw_lp_norms_without_fitting_a_rate() -> None:
    diagnostic = coefficient_moment_diagnostics(
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.5, 1.0]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([0.0, 0.0]),
        mesh_size=0.125,
        orders=(1.0, 2.0),
    )
    assert dict(diagnostic.intercept_lp)[1.0] == 0.5
    assert dict(diagnostic.intercept_lp)[2.0] == pytest.approx(math.sqrt(0.5))
    assert dict(diagnostic.slope_lp)[1.0] == 0.75


def test_barrier_obligation_summary_retains_fine_only_mesh_defect() -> None:
    log_barrier = math.log(0.5)
    fine_candidates = torch.tensor([[7.0, 5.0, 6.0, 3.0]], dtype=torch.float64)
    coarse_candidates = torch.tensor([[4.0, 2.0]], dtype=torch.float64)
    fine_intercept = torch.cat(
        (torch.zeros((1, 1), dtype=torch.float64), log_barrier - fine_candidates), dim=1
    )
    coarse_intercept = torch.cat(
        (torch.zeros((1, 1), dtype=torch.float64), log_barrier - coarse_candidates), dim=1
    )
    fine_slope = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    coarse_slope = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64)
    pathwise = evaluate_rbergomi_threshold_coupling(
        fine_intercept,
        fine_slope,
        coarse_intercept,
        coarse_slope,
        fine_step_dt=0.25,
        coarse_step_dt=0.5,
        task=DiscreteBarrierHitTask(0.5),
        denominator_floor=0.5,
    )
    summary = barrier_obligation_diagnostics(pathwise, active_time_cutoff=0.5)
    assert summary.active_before_cutoff_fraction == 1.0
    assert summary.mesh_enrichment_l1 == 2.0
    assert summary.mesh_enrichment_l2 == 4.0
    assert summary.maximum_exact_decomposition_violation == 0.0
