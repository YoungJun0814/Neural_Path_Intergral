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
