"""Finite-grid tests for adjacent rBergomi threshold diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    DiscreteBarrierHitTask,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    evaluate_rbergomi_dcs_adjacent,
    evaluate_rbergomi_threshold_coupling,
    simulate_coupled_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator


def _terminal_affine_paths() -> tuple[torch.Tensor, ...]:
    fine_intercept = torch.tensor(
        [[0.0, -0.1, -0.3, -0.6, -1.0], [0.0, 0.1, -0.1, -0.2, -0.4]],
        dtype=torch.float64,
    )
    fine_slope = torch.tensor(
        [[0.0, 0.2, 0.4, 0.6, 0.8], [0.0, 0.3, 0.5, 0.7, 0.9]],
        dtype=torch.float64,
    )
    coarse_intercept = torch.tensor([[0.0, -0.25, -0.9], [0.0, -0.05, -0.35]], dtype=torch.float64)
    coarse_slope = torch.tensor([[0.0, 0.39, 0.78], [0.0, 0.48, 0.87]], dtype=torch.float64)
    return fine_intercept, fine_slope, coarse_intercept, coarse_slope


def test_terminal_diagnostic_has_no_mesh_defect_and_respects_ratio_bound() -> None:
    diagnostics = evaluate_rbergomi_threshold_coupling(
        *_terminal_affine_paths(),
        fine_step_dt=0.25,
        coarse_step_dt=0.5,
        task=TerminalThresholdTask(level=1.0),
        denominator_floor=0.7,
    )
    assert diagnostics.task_kind == "terminal_threshold"
    assert torch.all(diagnostics.good_event)
    assert torch.equal(
        diagnostics.mesh_enrichment_defect,
        torch.zeros_like(diagnostics.mesh_enrichment_defect),
    )
    assert torch.allclose(
        diagnostics.threshold_error,
        diagnostics.common_candidate_error,
        atol=0.0,
        rtol=0.0,
    )
    assert diagnostics.maximum_good_event_bound_violation <= 2e-16
    assert diagnostics.maximum_exact_decomposition_violation == 0.0
    assert torch.equal(diagnostics.fine_active_index, torch.full((2,), 4))
    assert torch.equal(diagnostics.coarse_active_index, torch.full((2,), 2))
    assert torch.allclose(diagnostics.fine_active_time, torch.ones(2, dtype=torch.float64))
    assert torch.allclose(diagnostics.coarse_active_time, torch.ones(2, dtype=torch.float64))


def test_barrier_diagnostic_separates_new_monitoring_time_from_common_error() -> None:
    log_barrier = math.log(0.5)
    fine_candidates = torch.tensor([[7.0, 5.0, 6.0, 3.0]], dtype=torch.float64)
    coarse_candidates = torch.tensor([[4.0, 2.0]], dtype=torch.float64)
    fine_intercept = torch.cat(
        (
            torch.zeros((1, 1), dtype=torch.float64),
            log_barrier - fine_candidates,
        ),
        dim=1,
    )
    coarse_intercept = torch.cat(
        (
            torch.zeros((1, 1), dtype=torch.float64),
            log_barrier - coarse_candidates,
        ),
        dim=1,
    )
    fine_slope = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    coarse_slope = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64)
    diagnostics = evaluate_rbergomi_threshold_coupling(
        fine_intercept,
        fine_slope,
        coarse_intercept,
        coarse_slope,
        fine_step_dt=0.25,
        coarse_step_dt=0.5,
        task=DiscreteBarrierHitTask(barrier=0.5),
        denominator_floor=1.0,
    )
    assert diagnostics.task_kind == "discrete_barrier_hit"
    assert diagnostics.fine_threshold.item() == 7.0
    assert diagnostics.coarse_threshold.item() == 4.0
    assert diagnostics.common_candidate_error.item() == 1.0
    assert diagnostics.mesh_enrichment_defect.item() == 2.0
    assert diagnostics.threshold_error.item() == 3.0
    assert diagnostics.threshold_error_bound.item() == 3.0
    assert diagnostics.fine_active_index.item() == 1
    assert diagnostics.coarse_active_index.item() == 1
    assert diagnostics.fine_active_time.item() == 0.25
    assert diagnostics.coarse_active_time.item() == 0.5
    assert diagnostics.maximum_good_event_bound_violation == 0.0
    assert diagnostics.maximum_exact_decomposition_violation == 0.0


def test_bad_denominator_paths_are_not_claimed_by_the_good_event_bound() -> None:
    fine_intercept, fine_slope, coarse_intercept, coarse_slope = _terminal_affine_paths()
    diagnostics = evaluate_rbergomi_threshold_coupling(
        fine_intercept,
        fine_slope,
        coarse_intercept,
        coarse_slope,
        fine_step_dt=0.25,
        coarse_step_dt=0.5,
        task=TerminalThresholdTask(level=1.0),
        denominator_floor=0.85,
    )
    assert torch.equal(diagnostics.good_event, torch.tensor([False, True], dtype=torch.bool))
    assert diagnostics.maximum_good_event_bound_violation <= 2e-16


def test_diagnostic_rejects_nonadjacent_grids_and_preserves_infinite_thresholds() -> None:
    fine_intercept, fine_slope, coarse_intercept, coarse_slope = _terminal_affine_paths()
    with pytest.raises(ValueError, match="twice"):
        evaluate_rbergomi_threshold_coupling(
            fine_intercept[:, :-1],
            fine_slope[:, :-1],
            coarse_intercept,
            coarse_slope,
            fine_step_dt=1.0 / 3.0,
            coarse_step_dt=0.5,
            task=TerminalThresholdTask(level=1.0),
            denominator_floor=0.1,
        )
    diagnostics = evaluate_rbergomi_threshold_coupling(
        fine_intercept,
        fine_slope,
        coarse_intercept,
        coarse_slope,
        fine_step_dt=0.25,
        coarse_step_dt=0.5,
        task=DiscreteBarrierHitTask(barrier=1.0),
        denominator_floor=0.1,
    )
    assert torch.all(diagnostics.initially_hit)
    assert not torch.any(diagnostics.finite_threshold)
    assert torch.all(torch.isposinf(diagnostics.fine_threshold))
    assert torch.all(torch.isposinf(diagnostics.coarse_threshold))
    assert torch.equal(diagnostics.threshold_error, torch.zeros_like(diagnostics.threshold_error))
    assert not torch.any(diagnostics.good_event)
    assert torch.equal(diagnostics.fine_active_index, torch.zeros(2, dtype=torch.long))


@pytest.mark.parametrize(
    "task", [TerminalThresholdTask(level=95.0), DiscreteBarrierHitTask(barrier=92.0)]
)
def test_diagnostic_reconstructs_actual_dcs_threshold_difference(task) -> None:
    simulator = RBergomiSimulator(H=0.12, eta=1.1, xi=0.04, rho=-0.6, device="cpu")
    controls = (
        TimePiecewiseTwoDriverControl(((0.0, 0.0), (0.0, 0.0)), maturity=0.25),
        TimePiecewiseTwoDriverControl(((-0.4, -1.2), (-0.25, -0.7)), maturity=0.25),
    )
    torch.manual_seed(13_100_101)
    sample = simulate_coupled_rbergomi_mixture(
        simulator,
        controls,
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        fine_steps=16,
        num_paths=256,
        label_generator=torch.Generator().manual_seed(13_100_102),
        engine="fft",
    )
    evaluation = evaluate_rbergomi_dcs_adjacent(sample, task=task, rho=simulator.rho)
    minimum_slope = min(
        float(torch.amin(evaluation.fine.log_spot_slope[:, 1:])),
        float(torch.amin(evaluation.coarse.log_spot_slope[:, 1:])),
    )
    diagnostics = evaluate_rbergomi_threshold_coupling(
        evaluation.fine.log_spot_intercept,
        evaluation.fine.log_spot_slope,
        evaluation.coarse.log_spot_intercept,
        evaluation.coarse.log_spot_slope,
        fine_step_dt=sample.paths.fine.step_dt,
        coarse_step_dt=sample.paths.coarse.step_dt,
        task=task,
        denominator_floor=0.5 * minimum_slope,
    )
    assert torch.all(diagnostics.good_event)
    assert torch.allclose(
        diagnostics.fine_threshold, evaluation.fine.threshold, atol=2e-14, rtol=0.0
    )
    assert torch.allclose(
        diagnostics.coarse_threshold,
        evaluation.coarse.threshold,
        atol=2e-14,
        rtol=0.0,
    )
    assert torch.allclose(
        diagnostics.signed_threshold_difference,
        evaluation.threshold_difference,
        atol=3e-14,
        rtol=0.0,
    )
    assert diagnostics.maximum_good_event_bound_violation <= 2e-13
    assert diagnostics.maximum_exact_decomposition_violation <= 2e-13
