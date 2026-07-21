"""Adapter, task, and G10 characterization tests for G11 DCS-MGI."""

from __future__ import annotations

import pytest
import torch

from src.path_integral import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    evaluate_control_span_marginalized_adjacent_mixture,
    evaluate_control_span_marginalized_mixture,
    simulate_coupled_rbergomi_mixture,
    simulate_rbergomi_mixture,
)
from src.path_integral.rbergomi_dcs_mlmc import (
    evaluate_rbergomi_dcs_adjacent,
    evaluate_rbergomi_dcs_level,
)
from src.physics_engine import RBergomiSimulator


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.12, eta=1.1, xi=0.04, rho=-0.6, device="cpu")


def _controls() -> tuple[TimePiecewiseTwoDriverControl, TimePiecewiseTwoDriverControl]:
    return (
        TimePiecewiseTwoDriverControl(((0.0, 0.0), (0.0, 0.0)), maturity=0.25),
        TimePiecewiseTwoDriverControl(((-0.4, -1.2), (-0.25, -0.7)), maturity=0.25),
    )


def _downside() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=92.0,
        stress_level=97.0,
        minimum_occupation=1.0 / 64.0,
        hit_scale=3.0,
        occupation_scale=0.02,
    )


def _single_sample(paths: int = 4096):
    simulator = _simulator()
    torch.manual_seed(12_400_101)
    sample = simulate_rbergomi_mixture(
        simulator,
        _controls(),
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 64.0,
        num_paths=paths,
        label_generator=torch.Generator().manual_seed(12_400_102),
        engine="fft",
    )
    return simulator, sample


def _adjacent_sample(paths: int = 4096):
    simulator = _simulator()
    torch.manual_seed(12_500_101)
    sample = simulate_coupled_rbergomi_mixture(
        simulator,
        _controls(),
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        fine_steps=32,
        num_paths=paths,
        label_generator=torch.Generator().manual_seed(12_500_102),
        engine="fft",
    )
    return simulator, sample


@pytest.mark.parametrize(
    "task",
    [
        TerminalThresholdTask(level=95.0),
        DiscreteBarrierHitTask(barrier=92.0),
        _downside(),
    ],
)
def test_all_supported_tasks_have_exact_scalar_thresholds(task) -> None:
    simulator, sample = _single_sample()
    evaluation = evaluate_rbergomi_dcs_level(sample, task=task, rho=simulator.rho)
    assert torch.equal(evaluation.hard_event, evaluation.threshold_event)
    assert evaluation.maximum_path_reconstruction_error <= 2e-13
    assert evaluation.maximum_legacy_component_density_error <= 2e-13
    assert evaluation.maximum_legacy_mixture_density_error <= 2e-13
    assert evaluation.maximum_legacy_full_likelihood_error <= 2e-13
    assert evaluation.density.maximum_component_reconstruction_error <= 2e-13
    assert evaluation.density.maximum_mixture_reconstruction_error <= 2e-13
    assert evaluation.density.maximum_full_bound_violation <= 2e-13
    assert evaluation.density.maximum_residual_bound_violation <= 2e-13


def test_generic_single_adapter_is_numerically_identical_to_g10_downside() -> None:
    simulator, sample = _single_sample(8192)
    legacy = evaluate_control_span_marginalized_mixture(sample, task=_downside(), rho=simulator.rho)
    generic = evaluate_rbergomi_dcs_level(sample, task=_downside(), rho=simulator.rho)
    assert torch.equal(generic.hard_event, legacy.hard_event)
    assert torch.allclose(generic.threshold, legacy.threshold, atol=0.0, rtol=0.0)
    assert torch.allclose(
        generic.raw_contribution,
        legacy.raw_mixture_contribution,
        atol=2e-15,
        rtol=0.0,
    )
    assert torch.allclose(
        generic.marginalized_contribution,
        legacy.marginalized_contribution,
        atol=3e-15,
        rtol=0.0,
    )
    assert torch.allclose(
        generic.density.residual_log_likelihood,
        legacy.log_outer_likelihood,
        atol=2e-14,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "task",
    [
        TerminalThresholdTask(level=95.0),
        DiscreteBarrierHitTask(barrier=92.0),
        _downside(),
    ],
)
def test_adjacent_adapter_uses_one_coordinate_and_likelihood(task) -> None:
    simulator, sample = _adjacent_sample()
    evaluation = evaluate_rbergomi_dcs_adjacent(sample, task=task, rho=simulator.rho)
    assert torch.equal(evaluation.fine.hard_event, evaluation.fine.threshold_event)
    assert torch.equal(evaluation.coarse.hard_event, evaluation.coarse.threshold_event)
    assert evaluation.maximum_coordinate_mismatch <= 2e-15
    assert evaluation.fine.density is evaluation.coarse.density
    expected_raw = (
        evaluation.fine.hard_event.to(torch.float64)
        - evaluation.coarse.hard_event.to(torch.float64)
    ) * evaluation.fine.density.full_likelihood
    assert torch.allclose(evaluation.raw_correction, expected_raw, atol=2e-15, rtol=0.0)


def test_generic_adjacent_adapter_is_numerically_identical_to_g10_downside() -> None:
    simulator, sample = _adjacent_sample(8192)
    legacy = evaluate_control_span_marginalized_adjacent_mixture(
        sample, task=_downside(), rho=simulator.rho
    )
    generic = evaluate_rbergomi_dcs_adjacent(sample, task=_downside(), rho=simulator.rho)
    assert torch.allclose(generic.fine.threshold, legacy.fine_threshold, atol=0.0, rtol=0.0)
    assert torch.allclose(generic.coarse.threshold, legacy.coarse_threshold, atol=0.0, rtol=0.0)
    assert torch.allclose(generic.raw_correction, legacy.raw_correction, atol=3e-15, rtol=0.0)
    assert torch.allclose(
        generic.marginalized_correction,
        legacy.marginalized_correction,
        atol=4e-15,
        rtol=0.0,
    )


def test_task_types_reject_invalid_parameters_and_paths() -> None:
    with pytest.raises(ValueError, match="positive"):
        TerminalThresholdTask(level=0.0)
    with pytest.raises(ValueError, match="positive"):
        DiscreteBarrierHitTask(barrier=-1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        TerminalThresholdTask(level=90.0).hard_event(
            torch.tensor([[100.0, 0.0]], dtype=torch.float64), 0.1
        )


def test_terminal_and_barrier_cem_scores_have_exact_event_sign() -> None:
    spot = torch.tensor([[100.0, 95.0, 89.0], [100.0, 101.0, 102.0]], dtype=torch.float64)
    terminal = TerminalThresholdTask(level=90.0)
    barrier = DiscreteBarrierHitTask(barrier=90.0)
    assert torch.equal(terminal.score(spot, 0.1) >= 0.0, terminal.hard_event(spot, 0.1))
    assert torch.equal(barrier.score(spot, 0.1) >= 0.0, barrier.hard_event(spot, 0.1))


def test_adapter_rejects_degenerate_price_correlation() -> None:
    _simulator_instance, sample = _single_sample(128)
    with pytest.raises(ValueError, match="strictly between"):
        evaluate_rbergomi_dcs_level(sample, task=_downside(), rho=1.0)


def test_natural_proposal_uses_a_valid_fixed_event_direction() -> None:
    simulator = _simulator()
    torch.manual_seed(12_600_101)
    sample = simulate_rbergomi_mixture(
        simulator,
        (_controls()[0],),
        torch.ones(1, dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 32.0,
        num_paths=1024,
        label_generator=torch.Generator().manual_seed(12_600_102),
        engine="fft",
    )
    evaluation = evaluate_rbergomi_dcs_level(
        sample, task=TerminalThresholdTask(95.0), rho=simulator.rho
    )
    assert torch.equal(evaluation.hard_event, evaluation.threshold_event)
    assert torch.allclose(
        evaluation.density.residual_likelihood,
        torch.ones_like(evaluation.density.residual_likelihood),
        atol=0.0,
        rtol=0.0,
    )
