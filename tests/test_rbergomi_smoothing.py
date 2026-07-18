"""End-to-end exactness tests for Gaussian-smoothed rBergomi estimators."""

from __future__ import annotations

import math

import torch

from src.path_integral.controllers import TimePiecewiseTwoDriverControl
from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_smoothing import (
    simulate_smoothed_adjacent_rbergomi,
    simulate_smoothed_rbergomi,
)
from src.physics_engine import RBergomiSimulator


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


def _task() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=92.0,
        stress_level=97.0,
        minimum_occupation=0.125,
        hit_scale=2.0,
        occupation_scale=0.03,
    )


def _control() -> TimePiecewiseTwoDriverControl:
    return TimePiecewiseTwoDriverControl(((-0.4, -1.2), (-0.2, -0.8)), maturity=0.25)


def _assert_paired_mean_agreement(left: torch.Tensor, right: torch.Tensor) -> None:
    difference = left - right
    standard_error = math.sqrt(float(difference.var(unbiased=True)) / difference.numel())
    assert abs(float(difference.mean())) < 4.5 * standard_error + 2e-4


def test_single_grid_smoothing_reconstructs_event_likelihood_and_expectation() -> None:
    torch.manual_seed(9201)
    result = simulate_smoothed_rbergomi(
        _simulator(),
        S0=100.0,
        T=0.25,
        dt=0.25 / 16,
        num_paths=10_000,
        task=_task(),
        control_fn=_control(),
    )
    assert torch.equal(result.level.hard_event, result.level.threshold_event)
    assert result.maximum_likelihood_reconstruction_error < 3e-14
    assert result.maximum_residual_projection < 3e-15
    assert result.maximum_path_reconstruction_error < 3e-14
    _assert_paired_mean_agreement(result.level.smoothed_contribution, result.level.raw_contribution)
    assert float(result.level.smoothed_contribution.var(unbiased=True)) <= (
        1.05 * float(result.level.raw_contribution.var(unbiased=True))
    )


def test_positive_nonuniform_time_weights_preserve_exactness() -> None:
    steps = 16
    direction = torch.linspace(0.25, 1.75, steps, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    torch.manual_seed(9202)
    result = simulate_smoothed_rbergomi(
        _simulator(),
        S0=100.0,
        T=0.25,
        dt=0.25 / steps,
        num_paths=2_000,
        task=_task(),
        direction=direction,
    )
    assert torch.equal(result.level.hard_event, result.level.threshold_event)
    assert result.maximum_likelihood_reconstruction_error == 0.0


def test_adjacent_smoothing_is_exact_for_both_blp_marginals_and_correction() -> None:
    torch.manual_seed(9203)
    result = simulate_smoothed_adjacent_rbergomi(
        _simulator(),
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_paths=12_000,
        task=_task(),
        control_fn=_control(),
    )
    assert torch.equal(result.fine.hard_event, result.fine.threshold_event)
    assert torch.equal(result.coarse.hard_event, result.coarse.threshold_event)
    assert result.maximum_likelihood_reconstruction_error < 3e-14
    assert result.maximum_residual_projection < 3e-15
    assert result.maximum_fine_path_reconstruction_error < 3e-14
    assert result.maximum_coarse_path_reconstruction_error < 3e-14
    _assert_paired_mean_agreement(result.smoothed_correction, result.raw_correction)
    assert float(result.smoothed_correction.var(unbiased=True)) <= (
        1.08 * float(result.raw_correction.var(unbiased=True))
    )


class _UndeclaredFeedback:
    def __call__(
        self,
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack((torch.zeros_like(spot), 0.01 * (spot - 100.0)), dim=1)


def test_feedback_control_is_rejected_before_simulation() -> None:
    try:
        simulate_smoothed_rbergomi(
            _simulator(),
            S0=100.0,
            T=0.25,
            dt=0.25 / 8,
            num_paths=32,
            task=_task(),
            control_fn=_UndeclaredFeedback(),
        )
    except ValueError as error:
        assert "deterministic_time_control" in str(error)
    else:
        raise AssertionError("path-dependent control was not rejected")


class _FalselyDeclaredFeedback(_UndeclaredFeedback):
    is_deterministic_time_control = True


def test_recorded_path_dependence_is_rejected_even_if_controller_claims_determinism() -> None:
    try:
        simulate_smoothed_rbergomi(
            _simulator(),
            S0=100.0,
            T=0.25,
            dt=0.25 / 8,
            num_paths=32,
            task=_task(),
            control_fn=_FalselyDeclaredFeedback(),
            engine="reference",
        )
    except ValueError as error:
        assert "path-dependent" in str(error)
    else:
        raise AssertionError("falsely declared feedback control was not rejected")
