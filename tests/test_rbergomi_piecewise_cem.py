"""Exact-law and fitting tests for the time-piecewise rBergomi CEM baseline."""

from __future__ import annotations

import math

import torch

from src.path_integral import (
    DownsideExcursionTask,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    brownian_log_likelihood,
)
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_piecewise_cem import (
    _segment_sufficient_statistics,
    fit_rbergomi_piecewise_cem,
)


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


def _task() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=88.0,
        stress_level=94.0,
        minimum_occupation=0.10,
        hit_scale=3.0,
        occupation_scale=0.04,
    )


def test_piecewise_proposal_has_exact_augmented_likelihood() -> None:
    control = TimePiecewiseTwoDriverControl(((0.4, -0.3), (-0.7, 0.2)), maturity=0.25)
    torch.manual_seed(2601)
    result = _simulator().simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 32.0,
        num_paths=1_024,
        control_fn=control,
        record_augmented=True,
    )
    assert result.controls is not None
    assert result.proposal_brownian_increments is not None
    expected = brownian_log_likelihood(
        result.controls, result.proposal_brownian_increments, result.step_dt
    )
    assert torch.allclose(result.log_likelihood, expected, atol=1e-14, rtol=0.0)
    assert abs(float(torch.exp(result.log_likelihood).mean()) - 1.0) < 0.08


def test_segment_sufficient_statistics_are_duration_normalized() -> None:
    increments = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
        dtype=torch.float64,
    )
    result = _segment_sufficient_statistics(increments, segments=2, step_dt=0.5)
    assert torch.equal(result, torch.tensor([[[4.0, 6.0], [12.0, 14.0]]]))


def test_piecewise_cem_produces_finite_downside_control_and_history() -> None:
    simulator = _simulator()
    result = fit_rbergomi_piecewise_cem(
        simulator,
        _task(),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        initial_control=((0.5, -0.5), (0.5, -0.5)),
        num_paths=1_200,
        seed=2602,
        max_iterations=3,
        elite_quantile=0.85,
        smoothing=0.6,
        min_elite_paths=32,
        target_level_repetitions=1,
    )
    assert result.history
    assert all(math.isfinite(value) for pair in result.control for value in pair)
    assert all(
        math.isfinite(item.hard_probability_estimate) and item.elite_weight_ess > 0.0
        for item in result.history
    )
    perpendicular = math.sqrt(1.0 - simulator.rho**2)
    spot_drifts = [
        simulator.rho * first + perpendicular * second for first, second in result.control
    ]
    assert sum(spot_drifts) < 0.0


def test_piecewise_cem_accepts_terminal_threshold_task() -> None:
    result = fit_rbergomi_piecewise_cem(
        _simulator(),
        TerminalThresholdTask(95.0),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 8.0,
        initial_control=((0.0, -0.5),),
        num_paths=256,
        seed=2603,
        max_iterations=1,
        min_elite_paths=16,
        target_level_repetitions=1,
    )
    assert len(result.history) == 1
    assert all(math.isfinite(value) for pair in result.control for value in pair)
