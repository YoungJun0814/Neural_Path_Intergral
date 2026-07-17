"""Exact weighted-MLE tests for correction-event CEM."""

from __future__ import annotations

import pytest
import torch

from src.path_integral import DownsideExcursionTask, TimePiecewiseTwoDriverControl
from src.path_integral.rbergomi_coupling import simulate_coupled_rbergomi_adjacent
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_multilevel_cem import fit_rbergomi_correction_cem
from src.training.rbergomi_piecewise_cem import _segment_sufficient_statistics


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


def _task() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=90.0,
        stress_level=95.0,
        minimum_occupation=0.05,
        hit_scale=3.0,
        occupation_scale=0.03,
    )


def test_correction_cem_is_exact_likelihood_weighted_disagreement_mle() -> None:
    simulator = _simulator()
    task = _task()
    initial = ((-0.5, 0.2), (-0.3, 0.1))
    seed = 7301
    result = fit_rbergomi_correction_cem(
        simulator,
        task,
        spot=100.0,
        maturity=0.25,
        fine_steps=16,
        initial_control=initial,
        num_paths=5_000,
        seed=seed,
        max_iterations=1,
        smoothing=1.0,
        min_disagreement_paths=100,
        control_bound=20.0,
    )
    torch.manual_seed(seed)
    paths = simulate_coupled_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_paths=5_000,
        control_fn=TimePiecewiseTwoDriverControl(initial, maturity=0.25),
        record_augmented=True,
    )
    assert paths.target_fine_brownian_increments is not None
    fine = task.hard_event(paths.fine.spot, paths.fine.step_dt)
    coarse = task.hard_event(paths.coarse.spot, paths.coarse.step_dt)
    disagreement = fine != coarse
    normalized = torch.softmax(paths.log_likelihood[disagreement], dim=0)
    sufficient = _segment_sufficient_statistics(
        paths.target_fine_brownian_increments,
        segments=2,
        step_dt=paths.fine.step_dt,
    )
    expected = torch.sum(
        normalized[:, None, None] * sufficient[disagreement], dim=0
    )
    observed = torch.tensor(result.control, dtype=torch.float64)
    assert torch.allclose(observed, expected, atol=2e-14, rtol=0.0)
    assert result.history[0].disagreement_weight_ess <= float(disagreement.sum())


def test_correction_cem_refuses_an_unidentified_zero_disagreement_update() -> None:
    task = DownsideExcursionTask(
        hit_barrier=1.0,
        stress_level=2.0,
        minimum_occupation=0.25,
        hit_scale=1.0,
        occupation_scale=0.05,
    )
    with pytest.raises(RuntimeError, match="too few adjacent-level disagreements"):
        fit_rbergomi_correction_cem(
            _simulator(),
            task,
            spot=100.0,
            maturity=0.25,
            fine_steps=8,
            initial_control=((0.0, 0.0),),
            num_paths=256,
            seed=7302,
            max_iterations=1,
            min_disagreement_paths=8,
        )
