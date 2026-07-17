"""Finite-grid telescoping and MLMC allocation tests."""

from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.multilevel import (
    break_even_query_count,
    optimal_mlmc_sample_counts,
    single_level_online_work,
)
from src.path_integral import DownsideExcursionTask
from src.path_integral.rbergomi_coupling import simulate_coupled_rbergomi_adjacent
from src.physics_engine import RBergomiSimulator


def test_optimal_allocation_respects_budget_and_square_root_ratio() -> None:
    allocation = optimal_mlmc_sample_counts(
        [4.0, 1.0, 0.25],
        [1.0, 2.0, 4.0],
        variance_budget=0.01,
    )
    assert allocation.predicted_variance <= 0.01
    counts = allocation.sample_counts
    expected_ratio = math.sqrt((4.0 / 1.0) / (1.0 / 2.0))
    assert counts[0] / counts[1] == pytest.approx(expected_ratio, rel=0.01)
    single_count, single_work = single_level_online_work(
        4.0, 2.0, variance_budget=0.01
    )
    assert single_count == 400
    assert single_work == 800.0
    assert break_even_query_count(30.0, 10.0, 7.0) == 10.0
    assert math.isinf(break_even_query_count(30.0, 7.0, 10.0))


def test_hard_event_telescoping_recovers_finest_blp_expectation() -> None:
    simulator = RBergomiSimulator(
        H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu"
    )
    task = DownsideExcursionTask(
        hit_barrier=90.0,
        stress_level=95.0,
        minimum_occupation=0.05,
        hit_scale=3.0,
        occupation_scale=0.03,
    )
    paths = 20_000
    contributions: list[torch.Tensor] = []
    for level, steps in enumerate((8, 16, 32)):
        torch.manual_seed(7400 + level)
        pair = simulate_coupled_rbergomi_adjacent(
            simulator,
            S0=100.0,
            T=0.25,
            fine_steps=steps,
            num_paths=paths,
        )
        fine = task.hard_event(pair.fine.spot, pair.fine.step_dt).double()
        if level == 0:
            contributions.append(fine)
        else:
            coarse = task.hard_event(pair.coarse.spot, pair.coarse.step_dt).double()
            contributions.append(fine - coarse)
    telescoping = sum(float(value.mean()) for value in contributions)
    telescoping_variance = sum(float(value.var(unbiased=True)) / paths for value in contributions)

    torch.manual_seed(7499)
    direct = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=0.25 / 32,
        num_paths=paths,
        control_fn=None,
    )
    direct_event = task.hard_event(direct.spot, direct.step_dt).double()
    difference = telescoping - float(direct_event.mean())
    combined_se = math.sqrt(
        telescoping_variance + float(direct_event.var(unbiased=True)) / paths
    )
    assert abs(difference) < 4.0 * combined_se + 5e-4
