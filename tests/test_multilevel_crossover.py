"""Non-asymptotic single-level versus multilevel work comparisons."""

from __future__ import annotations

import pytest

from src.path_integral import (
    evaluate_multilevel_crossover,
    evaluate_total_work_crossover,
    optimal_sampling_work_coefficient,
)


def test_optimal_work_coefficient_matches_closed_form_allocation() -> None:
    assert optimal_sampling_work_coefficient([4.0, 1.0], [1.0, 4.0]) == 16.0


def test_crossover_selects_multilevel_only_when_total_coefficient_is_lower() -> None:
    decision = evaluate_multilevel_crossover(
        single_level_variances=[1.0, 1.0, 1.0],
        single_level_costs=[1.0, 2.0, 4.0],
        correction_variances=[0.01, 0.0025],
        correction_costs=[3.0, 6.0],
    )
    assert decision.multilevel_strictly_better
    assert decision.optimal_start_level < decision.finest_level
    assert decision.single_over_optimal_work_ratio > 1.0


def test_crossover_falls_back_to_finest_single_level_when_corrections_are_costly() -> None:
    decision = evaluate_multilevel_crossover(
        single_level_variances=[0.01, 0.005, 0.001],
        single_level_costs=[1.0, 2.0, 4.0],
        correction_variances=[1.0, 1.0],
        correction_costs=[3.0, 6.0],
    )
    assert not decision.multilevel_strictly_better
    assert decision.optimal_start_level == decision.finest_level
    assert decision.single_over_optimal_work_ratio == 1.0


def test_crossover_rejects_misaligned_level_arrays() -> None:
    with pytest.raises(ValueError, match="levels one through"):
        evaluate_multilevel_crossover(
            single_level_variances=[1.0, 1.0, 1.0],
            single_level_costs=[1.0, 2.0, 4.0],
            correction_variances=[0.1],
            correction_costs=[1.0],
        )


def test_training_cost_can_reverse_an_online_multilevel_advantage() -> None:
    common = {
        "single_level_variances": [1.0, 1.0, 1.0],
        "single_level_costs": [1.0, 2.0, 4.0],
        "correction_variances": [0.01, 0.0025],
        "correction_costs": [3.0, 6.0],
        "sampling_variance_target": 1.0,
    }
    online = evaluate_multilevel_crossover(
        single_level_variances=common["single_level_variances"],
        single_level_costs=common["single_level_costs"],
        correction_variances=common["correction_variances"],
        correction_costs=common["correction_costs"],
    )
    assert online.multilevel_strictly_better
    total = evaluate_total_work_crossover(
        **common,
        preprocessing_work_by_start_level=[100.0, 100.0, 0.0],
    )
    assert not total.multilevel_strictly_better
    assert total.optimal_start_level == total.online.finest_level
    assert total.single_over_optimal_total_work_ratio == 1.0
