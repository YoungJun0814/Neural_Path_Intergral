"""Oracle and contract tests for finite-look robust crossover selection."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    CandidateWorkInterval,
    advance_sequential_crossover,
    candidate_work_intervals,
    eliminate_dominated_candidates,
    freeze_crossover_decision,
    plug_in_relative_error_regret_bound,
    update_profile_intervals,
)


def _candidate(candidate_id: str, lower: float, point: float, upper: float):
    return CandidateWorkInterval(
        candidate_id=candidate_id,
        profile_ids=(f"{candidate_id}-profile",),
        sampling_variance_target=1.0,
        sampling_work_coefficient=(lower, point, upper),
        preprocessing_work=0.0,
        total_work_interval=(lower, upper),
        point_total_work=point,
    )


def test_hoeffding_profiles_cover_rademacher_oracle_at_every_fixed_look() -> None:
    generator = torch.Generator().manual_seed(14_100_101)
    sample = (2 * torch.randint(0, 2, (1024,), generator=generator) - 1).to(torch.float64)
    for count in (64, 256, 1024):
        profile = update_profile_intervals(
            {"rademacher": sample[:count]},
            absolute_bounds={"rademacher": 1.0},
            costs_per_sample={"rademacher": 3.0},
            familywise_alpha=0.05,
            total_predeclared_looks=3,
        )[0]
        assert profile.moments.mean_interval[0] <= 0.0 <= profile.moments.mean_interval[1]
        assert profile.moments.second_moment_interval[0] <= 1.0
        assert profile.moments.second_moment_interval[1] == 1.0
        assert profile.moments.variance_interval[0] <= 1.0
        assert profile.moments.variance_interval[1] == 1.0
        assert profile.moments.alpha_per_moment == pytest.approx(0.05 / 6.0)


def test_zero_observed_corrections_retain_positive_variance_upper_bound() -> None:
    profile = update_profile_intervals(
        {"correction": torch.zeros(128, dtype=torch.float64)},
        absolute_bounds={"correction": 5.0},
        costs_per_sample={"correction": 2.0},
        familywise_alpha=0.05,
        total_predeclared_looks=4,
    )[0]
    assert profile.moments.sample_variance == 0.0
    assert profile.moments.variance_interval[0] == 0.0
    assert profile.moments.variance_interval[1] > 0.0


def test_profile_rejects_data_outside_predeclared_defensive_bound() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        update_profile_intervals(
            {"term": [0.0, 1.01]},
            absolute_bounds={"term": 1.0},
            costs_per_sample={"term": 1.0},
            familywise_alpha=0.05,
            total_predeclared_looks=1,
        )


def test_candidate_work_propagation_counts_preprocessing_and_can_reverse_online_rank() -> None:
    observations = {
        "a": torch.tensor([-1.0, 1.0] * 512, dtype=torch.float64),
        "b": torch.tensor([-0.5, 0.5] * 512, dtype=torch.float64),
    }
    profiles = update_profile_intervals(
        observations,
        absolute_bounds={"a": 1.0, "b": 1.0},
        costs_per_sample={"a": 1.0, "b": 1.0},
        familywise_alpha=0.05,
        total_predeclared_looks=1,
    )
    without_training = candidate_work_intervals(
        profiles,
        candidate_profiles={"a_method": ("a",), "b_method": ("b",)},
        preprocessing_work={"a_method": 0.0, "b_method": 0.0},
        sampling_variance_target=0.01,
    )
    online = {item.candidate_id: item for item in without_training}
    assert online["b_method"].point_total_work < online["a_method"].point_total_work

    with_training = candidate_work_intervals(
        profiles,
        candidate_profiles={"a_method": ("a",), "b_method": ("b",)},
        preprocessing_work={"a_method": 0.0, "b_method": 100.0},
        sampling_variance_target=0.01,
    )
    total = {item.candidate_id: item for item in with_training}
    assert total["a_method"].point_total_work < total["b_method"].point_total_work


def test_elimination_is_one_sided_and_never_drops_interval_oracle() -> None:
    candidates = (
        _candidate("oracle", 9.0, 10.0, 11.0),
        _candidate("dominated", 20.0, 22.0, 25.0),
        _candidate("uncertain", 5.0, 15.0, 30.0),
    )
    result = eliminate_dominated_candidates(candidates, look_index=2)
    assert result.surviving_candidates == ("oracle", "uncertain")
    assert tuple(item.candidate_id for item in result.eliminated) == ("dominated",)
    assert result.best_upper_work == 11.0
    assert result.eliminated[0].candidate_lower_work > result.best_upper_work


def test_freeze_uses_simpler_endpoint_only_inside_predeclared_upper_tie() -> None:
    candidates = (
        _candidate("hybrid", 8.0, 9.0, 10.0),
        _candidate("slis", 8.5, 9.5, 10.05),
    )
    decision = freeze_crossover_decision(
        candidates,
        look_index=3,
        surviving_candidates=("hybrid", "slis"),
        simpler_candidate="slis",
        reason="pilot cap",
        upper_bound_tie_relative_tolerance=0.01,
    )
    assert decision.selected_candidate == "slis"
    assert decision.look_index == 3
    assert decision.worst_case_interval_regret_bound == pytest.approx(10.05 / 8.0)


def test_plugin_regret_formula_and_domain() -> None:
    assert plug_in_relative_error_regret_bound(0.0) == 1.0
    assert plug_in_relative_error_regret_bound(0.1) == pytest.approx(1.1 / 0.9)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        plug_in_relative_error_regret_bound(1.0)
    assert math.isfinite(plug_in_relative_error_regret_bound(0.99))


def test_sequential_state_rejects_optional_looks_and_freezes_at_declared_cap() -> None:
    common = {
        "absolute_bounds": {"hybrid_term": 1.0, "slis_term": 1.0},
        "costs_per_sample": {"hybrid_term": 1.0, "slis_term": 1.0},
        "candidate_profiles": {
            "hybrid": ("hybrid_term",),
            "slis": ("slis_term",),
        },
        "preprocessing_work": {"hybrid": 2.0, "slis": 0.0},
        "sampling_variance_target": 0.01,
        "predeclared_looks": (8, 16),
        "familywise_alpha": 0.05,
        "simpler_candidate": "slis",
    }
    first = advance_sequential_crossover(
        {
            "hybrid_term": torch.tensor([-1.0, 1.0] * 4),
            "slis_term": torch.zeros(8),
        },
        look_index=0,
        **common,
    )
    assert not first.stopped
    assert first.frozen_decision is None

    with pytest.raises(ValueError, match="declared look"):
        advance_sequential_crossover(
            {"hybrid_term": torch.zeros(12), "slis_term": torch.zeros(12)},
            look_index=1,
            previous_state=first,
            **common,
        )

    final = advance_sequential_crossover(
        {
            "hybrid_term": torch.tensor([-1.0, 1.0] * 8),
            "slis_term": torch.tensor([0.0] * 8 + [-1.0, 1.0] * 4),
        },
        look_index=1,
        previous_state=first,
        **common,
    )
    assert final.stopped
    assert final.stop_reason == "predeclared pilot cap reached"
    assert final.frozen_decision is not None
