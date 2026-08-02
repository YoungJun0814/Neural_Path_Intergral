from __future__ import annotations

import torch

from src.path_integral.baseline_diagnostics import (
    evaluate_baseline_likelihood_diagnostics,
    evaluate_coupling_flow_roundtrip,
)
from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    freeze_baseline_proposal,
)


def test_target_likelihood_diagnostics_are_exact() -> None:
    proposal = freeze_baseline_proposal(
        method="crude_mc",
        task_id="target",
        dimension=4,
        training_seed=1,
        training_cost=BaselineCostLedger(),
    )
    result = evaluate_baseline_likelihood_diagnostics(proposal, sample_count=1024, seed=2)
    assert result.normalization_mean == 1.0
    assert result.normalization_standard_error == 0.0
    assert result.normalization_z == 0.0
    assert result.effective_sample_size == 1024.0
    assert result.nonfinite_weight_count == 0
    assert result.component_counts == ()


def test_defensive_mixture_diagnostics_replay_component_occupancy() -> None:
    dimension = 4
    proposal = freeze_baseline_proposal(
        method="defensive_cem",
        task_id="mixture",
        dimension=dimension,
        training_seed=1,
        training_cost=BaselineCostLedger(training_samples=10, algorithmic_work_units=10.0),
        training_budget_work_units=10.0,
        component_means=(
            tuple(0.0 for _ in range(dimension)),
            (-1.0, 0.5, 0.0, -0.25),
        ),
        component_weights=(0.2, 0.8),
    )
    first = evaluate_baseline_likelihood_diagnostics(proposal, sample_count=2048, seed=3)
    replay = evaluate_baseline_likelihood_diagnostics(proposal, sample_count=2048, seed=3)
    assert first == replay
    assert sum(first.component_counts) == 2048
    assert all(count > 0 for count in first.component_counts)
    assert first.nonfinite_weight_count == 0
    assert torch.isfinite(torch.tensor(first.normalization_mean))
    assert first.normalization_z is not None
    assert abs(first.normalization_z) < 6.0
    assert first.effective_sample_size is not None
    assert 0.0 < first.effective_sample_size <= 2048.0


def test_coupling_flow_roundtrip_is_at_machine_precision() -> None:
    proposal = freeze_baseline_proposal(
        method="flow_is",
        task_id="flow",
        dimension=4,
        training_seed=1,
        training_cost=BaselineCostLedger(screening_samples=10, algorithmic_work_units=10.0),
        training_budget_work_units=10.0,
        location=(0.1, -0.2, 0.3, -0.4),
        flow_split=2,
        flow_scale_matrix=((0.2, -0.1), (0.1, 0.3)),
        flow_scale_bias=(0.1, -0.2),
        flow_shift_matrix=((0.4, 0.1), (-0.2, 0.3)),
        flow_shift_bias=(0.2, -0.1),
        flow_max_log_scale=0.5,
        conditional_integral="baseline_only",
    )
    result = evaluate_coupling_flow_roundtrip(proposal, sample_count=256, seed=7)
    assert result.maximum_reconstruction_error <= 1e-12
    assert result.maximum_log_jacobian_cancellation_error == 0.0
