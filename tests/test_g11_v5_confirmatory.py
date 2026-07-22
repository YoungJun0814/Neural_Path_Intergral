"""Fail-closed configuration and aggregate tests for the V5 G6 runner."""

from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.g11_v5_confirmatory import (
    _aggregate_records,
    _cell_reference,
    _qualification_gates,
)


def _record(cluster: int, estimate: float) -> dict:
    return {
        "cell_id": f"h012:terminal:cluster-{cluster}",
        "model_id": "h012",
        "task_name": "terminal",
        "reference": {"probability": 0.2, "standard_error": 0.001},
        "selection": {"stopped": True, "frozen_decision": {"selected_candidate": "start_4"}},
        "preparation": {"preparation_hash": f"{cluster + 1:064x}"},
        "seed_role_audit": {"selection_final_disjoint": True},
        "result": {
            "complete": True,
            "resource_censored": False,
            "design_target_attained": True,
            "empirical_target_attained": True,
            "requested_relative_sampling_rmse": 0.1,
            "estimate": estimate,
            "empirical_sampling_variance": 0.0001,
            "bounded_confidence_interval": [0.0, 1.0],
            "selected_candidate": "start_4",
            "work": {"entries": [{"work_units": 100.0 + cluster}]},
        },
    }


def _thresholds() -> dict:
    return {
        "minimum_empirical_target_attainment": 0.8,
        "maximum_relative_rmse_ratio": 1.1,
        "minimum_combined_asymptotic_coverage": 0.8,
    }


def test_formal_reference_is_model_and_task_specific() -> None:
    config = {
        "references": {
            "h012": {"terminal": {"probability": 0.2, "standard_error": 0.001}}
        }
    }
    actual = _cell_reference(
        config,
        model_id="h012",
        task_name="terminal",
        task_spec={},
        smoke=False,
    )
    assert actual == {"probability": 0.2, "standard_error": 0.001}


def test_formal_reference_rejects_legacy_task_fallback() -> None:
    task = {"reference_probability": 0.2, "reference_standard_error": 0.001}
    with pytest.raises(ValueError, match="missing qualified reference"):
        _cell_reference({}, model_id="h012", task_name="terminal", task_spec=task, smoke=False)


def test_aggregate_and_gates_accept_complete_two_cluster_cell() -> None:
    records = [_record(0, 0.19), _record(1, 0.21)]
    aggregates = _aggregate_records(records)
    gates = _qualification_gates(
        records,
        aggregates,
        expected_records=2,
        expected_cells=1,
        thresholds=_thresholds(),
    )
    assert len(aggregates) == 1
    assert aggregates[0]["empirical_relative_rmse_against_reference"] == pytest.approx(0.05)
    assert all(gates.values())


def test_gates_reject_censoring_seed_overlap_and_incomplete_matrix() -> None:
    record = deepcopy(_record(0, 0.2))
    record["result"]["complete"] = False
    record["result"]["resource_censored"] = True
    record["seed_role_audit"]["selection_final_disjoint"] = False
    aggregates = _aggregate_records([record])
    gates = _qualification_gates(
        [record],
        aggregates,
        expected_records=2,
        expected_cells=1,
        thresholds=_thresholds(),
    )
    assert not gates["complete_cluster_matrix"]
    assert not gates["no_resource_censoring"]
    assert not gates["all_runs_complete"]
    assert not gates["no_final_samples_reused_from_selection"]
