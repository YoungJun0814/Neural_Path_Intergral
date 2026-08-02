from __future__ import annotations

import math

import pytest

from src.path_integral.v9_terminal_protocol import (
    _cluster_geometric_summary,
    geometric_mean,
    work_to_target,
)


def test_work_to_target_charges_integer_units_and_amortizes_only_one_time_work() -> None:
    result = work_to_target(
        unit_variance=0.04,
        unit_work=3.0,
        one_time_work=100.0,
        query_count=10,
        target_estimator_variance=0.01,
    )
    assert result["required_units"] == 4
    assert result["evaluation_work"] == 12.0
    assert result["amortized_one_time_work"] == 10.0
    assert result["total_work"] == 22.0


def test_work_to_target_retains_two_units_and_rejects_invalid_values() -> None:
    result = work_to_target(
        unit_variance=0.0,
        unit_work=2.0,
        one_time_work=0.0,
        query_count=1,
        target_estimator_variance=1.0,
    )
    assert result["required_units"] == 2
    with pytest.raises(ValueError):
        work_to_target(
            unit_variance=-1.0,
            unit_work=2.0,
            one_time_work=0.0,
            query_count=1,
            target_estimator_variance=1.0,
        )


def test_cluster_summary_uses_clusters_not_cell_records_as_replicates() -> None:
    values = [(0, 2.0), (0, 8.0), (1, 4.0), (1, 4.0), (2, 8.0), (2, 2.0)]
    summary = _cluster_geometric_summary(values, confidence=0.95, multiplicity=1)
    assert summary["record_count"] == 6
    assert summary["cluster_count"] == 3
    assert summary["cluster_geometric_ratios"] == pytest.approx([4.0, 4.0, 4.0])
    assert summary["geometric_ratio"] == pytest.approx(4.0)
    assert summary["lower_confidence_bound"] == pytest.approx(4.0)


def test_geometric_mean_is_strict() -> None:
    assert geometric_mean([2.0, 8.0]) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        geometric_mean([1.0, math.inf])
