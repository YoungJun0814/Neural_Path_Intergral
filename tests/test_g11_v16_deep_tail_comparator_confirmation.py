import math

import torch

from experiments.g11_v16_deep_tail_comparator_confirmation import (
    _aggregate_clusters,
)
from src.path_integral.baseline_framework import BaselineCostLedger


def test_comparator_aggregation_charges_training_and_uses_between_cluster_se() -> None:
    clusters = (
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64),
    )

    def evaluate(index: int) -> tuple[torch.Tensor, BaselineCostLedger]:
        return clusters[index], BaselineCostLedger(algorithmic_work_units=5.0)

    result = _aggregate_clusters(
        method="test",
        training_cost=BaselineCostLedger(algorithmic_work_units=7.0),
        clusters=2,
        evaluate=evaluate,
        reference_estimate=2.0,
        reference_standard_error=0.1,
        query_count=10,
    )
    assert result["estimate"] == 2.0
    assert result["accuracy_z"] == 0.0
    assert result["between_cluster_standard_error"] == 1.0
    assert result["robust_standard_error"] == 1.0
    assert result["total_work_at_primary_query_count"] == 107.0
    assert math.isclose(
        result["work_normalized_variance"],
        result["sample_variance"] * 107.0 / 6,
    )
