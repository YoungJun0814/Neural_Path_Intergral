import math

from experiments.g11_v16_deep_tail_transport_development import (
    _qualify_comparator,
)


def _method(*, estimate: float, variance: float) -> dict:
    return {
        "method": "baseline",
        "estimate": estimate,
        "sample_variance": variance,
        "inferential_units": 100,
        "total_work_at_primary_query_count": 1000.0,
    }


def test_degenerate_missed_event_baseline_cannot_look_efficient() -> None:
    record = _qualify_comparator(
        _method(estimate=0.0, variance=0.0),
        reference_estimate=1e-8,
        reference_standard_error=1e-9,
        maximum_accuracy_z=4.0,
    )
    assert record["qualified"] is False
    assert record["work_normalized_variance"] is None


def test_only_reference_aligned_positive_variance_baseline_is_qualified() -> None:
    aligned = _qualify_comparator(
        _method(estimate=1.1e-8, variance=1e-16),
        reference_estimate=1e-8,
        reference_standard_error=1e-9,
        maximum_accuracy_z=4.0,
    )
    missed = _qualify_comparator(
        _method(estimate=1e-10, variance=1e-20),
        reference_estimate=1e-8,
        reference_standard_error=1e-9,
        maximum_accuracy_z=4.0,
    )
    assert aligned["qualified"] is True
    assert math.isfinite(aligned["work_normalized_variance"])
    assert missed["qualified"] is False
