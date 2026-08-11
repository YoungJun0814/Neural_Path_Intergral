from __future__ import annotations

import copy

import pytest

from src.path_integral.ecrpt_microstudy_audit import (
    recompute_summary,
    semantic_equal,
)


def test_recompute_summary_recovers_unbiased_ordinary_mean_statistics() -> None:
    summary = {
        "count": 4,
        "sum": 10.0,
        "sum_squares": 30.0,
        "estimate": 2.5,
        "variance": 5.0 / 3.0,
        "standard_error": (5.0 / 12.0) ** 0.5,
    }
    rebuilt = recompute_summary(summary)
    assert semantic_equal(rebuilt["estimate"], summary["estimate"])
    assert semantic_equal(rebuilt["variance"], summary["variance"])
    assert semantic_equal(rebuilt["standard_error"], summary["standard_error"])


def test_recompute_summary_rejects_impossible_mutation() -> None:
    summary = {"count": 4, "sum": 10.0, "sum_squares": 30.0}
    mutated = copy.deepcopy(summary)
    mutated["sum_squares"] = 1.0
    with pytest.raises(ValueError, match="negative variance"):
        recompute_summary(mutated)
