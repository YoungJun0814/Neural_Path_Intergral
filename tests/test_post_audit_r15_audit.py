"""Adversarial arithmetic checks for the R1.5 development audit."""

from __future__ import annotations

import math

import pytest

from experiments.post_audit_r15_audit import _cluster_estimate, _smc, _without_timing


def test_smc_rejects_stale_reported_precision() -> None:
    logs = [math.log(0.2), math.log(0.4), math.log(0.6)]
    record = {
        "log_replicate_estimates": logs,
        "mean": 0.4,
        "standard_error": 0.2 / math.sqrt(3),
        "relative_se": (0.2 / math.sqrt(3)) / 0.4,
    }
    _smc(record, "toy")
    record["relative_se"] = 0.001
    with pytest.raises(ValueError, match="relative SE"):
        _smc(record, "toy")


def test_cluster_audit_rejects_optimistic_rse() -> None:
    means = [0.1, 0.2, 0.4, 0.5]
    between_rse = 0.0
    record = {
        "log_mean": math.log(sum(means) / len(means)),
        "between_cluster_relative_se": between_rse,
        "iid_relative_se": 0.01,
        "relative_se": 0.01,
    }
    with pytest.raises(ValueError, match="cluster RSE"):
        _cluster_estimate(record, "toy", [math.log(x) for x in means])


def test_replay_excludes_timing_derived_ratio_but_not_accuracy_gate() -> None:
    payload = {
        "total_wall_seconds": 10.0,
        "fixed_precision_total_wall_ratio_smc_over_is": 1.3,
        "both_qualified": False,
        "relative_se": 0.2,
    }
    assert _without_timing(payload) == {
        "both_qualified": False,
        "relative_se": 0.2,
    }
