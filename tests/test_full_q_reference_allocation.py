import math

import torch

from experiments.post_audit_full_q_reference_allocation import (
    allocation_decision,
    expected_work,
    identity,
    jobs,
    seed,
    selections,
    summarize,
)
from src.path_integral.structural_v2_reference import log_moments


def rows():
    return [{"cell": {"id": c}, "parent_training_rep": rep}
            for c in ("canonical", "high-eta") for rep in range(5)]


def test_fixed_full_q_grid_seeds_and_accounting():
    grid = jobs(rows())
    names = [identity(j) for j in grid]
    streams = [seed(i, s) for i in names for s in ("path", "label", "whole")]
    assert len(grid) == len(set(names)) == 816
    assert len(streams) == len(set(streams))
    assert 256+sum(expected_work(j) for j in grid) == 71876864
    assert sum(j["target"] == "mu" for j in grid) == 16


def test_selection_uses_anchor_cost_not_new_pilot_outcomes():
    anchor = {"summaries": [{"cell": c, "target": t, "method": m,
                              "observed_cv2_wall_per_whole": score}
                             for c in ("a", "b") for t in ("mu", "risk")
                             for m, score in (("full-pcn", 2.), ("multiscale-block-pcn", 1.))]}
    assert selections(anchor) == {c: {t: "multiscale-block-pcn" for t in ("mu", "risk")} for c in ("a", "b")}


def test_mu_pooling_and_per_q_risk_remain_separate():
    records = []
    for j in jobs(rows()):
        if j["method"] == "guide-iid":
            moments = {"mu": log_moments(torch.zeros(2, dtype=torch.float64)),
                       "risk": log_moments(torch.full((2,), float(j["parent_training_rep"]), dtype=torch.float64))}
            records.append({**j, "moments": moments, "wall": .01})
        else:
            records.append({**j, "log_normalizer": 0., "wall": .01})
    summaries = summarize(records)
    assert len(summaries) == 34
    for s in summaries:
        if s["method"] == "guide-iid":
            assert s["summary"]["count"] == (640 if s["target"] == "mu" else 128)
            expected = 0. if s["target"] == "mu" else s["parent_training_rep"]
            assert abs(s["summary"]["log_mean"]-expected) < 1e-12


def test_allocation_never_qualifies_and_preserves_production_cap():
    s = {"cell": "a", "target": "risk", "parent_training_rep": 0, "method": "guide-free-smc",
         "allocation": {"status": "allocated"}, "forecast_work": 160000001, "forecast_wall": 1.}
    result = allocation_decision([s])
    assert result["status"] == "allocation_blocked"
    assert "original_production_work_cap" in result["failure_reasons"]
    assert not any(result[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    assert math.isfinite(result["forecast_wall_with_safety2"])
