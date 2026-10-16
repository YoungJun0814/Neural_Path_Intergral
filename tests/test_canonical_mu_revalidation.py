import copy

import pytest

from experiments.post_audit_canonical_family_validation import decision
from experiments.post_audit_canonical_family_validation import jobs as old_jobs
from experiments.post_audit_canonical_family_validation import seed as old_seed
from experiments.post_audit_canonical_mu_revalidation import COUNT, WORK, seed, summarize


def test_mu_fresh_fixed_allocation_and_disjoint_seeds():
    assert COUNT == 215 and COUNT*WORK == 54819840
    fresh = {seed(r) for r in range(COUNT)}
    old = {old_seed(j, s) for j in old_jobs() for s in (("path", "label") if j["method"] == "guide-iid" else ("whole",))}
    assert len(fresh) == COUNT and fresh.isdisjoint(old)


def test_component_inventory_cannot_rewrite_old_failure_or_open_training():
    rows = [{"method": m, "target": target,
             "summary": {"log_mean": -10., "relative_se": .001, "maximum_fraction": .001},
             "sensitivity": {"maximum_leave_one_out_relative_shift": .001, "between_unit_relative_se": .001}}
            for m in ("guide-iid", "free-full", "envelope") for target in ("mu", "risk-q0", "risk-q1", "risk-q2", "risk-q3", "risk-q4")]
    next(r for r in rows if r["method"] == "free-full" and r["target"] == "mu")["summary"]["relative_se"] = .0151
    anchor = {"summaries": rows, "decision": decision(rows)}
    original = copy.deepcopy(anchor)
    records = [{"rep": r, "log_mu": -10.} for r in range(COUNT)]
    out = summarize(records, anchor)
    assert out["fresh_mu_development_reference_pass"] and out["canonical_componentwise_development_recovery_pass"]
    assert not out["old_validation_pass"] and anchor == original
    assert not any(out[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    with pytest.raises(ValueError):
        summarize(records[:-1], anchor)
