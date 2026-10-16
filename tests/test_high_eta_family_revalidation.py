from experiments.post_audit_high_eta_family_revalidation import (
    TARGETS,
    allocation_from_anchor,
    identity,
    jobs,
    seed,
    umbrella_work,
)
from experiments.post_audit_high_eta_family_validation import jobs as old_jobs
from experiments.post_audit_high_eta_family_validation import seed as old_seed


def test_fixed_fresh_retry_count_work_and_namespace_separation():
    grid = jobs()
    streams = [seed(j, s) for j in grid for s in (("path", "label") if j["method"] == "guide-iid" else ("whole",))]
    previous = {old_seed(j, s) for j in old_jobs() for s in (("path", "label") if j["method"] == "guide-iid" else ("whole",))}
    assert len(grid) == len({identity(j) for j in grid}) == 532
    assert len(streams) == len(set(streams))
    assert not set(streams) & previous
    assert 128+sum(4096 if j["method"] == "guide-iid" else umbrella_work(j) for j in grid) == 95998080


def test_failure_anchor_safety_two_selects_frozen_160_not_final_top_up():
    anchor = {"summaries": [{"method": "free-full", "target": t,
                              "summary": {"relative_se": .0161333894838604 if t == "risk-q0" else .015}}
                             for t in TARGETS]}
    assert allocation_from_anchor(anchor) == 160
