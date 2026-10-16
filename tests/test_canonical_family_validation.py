import copy

from experiments.post_audit_canonical_family_validation import (
    TARGETS,
    decision,
    expected_work,
    identity,
    jobs,
    seed,
)
from experiments.post_audit_canonical_gradient_recovery import jobs as gradient_pilot_jobs
from experiments.post_audit_canonical_gradient_recovery import seed as gradient_pilot_seed


def test_fresh_fixed_canonical_reference_grid_and_forward_backward_budgets():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == 580
    streams = [seed(j, s) for j in grid for s in (("path", "label") if j["method"] == "guide-iid" else ("whole",))]
    assert len(streams) == len(set(streams))
    assert not set(streams) & {gradient_pilot_seed(j) for j in gradient_pilot_jobs()}
    assert 128+sum(expected_work(j)[0] for j in grid) == 116953216
    assert sum(expected_work(j)[1] for j in grid) == 69298176


def test_canonical_actual_pass_scope_never_opens_p2_or_training():
    rows = [{"method": method, "target": target,
             "summary": {"log_mean": -10., "relative_se": .001, "maximum_fraction": .001},
             "sensitivity": {"maximum_leave_one_out_relative_shift": .001, "between_unit_relative_se": .001}}
            for method in ("guide-iid", "free-full", "envelope") for target in TARGETS]
    d = decision(rows)
    assert d["canonical_five_q_development_reference_pass"]
    assert "high_eta_five_q_development_reference_pass" not in d
    assert not any(d[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    broken = copy.deepcopy(rows)
    broken[8]["summary"]["relative_se"] = .01501
    assert not decision(broken)["canonical_five_q_development_reference_pass"]
