from experiments.post_audit_high_eta_family_validation import (
    TARGETS,
    decision,
    identity,
    jobs,
    seed,
)
from experiments.post_audit_umbrella_reference_pilot import (
    expected_work,
)
from experiments.post_audit_umbrella_reference_pilot import (
    identity as pilot_identity,
)
from experiments.post_audit_umbrella_reference_pilot import (
    seed as pilot_seed,
)


def test_fixed_five_q_validation_work_and_disjoint_streams():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == 441
    work = 128+sum(4096 if j["method"] == "guide-iid" else expected_work(j) for j in grid)
    assert work == 49685632
    streams = [seed(j, s) for j in grid for s in (("path", "label") if j["method"] == "guide-iid" else ("whole",))]
    assert len(streams) == len(set(streams))
    for j in grid:
        if j["method"] != "guide-iid":
            assert seed(j, "whole") != pilot_seed(pilot_identity(j))


def test_pass_remains_scoped_and_one_failed_risk_blocks_all_five_q():
    summaries = [{"method": m, "target": target,
                  "summary": {"log_mean": -10., "relative_se": .005, "maximum_fraction": .001},
                  "sensitivity": {"maximum_leave_one_out_relative_shift": .001, "between_unit_relative_se": .005}}
                 for m in ("guide-iid", "free-full", "envelope") for target in TARGETS]
    good = decision(summaries)
    assert good["high_eta_five_q_development_reference_pass"]
    assert len(good["precision_checks"]) == 18
    assert len(good["agreements"]) == 16
    assert not any(good[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    summaries[-1]["summary"]["relative_se"] = .03
    assert not decision(summaries)["high_eta_five_q_development_reference_pass"]
