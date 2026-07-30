from __future__ import annotations

from experiments.g11_v8_p5_barrier_proposal_failure import (
    audit_barrier_proposal_failure,
)
from experiments.g11_v8_p5_barrier_proposal_falsification import ROOT

CONFIG = (
    ROOT
    / "configs/g11_v8/p5_barrier_reference_proposal_falsification_v1.yaml"
)
RESULT = (
    ROOT
    / "results/g11_v8_p5_barrier_proposal_falsification_v1_2026-07-31.json"
)


def test_barrier_proposal_failure_is_reproducible_and_fail_closed() -> None:
    report = audit_barrier_proposal_failure(CONFIG, RESULT)

    assert report["passed"] is True
    assert len(report["failing_method_optima"]) == 2
    assert {
        (entry["cell_id"], entry["method"])
        for entry in report["failing_method_optima"]
    } == {
        ("h0.20-discrete_lower_barrier-p1e-05", "raw_crosscheck"),
        ("h0.05-discrete_lower_barrier-p1e-05", "dcs_reference"),
    }
    assert report["decision"]["existing_candidate_reuse_authorized"] is False
    assert report["decision"]["new_cell_tuned_proposal_required"] is True
