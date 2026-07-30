from __future__ import annotations

from experiments.g11_v8_p5_all_failed_cells_cem_audit import (
    audit_all_failed_cells_cem,
)
from experiments.g11_v8_p5_cell_tuned_cem_proposal import ROOT

CONFIG = ROOT / "configs/g11_v8/p5_all_failed_cells_cem_proposal_v3.yaml"
RESULT = (
    ROOT / "results/g11_v8_p5_all_failed_cells_cem_proposal_v3_2026-07-31.json"
)


def test_v3_partial_success_is_exactly_audited_and_fail_closed() -> None:
    report = audit_all_failed_cells_cem(CONFIG, RESULT)

    assert report["passed"] is True
    assert report["selected_requirement_count"] == 9
    assert report["required_requirement_count"] == 11
    assert {
        (item["cell_id"], item["method"])
        for item in report["missing_requirements"]
    } == {
        ("h0.05-terminal_left_tail-p1e-05", "raw_crosscheck"),
        ("h0.05-discrete_lower_barrier-p1e-05", "raw_crosscheck"),
    }
    assert report["decision"]["proposal_manifest_freeze_authorized"] is False
