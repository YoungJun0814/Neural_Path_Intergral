from __future__ import annotations

from experiments.g11_v8_p5_cell_tuned_cem_proposal import ROOT
from experiments.g11_v8_p5_cell_tuned_cem_proposal_audit import (
    audit_cell_tuned_proposal,
)

CONFIG = ROOT / "configs/g11_v8/p5_cell_tuned_cem_proposal_v2.yaml"
RESULT = (
    ROOT / "results/g11_v8_p5_cell_tuned_cem_proposal_v2_2026-07-31.json"
)


def test_formal_cell_tuned_cem_result_is_exactly_auditable() -> None:
    report = audit_cell_tuned_proposal(CONFIG, RESULT)

    assert report["passed"] is True
    assert len(report["selected_proposal_summary"]) == 2
    assert all(
        entry["requested_to_cap_ratio"] < 0.15
        for entry in report["selected_proposal_summary"]
    )
    assert report["decision"]["proposal_manifest_freeze_authorized"] is True
    assert report["decision"]["new_full_pilot_authorized"] is False
