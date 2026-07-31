from __future__ import annotations

from pathlib import Path

from experiments.g11_v8_p5_method_role_proposal_audit import (
    audit_method_role_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_method_role_proposal_v1.yaml"
RESULT = ROOT / "results/g11_v8_p5_method_role_proposal_v1_2026-07-31.json"


def test_method_role_partial_falsification_is_fail_closed() -> None:
    report = audit_method_role_result(CONFIG, RESULT)
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["decision"]["reference_only_resource_escalation_required"] is True
    assert report["decision"]["partial_candidates_promoted"] is False
    assert report["decision"]["new_formal_pilot_authorized"] is False
    assert report["decision"]["final_execution_authorized"] is False
