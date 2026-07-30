from __future__ import annotations

from pathlib import Path

from experiments.g11_v8_p5_weight_optimized_proposal_audit import (
    audit_weight_optimized_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_weight_optimized_proposal_v1.yaml"
RESULT = (
    ROOT
    / "results/g11_v8_p5_weight_optimized_proposal_v1_2026-07-31.json"
)


def test_weight_optimization_partial_falsification_is_fail_closed() -> None:
    report = audit_weight_optimized_result(CONFIG, RESULT)
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["decision"]["method_role_precision_redesign_required"] is True
    assert report["decision"]["partial_candidates_promoted"] is False
    assert report["decision"]["new_formal_pilot_authorized"] is False
    assert report["decision"]["final_execution_authorized"] is False
