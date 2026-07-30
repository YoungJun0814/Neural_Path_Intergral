from __future__ import annotations

from pathlib import Path

from experiments.g11_v8_p5_production_scale_proposal_audit import (
    audit_production_scale_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_production_scale_proposal_v2.yaml"
RESULT = (
    ROOT
    / "results/g11_v8_p5_production_scale_proposal_v2_2026-07-31.json"
)
PERMUTED_CONFIG = (
    ROOT / "configs/g11_v8/p5_production_scale_proposal_v3.yaml"
)
PERMUTED_RESULT = (
    ROOT
    / "results/g11_v8_p5_production_scale_proposal_v3_2026-07-31.json"
)


def test_production_scale_falsification_audit_is_fail_closed() -> None:
    report = audit_production_scale_result(CONFIG, RESULT)
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["decision"]["permuted_block_protocol_required"] is True
    assert report["decision"]["candidate_promoted"] is False
    assert report["decision"]["new_formal_pilot_authorized"] is False
    assert report["decision"]["final_execution_authorized"] is False


def test_permuted_block_partial_falsification_is_exact() -> None:
    report = audit_production_scale_result(
        PERMUTED_CONFIG, PERMUTED_RESULT
    )
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["decision"]["v3_training_namespace_burned"] is True
    assert report["decision"]["v3_validation_namespace_burned"] is True
    assert report["decision"]["permuted_block_protocol_required"] is False
    assert report["decision"]["proposal_weight_optimization_required"] is True
    assert report["decision"]["candidate_promoted"] is False
    assert report["decision"]["new_formal_pilot_authorized"] is False
