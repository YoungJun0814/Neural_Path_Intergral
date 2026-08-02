from __future__ import annotations

import copy
from pathlib import Path

from src.path_integral.d1_audit import load_standard_json
from src.path_integral.d1_stage_b_audit import audit_d1_stage_b

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/d1_stage_b_v1.yaml"
RESULT = ROOT / "results/g11_v8_d1_stage_b_v1_2026-08-02.json"


def test_independent_stage_b_audit_accepts_valid_falsification_result() -> None:
    audit = audit_d1_stage_b(
        config_path=CONFIG,
        result=load_standard_json(RESULT),
        root=ROOT,
    )
    assert audit.passed, audit.failures
    assert audit.recomputed_aggregate["stage_b_complete"] is False
    assert audit.recomputed_aggregate["mechanism_pass"] is False


def test_stage_b_audit_detects_aggregate_and_amortization_mutations() -> None:
    mutated = copy.deepcopy(load_standard_json(RESULT))
    mutated["aggregate"]["mechanism_pass"] = True
    mutated["paired_records"][0]["amortized_total_work"]["10"]["dcs"] += 1.0
    audit = audit_d1_stage_b(config_path=CONFIG, result=mutated, root=ROOT)
    assert not audit.passed
    assert "aggregate_recomputation" in audit.failures
    assert "bank_and_amortization" in audit.failures
