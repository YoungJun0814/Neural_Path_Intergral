from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.path_integral.d1_audit import audit_d1_stage_a, load_standard_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/d1_p7_falsification_stage_a_v2.yaml"
RESULT = ROOT / "results/g11_v8_d1_p7_falsification_stage_a_v2_2026-08-02.json"


def test_independent_d1_stage_a_audit_supersedes_v2_missing_accuracy_gate() -> None:
    result = load_standard_json(RESULT)
    audit = audit_d1_stage_a(config_path=CONFIG, result=result, root=ROOT)
    assert not audit.passed
    assert "aggregate_recomputation" in audit.failures
    assert audit.recomputed_aggregate["primary_accuracy_pass"] is False


def test_d1_audit_detects_mutated_gate_and_cost() -> None:
    result = load_standard_json(RESULT)
    mutated = copy.deepcopy(result)
    mutated["aggregate"]["mechanism_pass"] = False
    mutated["external_records"][0]["total_algorithmic_work_units_including_diagnostic"] += 1.0
    audit = audit_d1_stage_a(config_path=CONFIG, result=mutated, root=ROOT)
    assert not audit.passed
    assert "aggregate_recomputation" in audit.failures
    assert "cost_conservation" in audit.failures


def test_standard_json_loader_rejects_infinity(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"valid": True})[:-1] + ', "bad": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard"):
        load_standard_json(path)
