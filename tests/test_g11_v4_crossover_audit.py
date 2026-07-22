"""Independent-audit tests for the frozen G11 V4 crossover qualification."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.g11_v4_crossover_audit import (
    _cell_failures,
    _expected_cells,
    _load_config,
    run,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v4_crossover_qualification.yaml"
RESULT = ROOT / "results" / "g11_v4_crossover_qualification_v1_2026-07-22.json"


def test_committed_v4_result_passes_independent_audit() -> None:
    audit = run(CONFIG, RESULT)
    assert audit["integrity_failures"] == []
    assert audit["integrity_passed"] is True
    assert audit["qualification_gate_passed"] is True
    assert audit["independent_summary"]["cell_count"] == 27
    assert audit["independent_summary"]["run_count"] == 135


def test_v4_audit_treats_manifest_separators_portably(tmp_path: Path) -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    for item in result["input_artifacts"]:
        item["path"] = item["path"].replace("\\", "/")
    portable_result = tmp_path / "portable-result.json"
    portable_result.write_text(json.dumps(result), encoding="utf-8")
    audit = run(CONFIG, portable_result)
    assert "input artifact manifest mismatch" not in audit["integrity_failures"]


def test_v4_audit_detects_a_tampered_crossover_decision() -> None:
    config, _config_hash = _load_config(CONFIG)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    cell = copy.deepcopy(result["cells"][0])
    contract = _expected_cells(config)[(cell["regime"], cell["task"])]
    cell["runs"][0]["decisions"]["0.20"]["dcs"]["optimal_start_level"] = 99
    failures = _cell_failures(config, cell, contract)
    assert any("independent decision mismatch" in item for item in failures)
