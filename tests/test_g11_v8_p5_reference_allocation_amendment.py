from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.g11_v8_p5_reference_allocation_amendment import (
    audit_amendment,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v6.yaml"
PACKAGE = ROOT / "results/g11_v8_p5_reference_pilot_package_v6_2026-07-31.json"
FAILURE = (
    ROOT / "results/g11_v8_p5_reference_allocation_failure_v6_2026-07-31.json"
)
AMENDMENT = (
    ROOT
    / "results/g11_v8_p5_reference_allocation_amendment_receipt_v1_2026-07-31.json"
)


def test_exact_count_cap_amendment_is_reproducible_and_fail_closed() -> None:
    report = audit_amendment(CONFIG, PACKAGE, FAILURE, AMENDMENT)
    assert report["passed"] is True
    assert report["decision"]["statistical_allocation_complete"] is True
    assert report["decision"]["hardware_execution_authorized"] is False


def test_amendment_rejects_requested_count_tampering(tmp_path: Path) -> None:
    receipt = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(receipt)
    mutated["trigger_requested_final_samples"] -= 1
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    report = audit_amendment(CONFIG, PACKAGE, FAILURE, path)
    assert report["passed"] is False
    assert "minimal_power_of_two_cap_rule_exact" in report["failures"]
