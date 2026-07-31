from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.g11_v8_p5_reference_allocation_failure import (
    audit_failure_evidence,
)
from experiments.g11_v8_p5_sharded_reference_common import ROOT
from src.path_integral.reference_protocol import canonical_sha256

CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
PACKAGE = ROOT / "results/g11_v8_p5_reference_pilot_package_v3_2026-07-31.json"
RECEIPT = (
    ROOT / "results/g11_v8_p5_reference_allocation_failure_v3_2026-07-31.json"
)
METHOD_ROLE_CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v5.yaml"
METHOD_ROLE_PACKAGE = (
    ROOT / "results/g11_v8_p5_reference_pilot_package_v5_2026-07-31.json"
)
METHOD_ROLE_RECEIPT = (
    ROOT / "results/g11_v8_p5_reference_allocation_failure_v5_2026-07-31.json"
)


def test_allocation_failure_is_exactly_reproducible_and_fail_closed() -> None:
    report = audit_failure_evidence(CONFIG, PACKAGE, RECEIPT)

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["decision"]["status"] == (
        "r2_development_allocation_resource_failure"
    )
    assert report["decision"]["final_execution_authorized"] is False
    assert report["decision"]["new_reference_design_required"] is True


def test_failure_audit_rejects_pilot_statistic_tampering(tmp_path: Path) -> None:
    package = json.loads(PACKAGE.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(package)
    mutated["shards"][0]["artifact"]["contribution"]["mean"] += 1.0
    changed_receipt = copy.deepcopy(receipt)
    changed_receipt["pilot_package_sha256"] = canonical_sha256(mutated)
    package_path = tmp_path / "package.json"
    receipt_path = tmp_path / "receipt.json"
    package_path.write_text(json.dumps(mutated), encoding="utf-8")
    receipt_path.write_text(json.dumps(changed_receipt), encoding="utf-8")

    report = audit_failure_evidence(CONFIG, package_path, receipt_path)
    assert report["passed"] is False
    assert "allocation_exactly_reconstructed" in report["failures"]


def test_method_role_failure_reconstructs_unequal_targets_and_caps_if_present() -> None:
    if not METHOD_ROLE_PACKAGE.exists() or not METHOD_ROLE_RECEIPT.exists():
        return
    report = audit_failure_evidence(
        METHOD_ROLE_CONFIG,
        METHOD_ROLE_PACKAGE,
        METHOD_ROLE_RECEIPT,
    )
    assert report["passed"] is True
    assert report["decision"]["final_execution_authorized"] is False
