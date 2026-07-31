from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.g11_v8_p5_reference_resource_escalated_manifest import (
    audit_manifest,
    load_manifest_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_reference_resource_escalated_manifest_v1.yaml"
FROZEN_MANIFEST = (
    ROOT / "results/g11_v8_p5_reference_proposal_manifest_v2_2026-07-31.json"
)


def test_resource_escalated_config_loads() -> None:
    config, digest = load_manifest_config(CONFIG)
    assert config["reference_protocol"]["primary_method"] == "dcs_reference"
    assert config["method_role_precision"]["relative_standard_error_targets"] == {
        "dcs_reference": 0.02,
        "raw_crosscheck": 0.05,
    }
    assert len(digest) == 64


def test_frozen_resource_escalated_manifest_audits_if_present() -> None:
    if not FROZEN_MANIFEST.exists():
        pytest.skip("frozen resource-escalated manifest has not been built yet")
    report = audit_manifest(CONFIG, FROZEN_MANIFEST)
    assert report["passed"] is True
    assert report["source_counts"] == {
        "prior_v4_resource_feasible": 40,
        "retained_frozen_pass": 4,
        "resource_only_after_falsification": 4,
    }


def test_audit_rejects_unreported_resource_only_success(
    tmp_path: Path,
) -> None:
    if not FROZEN_MANIFEST.exists():
        pytest.skip("frozen resource-escalated manifest has not been built yet")
    payload = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    target = next(
        entry
        for entry in mutated["entries"]
        if entry["source_kind"] == "resource_only_after_falsification"
    )
    target["development_gates"]["contribution_concentration_pass"] = True
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    report = audit_manifest(CONFIG, path)
    assert report["passed"] is False
    assert "resource_only_failure_disclosed" in report["failures"]
