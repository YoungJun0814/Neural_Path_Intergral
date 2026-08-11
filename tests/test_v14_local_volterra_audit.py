from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.g11_v14_local_volterra_development import load_config
from experiments.g11_v14_local_volterra_qualification import (
    validate_qualification_config,
)
from src.path_integral.v14_local_volterra_audit import audit_v14_local_volterra

ROOT = Path(__file__).resolve().parents[1]
DEV_CONFIG = ROOT / "configs/g11_v14/local_volterra_development_v4.yaml"
DEV_RESULT = ROOT / "results/g11_v14_local_volterra_development_v4_2026-08-11.json"
QUAL_CONFIG = ROOT / "configs/g11_v14/local_volterra_qualification_v1.yaml"
QUAL_RESULT = ROOT / "results/g11_v14_local_volterra_qualification_v1_2026-08-11.json"


def _audit(config_path: Path, result_path: Path, stage: str):
    config, digest = load_config(config_path)
    raw = result_path.read_bytes()
    result = json.loads(raw.decode("utf-8"))
    return audit_v14_local_volterra(
        config=config,
        config_sha256=digest,
        result=result,
        result_bytes=raw,
        expected_stage=stage,
    )


def test_v14_development_and_qualification_artifacts_pass_audit() -> None:
    development = _audit(DEV_CONFIG, DEV_RESULT, "development")
    qualification = _audit(QUAL_CONFIG, QUAL_RESULT, "qualification")
    assert development.passed
    assert qualification.passed
    result = json.loads(QUAL_RESULT.read_text(encoding="utf-8"))
    assert result["aggregate"]["stage_pass"]
    assert result["aggregate"]["descriptive_one_sided_lower_ratio"] > 1.0


def test_v14_audit_rejects_proposal_tampering() -> None:
    config, digest = load_config(DEV_CONFIG)
    result = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(result)
    tampered["records"][0]["candidate"]["proposal"]["component_means"][1][0] += 0.1
    audit = audit_v14_local_volterra(
        config=config,
        config_sha256=digest,
        result=tampered,
        expected_stage="development",
    )
    assert not audit.passed
    assert "proposal_hash" in audit.failures


def test_qualification_is_hash_bound_to_authorized_development() -> None:
    config, _ = load_config(QUAL_CONFIG)
    validate_qualification_config(config, root=ROOT)
    tampered = copy.deepcopy(config)
    tampered["candidate_training"]["target_powers"] = [0.2]
    try:
        validate_qualification_config(tampered, root=ROOT)
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("qualification architecture tampering was accepted")
