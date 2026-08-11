from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from experiments.g11_v13_structured_ecrpt_development import load_config
from src.path_integral.v13_development_audit import audit_v13_development

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v13/structured_ecrpt_development_v1.yaml"
RESULT = ROOT / "results/g11_v13_structured_ecrpt_development_v1_2026-08-11.json"


def test_committed_v13_development_result_passes_independent_audit() -> None:
    config, digest = load_config(CONFIG)
    raw = RESULT.read_bytes()
    result = json.loads(raw.decode("utf-8"))
    audit = audit_v13_development(
        config=config, config_sha256=digest, result=result, result_bytes=raw
    )
    assert audit.passed
    assert not result["decision"]["qualification_authorized"]
    assert not result["aggregate"]["stage_pass"]


def test_v13_audit_detects_proposal_and_smc_tampering() -> None:
    config, digest = load_config(CONFIG)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(result)
    tampered["records"][0]["candidate"]["proposal"]["layers"][0]["scale_bias"][0] += 0.01
    tampered["records"][1]["candidate"]["smc"]["final_beta"] = 0.9
    audit = audit_v13_development(config=config, config_sha256=digest, result=tampered)
    assert not audit.passed
    assert "proposal_hash" in audit.failures
    assert "smc_contract" in audit.failures


def test_qualification_manifest_is_fail_closed_and_hash_bound() -> None:
    manifest = yaml.safe_load(
        (ROOT / "configs/g11_v13/structured_ecrpt_qualification_blocked_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "blocked"
    assert not manifest["authorization_observed"]
    assert not manifest["qualification_executed"]
    assert manifest["source_development"]["sha256"] == (
        "65dae19f013ec18dd4945efb95df899fa804842b97d9ac2a5b7980eb537f66fd"
    )
