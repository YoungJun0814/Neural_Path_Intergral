from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from experiments.g11_v8_b1_baseline_audit import audit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/g11_v8/b1_baseline_implementation_v1.yaml"
RESULT_PATH = ROOT / "results/g11_v8_b1_baseline_implementation_v1_2026-08-02.json"


def _inputs():
    raw = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    return config, hashlib.sha256(raw).hexdigest(), result


def test_independent_b1_auditor_accepts_frozen_implementation_artifact() -> None:
    config, digest, result = _inputs()
    report = audit(config, digest, result)
    assert report["passed"]
    assert report["failures"] == []
    assert all(report["checks"].values())
    assert report["decision"]["b1_implementation_complete"]
    assert report["decision"]["d1_falsification_authorized"]
    assert not report["decision"]["performance_claim_authorized"]


def test_independent_b1_auditor_rejects_likelihood_corruption() -> None:
    config, digest, result = _inputs()
    corrupted = copy.deepcopy(result)
    corrupted["records"][0]["artifact"]["proposal"]["self_normalized"] = True
    report = audit(config, digest, corrupted)
    assert not report["passed"]
    assert "method_family_exact" in report["failures"]
    assert "proposal_hashes_independently_recomputed" in report["failures"]


def test_independent_b1_auditor_rejects_seed_collision() -> None:
    config, digest, result = _inputs()
    corrupted = copy.deepcopy(result)
    first = corrupted["records"][0]["artifact"]["proposal"]["training_seed"]
    corrupted["records"][1]["artifact"]["proposal"]["training_seed"] = first
    report = audit(config, digest, corrupted)
    assert not report["passed"]
    assert "all_seeds_global_disjoint" in report["failures"]


def test_independent_b1_auditor_rejects_missing_conditional_cost() -> None:
    config, digest, result = _inputs()
    corrupted = copy.deepcopy(result)
    record = next(item for item in corrupted["records"] if item["method"] == "conditional_rbergomi")
    record["artifact"]["estimate"]["final_cost"]["cdf_calls"] = 0
    report = audit(config, digest, corrupted)
    assert not report["passed"]
    assert "method_specific_contracts_pass" in report["failures"]


def test_independent_b1_auditor_rejects_performance_authorization() -> None:
    config, digest, result = _inputs()
    corrupted = copy.deepcopy(result)
    corrupted["decision"]["performance_claim_authorized"] = True
    report = audit(config, digest, corrupted)
    assert not report["passed"]
    assert "executor_decision_fail_closed" in report["failures"]
