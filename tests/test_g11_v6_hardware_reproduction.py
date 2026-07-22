"""Cross-environment V6 reproduction-audit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.g11_v6_hardware_reproduction import _load_config, run
from src.path_integral import SeedKey, SeedLedger

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "hardware_reproduction_development.yaml"


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(schema: str, *, protocol: str, os_name: str) -> dict:
    ledger = SeedLedger()
    ledger.allocate(SeedKey(protocol, "final", "cell", "method", 0, 0, "proposal"))
    return {
        "schema": schema,
        "manifest_sha256": "1" * 64,
        "reference_artifact_sha256": "2" * 64,
        "source_commit": "3" * 40,
        "environment": {"os": os_name},
        "seed_ledger": ledger.to_dict(),
    }


def _confirmation(baseline_hash: str, policy_hash: str, effect: float) -> dict:
    return {
        "schema": "npi.g11.v6-confirmatory.v1",
        "result_artifact_sha256": {
            "baseline": baseline_hash,
            "policy": policy_hash,
        },
        "primary_efficiency": {
            "mean_log_ratio": effect,
            "standard_error": 0.05,
        },
        "scientific_gates_passed": True,
        "confirmation_passed": False,
    }


def test_hardware_reproduction_config_is_strict() -> None:
    config, digest = _load_config(CONFIG)
    assert config["statistics"]["maximum_effect_z_score"] == 3.0
    assert len(digest) == 64


def test_hardware_reproduction_checks_platform_effect_and_seed_separation(
    tmp_path: Path,
) -> None:
    canonical_baseline = _source(
        "npi.g11.v6-baseline-qualification.v1",
        protocol="canonical-baseline",
        os_name="Windows-11",
    )
    canonical_policy = _source(
        "npi.g11.v6-routed-policy.v1",
        protocol="canonical-policy",
        os_name="Windows-11",
    )
    reproduction_baseline = _source(
        "npi.g11.v6-baseline-qualification.v1",
        protocol="linux-baseline",
        os_name="Linux-6.8",
    )
    reproduction_policy = _source(
        "npi.g11.v6-routed-policy.v1",
        protocol="linux-policy",
        os_name="Linux-6.8",
    )
    paths = {
        "cb": tmp_path / "canonical-baseline.json",
        "cp": tmp_path / "canonical-policy.json",
        "rb": tmp_path / "reproduction-baseline.json",
        "rp": tmp_path / "reproduction-policy.json",
    }
    hashes = {
        "cb": _write(paths["cb"], canonical_baseline),
        "cp": _write(paths["cp"], canonical_policy),
        "rb": _write(paths["rb"], reproduction_baseline),
        "rp": _write(paths["rp"], reproduction_policy),
    }
    canonical_confirmation = _confirmation(hashes["cb"], hashes["cp"], 0.50)
    reproduction_confirmation = _confirmation(hashes["rb"], hashes["rp"], 0.55)
    canonical_confirmation_path = tmp_path / "canonical-confirmation.json"
    reproduction_confirmation_path = tmp_path / "reproduction-confirmation.json"
    _write(canonical_confirmation_path, canonical_confirmation)
    _write(reproduction_confirmation_path, reproduction_confirmation)
    result = run(
        CONFIG,
        canonical_confirmation_path,
        reproduction_confirmation_path,
        paths["cb"],
        paths["cp"],
        paths["rb"],
        paths["rp"],
    )
    assert result["gates"]["confirmation_source_hashes"]
    assert result["gates"]["disjoint_seed_streams"]
    assert result["gates"]["linux_reproduction"]
    assert result["gates"]["different_operating_system"]
    assert result["gates"]["effect_consistent_within_z_limit"]
    assert result["reproduction_gates_passed"]
    assert not result["hardware_reproduction_passed"]
