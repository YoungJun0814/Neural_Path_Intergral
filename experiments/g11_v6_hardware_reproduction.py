"""Cross-environment V6 reproduction audit with independent random streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.seed_ledger import SeedLedger

_SCHEMA = "npi.g11.v6-hardware-reproduction.config.v1"
_EXPECTED_FIELDS = {
    "canonical_confirmation",
    "reproduction_baseline_config",
    "reproduction_policy_config",
    "manifest",
    "reference",
    "power",
    "audit_config",
}


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 hardware-reproduction config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "frozen",
        "expected_sha256",
        "statistics",
        "requirements",
    }:
        raise ValueError("malformed V6 hardware-reproduction config fields")
    hashes = payload["expected_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != _EXPECTED_FIELDS:
        raise ValueError("hardware-reproduction hashes are malformed")
    if payload["frozen"]:
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        ):
            raise ValueError("frozen reproduction requires every protocol hash")
    return payload, hashlib.sha256(raw).hexdigest()


def _load(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema} artifact")
    return payload, hashlib.sha256(raw).hexdigest()


def _source_hashes_match(
    confirmation: dict[str, Any],
    baseline_hash: str,
    policy_hash: str,
) -> bool:
    result_hashes = confirmation.get("result_artifact_sha256", {})
    return (
        result_hashes.get("baseline") == baseline_hash
        and result_hashes.get("policy") == policy_hash
    )


def _seed_sets(*artifacts: dict[str, Any]) -> tuple[set[int], set[tuple[Any, ...]]]:
    seeds: set[int] = set()
    keys: set[tuple[Any, ...]] = set()
    for artifact in artifacts:
        ledger = SeedLedger.from_dict(artifact["seed_ledger"])
        for record in ledger.records:
            key = record.key
            identity = (
                key.protocol,
                key.role,
                key.regime,
                key.task,
                key.level,
                key.replicate,
                key.stream,
            )
            if record.seed in seeds or identity in keys:
                raise ValueError("duplicate stream inside one hardware run")
            seeds.add(record.seed)
            keys.add(identity)
    return seeds, keys


def _effect_consistency(
    canonical: dict[str, Any], reproduction: dict[str, Any]
) -> tuple[float, bool]:
    canonical_effect = canonical["primary_efficiency"]
    reproduction_effect = reproduction["primary_efficiency"]
    difference = abs(
        float(canonical_effect["mean_log_ratio"])
        - float(reproduction_effect["mean_log_ratio"])
    )
    canonical_se = canonical_effect["standard_error"]
    reproduction_se = reproduction_effect["standard_error"]
    if canonical_se is None or reproduction_se is None:
        return math.inf, False
    denominator = math.hypot(float(canonical_se), float(reproduction_se))
    if denominator == 0.0:
        return (0.0, True) if difference == 0.0 else (math.inf, False)
    return difference / denominator, True


def run(
    config_path: Path,
    canonical_confirmation_path: Path,
    reproduction_confirmation_path: Path,
    canonical_baseline_path: Path,
    canonical_policy_path: Path,
    reproduction_baseline_path: Path,
    reproduction_policy_path: Path,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    canonical, canonical_hash = _load(
        canonical_confirmation_path, "npi.g11.v6-confirmatory.v1"
    )
    reproduction, reproduction_hash = _load(
        reproduction_confirmation_path, "npi.g11.v6-confirmatory.v1"
    )
    canonical_baseline, canonical_baseline_hash = _load(
        canonical_baseline_path, "npi.g11.v6-baseline-qualification.v1"
    )
    canonical_policy, canonical_policy_hash = _load(
        canonical_policy_path, "npi.g11.v6-routed-policy.v1"
    )
    reproduction_baseline, reproduction_baseline_hash = _load(
        reproduction_baseline_path, "npi.g11.v6-baseline-qualification.v1"
    )
    reproduction_policy, reproduction_policy_hash = _load(
        reproduction_policy_path, "npi.g11.v6-routed-policy.v1"
    )
    source_hashes_valid = _source_hashes_match(
        canonical, canonical_baseline_hash, canonical_policy_hash
    ) and _source_hashes_match(
        reproduction, reproduction_baseline_hash, reproduction_policy_hash
    )

    canonical_seeds, canonical_keys = _seed_sets(canonical_baseline, canonical_policy)
    reproduction_seeds, reproduction_keys = _seed_sets(
        reproduction_baseline, reproduction_policy
    )
    seeds_disjoint = canonical_seeds.isdisjoint(reproduction_seeds) and canonical_keys.isdisjoint(
        reproduction_keys
    )
    manifest_reference_match = (
        canonical_baseline.get("manifest_sha256")
        == canonical_policy.get("manifest_sha256")
        == reproduction_baseline.get("manifest_sha256")
        == reproduction_policy.get("manifest_sha256")
        and canonical_baseline.get("reference_artifact_sha256")
        == canonical_policy.get("reference_artifact_sha256")
        == reproduction_baseline.get("reference_artifact_sha256")
        == reproduction_policy.get("reference_artifact_sha256")
    )
    source_commit_match = (
        canonical_baseline.get("source_commit")
        == canonical_policy.get("source_commit")
        == reproduction_baseline.get("source_commit")
        == reproduction_policy.get("source_commit")
    )
    canonical_os = str(canonical_baseline.get("environment", {}).get("os", ""))
    reproduction_os = str(reproduction_baseline.get("environment", {}).get("os", ""))
    effect_z, effect_defined = _effect_consistency(canonical, reproduction)
    expected = config["expected_sha256"]
    reproduction_protocol = reproduction.get("frozen_protocol_sha256", {})
    actual_frozen = {
        "canonical_confirmation": canonical_hash,
        "reproduction_baseline_config": reproduction_protocol.get("baseline_config"),
        "reproduction_policy_config": reproduction_protocol.get("policy_config"),
        "manifest": reproduction_protocol.get("manifest"),
        "reference": reproduction_protocol.get("reference"),
        "power": reproduction_protocol.get("power"),
        "audit_config": reproduction_protocol.get("audit_config"),
    }
    frozen_hashes_match = all(
        expected[name] is None or expected[name] == actual_frozen[name]
        for name in _EXPECTED_FIELDS
    )
    requirements = config["requirements"]
    gates = {
        "confirmation_source_hashes": source_hashes_valid,
        "frozen_confirmation_hashes": frozen_hashes_match,
        "same_estimand_manifest_and_reference": manifest_reference_match,
        "same_source_commit": source_commit_match,
        "disjoint_seed_streams": seeds_disjoint,
        "linux_reproduction": "linux" in reproduction_os.lower(),
        "different_operating_system": canonical_os != reproduction_os,
        "both_scientific_gates": bool(canonical.get("scientific_gates_passed"))
        and bool(reproduction.get("scientific_gates_passed")),
        "effect_consistent_within_z_limit": effect_defined
        and effect_z <= float(config["statistics"]["maximum_effect_z_score"]),
    }
    if not bool(requirements["require_disjoint_seed_streams"]):
        gates["disjoint_seed_streams"] = True
    if not bool(requirements["require_linux_reproduction"]):
        gates["linux_reproduction"] = True
    if not bool(requirements["require_different_operating_system"]):
        gates["different_operating_system"] = True
    if not bool(requirements["require_both_scientific_gates"]):
        gates["both_scientific_gates"] = True
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "canonical_confirmation_passed": bool(canonical.get("confirmation_passed")),
        "reproduction_confirmation_passed": bool(reproduction.get("confirmation_passed")),
        "clean_source": not bool(provenance["dirty_worktree"]),
    }
    return {
        "schema": "npi.g11.v6-hardware-reproduction.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "canonical_confirmation_sha256": canonical_hash,
        "reproduction_confirmation_sha256": reproduction_hash,
        "reproduction_protocol_sha256": actual_frozen,
        "canonical_os": canonical_os,
        "reproduction_os": reproduction_os,
        "effect_difference_z_score": effect_z,
        "canonical_seed_count": len(canonical_seeds),
        "reproduction_seed_count": len(reproduction_seeds),
        "gates": gates,
        "formal_readiness": formal,
        "reproduction_gates_passed": all(gates.values()),
        "hardware_reproduction_passed": all(gates.values()) and all(formal.values()),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/hardware_reproduction_development.yaml"),
    )
    parser.add_argument("--canonical-confirmation", type=Path, required=True)
    parser.add_argument("--reproduction-confirmation", type=Path, required=True)
    parser.add_argument("--canonical-baseline", type=Path, required=True)
    parser.add_argument("--canonical-policy", type=Path, required=True)
    parser.add_argument("--reproduction-baseline", type=Path, required=True)
    parser.add_argument("--reproduction-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.canonical_confirmation,
        arguments.reproduction_confirmation,
        arguments.canonical_baseline,
        arguments.canonical_policy,
        arguments.reproduction_baseline,
        arguments.reproduction_policy,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"passed": result["hardware_reproduction_passed"], **result["gates"]}))


if __name__ == "__main__":
    main()
