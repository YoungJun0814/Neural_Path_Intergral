"""Cross-environment audit for the V7 mechanism confirmation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from src.path_integral.provenance import runtime_provenance, source_provenance


def _load(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _seeds(payload: dict[str, Any]) -> set[int]:
    values = [int(record["seed"]) for record in payload["seed_ledger"]["records"]]
    if len(values) != len(set(values)):
        raise ValueError("hardware artifact contains duplicate seeds")
    return set(values)


def _effect_z(
    canonical: dict[str, Any],
    reproduction: dict[str, Any],
) -> float:
    difference = abs(float(canonical["mean_log_ratio"]) - float(reproduction["mean_log_ratio"]))
    denominator = math.hypot(
        float(canonical["standard_error"]),
        float(reproduction["standard_error"]),
    )
    if denominator == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / denominator


def audit(
    freeze_receipt_path: Path,
    reproduction_execution_freeze_path: Path,
    canonical_audit_path: Path,
    reproduction_audit_path: Path,
    canonical_probe_path: Path,
    canonical_fixed_path: Path,
    canonical_joint_path: Path,
    canonical_accuracy_path: Path,
    reproduction_probe_path: Path,
    reproduction_fixed_path: Path,
    reproduction_joint_path: Path,
    reproduction_accuracy_path: Path,
) -> dict[str, Any]:
    receipt, receipt_hash = _load(
        freeze_receipt_path,
        "npi.g11.v7-linux-reproduction-freeze.v1",
    )
    reproduction_freeze, reproduction_freeze_hash = _load(
        reproduction_execution_freeze_path,
        "npi.g11.v7-confirmation-freeze.v1",
    )
    canonical_audit, canonical_audit_hash = _load(
        canonical_audit_path,
        "npi.g11.v7-confirmation-audit.v1",
    )
    reproduction_audit, reproduction_audit_hash = _load(
        reproduction_audit_path,
        "npi.g11.v7-confirmation-audit.v1",
    )
    inputs = {
        "canonical_probe": _load(
            canonical_probe_path,
            "npi.g11.v7-mechanism-probe.v1",
        ),
        "canonical_fixed": _load(
            canonical_fixed_path,
            "npi.g11.v6-secondary-baselines.v1",
        ),
        "canonical_joint": _load(
            canonical_joint_path,
            "npi.g11.v7-mechanism-analysis.v1",
        ),
        "canonical_accuracy": _load(
            canonical_accuracy_path,
            "npi.g11.v7-accuracy.v1",
        ),
        "reproduction_probe": _load(
            reproduction_probe_path,
            "npi.g11.v7-mechanism-probe.v1",
        ),
        "reproduction_fixed": _load(
            reproduction_fixed_path,
            "npi.g11.v6-secondary-baselines.v1",
        ),
        "reproduction_joint": _load(
            reproduction_joint_path,
            "npi.g11.v7-mechanism-analysis.v1",
        ),
        "reproduction_accuracy": _load(
            reproduction_accuracy_path,
            "npi.g11.v7-accuracy.v1",
        ),
    }
    payloads = {name: value[0] for name, value in inputs.items()}
    hashes = {name: value[1] for name, value in inputs.items()}
    failures = []
    if receipt["canonical_sha256"]["aggregate_audit"] != canonical_audit_hash:
        failures.append("canonical audit hash differs from freeze")
    if receipt["reproduction_sha256"]["execution_freeze"] != reproduction_freeze_hash:
        failures.append("reproduction execution freeze hash drifted")
    if not canonical_audit.get("confirmation_audit_passed") or not reproduction_audit.get(
        "confirmation_audit_passed"
    ):
        failures.append("one environment failed aggregate confirmation audit")

    audit_links = {
        "canonical_probe": "mechanism_probe",
        "canonical_fixed": "fixed_estimators",
        "canonical_joint": "joint_analysis",
        "canonical_accuracy": "simultaneous_accuracy",
        "reproduction_probe": "mechanism_probe",
        "reproduction_fixed": "fixed_estimators",
        "reproduction_joint": "joint_analysis",
        "reproduction_accuracy": "simultaneous_accuracy",
    }
    for name, audit_key in audit_links.items():
        audit_payload = canonical_audit if name.startswith("canonical") else reproduction_audit
        if audit_payload["artifact_sha256"][audit_key] != hashes[name]:
            failures.append(f"{name} differs from its aggregate audit")

    canonical_source = payloads["canonical_fixed"]["source_commit"]
    reproduction_source = payloads["reproduction_fixed"]["source_commit"]
    expected_source = receipt["execution_source_commit"]
    if canonical_source != reproduction_source or canonical_source != expected_source:
        failures.append("execution source commits differ")
    canonical_os = str(payloads["canonical_fixed"]["environment"]["os"])
    reproduction_os = str(payloads["reproduction_fixed"]["environment"]["os"])
    if "linux" not in reproduction_os.lower() or canonical_os == reproduction_os:
        failures.append("reproduction environment is not an independent Linux OS")

    identity_fields = (
        "manifest_sha256",
        "reference_artifact_sha256",
    )
    for field in identity_fields:
        values = {
            payloads[name][field]
            for name in (
                "canonical_probe",
                "canonical_fixed",
                "reproduction_probe",
                "reproduction_fixed",
            )
        }
        if len(values) != 1:
            failures.append(f"{field} differs across environments")
    proposal_values = {
        payloads[name]["proposal_training_audit"]["source_artifact_sha256"]
        for name in (
            "canonical_probe",
            "canonical_fixed",
            "reproduction_probe",
            "reproduction_fixed",
        )
    }
    if len(proposal_values) != 1:
        failures.append("proposal source differs across environments")

    seed_sets = {
        name: _seeds(payloads[name])
        for name in (
            "canonical_probe",
            "canonical_fixed",
            "reproduction_probe",
            "reproduction_fixed",
        )
    }
    intersections = {}
    for left, right in itertools.combinations(sorted(seed_sets), 2):
        label = f"{left}__{right}"
        intersections[label] = len(seed_sets[left] & seed_sets[right])
        if intersections[label]:
            failures.append(f"{label} has overlapping seeds")

    canonical_probe_effect = payloads["canonical_probe"]["paired_cluster_effect"]
    reproduction_probe_effect = payloads["reproduction_probe"]["paired_cluster_effect"]
    effect_z = {
        "probe_variance": _effect_z(
            {
                "mean_log_ratio": canonical_probe_effect["mean_log_raw_over_dcs_variance"],
                "standard_error": canonical_probe_effect["standard_error"],
            },
            {
                "mean_log_ratio": reproduction_probe_effect["mean_log_raw_over_dcs_variance"],
                "standard_error": reproduction_probe_effect["standard_error"],
            },
        )
    }
    for metric in (
        "execution_variance",
        "final_work",
        "training_inclusive_work",
        "nontraining_work",
        "final_wall",
    ):
        effect_z[metric] = _effect_z(
            payloads["canonical_joint"]["effects"][metric],
            payloads["reproduction_joint"]["effects"][metric],
        )
    maximum_z = float(receipt["maximum_effect_difference_z"])
    for metric in receipt["gated_effects"]:
        if effect_z[metric] > maximum_z:
            failures.append(f"{metric} effect differs beyond frozen z limit")

    if (
        not payloads["canonical_joint"]["formal_mechanism_qualified"]
        or not payloads["reproduction_joint"]["formal_mechanism_qualified"]
        or not payloads["canonical_accuracy"]["accuracy_qualified"]
        or not payloads["reproduction_accuracy"]["accuracy_qualified"]
    ):
        failures.append("one environment failed scientific co-gates")
    provenance = source_provenance()
    return {
        "schema": "npi.g11.v7-hardware-reproduction.v1",
        "freeze_receipt_sha256": receipt_hash,
        "reproduction_execution_freeze_sha256": reproduction_freeze_hash,
        "canonical_audit_sha256": canonical_audit_hash,
        "reproduction_audit_sha256": reproduction_audit_hash,
        "artifact_sha256": hashes,
        "canonical_os": canonical_os,
        "reproduction_os": reproduction_os,
        "execution_source_commit": expected_source,
        "seed_counts": {name: len(values) for name, values in seed_sets.items()},
        "pairwise_seed_intersections": intersections,
        "effect_difference_z": effect_z,
        "maximum_gated_effect_difference_z": maximum_z,
        "failures": failures,
        "hardware_reproduction_passed": not failures,
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--reproduction-execution-freeze", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--reproduction-audit", type=Path, required=True)
    parser.add_argument("--canonical-probe", type=Path, required=True)
    parser.add_argument("--canonical-fixed", type=Path, required=True)
    parser.add_argument("--canonical-joint", type=Path, required=True)
    parser.add_argument("--canonical-accuracy", type=Path, required=True)
    parser.add_argument("--reproduction-probe", type=Path, required=True)
    parser.add_argument("--reproduction-fixed", type=Path, required=True)
    parser.add_argument("--reproduction-joint", type=Path, required=True)
    parser.add_argument("--reproduction-accuracy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(
        arguments.freeze_receipt,
        arguments.reproduction_execution_freeze,
        arguments.canonical_audit,
        arguments.reproduction_audit,
        arguments.canonical_probe,
        arguments.canonical_fixed,
        arguments.canonical_joint,
        arguments.canonical_accuracy,
        arguments.reproduction_probe,
        arguments.reproduction_fixed,
        arguments.reproduction_joint,
        arguments.reproduction_accuracy,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "hardware_reproduction_passed": result["hardware_reproduction_passed"],
                "failure_count": len(result["failures"]),
                "effect_difference_z": result["effect_difference_z"],
            }
        )
    )


if __name__ == "__main__":
    main()
