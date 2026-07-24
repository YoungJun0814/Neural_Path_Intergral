"""Independent aggregate audit for the complete V7 confirmation package.

The numerical recomputation helpers come from the JSON-only qualification auditor,
not from any production analyzer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.g11_v7_qualification_audit import (
    _close,
    _config_hash,
    _load,
    _recompute_accuracy,
    _recompute_joint_effects,
    _seed_values,
)
from src.path_integral.provenance import runtime_provenance, source_provenance


def audit(
    freeze_path: Path,
    probe_path: Path,
    probe_audit_path: Path,
    fixed_path: Path,
    fixed_audit_path: Path,
    resources_path: Path,
    joint_path: Path,
    accuracy_path: Path,
    probe_config_path: Path,
    fixed_config_path: Path,
    joint_config_path: Path,
    accuracy_config_path: Path,
) -> dict[str, Any]:
    freeze, freeze_hash = _load(freeze_path, "npi.g11.v7-confirmation-freeze.v1")
    probe, probe_hash = _load(probe_path, "npi.g11.v7-mechanism-probe.v1")
    probe_audit, probe_audit_hash = _load(probe_audit_path, "npi.g11.v7-mechanism-probe-audit.v1")
    fixed, fixed_hash = _load(fixed_path, "npi.g11.v6-secondary-baselines.v1")
    fixed_audit, fixed_audit_hash = _load(fixed_audit_path, "npi.g11.v6-independent-audit.v1")
    resources, resources_hash = _load(resources_path, "npi.g11.v6-resource-supplement.v1")
    joint, joint_hash = _load(joint_path, "npi.g11.v7-mechanism-analysis.v1")
    accuracy, accuracy_hash = _load(accuracy_path, "npi.g11.v7-accuracy.v1")
    probe_config, probe_config_hash = _config_hash(
        probe_config_path, "npi.g11.v7-mechanism-probe.config.v2"
    )
    fixed_config, fixed_config_hash = _config_hash(
        fixed_config_path, "npi.g11.v6-secondary-baselines.config.v2"
    )
    joint_config, joint_config_hash = _config_hash(
        joint_config_path, "npi.g11.v7-mechanism-analysis.config.v1"
    )
    accuracy_config, accuracy_config_hash = _config_hash(
        accuracy_config_path, "npi.g11.v7-accuracy.config.v1"
    )
    failures: list[str] = []
    expected_config_hashes = {
        "mechanism_probe": probe_config_hash,
        "fixed_estimators": fixed_config_hash,
        "joint_analysis": joint_config_hash,
        "simultaneous_accuracy": accuracy_config_hash,
    }
    if freeze.get("confirmation_config_sha256") != expected_config_hashes:
        failures.append("confirmation config hashes differ from freeze")
    configs = (
        probe_config,
        fixed_config,
        joint_config,
        accuracy_config,
    )
    if any(
        config.get("phase") != "confirmation" or config.get("frozen") is not True
        for config in configs
    ):
        failures.append("confirmation phase or frozen flag drift")

    source_commit = freeze.get("source_commit")
    formal_artifacts = (
        probe,
        probe_audit,
        fixed,
        fixed_audit,
        resources,
        joint,
        accuracy,
    )
    if any(
        artifact.get("source_commit") != source_commit or bool(artifact.get("dirty_worktree"))
        for artifact in formal_artifacts
    ):
        failures.append("formal source commit or cleanliness drift")

    identities = freeze["input_identity"]
    for name, artifact in (("probe", probe), ("fixed", fixed)):
        if artifact.get("manifest_sha256") != identities["manifest"]:
            failures.append(f"{name} manifest differs from freeze")
        if artifact.get("reference_artifact_sha256") != identities["reference"]:
            failures.append(f"{name} reference differs from freeze")
        if (
            artifact.get("proposal_training_audit", {}).get("source_artifact_sha256")
            != identities["proposal_training_source"]
        ):
            failures.append(f"{name} proposal source differs from freeze")

    expected_cells = int(joint_config["matrix"]["expected_cells"])
    expected_clusters = int(joint_config["matrix"]["expected_clusters"])
    expected_probe_records = expected_cells * expected_clusters
    expected_fixed_records = expected_probe_records * len(joint_config["matrix"]["methods"])
    if len(probe.get("records", [])) != expected_probe_records:
        failures.append("confirmation probe record count mismatch")
    if len(fixed.get("records", [])) != expected_fixed_records:
        failures.append("confirmation fixed-estimator record count mismatch")

    links = {
        "probe audit": probe_audit.get("source_artifact_sha256") == probe_hash,
        "fixed audit": fixed_audit.get("source_artifact_sha256") == fixed_hash,
        "resource supplement": resources.get("source_artifact_sha256") == fixed_hash,
        "joint probe": joint.get("source_artifact_sha256", {}).get("mechanism_probe") == probe_hash,
        "joint fixed": joint.get("source_artifact_sha256", {}).get("fixed_estimators")
        == fixed_hash,
        "accuracy fixed": accuracy.get("source_artifact_sha256") == fixed_hash,
    }
    if not all(links.values()):
        failures.append("confirmation artifact hash link failed")
    decisions = {
        "freeze": bool(freeze.get("confirmation_authorized")),
        "probe": bool(probe.get("formal_mechanism_passed")),
        "probe_audit": bool(probe_audit.get("passed")),
        "fixed": bool(fixed.get("secondary_baselines_qualified")),
        "fixed_audit": bool(fixed_audit.get("qualification_audit_passed")),
        "resources": bool(resources.get("passed")),
        "joint": bool(joint.get("formal_mechanism_qualified")),
        "accuracy": bool(accuracy.get("accuracy_qualified")),
    }
    if not all(decisions.values()):
        failures.append("one or more confirmation decisions failed")
    if not _seed_values(probe).isdisjoint(_seed_values(fixed)):
        failures.append("mechanism and fixed-estimator seeds overlap")

    recomputed_effects, recomputed_floors = _recompute_joint_effects(
        fixed,
        confidence=float(joint_config["statistics"]["confidence_level"]),
    )
    for metric, recomputed in recomputed_effects.items():
        recorded = joint["effects"][metric]
        if any(not _close(float(recorded[field]), value) for field, value in recomputed.items()):
            failures.append(f"joint {metric} effect mismatch")
    if joint.get("floor_counts") != recomputed_floors:
        failures.append("joint floor counts mismatch")

    recomputed_accuracy = _recompute_accuracy(fixed, accuracy_config)
    recorded_accuracy = accuracy.get("accuracy")
    if not isinstance(recorded_accuracy, list) or len(recorded_accuracy) != len(
        recomputed_accuracy
    ):
        failures.append("accuracy group count mismatch")
    else:
        for index, (recorded, recomputed) in enumerate(
            zip(recorded_accuracy, recomputed_accuracy, strict=True)
        ):
            for field, expected in recomputed.items():
                observed = recorded.get(field)
                if isinstance(expected, float):
                    if (
                        isinstance(observed, bool)
                        or not isinstance(observed, (int, float))
                        or not _close(float(observed), expected)
                    ):
                        failures.append(f"accuracy group {index} field {field} mismatch")
                elif observed != expected:
                    failures.append(f"accuracy group {index} field {field} mismatch")
    provenance = source_provenance()
    return {
        "schema": "npi.g11.v7-confirmation-audit.v1",
        "freeze_sha256": freeze_hash,
        "artifact_sha256": {
            "mechanism_probe": probe_hash,
            "mechanism_probe_audit": probe_audit_hash,
            "fixed_estimators": fixed_hash,
            "fixed_estimators_audit": fixed_audit_hash,
            "resource_supplement": resources_hash,
            "joint_analysis": joint_hash,
            "simultaneous_accuracy": accuracy_hash,
        },
        "links": links,
        "decisions": decisions,
        "recomputed_effects": recomputed_effects,
        "recomputed_floor_counts": recomputed_floors,
        "recomputed_accuracy_groups": len(recomputed_accuracy),
        "failures": failures,
        "confirmation_audit_passed": not failures,
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-audit", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--fixed-audit", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--fixed-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--accuracy-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(
        arguments.freeze,
        arguments.probe,
        arguments.probe_audit,
        arguments.fixed,
        arguments.fixed_audit,
        arguments.resources,
        arguments.joint,
        arguments.accuracy,
        arguments.probe_config,
        arguments.fixed_config,
        arguments.joint_config,
        arguments.accuracy_config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "confirmation_audit_passed": result["confirmation_audit_passed"],
                "failure_count": len(result["failures"]),
            }
        )
    )


if __name__ == "__main__":
    main()
