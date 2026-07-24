"""Outcome-locked freeze receipt for the V7 confirmation study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.g11_v6_baseline_qualification import _load_references
from experiments.g11_v6_reference import _load_manifest
from experiments.g11_v6_secondary_baselines import (
    _load_config as load_fixed_config,
)
from experiments.g11_v7_accuracy_analysis import (
    _load_config as load_accuracy_config,
)
from experiments.g11_v7_mechanism_analysis import (
    _load_config as load_analysis_config,
)
from experiments.g11_v7_mechanism_probe import (
    _load_config as load_probe_config,
)
from src.path_integral.provenance import runtime_provenance, source_provenance


def _load_json(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _without(mapping: dict[str, Any], *paths: tuple[str, ...]) -> dict[str, Any]:
    result = copy.deepcopy(mapping)
    for path in paths:
        parent: Any = result
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]
    return result


def validate_config_transition(
    *,
    qualification_probe: dict[str, Any],
    confirmation_probe: dict[str, Any],
    qualification_fixed: dict[str, Any],
    confirmation_fixed: dict[str, Any],
    qualification_analysis: dict[str, Any],
    confirmation_analysis: dict[str, Any],
    qualification_accuracy: dict[str, Any],
    confirmation_accuracy: dict[str, Any],
) -> None:
    """Reject any outcome-dependent design drift other than declared metadata."""

    pairs = (
        (
            _without(
                qualification_probe,
                ("protocol_id",),
                ("phase",),
                ("sampling", "clusters"),
            ),
            _without(
                confirmation_probe,
                ("protocol_id",),
                ("phase",),
                ("sampling", "clusters"),
            ),
            "mechanism-probe",
        ),
        (
            _without(
                qualification_fixed,
                ("protocol_id",),
                ("phase",),
                ("proposal", "training_amortization_record_count"),
                ("sampling", "clusters"),
                (
                    "qualification_decision",
                    "aggregate_accuracy_protocol_id",
                ),
            ),
            _without(
                confirmation_fixed,
                ("protocol_id",),
                ("phase",),
                ("proposal", "training_amortization_record_count"),
                ("sampling", "clusters"),
                (
                    "qualification_decision",
                    "aggregate_accuracy_protocol_id",
                ),
            ),
            "fixed-estimator",
        ),
        (
            _without(
                qualification_analysis,
                ("protocol_id",),
                ("phase",),
                ("matrix", "expected_clusters"),
            ),
            _without(
                confirmation_analysis,
                ("protocol_id",),
                ("phase",),
                ("matrix", "expected_clusters"),
            ),
            "joint-analysis",
        ),
        (
            _without(
                qualification_accuracy,
                ("protocol_id",),
                ("phase",),
                ("matrix", "expected_clusters"),
                ("statistics", "bootstrap_seed"),
            ),
            _without(
                confirmation_accuracy,
                ("protocol_id",),
                ("phase",),
                ("matrix", "expected_clusters"),
                ("statistics", "bootstrap_seed"),
            ),
            "accuracy",
        ),
    )
    for qualification, confirmation, name in pairs:
        if qualification != confirmation:
            raise ValueError(f"V7 {name} design drifted after qualification")

    configs = (
        confirmation_probe,
        confirmation_fixed,
        confirmation_analysis,
        confirmation_accuracy,
    )
    if any(
        config.get("phase") != "confirmation" or config.get("frozen") is not True
        for config in configs
    ):
        raise ValueError("every V7 confirmation config must be frozen")
    expected_clusters = 64
    cluster_counts = (
        int(confirmation_probe["sampling"]["clusters"]),
        int(confirmation_fixed["sampling"]["clusters"]),
        int(confirmation_analysis["matrix"]["expected_clusters"]),
        int(confirmation_accuracy["matrix"]["expected_clusters"]),
    )
    if cluster_counts != (expected_clusters,) * 4:
        raise ValueError("V7 confirmation requires 64 clusters everywhere")
    expected_amortization = (
        int(confirmation_analysis["matrix"]["expected_cells"]) * expected_clusters
    )
    if (
        int(confirmation_fixed["proposal"]["training_amortization_record_count"])
        != expected_amortization
    ):
        raise ValueError("V7 confirmation training amortization drifted")
    if (
        confirmation_fixed["qualification_decision"]["aggregate_accuracy_protocol_id"]
        != confirmation_accuracy["protocol_id"]
    ):
        raise ValueError("V7 confirmation accuracy protocol link drifted")
    protocol_ids = {str(config["protocol_id"]) for config in configs}
    if len(protocol_ids) != len(configs):
        raise ValueError("V7 confirmation protocol IDs must be distinct")


def freeze(
    qualification_freeze_path: Path,
    qualification_audit_path: Path,
    qualification_probe_config_path: Path,
    qualification_fixed_config_path: Path,
    qualification_analysis_config_path: Path,
    qualification_accuracy_config_path: Path,
    confirmation_probe_config_path: Path,
    confirmation_fixed_config_path: Path,
    confirmation_analysis_config_path: Path,
    confirmation_accuracy_config_path: Path,
    confirmation_manifest_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    qualification_freeze, qualification_freeze_hash = _load_json(
        qualification_freeze_path,
        "npi.g11.v7-qualification-freeze.v1",
    )
    qualification_audit, qualification_audit_hash = _load_json(
        qualification_audit_path,
        "npi.g11.v7-qualification-audit.v1",
    )
    if (
        not qualification_freeze.get("qualification_authorized")
        or not qualification_audit.get("qualification_audit_passed")
        or qualification_audit.get("failures")
        or qualification_audit.get("freeze_sha256") != qualification_freeze_hash
    ):
        raise ValueError("audited V7 qualification did not authorize confirmation")

    qualification_probe, _ = load_probe_config(qualification_probe_config_path)
    qualification_fixed, _ = load_fixed_config(qualification_fixed_config_path)
    qualification_analysis, _ = load_analysis_config(qualification_analysis_config_path)
    qualification_accuracy, _ = load_accuracy_config(qualification_accuracy_config_path)
    confirmation_probe, confirmation_probe_hash = load_probe_config(confirmation_probe_config_path)
    confirmation_fixed, confirmation_fixed_hash = load_fixed_config(confirmation_fixed_config_path)
    confirmation_analysis, confirmation_analysis_hash = load_analysis_config(
        confirmation_analysis_config_path
    )
    confirmation_accuracy, confirmation_accuracy_hash = load_accuracy_config(
        confirmation_accuracy_config_path
    )
    validate_config_transition(
        qualification_probe=qualification_probe,
        confirmation_probe=confirmation_probe,
        qualification_fixed=qualification_fixed,
        confirmation_fixed=confirmation_fixed,
        qualification_analysis=qualification_analysis,
        confirmation_analysis=confirmation_analysis,
        qualification_accuracy=qualification_accuracy,
        confirmation_accuracy=confirmation_accuracy,
    )

    manifest = _load_manifest(confirmation_manifest_path)
    if len(manifest.cells) != 18:
        raise ValueError("V7 confirmation manifest must contain 18 cells")
    _, reference_hash = _load_references(reference_path)
    identity = qualification_freeze["input_identity"]
    if reference_hash != identity["reference"]:
        raise ValueError("V7 confirmation reference artifact drifted")
    proposal_hashes = {
        confirmation_probe["proposal"]["training_source_artifact_sha256"],
        confirmation_fixed["proposal"]["training_source_artifact_sha256"],
        identity["proposal_training_source"],
    }
    if len(proposal_hashes) != 1:
        raise ValueError("V7 confirmation proposal source drifted")

    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise ValueError("V7 confirmation freeze requires a clean source tree")
    return {
        "schema": "npi.g11.v7-confirmation-freeze.v1",
        "source_commit": provenance["source_commit"],
        "dirty_worktree": provenance["dirty_worktree"],
        "planned_clusters": 64,
        "qualification_authorization_sha256": {
            "freeze": qualification_freeze_hash,
            "independent_audit": qualification_audit_hash,
        },
        "confirmation_config_sha256": {
            "mechanism_probe": confirmation_probe_hash,
            "fixed_estimators": confirmation_fixed_hash,
            "joint_analysis": confirmation_analysis_hash,
            "simultaneous_accuracy": confirmation_accuracy_hash,
        },
        "input_identity": {
            "manifest": manifest.sha256,
            "reference": reference_hash,
            "proposal_training_source": proposal_hashes.pop(),
        },
        "allowed_design_changes": {
            "clusters": {"qualification": 24, "confirmation": 64},
            "training_amortization_record_count": {
                "qualification": 432,
                "confirmation": 1152,
            },
            "bootstrap_seed_namespace_changed": True,
            "protocol_seed_namespaces_changed": True,
        },
        "confirmation_authorized": True,
        "environment": runtime_provenance(dtype="serialized-float64"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-freeze", type=Path, required=True)
    parser.add_argument("--qualification-audit", type=Path, required=True)
    parser.add_argument("--qualification-probe-config", type=Path, required=True)
    parser.add_argument("--qualification-fixed-config", type=Path, required=True)
    parser.add_argument("--qualification-analysis-config", type=Path, required=True)
    parser.add_argument("--qualification-accuracy-config", type=Path, required=True)
    parser.add_argument("--confirmation-probe-config", type=Path, required=True)
    parser.add_argument("--confirmation-fixed-config", type=Path, required=True)
    parser.add_argument("--confirmation-analysis-config", type=Path, required=True)
    parser.add_argument("--confirmation-accuracy-config", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = freeze(
        arguments.qualification_freeze,
        arguments.qualification_audit,
        arguments.qualification_probe_config,
        arguments.qualification_fixed_config,
        arguments.qualification_analysis_config,
        arguments.qualification_accuracy_config,
        arguments.confirmation_probe_config,
        arguments.confirmation_fixed_config,
        arguments.confirmation_analysis_config,
        arguments.confirmation_accuracy_config,
        arguments.confirmation_manifest,
        arguments.reference,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "confirmation_authorized": result["confirmation_authorized"],
                "planned_clusters": result["planned_clusters"],
            }
        )
    )


if __name__ == "__main__":
    main()
