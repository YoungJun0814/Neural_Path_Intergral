"""Freeze disjoint-seed Linux reproduction configs before Linux outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

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

_OUTPUT_NAMES = {
    "mechanism_probe": "mechanism_probe_linux_reproduction_v1.yaml",
    "fixed_estimators": "fixed_estimators_linux_reproduction_v1.yaml",
    "joint_analysis": "mechanism_analysis_linux_reproduction_v1.yaml",
    "simultaneous_accuracy": "accuracy_linux_reproduction_v1.yaml",
}


def _load_json(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(config: dict[str, Any], kind: str) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("protocol_id")
    if kind == "fixed_estimators":
        result["qualification_decision"].pop("aggregate_accuracy_protocol_id")
    if kind == "simultaneous_accuracy":
        result["statistics"].pop("bootstrap_seed")
    return result


def build_reproduction_configs(
    *,
    probe: dict[str, Any],
    fixed: dict[str, Any],
    analysis: dict[str, Any],
    accuracy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    configs = {
        "mechanism_probe": copy.deepcopy(probe),
        "fixed_estimators": copy.deepcopy(fixed),
        "joint_analysis": copy.deepcopy(analysis),
        "simultaneous_accuracy": copy.deepcopy(accuracy),
    }
    identifiers = {
        "mechanism_probe": "g11-v7-mechanism-probe-linux-reproduction-v1",
        "fixed_estimators": "g11-v7-fixed-estimators-linux-reproduction-v1",
        "joint_analysis": "g11-v7-mechanism-analysis-linux-reproduction-v1",
        "simultaneous_accuracy": "g11-v7-accuracy-linux-reproduction-v1",
    }
    for kind, config in configs.items():
        config["protocol_id"] = identifiers[kind]
    configs["fixed_estimators"]["qualification_decision"]["aggregate_accuracy_protocol_id"] = (
        identifiers["simultaneous_accuracy"]
    )
    configs["simultaneous_accuracy"]["statistics"]["bootstrap_seed"] = 24072511
    originals = {
        "mechanism_probe": probe,
        "fixed_estimators": fixed,
        "joint_analysis": analysis,
        "simultaneous_accuracy": accuracy,
    }
    for kind, config in configs.items():
        if _normalized(config, kind) != _normalized(originals[kind], kind):
            raise ValueError(f"V7 Linux {kind} design drifted")
        if config.get("phase") != "confirmation" or config.get("frozen") is not True:
            raise ValueError("Linux reproduction keeps the frozen confirmation phase")
    return configs


def freeze(
    canonical_freeze_path: Path,
    canonical_audit_path: Path,
    probe_config_path: Path,
    fixed_config_path: Path,
    analysis_config_path: Path,
    accuracy_config_path: Path,
    *,
    container_image_id: str,
    output_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_freeze, canonical_freeze_hash = _load_json(
        canonical_freeze_path,
        "npi.g11.v7-confirmation-freeze.v1",
    )
    canonical_audit, canonical_audit_hash = _load_json(
        canonical_audit_path,
        "npi.g11.v7-confirmation-audit.v1",
    )
    if (
        not canonical_freeze.get("confirmation_authorized")
        or not canonical_audit.get("confirmation_audit_passed")
        or canonical_audit.get("failures")
        or canonical_audit.get("freeze_sha256") != canonical_freeze_hash
    ):
        raise ValueError("passing canonical V7 confirmation is required")
    if (
        not container_image_id.startswith("sha256:")
        or len(container_image_id) != 71
        or any(character not in "0123456789abcdef" for character in container_image_id[7:])
    ):
        raise ValueError("container image ID must be a lowercase sha256 digest")

    probe, _ = load_probe_config(probe_config_path)
    fixed, _ = load_fixed_config(fixed_config_path)
    analysis, _ = load_analysis_config(analysis_config_path)
    accuracy, _ = load_accuracy_config(accuracy_config_path)
    configs = build_reproduction_configs(
        probe=probe,
        fixed=fixed,
        analysis=analysis,
        accuracy=accuracy,
    )
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise ValueError("Linux reproduction freeze requires a clean source tree")
    if output_directory.exists():
        raise FileExistsError("Linux reproduction freeze refuses to overwrite")
    output_directory.mkdir(parents=True)
    config_hashes = {}
    for kind, config in configs.items():
        path = output_directory / _OUTPUT_NAMES[kind]
        path.write_text(
            yaml.safe_dump(
                config,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        config_hashes[kind] = _sha256(path)

    execution_freeze = {
        "schema": "npi.g11.v7-confirmation-freeze.v1",
        "source_commit": canonical_freeze["source_commit"],
        "dirty_worktree": False,
        "planned_clusters": 64,
        "qualification_authorization_sha256": {
            "canonical_confirmation_freeze": canonical_freeze_hash,
            "canonical_confirmation_audit": canonical_audit_hash,
        },
        "confirmation_config_sha256": config_hashes,
        "input_identity": canonical_freeze["input_identity"],
        "allowed_design_changes": {
            "protocol_seed_namespaces_changed": True,
            "bootstrap_seed_namespace_changed": True,
            "all_scientific_fields_unchanged": True,
        },
        "confirmation_authorized": True,
        "environment": runtime_provenance(dtype="serialized-float64"),
    }
    execution_freeze_path = output_directory / "execution_freeze.json"
    execution_freeze_path.write_text(
        json.dumps(
            execution_freeze,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema": "npi.g11.v7-linux-reproduction-freeze.v1",
        "analysis_source_commit": provenance["source_commit"],
        "execution_source_commit": canonical_freeze["source_commit"],
        "container_image_id": container_image_id,
        "canonical_sha256": {
            "freeze": canonical_freeze_hash,
            "aggregate_audit": canonical_audit_hash,
        },
        "reproduction_sha256": {
            **config_hashes,
            "execution_freeze": _sha256(execution_freeze_path),
        },
        "maximum_effect_difference_z": 3.0,
        "gated_effects": [
            "probe_variance",
            "execution_variance",
            "final_work",
            "training_inclusive_work",
        ],
        "linux_reproduction_authorized": True,
        "environment": runtime_provenance(dtype="serialized-float64"),
    }
    receipt_path = output_directory / "freeze_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return execution_freeze, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-freeze", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--fixed-config", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--accuracy-config", type=Path, required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    execution_freeze, receipt = freeze(
        arguments.canonical_freeze,
        arguments.canonical_audit,
        arguments.probe_config,
        arguments.fixed_config,
        arguments.analysis_config,
        arguments.accuracy_config,
        container_image_id=arguments.container_image_id,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "linux_reproduction_authorized": receipt["linux_reproduction_authorized"],
                "execution_source_commit": execution_freeze["source_commit"],
                "container_image_id": receipt["container_image_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
