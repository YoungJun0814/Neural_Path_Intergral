"""Predeclared simultaneous accuracy analysis for V7 fixed estimators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_confirmatory import _accuracy_groups
from src.path_integral.provenance import runtime_provenance, source_provenance

_SCHEMA = "npi.g11.v7-accuracy.config.v1"
_SOURCE_SCHEMA = "npi.g11.v6-secondary-baselines.v1"


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V7 accuracy config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "matrix",
        "statistics",
        "accuracy",
        "requirements",
    }:
        raise ValueError("malformed V7 accuracy config fields")
    if payload["phase"] not in {"qualification", "confirmation"}:
        raise ValueError("V7 simultaneous accuracy is formal-only")
    if payload["frozen"] is not True:
        raise ValueError("V7 simultaneous accuracy config must be frozen")
    matrix = payload["matrix"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "expected_cells",
        "expected_clusters",
        "methods",
    }:
        raise ValueError("malformed V7 accuracy matrix")
    if (
        isinstance(matrix["expected_cells"], bool)
        or not isinstance(matrix["expected_cells"], int)
        or matrix["expected_cells"] < 1
        or isinstance(matrix["expected_clusters"], bool)
        or not isinstance(matrix["expected_clusters"], int)
        or matrix["expected_clusters"] < 2
        or matrix["methods"]
        != ["fixed_dcs_slis", "fixed_raw_defensive"]
    ):
        raise ValueError("invalid V7 accuracy matrix")
    statistics = payload["statistics"]
    if not isinstance(statistics, dict) or set(statistics) != {
        "familywise_alpha",
        "bootstrap_repetitions",
        "bootstrap_seed",
    }:
        raise ValueError("malformed V7 accuracy statistics")
    alpha = statistics["familywise_alpha"]
    repetitions = statistics["bootstrap_repetitions"]
    seed = statistics["bootstrap_seed"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 0.5
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("invalid V7 accuracy statistics")
    accuracy = payload["accuracy"]
    if not isinstance(accuracy, dict) or set(accuracy) != {
        "rmse_engineering_multiplier",
        "minimum_target_attainment_rate",
    }:
        raise ValueError("malformed V7 accuracy thresholds")
    multiplier = accuracy["rmse_engineering_multiplier"]
    attainment = accuracy["minimum_target_attainment_rate"]
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not math.isfinite(float(multiplier))
        or float(multiplier) <= 0.0
        or isinstance(attainment, bool)
        or not isinstance(attainment, (int, float))
        or not 0.0 < float(attainment) < 1.0
    ):
        raise ValueError("invalid V7 accuracy thresholds")
    requirements = payload["requirements"]
    if not isinstance(requirements, dict) or set(requirements) != {
        "expected_accuracy_gates_per_group",
        "expected_claims",
    }:
        raise ValueError("malformed V7 accuracy requirements")
    gates = requirements["expected_accuracy_gates_per_group"]
    claims = requirements["expected_claims"]
    implied = matrix["expected_cells"] * len(matrix["methods"]) * gates
    if (
        isinstance(gates, bool)
        or not isinstance(gates, int)
        or gates != 2
        or isinstance(claims, bool)
        or not isinstance(claims, int)
        or claims != implied
    ):
        raise ValueError("V7 accuracy claim family is inconsistent")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_source(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SOURCE_SCHEMA:
        raise ValueError("unsupported V7 fixed-estimator source")
    return payload, hashlib.sha256(raw).hexdigest()


def run(config_path: Path, source_path: Path) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    source, source_hash = _load_source(source_path)
    if source.get("phase") != config["phase"]:
        raise ValueError("V7 accuracy phase differs from the estimator phase")
    if (
        source.get("qualification_contract", {}).get(
            "aggregate_accuracy_protocol_id"
        )
        != config["protocol_id"]
    ):
        raise ValueError("fixed estimator did not predeclare this accuracy protocol")
    matrix = config["matrix"]
    methods = list(matrix["methods"])
    records = list(source["records"])
    key_list = [
        (
            str(record["cell_id"]),
            int(record["cluster"]),
            str(record["method"]),
        )
        for record in records
    ]
    keys = set(key_list)
    if len(keys) != len(key_list):
        raise ValueError("V7 accuracy source contains duplicate records")
    cells = sorted({key[0] for key in keys})
    clusters = sorted({key[1] for key in keys})
    expected_keys = {
        (cell, cluster, method)
        for cell in cells
        for cluster in clusters
        for method in methods
    }
    if (
        len(cells) != int(matrix["expected_cells"])
        or len(clusters) != int(matrix["expected_clusters"])
        or keys != expected_keys
    ):
        raise ValueError("V7 accuracy source matrix is incomplete")
    requirements = config["requirements"]
    claim_count = int(requirements["expected_claims"])
    familywise_alpha = float(config["statistics"]["familywise_alpha"])
    per_claim_alpha = familywise_alpha / claim_count
    simultaneous_confidence = 1.0 - per_claim_alpha
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    seed = int(config["statistics"]["bootstrap_seed"])
    accuracy = config["accuracy"]
    groups = []
    for method_index, method in enumerate(methods):
        method_records = [
            record for record in records if record["method"] == method
        ]
        groups.extend(
            _accuracy_groups(
                method_records,
                method_label=method,
                multiplier=float(accuracy["rmse_engineering_multiplier"]),
                minimum_attainment=float(
                    accuracy["minimum_target_attainment_rate"]
                ),
                confidence_level=simultaneous_confidence,
                repetitions=repetitions,
                seed=seed + method_index * 100_000,
            )
        )
    gates = {
        "complete_claim_family": len(groups)
        * int(requirements["expected_accuracy_gates_per_group"])
        == claim_count,
        "source_qualified": bool(source.get("secondary_baselines_qualified")),
        "all_simultaneous_attainment_gates": all(
            bool(group["attainment_gate"]) for group in groups
        ),
        "all_simultaneous_rmse_gates": all(
            bool(group["rmse_gate"]) for group in groups
        ),
    }
    attainment_lowers = [
        float(group["one_sided_exact_attainment_lower"]) for group in groups
    ]
    rmse_ratios = [
        float(group["bootstrap_rmse_upper"])
        / float(group["rmse_tolerance_including_reference"])
        for group in groups
    ]
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "clean_analysis_source": not bool(provenance["dirty_worktree"]),
        "clean_estimator_source": not bool(source.get("dirty_worktree")),
        "non_smoke_source": not bool(source.get("smoke")),
        "formal_estimator_source": all(
            bool(value)
            for value in source.get("formal_readiness", {}).values()
        ),
    }
    return {
        "schema": "npi.g11.v7-accuracy.v1",
        "protocol_id": config["protocol_id"],
        "phase": config["phase"],
        "config_sha256": config_hash,
        "source_artifact_sha256": source_hash,
        "cell_count": len(cells),
        "cluster_count": len(clusters),
        "method_count": len(methods),
        "claim_count": claim_count,
        "familywise_alpha": familywise_alpha,
        "per_claim_alpha": per_claim_alpha,
        "simultaneous_confidence": simultaneous_confidence,
        "bootstrap_repetitions": repetitions,
        "expected_upper_tail_bootstrap_draws": repetitions * per_claim_alpha,
        "accuracy": groups,
        "minimum_simultaneous_attainment_lower": min(attainment_lowers),
        "maximum_simultaneous_rmse_ratio": max(rmse_ratios),
        "gates": gates,
        "formal_readiness": formal,
        "accuracy_qualified": all(gates.values()) and all(formal.values()),
        "interpretation": (
            "predeclared Bonferroni nominal family-wise accuracy analysis; "
            "Clopper-Pearson attainment is finite-sample exact, while the RMSE "
            "percentile-bootstrap coverage remains approximate"
        ),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.source)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"accuracy_qualified": result["accuracy_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
