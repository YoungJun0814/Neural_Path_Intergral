"""Prespecified method-by-cell accuracy co-gates for V6 secondary methods."""

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

_SCHEMA = "npi.g11.v6-accuracy-analysis.config.v1"


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 accuracy-analysis config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "frozen",
        "accepted_source_schema",
        "methods",
        "accuracy",
        "requirements",
    }:
        raise ValueError("malformed V6 accuracy-analysis config")
    methods = payload["methods"]
    if (
        not isinstance(methods, list)
        or not methods
        or len(methods) != len(set(methods))
        or any(not isinstance(method, str) or not method for method in methods)
    ):
        raise ValueError("accuracy-analysis methods must be nonempty and unique")
    accuracy = payload["accuracy"]
    if not isinstance(accuracy, dict) or set(accuracy) != {
        "confidence_level",
        "rmse_engineering_multiplier",
        "minimum_target_attainment_rate",
        "bootstrap_repetitions",
        "bootstrap_seed",
    }:
        raise ValueError("malformed V6 accuracy-analysis statistics")
    confidence = float(accuracy["confidence_level"])
    multiplier = float(accuracy["rmse_engineering_multiplier"])
    minimum = float(accuracy["minimum_target_attainment_rate"])
    repetitions = accuracy["bootstrap_repetitions"]
    seed = accuracy["bootstrap_seed"]
    if (
        not 0.5 < confidence < 1.0
        or not math.isfinite(multiplier)
        or multiplier <= 0.0
        or not 0.0 < minimum < 1.0
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 200
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("invalid V6 accuracy-analysis design")
    requirements = payload["requirements"]
    if not isinstance(requirements, dict) or set(requirements) != {
        "require_operationally_qualified_source",
        "require_qualification_audit",
    }:
        raise ValueError("malformed V6 accuracy-analysis requirements")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("V6 accuracy source must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def run(
    config_path: Path, source_path: Path, audit_path: Path
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    source, source_hash = _load_json(source_path)
    audit, audit_hash = _load_json(audit_path)
    if source.get("schema") != config["accepted_source_schema"]:
        raise ValueError("accuracy source schema does not match the frozen config")
    methods = list(config["methods"])
    source_records = source.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("accuracy source contains no records")
    if {str(record.get("method")) for record in source_records} != set(methods):
        raise ValueError("accuracy source method set does not match the config")
    accuracy = config["accuracy"]
    summaries = []
    for index, method in enumerate(methods):
        records = [
            record
            for record in source_records
            if str(record.get("method")) == method
        ]
        summaries.extend(
            _accuracy_groups(
                records,
                method_label=method,
                multiplier=float(
                    accuracy["rmse_engineering_multiplier"]
                ),
                minimum_attainment=float(
                    accuracy["minimum_target_attainment_rate"]
                ),
                confidence_level=float(accuracy["confidence_level"]),
                repetitions=int(accuracy["bootstrap_repetitions"]),
                seed=int(accuracy["bootstrap_seed"]) + 100_000 * index,
            )
        )
    cells = sorted(
        {str(record["cell_id"]) for record in source_records}
    )
    requirements = config["requirements"]
    gates = {
        "complete_method_cell_matrix": len(summaries)
        == len(methods) * len(cells),
        "all_accuracy_co_gates": all(
            record["attainment_gate"] and record["rmse_gate"]
            for record in summaries
        ),
        "source_operationally_qualified": (
            bool(source.get("secondary_baselines_qualified"))
            or not bool(
                requirements["require_operationally_qualified_source"]
            )
        ),
        "independent_audit_matches_source": (
            audit.get("source_artifact_sha256") == source_hash
            and audit.get("qualification_audit_passed") is True
        )
        or not bool(requirements["require_qualification_audit"]),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke_source": not bool(source.get("smoke")),
    }
    return {
        "schema": "npi.g11.v6-accuracy-analysis.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "source_artifact_sha256": source_hash,
        "audit_artifact_sha256": audit_hash,
        "methods": methods,
        "cell_count": len(cells),
        "accuracy": summaries,
        "gates": gates,
        "formal_readiness": formal,
        "accuracy_qualified": all(gates.values()) and all(formal.values()),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.source, arguments.audit)
    if arguments.output.exists():
        raise FileExistsError("accuracy analysis refuses to overwrite an artifact")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"qualified": result["accuracy_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
