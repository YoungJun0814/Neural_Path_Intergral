"""Re-audit V16 deep-tail methods against an independent high-SNR SMC reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.v15_baseline_protocol import REQUIRED_PRIMARY_COMPARATORS
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance

ROOT = Path(__file__).resolve().parents[1]


def _load_bound(binding: dict[str, str]) -> dict[str, Any]:
    path = ROOT / binding["path"]
    if not path.is_file() or file_sha256(path) != binding["sha256"]:
        raise RuntimeError(f"artifact binding failed: {binding['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _accuracy_z(method: dict[str, Any], reference: dict[str, Any]) -> float:
    denominator = math.sqrt(
        float(method["standard_error"]) ** 2
        + float(reference["standard_error"]) ** 2
    )
    return abs(float(method["estimate"]) - float(reference["estimate"])) / max(
        denominator,
        float.fromhex("0x1.0p-1022"),
    )


def _work_normalized_variance(method: dict[str, Any]) -> float:
    return (
        float(method["sample_variance"])
        * float(method["total_work_at_primary_query_count"])
        / int(method["inferential_units"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = _load_bound(config["artifacts"]["experiment"])
    reference = _load_bound(config["artifacts"]["independent_reference"])
    failures = []
    if reference.get("passed") is not True or reference.get("failures"):
        failures.append("independent reference artifact did not pass its own gates")
    reference_by_cell = {item["cell_id"]: item for item in reference["records"]}
    records = []
    gates = config["gates"]
    required_qualified = set(gates["required_accuracy_qualified"])
    for cell in experiment["cells"]:
        cell_id = str(cell["cell_id"])
        external = reference_by_cell.get(cell_id)
        if external is None:
            failures.append(f"{cell_id}: independent reference missing")
            continue
        if float(external["signal_to_noise"]) < float(
            gates["minimum_reference_signal_to_noise"]
        ):
            failures.append(f"{cell_id}: independent reference SNR failed")
        methods = {item["method"]: item for item in cell["methods"]}
        missing = REQUIRED_PRIMARY_COMPARATORS - set(methods)
        if missing:
            failures.append(f"{cell_id}: missing comparators {sorted(missing)}")
        candidate = methods.get("v15_cm_transport")
        if candidate is None:
            failures.append(f"{cell_id}: candidate missing")
            continue
        exactness = candidate.get("exactness", {})
        if exactness.get("proposal_hash_unchanged") is not True:
            failures.append(f"{cell_id}: candidate proposal changed")
        if float(exactness.get("maximum_likelihood_bound_violation", math.inf)) > float(
            gates["maximum_likelihood_bound_violation"]
        ):
            failures.append(f"{cell_id}: candidate likelihood bound failed")
        candidate_z = _accuracy_z(candidate, external)
        if candidate_z > float(gates["maximum_accuracy_z"]):
            failures.append(f"{cell_id}: candidate external accuracy failed")
        candidate_wnv = _work_normalized_variance(candidate)
        comparator_records = []
        accurate = set()
        ratios = {}
        for method_name in sorted(REQUIRED_PRIMARY_COMPARATORS):
            method = methods[method_name]
            z_score = _accuracy_z(method, external)
            qualified = (
                z_score <= float(gates["maximum_accuracy_z"])
                and float(method["sample_variance"]) > 0.0
            )
            ratio = None
            if qualified:
                accurate.add(method_name)
                ratio = _work_normalized_variance(method) / candidate_wnv
                ratios[method_name] = ratio
            comparator_records.append(
                {
                    "method": method_name,
                    "estimate": float(method["estimate"]),
                    "standard_error": float(method["standard_error"]),
                    "external_accuracy_z": z_score,
                    "accuracy_qualified": qualified,
                    "work_ratio_over_v16": ratio,
                }
            )
        if len(accurate) < int(gates["minimum_accuracy_qualified_comparators"]):
            failures.append(f"{cell_id}: too few accuracy-qualified comparators")
        missing_required = required_qualified - accurate
        if missing_required:
            failures.append(
                f"{cell_id}: required comparators failed accuracy {sorted(missing_required)}"
            )
        minimum_ratio = min(ratios.values()) if ratios else 0.0
        if minimum_ratio < float(gates["minimum_work_ratio"]):
            failures.append(f"{cell_id}: external-reference work gate failed")
        records.append(
            {
                "cell_id": cell_id,
                "independent_reference_estimate": float(external["estimate"]),
                "independent_reference_standard_error": float(
                    external["standard_error"]
                ),
                "independent_reference_signal_to_noise": float(
                    external["signal_to_noise"]
                ),
                "candidate_estimate": float(candidate["estimate"]),
                "candidate_standard_error": float(candidate["standard_error"]),
                "candidate_external_accuracy_z": candidate_z,
                "accuracy_qualified_comparators": sorted(accurate),
                "minimum_qualified_comparator_over_v16_work_ratio": minimum_ratio,
                "comparators": comparator_records,
            }
        )
    payload = {
        "schema": "npi.g11.v16-deep-tail-closure-audit.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "artifact_bindings": config["artifacts"],
        "source_provenance": git_source_provenance(ROOT),
        "passed": not failures,
        "failures": failures,
        "cells": records,
        "claim_boundary": {
            "supersedes_v1_low_SNR_reference_decision": True,
            "independent_reference_family": True,
            "deep_tail_development_evidence": True,
            "untouched_external_reproduction": False,
            "universal_dominance_claim": False,
        },
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": not failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
