"""Fail-closed audit of the V8 P5 outcome-blind reference/matrix design."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.baseline_framework import BASELINE_METHODS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-p5-reference-matrix-design.v1"
REPORT = "npi.g11.v8-p5-reference-matrix-audit.v1"
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "baseline_framework_ledger_sha256",
    "primary_model",
    "primary_tasks",
    "nominal_probabilities",
    "threshold_policy",
    "reference_contract",
    "baseline_matrix",
    "robustness_one_factor_at_a_time",
    "mesh_matrix",
    "decision",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_design(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unexpected P5 reference-matrix schema")
    return value, hashlib.sha256(raw).hexdigest()


def audit_design(design: dict[str, Any], digest: str) -> dict[str, Any]:
    model = design.get("primary_model", {})
    reference = design.get("reference_contract", {})
    baseline = design.get("baseline_matrix", {})
    threshold = design.get("threshold_policy", {})
    mesh = design.get("mesh_matrix", {})
    decision = design.get("decision", {})
    upstream = ROOT / "configs/g11_v8/baseline_framework_ledger_v1.yaml"
    hursts = model.get("hurst_values", [])
    tasks = design.get("primary_tasks", [])
    probabilities = design.get("nominal_probabilities", [])
    namespaces = [
        threshold.get("calibration_seed_namespace"),
        reference.get("reference_seed_namespace"),
        reference.get("final_seed_namespace"),
        baseline.get("training_seed_namespace"),
        baseline.get("pilot_seed_namespace"),
        baseline.get("final_seed_namespace"),
    ]
    checks = {
        "schema_exact": design.get("schema") == SCHEMA,
        "root_keys_exact": set(design) == ROOT_KEYS,
        "phase_exact": design.get("phase") == "p5_development",
        "outcome_blind": design.get("outcome_data_used") is False,
        "upstream_hash_bound": upstream.is_file()
        and design.get("baseline_framework_ledger_sha256") == _sha(upstream),
        "primary_hursts_exact": hursts == [0.05, 0.12, 0.20],
        "primary_tasks_exact": tasks == ["terminal_left_tail", "discrete_lower_barrier"],
        "primary_probabilities_exact": probabilities == [0.01, 0.001, 0.0001, 0.00001],
        "primary_cell_count_exact": len(hursts) * len(tasks) * len(probabilities) == 24,
        "threshold_calibration_required": threshold.get("calibration_required_before_reference") is True,
        "threshold_calibration_independent": threshold.get("calibration_must_be_independent_of_reference_and_final") is True,
        "threshold_hash_required": threshold.get("calibrated_threshold_is_bound_by_hash") is True,
        "reference_methods_exact": reference.get("methods") == ["dcs_reference", "raw_crosscheck"],
        "reference_paths_independent": reference.get("separate_code_paths_from_final_methods") is True,
        "reference_se_strict": reference.get("maximum_reference_se_fraction_of_final_target") == 0.10,
        "reference_uncertainty_included": reference.get("reference_uncertainty_enters_accuracy_calculation") is True,
        "crosscheck_count_exact": reference.get("minimum_crosscheck_methods") == 2,
        "crosscheck_z_bound_exact": reference.get("maximum_combined_z_score") == 4.0,
        "baseline_methods_exact": baseline.get("methods") == list(BASELINE_METHODS),
        "fresh_training_required": baseline.get("fresh_training_per_task_and_cell") is True,
        "seed_namespaces_unique": all(isinstance(x, str) and x for x in namespaces) and len(set(namespaces)) == len(namespaces),
        "mesh_steps_exact": mesh.get("steps") == [32, 64, 128, 256, 512],
        "mesh_diagnostics_complete": mesh.get("diagnostics") == ["coefficient", "active_time", "fine_only_crossing", "threshold", "weak_bias", "correction_variance", "cost"],
        "actual_thresholds_open": decision.get("actual_thresholds_bound") is False,
        "actual_references_open": decision.get("actual_references_complete") is False,
        "performance_refused": decision.get("performance_claim_authorized") is False,
        "p6_authorized": decision.get("p6_statistical_design_authorized") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {"schema": REPORT, "design_sha256": digest, "checks": checks, "failure_count": len(failures), "failures": failures, "passed": not failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    design, digest = load_design(args.design)
    report = audit_design(design, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
