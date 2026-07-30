"""Audit the V3 all-failed-cells CEM development result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_cell_tuned_cem_proposal import (
    EXPECTED_V3_TARGETS,
    RESULT_SCHEMAS,
    SCHEMA_V3,
    _training_seed,
    _validation_seeds,
    load_cell_tuned_config,
)
from src.path_integral.reference_protocol import canonical_sha256

AUDIT_SCHEMA = "npi.g11.v8-p5-all-failed-cells-cem-audit.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V3 CEM result must be a mapping")
    return value


def audit_all_failed_cells_cem(
    config_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    config, config_sha256 = load_cell_tuned_config(config_path)
    if config["schema"] != SCHEMA_V3:
        raise ValueError("all-failed-cells audit requires the V3 protocol")
    result = _load_json(result_path)
    validation = config["validation"]
    fits = result.get("training_fits", [])
    expected_fit_keys = {
        (cell_id, replicate)
        for cell_id in EXPECTED_V3_TARGETS
        for replicate in range(int(config["training"]["training_seed_replicates"]))
    }
    actual_fit_keys = {
        (fit.get("cell_id"), fit.get("training_replicate"))
        for fit in fits
        if isinstance(fit, dict)
    }
    fit_roster_exact = actual_fit_keys == expected_fit_keys
    fit_constraints_exact = fit_roster_exact and all(
        fit.get("converged") is True
        and len(fit.get("history", [])) >= 2
        and all(
            math.isfinite(float(value))
            for pair in fit.get("control", [])
            for value in pair
        )
        and all(float(pair[1]) <= -0.05 for pair in fit.get("control", []))
        for fit in fits
    )
    fit_map = {
        (fit["cell_id"], fit["training_replicate"]): fit for fit in fits
    }
    candidates = result.get("candidates", [])
    expected_candidate_count = (
        len(expected_fit_keys) * len(config["proposal_families"])
    )
    candidate_roster_exact = len(candidates) == expected_candidate_count
    rank_one_exact = candidate_roster_exact
    allocation_exact = candidate_roster_exact
    diagnostic_gates_exact = candidate_roster_exact
    for candidate in candidates:
        parts = candidate["candidate_id"].split("/")
        fit = fit_map[(parts[0], int(parts[1][6:]))]
        family = next(
            family
            for family in config["proposal_families"]
            if family["id"] == parts[2]
        )
        expected_schedules = [
            [
                [float(scale) * float(pair[0]), float(scale) * float(pair[1])]
                for pair in fit["control"]
            ]
            for scale in family["scales"]
        ]
        rank_one_exact = rank_one_exact and (
            candidate.get("schedules") == expected_schedules
            and candidate.get("weights") == family["weights"]
            and all(
                float(pair[1]) < 0.0
                for schedule in expected_schedules[1:]
                for pair in schedule
            )
        )
        entries = candidate.get("entries", [])
        if candidate.get("numerical_failure") is not None or len(entries) != 2:
            allocation_exact = False
            diagnostic_gates_exact = False
            continue
        entry_by_method = {entry["method"]: entry for entry in entries}
        for entry in entries:
            variance = max(float(value) for value in entry["replicate_variances"])
            requested = max(
                8192,
                math.ceil(
                    float(validation["allocation_safety_factor"])
                    * variance
                    / float(entry["target_standard_error"]) ** 2
                ),
            )
            ordered = sorted(float(value) for value in entry["replicate_variances"])
            median = 0.5 * (ordered[3] + ordered[4])
            variance_ratio = (
                variance / median
                if median > 0.0
                else (1.0 if variance == 0.0 else math.inf)
            )
            allocation_exact = allocation_exact and (
                variance == float(entry["allocation_design_variance"])
                and requested == int(entry["projected_final_samples"])
                and variance_ratio
                == float(entry["replicate_variance_to_median_ratio"])
            )
        raw_entry = entry_by_method["raw_crosscheck"]
        expected_common = {
            "raw_coverage_pass": min(raw_entry["raw_nonzero_counts"])
            >= int(validation["minimum_raw_nonzero_contributions_per_replicate"]),
            "likelihood_normalization_pass": abs(
                float(candidate["normalization_z"])
            )
            <= float(validation["maximum_likelihood_normalization_absolute_z"]),
        }
        expected_method_gates = {}
        for method in candidate["target_methods"]:
            entry = entry_by_method[method]
            expected_method_gates[method] = {
                "target_method_margin_pass": float(
                    entry["requested_to_cap_ratio"]
                )
                <= float(
                    validation[
                        "selected_method_maximum_requested_to_cap_ratio"
                    ]
                ),
                "replicate_variance_stability_pass": float(
                    entry["replicate_variance_to_median_ratio"]
                )
                <= float(
                    validation["maximum_replicate_variance_to_median_ratio"]
                ),
                "single_contribution_concentration_pass": max(
                    float(value)
                    for value in entry[
                        "replicate_maximum_contribution_shares"
                    ]
                )
                <= float(validation["maximum_single_contribution_share"]),
            }
        expected_gates = {
            **expected_common,
            "all_target_methods_pass": all(
                all(gates.values()) for gates in expected_method_gates.values()
            ),
        }
        diagnostic_gates_exact = diagnostic_gates_exact and (
            candidate["method_gates"] == expected_method_gates
            and candidate["gates"] == expected_gates
            and bool(candidate["passes"]) == all(expected_gates.values())
        )
    expected_seed_records: list[dict[str, Any]] = []
    for cell in config["cells"]:
        for replicate in range(int(config["training"]["training_seed_replicates"])):
            key, seed = _training_seed(config, cell["cell_id"], replicate)
            expected_seed_records.append({"key": asdict(key), "seed": seed})
    for fit in fits:
        for family in config["proposal_families"]:
            candidate_id = (
                f"{fit['cell_id']}/train-{fit['training_replicate']}/"
                f"{family['id']}"
            )
            for replicate in range(int(validation["replicates"])):
                expected_seed_records.extend(
                    {"key": asdict(key), "seed": seed}
                    for key, seed in _validation_seeds(
                        config, fit["cell_id"], candidate_id, replicate
                    )
                )
    selected = result.get("selected_proposals", {})
    selected_pairs = {
        (cell_id, method)
        for cell_id, methods in selected.items()
        for method in methods
    }
    required_pairs = {
        (cell_id, method)
        for cell_id, methods in EXPECTED_V3_TARGETS.items()
        for method in methods
    }
    missing_pairs = sorted(required_pairs - selected_pairs)
    selection_exact = True
    for cell_id, method in selected_pairs:
        passing = [
            candidate
            for candidate in candidates
            if candidate["cell_id"] == cell_id
            and candidate.get("numerical_failure") is None
            and candidate["gates"]["raw_coverage_pass"]
            and candidate["gates"]["likelihood_normalization_pass"]
            and all(candidate["method_gates"][method].values())
        ]
        expected = min(
            passing,
            key=lambda candidate: next(
                entry["requested_to_cap_ratio"]
                for entry in candidate["entries"]
                if entry["method"] == method
            ),
        )
        selection_exact = selection_exact and (
            selected[cell_id][method]["candidate_id"] == expected["candidate_id"]
        )
    seed_values = [record["seed"] for record in expected_seed_records]
    checks = {
        "schema_protocol_config_exact": result.get("schema")
        == RESULT_SCHEMAS[SCHEMA_V3]
        and result.get("protocol_id") == config["protocol_id"]
        and result.get("config_sha256") == config_sha256,
        "formal_clean_execution": result.get("smoke") is False
        and result.get("dirty_worktree") is False
        and isinstance(result.get("source_commit"), str)
        and len(result["source_commit"]) == 40,
        "fit_roster_and_constraints_exact": fit_constraints_exact,
        "candidate_roster_exact": candidate_roster_exact,
        "rank_one_schedules_exact": rank_one_exact,
        "allocation_formula_exact": allocation_exact,
        "diagnostic_gates_exact": diagnostic_gates_exact,
        "seed_roster_exact_and_unique": result.get("seed_records")
        == expected_seed_records
        and result.get("seed_count") == len(expected_seed_records)
        and len(seed_values) == len(set(seed_values)),
        "selection_rule_exact": selection_exact,
        "nine_of_eleven_requirements_pass": len(selected_pairs) == 9
        and len(missing_pairs) == 2,
        "two_missing_requirements_exact": missing_pairs
        == [
            ("h0.05-discrete_lower_barrier-p1e-05", "raw_crosscheck"),
            ("h0.05-terminal_left_tail-p1e-05", "raw_crosscheck"),
        ],
        "decision_fail_closed": result.get("passed") is False
        and result.get("decision")
        == {
            "status": "cell_tuned_cem_proposal_falsification_fail",
            "selected_proposal_frozen": False,
            "new_full_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
        },
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "config_sha256": config_sha256,
        "result_file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "result_canonical_sha256": canonical_sha256(result),
        "checks": checks,
        "selected_requirement_count": len(selected_pairs),
        "required_requirement_count": len(required_pairs),
        "missing_requirements": [
            {"cell_id": cell_id, "method": method}
            for cell_id, method in missing_pairs
        ],
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "all_failed_cells_cem_audit_pass"
                if not failures
                else "all_failed_cells_cem_audit_fail"
            ),
            "partial_selections_are_development_only": True,
            "proposal_manifest_freeze_authorized": False,
            "targeted_dense_mixture_development_required": not failures,
            "new_full_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit_all_failed_cells_cem(arguments.config, arguments.result)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(
                f"refusing to overwrite all-failed-cells audit: {arguments.output}"
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
