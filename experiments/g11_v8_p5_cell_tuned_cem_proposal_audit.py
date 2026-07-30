"""Independent audit for the V2 cell-tuned CEM proposal result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_cell_tuned_cem_proposal import (
    RESULT_SCHEMAS,
    SCHEMA_V2,
    _training_seed,
    _validation_seeds,
    load_cell_tuned_config,
)
from src.path_integral.reference_protocol import canonical_sha256

AUDIT_SCHEMA = "npi.g11.v8-p5-cell-tuned-cem-proposal-audit.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cell-tuned proposal result must be a mapping")
    return value


def audit_cell_tuned_proposal(
    config_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    config, config_sha256 = load_cell_tuned_config(config_path)
    if config["schema"] != SCHEMA_V2:
        raise ValueError("the formal cell-tuned audit requires the V2 protocol")
    result = _load_json(result_path)
    validation = config["validation"]
    expected_fits = {
        (cell["cell_id"], replicate)
        for cell in config["cells"]
        for replicate in range(int(config["training"]["training_seed_replicates"]))
    }
    fits = result.get("training_fits", [])
    fit_matrix_exact = {
        (fit.get("cell_id"), fit.get("training_replicate"))
        for fit in fits
        if isinstance(fit, dict)
    } == expected_fits
    fit_map = {
        (fit["cell_id"], fit["training_replicate"]): fit
        for fit in fits
        if isinstance(fit, dict)
    }
    fit_constraints_exact = fit_matrix_exact and all(
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
    expected_candidates = {
        (
            cell["cell_id"],
            replicate,
            family["id"],
        )
        for cell in config["cells"]
        for replicate in range(int(config["training"]["training_seed_replicates"]))
        for family in config["proposal_families"]
    }
    candidates = result.get("candidates", [])
    candidate_map: dict[tuple[str, int, str], dict[str, Any]] = {}
    for candidate in candidates:
        parts = str(candidate.get("candidate_id", "")).split("/")
        if len(parts) != 3 or not parts[1].startswith("train-"):
            continue
        candidate_map[(parts[0], int(parts[1][6:]), parts[2])] = candidate
    candidate_matrix_exact = set(candidate_map) == expected_candidates
    rank_one_schedules_exact = candidate_matrix_exact
    allocation_formula_exact = candidate_matrix_exact
    gate_logic_exact = candidate_matrix_exact
    for (cell_id, replicate, family_id), candidate in candidate_map.items():
        fit = fit_map[(cell_id, replicate)]
        family = next(
            family
            for family in config["proposal_families"]
            if family["id"] == family_id
        )
        expected_schedules = [
            [
                [float(scale) * float(pair[0]), float(scale) * float(pair[1])]
                for pair in fit["control"]
            ]
            for scale in family["scales"]
        ]
        rank_one_schedules_exact = rank_one_schedules_exact and (
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
            allocation_formula_exact = False
            gate_logic_exact = False
            continue
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
            allocation_formula_exact = allocation_formula_exact and (
                variance == float(entry["allocation_design_variance"])
                and requested == int(entry["projected_final_samples"])
                and math.isclose(
                    requested / int(validation["maximum_final_samples"]),
                    float(entry["requested_to_cap_ratio"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and bool(entry["projected_cap_pass"])
                == (requested <= int(validation["maximum_final_samples"]))
            )
        target_entry = next(
            entry
            for entry in entries
            if entry["method"] == candidate["target_method"]
        )
        raw_entry = next(
            entry for entry in entries if entry["method"] == "raw_crosscheck"
        )
        expected_gates = {
            "target_method_margin_pass": float(
                target_entry["requested_to_cap_ratio"]
            )
            <= float(
                validation["selected_method_maximum_requested_to_cap_ratio"]
            ),
            "raw_coverage_pass": bool(raw_entry["raw_coverage_pass"]),
            "likelihood_normalization_pass": abs(
                float(candidate["normalization_z"])
            )
            <= float(validation["maximum_likelihood_normalization_absolute_z"]),
        }
        gate_logic_exact = gate_logic_exact and (
            candidate["gates"] == expected_gates
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
    selection_exact = set(selected) == {
        cell["cell_id"] for cell in config["cells"]
    }
    selected_summary: list[dict[str, Any]] = []
    for specification in config["cells"]:
        cell_id = specification["cell_id"]
        target_method = specification["target_method"]
        passing = [
            candidate
            for candidate in candidates
            if candidate["cell_id"] == cell_id and candidate["passes"]
        ]
        expected = min(
            passing,
            key=lambda candidate: next(
                entry["requested_to_cap_ratio"]
                for entry in candidate["entries"]
                if entry["method"] == target_method
            ),
        )
        actual = selected.get(cell_id, {})
        selection_exact = selection_exact and (
            actual.get("candidate_id") == expected["candidate_id"]
            and actual.get("target_method") == target_method
            and actual.get("weights") == expected["weights"]
            and actual.get("schedules") == expected["schedules"]
        )
        selected_summary.append(
            {
                "cell_id": cell_id,
                "target_method": target_method,
                "candidate_id": actual.get("candidate_id"),
                "requested_to_cap_ratio": actual.get(
                    "target_method_entry", {}
                ).get("requested_to_cap_ratio"),
                "projected_final_samples": actual.get(
                    "target_method_entry", {}
                ).get("projected_final_samples"),
            }
        )
    seed_values = [record["seed"] for record in expected_seed_records]
    checks = {
        "schema_protocol_config_exact": result.get("schema")
        == RESULT_SCHEMAS[SCHEMA_V2]
        and result.get("protocol_id") == config["protocol_id"]
        and result.get("config_sha256") == config_sha256,
        "formal_clean_execution": result.get("smoke") is False
        and result.get("dirty_worktree") is False
        and isinstance(result.get("source_commit"), str)
        and len(result["source_commit"]) == 40,
        "namespace_exact": result.get("training_namespace")
        == config["training_namespace"]
        and result.get("validation_namespace") == config["validation_namespace"],
        "fit_matrix_and_constraints_exact": fit_constraints_exact,
        "candidate_matrix_exact": candidate_matrix_exact,
        "rank_one_schedules_exact": rank_one_schedules_exact,
        "allocation_formula_exact": allocation_formula_exact,
        "gate_logic_exact": gate_logic_exact,
        "seed_roster_exact_and_unique": result.get("seed_records")
        == expected_seed_records
        and result.get("seed_count") == len(expected_seed_records)
        and len(seed_values) == len(set(seed_values)),
        "selection_rule_exact": selection_exact,
        "large_selection_margin": all(
            float(item["requested_to_cap_ratio"]) <= 0.15
            for item in selected_summary
        ),
        "decision_fail_closed": result.get("passed") is True
        and result.get("decision")
        == {
            "status": "cell_tuned_cem_proposal_falsification_pass",
            "selected_proposal_frozen": False,
            "new_full_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
        },
        "development_provenance_exact": result.get(
            "design_informed_by_prior_development_outcomes"
        )
        is True
        and result.get("current_namespace_outcomes_inspected_before_freeze") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "config_sha256": config_sha256,
        "result_file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "result_canonical_sha256": canonical_sha256(result),
        "checks": checks,
        "selected_proposal_summary": selected_summary,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "cell_tuned_cem_proposal_audit_pass"
                if not failures
                else "cell_tuned_cem_proposal_audit_fail"
            ),
            "selection_is_development_only": True,
            "proposal_manifest_freeze_authorized": not failures,
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
    report = audit_cell_tuned_proposal(arguments.config, arguments.result)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(
                f"refusing to overwrite cell-tuned audit: {arguments.output}"
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
