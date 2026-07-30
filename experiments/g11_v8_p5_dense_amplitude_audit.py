"""Audit the partially successful dense-amplitude proposal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_cell_tuned_cem_proposal import (
    _validation_seeds,
)
from experiments.g11_v8_p5_dense_amplitude_proposal import (
    EXPECTED_CELLS,
    RESULT_SCHEMA,
    _load_v3_result,
    load_dense_config,
)
from src.path_integral.reference_protocol import canonical_sha256

AUDIT_SCHEMA = "npi.g11.v8-p5-dense-amplitude-proposal-audit.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dense-amplitude result must be a mapping")
    return value


def audit_dense_amplitude(
    config_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    config, config_sha256 = load_dense_config(config_path)
    source = _load_v3_result(config)
    result = _load_json(result_path)
    validation = config["validation"]
    profiles = {
        (fit["cell_id"], fit["training_replicate"]): fit["control"]
        for fit in source["training_fits"]
        if fit["cell_id"] in EXPECTED_CELLS
    }
    candidates = result.get("candidates", [])
    expected_candidate_ids = {
        f"{cell_id}/v3-train-{replicate}/{family['id']}"
        for cell_id in EXPECTED_CELLS
        for replicate in range(3)
        for family in config["proposal_families"]
    }
    candidate_roster_exact = {
        candidate.get("candidate_id") for candidate in candidates
    } == expected_candidate_ids
    structure_exact = candidate_roster_exact
    allocation_and_gates_exact = candidate_roster_exact
    for candidate in candidates:
        cell_id, train_part, family_id = candidate["candidate_id"].split("/")
        replicate = int(train_part[9:])
        family = next(
            family
            for family in config["proposal_families"]
            if family["id"] == family_id
        )
        expected_schedules = [
            [
                [float(scale) * float(pair[0]), float(scale) * float(pair[1])]
                for pair in profiles[(cell_id, replicate)]
            ]
            for scale in family["scales"]
        ]
        structure_exact = structure_exact and (
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
            allocation_and_gates_exact = False
            continue
        raw = next(entry for entry in entries if entry["method"] == "raw_crosscheck")
        variance = max(float(value) for value in raw["replicate_variances"])
        requested = max(
            8192,
            math.ceil(
                float(validation["allocation_safety_factor"])
                * variance
                / float(raw["target_standard_error"]) ** 2
            ),
        )
        method_gates = {
            "target_method_margin_pass": float(raw["requested_to_cap_ratio"])
            <= float(validation["selected_method_maximum_requested_to_cap_ratio"]),
            "replicate_variance_stability_pass": float(
                raw["replicate_variance_to_median_ratio"]
            )
            <= float(validation["maximum_replicate_variance_to_median_ratio"]),
            "single_contribution_concentration_pass": max(
                float(value)
                for value in raw["replicate_maximum_contribution_shares"]
            )
            <= float(validation["maximum_single_contribution_share"]),
        }
        common_gates = {
            "raw_coverage_pass": min(raw["raw_nonzero_counts"])
            >= int(validation["minimum_raw_nonzero_contributions_per_replicate"]),
            "likelihood_normalization_pass": abs(
                float(candidate["normalization_z"])
            )
            <= float(validation["maximum_likelihood_normalization_absolute_z"]),
            "all_target_methods_pass": all(method_gates.values()),
        }
        allocation_and_gates_exact = allocation_and_gates_exact and (
            requested == raw["projected_final_samples"]
            and variance == raw["allocation_design_variance"]
            and candidate["method_gates"]["raw_crosscheck"] == method_gates
            and candidate["gates"] == common_gates
            and candidate["passes"] == all(common_gates.values())
        )
    expected_seed_records: list[dict[str, Any]] = []
    for candidate in candidates:
        for replicate in range(int(validation["replicates"])):
            expected_seed_records.extend(
                {"key": asdict(key), "seed": seed}
                for key, seed in _validation_seeds(
                    config,
                    candidate["cell_id"],
                    candidate["candidate_id"],
                    replicate,
                )
            )
    selected = result.get("selected_proposals", {})
    selected_exact = set(selected) == {
        "h0.05-discrete_lower_barrier-p1e-05"
    }
    if selected_exact:
        cell_id = "h0.05-discrete_lower_barrier-p1e-05"
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
                if entry["method"] == "raw_crosscheck"
            ),
        )
        selected_exact = (
            selected[cell_id]["candidate_id"] == expected["candidate_id"]
        )
    seed_values = [record["seed"] for record in expected_seed_records]
    checks = {
        "schema_protocol_config_exact": result.get("schema") == RESULT_SCHEMA
        and result.get("protocol_id") == config["protocol_id"]
        and result.get("config_sha256") == config_sha256,
        "formal_clean_execution": result.get("smoke") is False
        and result.get("dirty_worktree") is False,
        "candidate_roster_exact": candidate_roster_exact,
        "rank_one_structure_exact": structure_exact,
        "allocation_and_gates_exact": allocation_and_gates_exact,
        "seed_roster_exact_and_unique": result.get("seed_records")
        == expected_seed_records
        and result.get("seed_count") == len(expected_seed_records)
        and len(seed_values) == len(set(seed_values)),
        "barrier_selected_terminal_missing": selected_exact,
        "decision_fail_closed": result.get("passed") is False
        and result.get("decision")
        == {
            "status": "dense_amplitude_proposal_falsification_fail",
            "selected_proposal_frozen": False,
            "proposal_manifest_freeze_authorized": False,
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
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "dense_amplitude_audit_pass"
                if not failures
                else "dense_amplitude_audit_fail"
            ),
            "barrier_raw_development_selection_available": not failures,
            "terminal_raw_redesign_required": not failures,
            "proposal_manifest_freeze_authorized": False,
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
    report = audit_dense_amplitude(arguments.config, arguments.result)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(
                f"refusing to overwrite dense-amplitude audit: {arguments.output}"
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
