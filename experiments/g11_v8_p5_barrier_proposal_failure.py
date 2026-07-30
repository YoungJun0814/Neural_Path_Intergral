"""Independent audit of the failed rank-one barrier-proposal falsification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_barrier_proposal_falsification import (
    RESULT_SCHEMA,
    _seeds,
    load_falsification_config,
)
from src.path_integral import derive_seed
from src.path_integral.reference_protocol import canonical_sha256

AUDIT_SCHEMA = "npi.g11.v8-p5-barrier-reference-proposal-failure-audit.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("barrier-proposal result must be a mapping")
    return value


def _candidate_method_optima(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    optima: list[dict[str, Any]] = []
    all_method_optima_pass = True
    for cell_id in result["cells"]:
        method_records: dict[str, dict[str, Any]] = {}
        joint_records: list[tuple[float, str]] = []
        for candidate in result["candidates"]:
            entries = {
                entry["method"]: entry
                for entry in candidate["entries"]
                if entry["cell_id"] == cell_id
            }
            joint_records.append(
                (
                    max(
                        float(entry["requested_to_cap_ratio"])
                        for entry in entries.values()
                    ),
                    str(candidate["candidate_id"]),
                )
            )
            for method, entry in entries.items():
                record = {
                    "candidate_id": candidate["candidate_id"],
                    "requested_to_cap_ratio": entry["requested_to_cap_ratio"],
                    "projected_final_samples": entry["projected_final_samples"],
                    "projected_cap_pass": entry["projected_cap_pass"],
                }
                previous = method_records.get(method)
                if previous is None or float(record["requested_to_cap_ratio"]) < float(
                    previous["requested_to_cap_ratio"]
                ):
                    method_records[method] = record
        joint_ratio, joint_candidate = min(joint_records)
        method_optima_pass = all(
            bool(record["projected_cap_pass"]) for record in method_records.values()
        )
        all_method_optima_pass = all_method_optima_pass and method_optima_pass
        optima.append(
            {
                "cell_id": cell_id,
                "best_joint_candidate_id": joint_candidate,
                "best_joint_requested_to_cap_ratio": joint_ratio,
                "method_optima": method_records,
                "all_method_optima_pass": method_optima_pass,
            }
        )
    return optima, all_method_optima_pass


def audit_barrier_proposal_failure(
    config_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    config, config_sha256 = load_falsification_config(config_path)
    result = _load_json(result_path)
    gates = config["gates"]
    expected_seed_records = []
    for candidate in config["candidates"]:
        for cell_id in config["cells"]:
            for replicate in range(int(config["sampling"]["replicates"])):
                for key in _seeds(
                    config["protocol_id"],
                    config["namespace"],
                    candidate["id"],
                    cell_id,
                    replicate,
                ):
                    expected_seed_records.append(
                        {"key": key.__dict__, "seed": derive_seed(key)}
                    )
    entry_matrix_exact = True
    allocation_formula_exact = True
    candidate_gate_logic_exact = True
    for candidate in result.get("candidates", []):
        entries = candidate.get("entries", [])
        expected_matrix = {
            (cell_id, method)
            for cell_id in config["cells"]
            for method in ("dcs_reference", "raw_crosscheck")
        }
        if {
            (entry.get("cell_id"), entry.get("method"))
            for entry in entries
            if isinstance(entry, dict)
        } != expected_matrix:
            entry_matrix_exact = False
            continue
        for entry in entries:
            variance = max(float(value) for value in entry["replicate_variances"])
            requested = max(
                8192,
                math.ceil(
                    float(gates["allocation_safety_factor"])
                    * variance
                    / float(entry["target_standard_error"]) ** 2
                ),
            )
            allocation_formula_exact = allocation_formula_exact and (
                math.isclose(
                    variance,
                    float(entry["allocation_design_variance"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and requested == int(entry["projected_final_samples"])
                and bool(entry["projected_cap_pass"])
                == (requested <= int(gates["maximum_final_samples"]))
            )
        expected_gates = {
            "all_projected_caps_pass": all(
                bool(entry["projected_cap_pass"]) for entry in entries
            ),
            "all_raw_coverage_pass": all(
                bool(entry["raw_coverage_pass"]) for entry in entries
            ),
            "likelihood_normalization_pass": abs(
                float(candidate["normalization_z"])
            )
            <= float(gates["maximum_likelihood_normalization_absolute_z"]),
            "worst_ratio_pass": float(candidate["worst_requested_to_cap_ratio"])
            <= float(gates["selected_worst_requested_to_cap_ratio"]),
        }
        candidate_gate_logic_exact = candidate_gate_logic_exact and (
            candidate["gates"] == expected_gates
            and bool(candidate["passes"]) == all(expected_gates.values())
        )
    optima, all_method_optima_pass = _candidate_method_optima(result)
    failing_method_optima = [
        {
            "cell_id": cell["cell_id"],
            "method": method,
            **record,
        }
        for cell in optima
        for method, record in cell["method_optima"].items()
        if not record["projected_cap_pass"]
    ]
    checks = {
        "schema_and_protocol_exact": result.get("schema") == RESULT_SCHEMA
        and result.get("protocol_id") == config["protocol_id"],
        "config_and_namespace_exact": result.get("config_sha256") == config_sha256
        and result.get("namespace") == config["namespace"],
        "formal_sampling_roster_exact": result.get("smoke") is False
        and result.get("cells") == config["cells"]
        and result.get("replicates") == config["sampling"]["replicates"]
        and result.get("paths_per_replicate")
        == config["sampling"]["paths_per_replicate"],
        "candidate_roster_exact": [
            candidate.get("candidate_id") for candidate in result.get("candidates", [])
        ]
        == [candidate["id"] for candidate in config["candidates"]],
        "entry_matrix_exact": entry_matrix_exact,
        "allocation_formula_exact": allocation_formula_exact,
        "candidate_gate_logic_exact": candidate_gate_logic_exact,
        "seed_roster_exact_and_unique": result.get("seed_records")
        == expected_seed_records
        and result.get("seed_count") == len(expected_seed_records)
        and len({record["seed"] for record in expected_seed_records})
        == len(expected_seed_records),
        "clean_frozen_source": result.get("dirty_worktree") is False
        and isinstance(result.get("source_commit"), str)
        and len(result["source_commit"]) == 40,
        "no_proposal_passed": result.get("passed") is False
        and result.get("selected_candidate_id") is None
        and not any(
            bool(candidate.get("passes"))
            for candidate in result.get("candidates", [])
        ),
        "method_specific_reuse_also_falsified": all_method_optima_pass is False
        and len(failing_method_optima) == 2,
        "decision_fail_closed": result.get("decision")
        == {
            "status": "barrier_proposal_falsification_fail",
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
        "cell_and_method_optima": optima,
        "failing_method_optima": failing_method_optima,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": "rank_one_candidate_roster_falsified",
            "existing_candidate_reuse_authorized": False,
            "new_cell_tuned_proposal_required": True,
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
    report = audit_barrier_proposal_failure(arguments.config, arguments.result)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(
                f"refusing to overwrite barrier-proposal failure audit: {arguments.output}"
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
