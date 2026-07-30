"""Audit weight-optimized proposal development without promoting partial wins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_weight_optimized_proposal import (
    DCS_METHOD,
    RAW_METHOD,
    load_weight_optimized_config,
)

AUDIT_SCHEMA = "npi.g11.v8-p5-weight-optimized-proposal-audit.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def audit_weight_optimized_result(
    config_path: Path, result_path: Path
) -> dict[str, Any]:
    config, config_sha256 = load_weight_optimized_config(config_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("weight-optimized result must be a mapping")
    candidates = result.get("candidates")
    fits = result.get("weight_fits")
    seeds = result.get("seed_records")
    if (
        not isinstance(candidates, list)
        or not isinstance(fits, list)
        or not isinstance(seeds, list)
    ):
        raise ValueError("weight-optimized result roster is malformed")
    passing: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        if candidate.get("passes") is True:
            passing.setdefault(
                (str(candidate["cell_id"]), str(candidate["method"])), []
            ).append(candidate)
    expected_new = {
        key: min(
            values,
            key=lambda candidate: float(candidate["requested_to_cap_ratio"]),
        )["candidate_id"]
        for key, values in passing.items()
    }
    actual_new = {
        (str(cell_id), str(method)): selected["candidate_id"]
        for cell_id, methods in result.get("new_selected_proposals", {}).items()
        for method, selected in methods.items()
    }
    derived_seeds = [int(record["seed"]) for record in seeds]
    checks = {
        "schema_config_protocol_clean_source_exact": result.get("schema")
        == "npi.g11.v8-p5-weight-optimized-proposal-result.v1"
        and result.get("config_sha256") == config_sha256
        and result.get("protocol_id") == config["protocol_id"]
        and result.get("dirty_worktree") is False
        and isinstance(result.get("source_commit"), str)
        and len(result["source_commit"]) == 40,
        "strict_json_finite": _finite(result),
        "candidate_and_seed_rosters_exact": len(candidates) == 17
        and len({candidate["candidate_id"] for candidate in candidates}) == 17
        and sum(candidate["method"] == RAW_METHOD for candidate in candidates) == 3
        and sum(candidate["method"] == DCS_METHOD for candidate in candidates) == 14
        and len(derived_seeds) == 432
        and len(derived_seeds) == len(set(derived_seeds)),
        "raw_weight_fit_contract_exact": len(fits) == 3
        and all(
            math.isclose(sum(float(value) for value in fit["optimized_weights"]), 1.0)
            and math.isclose(float(fit["optimized_weights"][0]), 0.08)
            and all(float(value) > 0.0 for value in fit["optimized_weights"])
            and float(fit["optimized_empirical_second_moment"])
            <= float(fit["base_empirical_second_moment"]) * (1.0 + 1e-12)
            for fit in fits
        ),
        "candidate_gates_and_selection_exact": all(
            bool(candidate["passes"])
            == all(bool(value) for value in candidate["gates"].values())
            for candidate in candidates
        )
        and actual_new == expected_new,
        "partial_falsification_exact": result.get("passed") is False
        and int(result.get("selected_count", -1)) == 4
        and int(result.get("required_selection_count", -1)) == 8
        and len(actual_new) == 1,
        "decision_fail_closed": result.get("decision", {}).get(
            "proposal_manifest_build_authorized"
        )
        is False
        and result.get("decision", {}).get("new_formal_pilot_authorized") is False
        and result.get("decision", {}).get("final_execution_authorized") is False
        and result.get("decision", {}).get("performance_claim_authorized") is False
        and result.get("decision", {}).get("submission_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "config_sha256": config_sha256,
        "result_file_sha256": _sha256(result_path),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "weight_optimization_partial_falsification_confirmed"
                if not failures
                else "weight_optimization_audit_failure"
            ),
            "weight_training_namespace_burned": True,
            "validation_namespace_burned": True,
            "partial_candidates_promoted": False,
            "method_role_precision_redesign_required": True,
            "proposal_manifest_build_authorized": False,
            "new_formal_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"refusing to overwrite weight-optimized audit: {arguments.output}"
        )
    report = audit_weight_optimized_result(
        arguments.config, arguments.result
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], **report["decision"]}))


if __name__ == "__main__":
    main()
