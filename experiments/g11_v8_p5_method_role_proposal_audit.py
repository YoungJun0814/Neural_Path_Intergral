"""Audit the method-role proposal result and preserve its failed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_method_role_proposal import (
    DCS_METHOD,
    RAW_METHOD,
    load_method_role_config,
)

AUDIT_SCHEMA = "npi.g11.v8-p5-method-role-proposal-audit.v1"


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


def _nominal_probability(cell_id: str) -> float:
    suffix = cell_id.rsplit("-p", 1)[-1]
    if not suffix.startswith("1e-"):
        raise ValueError("cell id lacks a supported nominal probability")
    return 10.0 ** (-int(suffix[3:]))


def audit_method_role_result(
    config_path: Path, result_path: Path
) -> dict[str, Any]:
    config, config_sha256 = load_method_role_config(config_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("method-role result must be a mapping")
    candidates = result.get("candidates")
    seed_records = result.get("seed_records")
    if not isinstance(candidates, list) or not isinstance(seed_records, list):
        raise ValueError("method-role candidate or seed roster is malformed")
    seeds = [int(record["seed"]) for record in seed_records]
    precision_exact = all(
        math.isclose(
            float(candidate["target_standard_error"]),
            _nominal_probability(str(candidate["cell_id"]))
            * (0.02 if candidate["method"] == DCS_METHOD else 0.05),
        )
        for candidate in candidates
    )
    checks = {
        "schema_config_protocol_clean_source_exact": result.get("schema")
        == "npi.g11.v8-p5-method-role-proposal-result.v1"
        and result.get("config_sha256") == config_sha256
        and result.get("protocol_id") == config["protocol_id"]
        and result.get("dirty_worktree") is False
        and isinstance(result.get("source_commit"), str)
        and len(result["source_commit"]) == 40,
        "strict_json_finite": _finite(result),
        "candidate_seed_roster_exact": len(candidates) == 17
        and len({candidate["candidate_id"] for candidate in candidates}) == 17
        and sum(candidate["method"] == RAW_METHOD for candidate in candidates) == 9
        and sum(candidate["method"] == DCS_METHOD for candidate in candidates) == 8
        and len(seeds) == 408
        and len(seeds) == len(set(seeds)),
        "method_precision_exact": precision_exact
        and result.get("method_role_precision", {}).get(
            "raw_may_replace_primary_reference"
        )
        is False
        and result.get("method_role_precision", {}).get("self_normalized") is False,
        "candidate_gates_exact": all(
            bool(candidate["passes"])
            == all(bool(value) for value in candidate["gates"].values())
            for candidate in candidates
        ),
        "falsification_exact": result.get("passed") is False
        and int(result.get("selected_count", -1)) == 4
        and int(result.get("required_selection_count", -1)) == 8
        and result.get("new_selected_proposals") == {},
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
                "method_role_partial_falsification_confirmed"
                if not failures
                else "method_role_audit_failure"
            ),
            "validation_namespace_burned": True,
            "partial_candidates_promoted": False,
            "reference_only_resource_escalation_required": True,
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
            f"refusing to overwrite method-role audit: {arguments.output}"
        )
    report = audit_method_role_result(arguments.config, arguments.result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], **report["decision"]}))


if __name__ == "__main__":
    main()
