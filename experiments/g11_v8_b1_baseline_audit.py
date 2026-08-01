"""Independent fail-closed audit of the B1 implementation smoke artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "npi.g11.v8-b1-baseline-implementation.v1"
RESULT_SCHEMA = "npi.g11.v8-b1-baseline-implementation-result.v1"
AUDIT_SCHEMA = "npi.g11.v8-b1-baseline-implementation-audit.v1"
METHOD_FAMILY = {
    "crude_mc": "target_gaussian",
    "antithetic_mc": "target_gaussian",
    "conditional_rbergomi": "target_gaussian",
    "pure_cem": "gaussian_shift",
    "defensive_cem": "gaussian_mixture_shift",
    "smoothing_rqmc": "rqmc_target_gaussian",
    "ld_subspace_is": "gaussian_mixture_shift",
    "flow_is": "coupling_flow",
}
METHOD_UNIT = {
    "crude_mc": "iid_path",
    "antithetic_mc": "antithetic_pair",
    "conditional_rbergomi": "iid_path",
    "pure_cem": "iid_path",
    "defensive_cem": "iid_path",
    "smoothing_rqmc": "rqmc_randomization",
    "ld_subspace_is": "iid_path",
    "flow_is": "iid_path",
}
TRAINED = {"pure_cem", "defensive_cem", "ld_subspace_is", "flow_is"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proposal_hash(proposal: dict[str, Any]) -> str:
    fields = (
        "schema",
        "method",
        "family",
        "task_id",
        "dimension",
        "training_seed",
        "training_budget_work_units",
        "location",
        "component_means",
        "component_weights",
        "flow_split",
        "flow_scale_matrix",
        "flow_scale_bias",
        "flow_shift_matrix",
        "flow_shift_bias",
        "flow_max_log_scale",
        "exact_likelihood",
        "self_normalized",
        "dcs_extension_eligible",
        "conditional_integral",
        "frozen",
        "training_cost",
    )
    payload = {name: proposal.get(name) for name in fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _source_is_ancestor(source_commit: Any) -> bool:
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        return False
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", source_commit, "HEAD"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def audit(config: dict[str, Any], config_sha256: str, result: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    def record(name: str, value: bool) -> None:
        checks[name] = bool(value)

    raw_records = result.get("records")
    records: list[dict[str, Any]] = (
        [item for item in raw_records if isinstance(item, dict)]
        if isinstance(raw_records, list)
        else []
    )
    raw_methods = config.get("methods")
    expected_methods = set(raw_methods) if isinstance(raw_methods, list) else set()
    terminal_id = "h0.12-terminal_left_tail-p1e-02"
    barrier_id = "h0.12-discrete_lower_barrier-p1e-02"
    terminal = [item for item in records if item.get("cell_id") == terminal_id]
    barrier = [item for item in records if item.get("cell_id") == barrier_id]
    record("config_schema_exact", config.get("schema") == CONFIG_SCHEMA)
    record("result_schema_exact", result.get("schema") == RESULT_SCHEMA)
    record("config_hash_exact", result.get("config_sha256") == config_sha256)
    record("protocol_exact", result.get("protocol_id") == config.get("protocol_id"))
    record("namespace_exact", result.get("namespace") == config.get("namespace"))
    record("source_commit_is_ancestor", _source_is_ancestor(result.get("source_commit")))
    record("run_count_exact", len(records) == int(config["gate"]["expected_run_count"]))
    record("terminal_roster_exact", {item.get("method") for item in terminal} == expected_methods)
    record(
        "barrier_roster_exact",
        {item.get("method") for item in barrier} == expected_methods - {"conditional_rbergomi"},
    )

    proposal_hashes = []
    all_seeds: list[int] = []
    structure_ok = bool(records)
    family_ok = bool(records)
    hashes_ok = bool(records)
    lifecycle_ok = bool(records)
    allocation_ok = bool(records)
    cost_ok = bool(records)
    method_contracts_ok = bool(records)
    estimates_ok = bool(records)
    for item in records:
        method = item.get("method")
        artifact = item.get("artifact", {})
        proposal = artifact.get("proposal", {})
        plan = artifact.get("plan", {})
        estimate = artifact.get("estimate", {})
        lifecycle = artifact.get("audit", {})
        structure_ok &= all(
            isinstance(value, dict) for value in (artifact, proposal, plan, estimate)
        )
        family_ok &= (
            method in METHOD_FAMILY
            and proposal.get("method") == method
            and proposal.get("family") == METHOD_FAMILY.get(method)
            and proposal.get("exact_likelihood") is True
            and proposal.get("self_normalized") is False
            and proposal.get("frozen") is True
        )
        observed_hash = proposal.get("sha256")
        hashes_ok &= isinstance(observed_hash, str) and observed_hash == _proposal_hash(proposal)
        if isinstance(observed_hash, str):
            proposal_hashes.append(observed_hash)
        audit_checks = lifecycle.get("checks", [])
        lifecycle_ok &= (
            lifecycle.get("passed") is True
            and bool(audit_checks)
            and all(
                isinstance(pair, list) and len(pair) == 2 and pair[1] is True
                for pair in audit_checks
            )
        )
        points = plan.get("points_per_unit")
        units = plan.get("planned_units")
        final_samples = plan.get("planned_final_samples")
        expected_unit = METHOD_UNIT.get(method) if isinstance(method, str) else None
        allocation_ok &= (
            plan.get("frozen") is True
            and plan.get("proposal_sha256") == observed_hash
            and estimate.get("proposal_sha256") == observed_hash
            and estimate.get("inferential_unit") == expected_unit
            and estimate.get("ordinary_mean") is True
            and estimate.get("likelihood_clipped") is False
            and isinstance(points, int)
            and isinstance(units, int)
            and final_samples == points * units == estimate.get("final_sample_count")
        )
        training_cost = proposal.get("training_cost", {})
        planning_cost = plan.get("planning_cost", {})
        final_cost = estimate.get("final_cost", {})
        trained_cost_ok = method not in TRAINED or (
            training_cost.get("algorithmic_work_units", 0.0) > 0.0
            and training_cost.get("wall_seconds", 0.0) > 0.0
            and training_cost.get("peak_memory_bytes", 0) > 0
        )
        cost_ok &= (
            trained_cost_ok
            and planning_cost.get("planning_samples", 0) > 0
            and planning_cost.get("algorithmic_work_units", 0.0) > 0.0
            and final_cost.get("final_samples") == final_samples
            and final_cost.get("algorithmic_work_units", 0.0) > 0.0
            and final_cost.get("wall_seconds", 0.0) > 0.0
            and final_cost.get("peak_memory_bytes", 0) > 0
        )
        if method == "conditional_rbergomi":
            method_contracts_ok &= (
                item.get("cell_id") == terminal_id
                and proposal.get("conditional_integral") == "analytic_gaussian_cdf"
                and final_cost.get("cdf_calls") == final_samples
            )
        elif method == "smoothing_rqmc":
            method_contracts_ok &= (
                proposal.get("conditional_integral") == "analytic_gaussian_cdf"
                and isinstance(points, int)
                and points > 0
                and points & (points - 1) == 0
                and final_cost.get("cdf_calls") == final_samples
            )
        elif method == "defensive_cem":
            means = proposal.get("component_means", [])
            method_contracts_ok &= any(
                isinstance(mean, list) and mean and all(value == 0.0 for value in mean)
                for mean in means
            )
        elif method == "ld_subspace_is":
            weights = proposal.get("component_weights", [])
            means = proposal.get("component_means", [])
            method_contracts_ok &= (
                len(weights) == 2
                and all(value > 0.0 for value in weights)
                and any(all(value == 0.0 for value in mean) for mean in means)
            )
        elif method == "flow_is":
            method_contracts_ok &= (
                proposal.get("conditional_integral") == "baseline_only"
                and proposal.get("dcs_extension_eligible") is False
                and proposal.get("flow_max_log_scale", 0.0) > 0.0
            )
        estimate_value = estimate.get("estimate")
        estimates_ok &= (
            isinstance(estimate_value, (int, float))
            and math.isfinite(float(estimate_value))
            and estimate_value >= 0.0
        )
        for key in ("training_seed",):
            value = proposal.get(key)
            if isinstance(value, int):
                all_seeds.append(value)
        for key in ("pilot_seed", "final_seed"):
            value = plan.get(key)
            if isinstance(value, int):
                all_seeds.append(value)

    record("record_structure_valid", structure_ok)
    record("method_family_exact", family_ok)
    record("proposal_hashes_independently_recomputed", hashes_ok)
    record("proposal_hashes_unique", len(proposal_hashes) == len(set(proposal_hashes)))
    record("all_lifecycle_checks_pass", lifecycle_ok)
    record("allocation_and_units_exact", allocation_ok)
    record("training_planning_final_costs_charged", cost_ok)
    record("method_specific_contracts_pass", method_contracts_ok)
    record("all_estimates_finite_nonnegative", estimates_ok)
    record("all_seeds_global_disjoint", len(all_seeds) == 3 * len(records) == len(set(all_seeds)))
    decision = result.get("decision", {})
    record(
        "executor_decision_fail_closed",
        result.get("passed") is True
        and not result.get("failures")
        and decision.get("b1_implementation_complete") is True
        and decision.get("d1_falsification_authorized") is True
        and decision.get("performance_claim_authorized") is False
        and decision.get("p8_qualification_authorized") is False
        and decision.get("submission_authorized") is False,
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": AUDIT_SCHEMA,
        "config_sha256": config_sha256,
        "result_sha256": hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "b1_implementation_complete": not failures,
            "d1_falsification_authorized": not failures,
            "performance_claim_authorized": False,
            "p8_qualification_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config_raw = args.config.read_bytes()
    config = yaml.safe_load(config_raw.decode("utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    report = audit(config, hashlib.sha256(config_raw).hexdigest(), result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
