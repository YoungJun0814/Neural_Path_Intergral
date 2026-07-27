"""Fail-closed V8 P6 multiplicity and pre-outcome power-design audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import scipy.stats
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-p6-statistical-design.v1"
REPORT = "npi.g11.v8-p6-statistical-design-audit.v1"
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "claim_contract_sha256",
    "p5_reference_matrix_design_sha256",
    "inference",
    "primary_methods",
    "efficiency_family",
    "accuracy_family",
    "cluster_design",
    "seed_namespaces",
    "failure_rules",
    "decision",
}
EFFICIENCY_ENDPOINTS = [
    "raw_probe_variance",
    "raw_execution_variance",
    "raw_final_work",
    "pure_cem_training_inclusive_work",
    "smoothing_rqmc_training_inclusive_work",
]
PRIMARY_COMPARATORS = [
    "fixed_raw_defensive",
    "task_tuned_pure_cem",
    "numerical_smoothing_rqmc",
]
ACCURACY_METHODS = [
    "defensive_conditional_path_integration",
    *PRIMARY_COMPARATORS,
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_design(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unexpected P6 statistical-design schema")
    return value, hashlib.sha256(raw).hexdigest()


def required_one_sided_clusters(
    *,
    true_ratio: float,
    minimum_lower_ratio: float,
    cluster_log_sd: float,
    alpha: float,
    power: float,
) -> int:
    """Normal-approximation count for a one-sided lower-ratio gate.

    The tested parameter is the mean cluster log-ratio.  The alternative must
    strictly exceed the prespecified lower-ratio boundary; this is a planning
    calculation, never an observed-effect estimate.
    """

    values = (true_ratio, minimum_lower_ratio, cluster_log_sd, alpha, power)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("power inputs must be finite")
    if true_ratio <= minimum_lower_ratio or minimum_lower_ratio <= 0.0:
        raise ValueError("planning true ratio must exceed a positive lower boundary")
    if cluster_log_sd <= 0.0 or not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("invalid standard deviation, alpha, or power")
    effect_above_boundary = math.log(true_ratio) - math.log(minimum_lower_ratio)
    z_alpha = float(scipy.stats.norm.ppf(1.0 - alpha))
    z_power = float(scipy.stats.norm.ppf(power))
    required = math.ceil(
        ((z_alpha + z_power) * cluster_log_sd / effect_above_boundary) ** 2
    )
    return max(2, required)


def _p5_seed_namespaces() -> set[str]:
    path = ROOT / "configs/g11_v8/p5_reference_matrix_design_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P5 design must be a mapping")
    threshold = payload.get("threshold_policy", {})
    reference = payload.get("reference_contract", {})
    baseline = payload.get("baseline_matrix", {})
    values = {
        threshold.get("calibration_seed_namespace"),
        reference.get("reference_seed_namespace"),
        reference.get("final_seed_namespace"),
        baseline.get("training_seed_namespace"),
        baseline.get("pilot_seed_namespace"),
        baseline.get("final_seed_namespace"),
    }
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError("P5 must define nonempty seed namespaces")
    return values


def audit_design(design: dict[str, Any], digest: str) -> dict[str, Any]:
    claim_contract = ROOT / "configs/g11_v8/top_journal_claim_contract_v1.yaml"
    p5_design = ROOT / "configs/g11_v8/p5_reference_matrix_design_v1.yaml"
    inference = design.get("inference", {})
    primary_methods = design.get("primary_methods", {})
    efficiency = design.get("efficiency_family", {})
    accuracy = design.get("accuracy_family", {})
    cluster_design = design.get("cluster_design", {})
    failure_rules = design.get("failure_rules", {})
    decision = design.get("decision", {})
    endpoints = efficiency.get("endpoints", [])
    endpoint_ids = [item.get("id") for item in endpoints if isinstance(item, dict)]
    efficiency_alpha = float(efficiency.get("familywise_alpha", math.nan))
    accuracy_alpha = float(accuracy.get("familywise_alpha", math.nan))
    per_efficiency_alpha = efficiency_alpha / len(endpoints) if endpoints else math.nan
    accuracy_methods = accuracy.get("methods", [])
    method_cell_claims = accuracy.get("method_cell_claims")
    expected_accuracy_claims = (
        len(accuracy_methods) * int(inference.get("primary_cell_count", 0)) * method_cell_claims
        if isinstance(method_cell_claims, int) and not isinstance(method_cell_claims, bool)
        else -1
    )
    per_accuracy_alpha = (
        accuracy_alpha / expected_accuracy_claims if expected_accuracy_claims > 0 else math.nan
    )
    forecasts: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        try:
            required = required_one_sided_clusters(
                true_ratio=float(endpoint["planning_true_ratio"]),
                minimum_lower_ratio=float(endpoint["minimum_lower_ratio"]),
                cluster_log_sd=float(endpoint["planning_cluster_log_sd"]),
                alpha=per_efficiency_alpha,
                power=float(efficiency["target_power"]),
            )
        except (KeyError, TypeError, ValueError):
            required = None
        forecasts.append({"id": endpoint.get("id"), "required_clusters": required})
    planned_qualification = cluster_design.get("p8_qualification_clusters")
    planned_confirmation = cluster_design.get("p10_confirmation_clusters")
    namespace_values = list(design.get("seed_namespaces", {}).values())
    try:
        p5_namespaces = _p5_seed_namespaces()
    except ValueError:
        p5_namespaces = set()
    endpoint_by_id = {item["id"]: item for item in endpoints if isinstance(item, dict) and "id" in item}
    checks = {
        "schema_exact": design.get("schema") == SCHEMA,
        "root_keys_exact": set(design) == ROOT_KEYS,
        "phase_exact": design.get("phase") == "p6_development",
        "outcome_blind": design.get("outcome_data_used") is False,
        "claim_contract_hash_bound": claim_contract.is_file()
        and design.get("claim_contract_sha256") == _sha(claim_contract),
        "p5_hash_bound": p5_design.is_file()
        and design.get("p5_reference_matrix_design_sha256") == _sha(p5_design),
        "cluster_inference_exact": inference.get("unit") == "independent_seed_cluster",
        "equal_cell_weighting_required": inference.get("within_cluster_cell_aggregation")
        == "equal_weight_mean_log_ratio",
        "path_pseudoreplication_refused": inference.get("path_level_pseudoreplication_allowed") is False,
        "finite_grid_exact": inference.get("primary_grid_steps") == 128,
        "primary_cell_count_exact": inference.get("primary_cell_count") == 24,
        "primary_estimator_exact": primary_methods.get("estimator")
        == "defensive_conditional_path_integration",
        "primary_comparators_exact": primary_methods.get("comparators") == PRIMARY_COMPARATORS,
        "efficiency_alpha_exact": efficiency_alpha == 0.025,
        "efficiency_bonferroni_exact": efficiency.get("multiplicity") == "bonferroni",
        "efficiency_one_sided_t_exact": efficiency.get("interval") == "one_sided_student_t_lower",
        "efficiency_endpoints_exact": endpoint_ids == EFFICIENCY_ENDPOINTS,
        "efficiency_per_endpoint_alpha_exact": math.isclose(per_efficiency_alpha, 0.005),
        "efficiency_thresholds_exact": all(
            (
                endpoint_by_id.get("raw_probe_variance", {}).get("minimum_lower_ratio")
                == 2.0,
                endpoint_by_id.get("raw_execution_variance", {}).get("minimum_lower_ratio")
                == 2.0,
                endpoint_by_id.get("raw_final_work", {}).get("minimum_lower_ratio") == 1.5,
                endpoint_by_id.get("pure_cem_training_inclusive_work", {}).get(
                    "minimum_lower_ratio"
                )
                == 1.2,
                endpoint_by_id.get("smoothing_rqmc_training_inclusive_work", {}).get(
                    "minimum_lower_ratio"
                )
                == 1.2,
            )
        ),
        "accuracy_alpha_exact": accuracy_alpha == 0.025,
        "accuracy_bonferroni_exact": accuracy.get("multiplicity") == "bonferroni",
        "accuracy_methods_exact": accuracy_methods == ACCURACY_METHODS,
        "accuracy_claim_shape_exact": method_cell_claims == 2
        and expected_accuracy_claims == 192
        and accuracy.get("expected_claim_count") == 192,
        "accuracy_per_claim_alpha_exact": math.isclose(per_accuracy_alpha, 0.025 / 192.0),
        "attainment_interval_exact": accuracy.get("exact_attainment", {}).get("method")
        == "one_sided_clopper_pearson_lower",
        "attainment_lower_exact": accuracy.get("exact_attainment", {}).get("minimum_lower_bound") == 0.80,
        "rmse_bound_honestly_nominal": accuracy.get("rmse", {}).get("coverage_description")
        == "nominal_simultaneous_bootstrap_upper",
        "rmse_threshold_exact": accuracy.get("rmse", {}).get("maximum_upper_to_tolerance_ratio") == 1.0,
        "bootstrap_repetitions_sufficient": accuracy.get("rmse", {}).get("bootstrap_repetitions", 0)
        >= 100000,
        "qualification_cluster_count_exact": planned_qualification == 32,
        "confirmation_cluster_count_exact": planned_confirmation == 48,
        "preoutcome_power_label_exact": cluster_design.get("confirmation_power_source")
        == "pre_outcome_normal_approximation_only",
        "post_p8_reestimation_refused": cluster_design.get("confirmation_reestimation_after_p8")
        == "prohibited",
        "normal_planning_scenarios_valid": len(forecasts) == len(endpoints)
        and all(item["required_clusters"] is not None for item in forecasts),
        "qualification_scenarios_powered": isinstance(planned_qualification, int)
        and all(
            item["required_clusters"] is not None
            and int(item["required_clusters"]) <= planned_qualification
            for item in forecasts
        ),
        "confirmation_scenarios_powered": isinstance(planned_confirmation, int)
        and all(
            item["required_clusters"] is not None
            and int(item["required_clusters"]) <= planned_confirmation
            for item in forecasts
        ),
        "seed_namespaces_unique": len(namespace_values) == 6
        and all(isinstance(value, str) and value for value in namespace_values)
        and len(set(namespace_values)) == len(namespace_values),
        "p5_p6_seed_sets_disjoint": bool(p5_namespaces)
        and not (set(namespace_values) & p5_namespaces),
        "no_primary_censoring": failure_rules.get("resource_censoring_allowed_for_primary_claim") is False,
        "no_record_deletion": failure_rules.get("incomplete_record_deletion_allowed") is False,
        "retry_cost_charged": failure_rules.get("failed_retry_cost_charged") is True,
        "cross_phase_seed_limit_exact": failure_rules.get("cross_phase_seed_intersection_maximum") == 0,
        "actual_power_refused": decision.get("actual_power_estimated") is False,
        "p7_authorized": decision.get("p7_falsification_authorized") is True,
        "p8_not_prematurely_authorized": decision.get("p8_qualification_authorized") is False,
        "performance_refused": decision.get("performance_claim_authorized") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT,
        "design_sha256": digest,
        "efficiency_per_endpoint_alpha": per_efficiency_alpha,
        "accuracy_per_claim_alpha": per_accuracy_alpha,
        "power_forecasts": forecasts,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


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
