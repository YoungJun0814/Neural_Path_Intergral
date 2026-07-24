"""Fail-closed audit for the G11 V8 top-journal claim contract.

This audit intentionally validates only the pre-experimental research contract.
It does not inspect, rank, or certify empirical outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SCHEMA = "npi.g11.v8-top-journal-claim-contract.v1"
REPORT_SCHEMA = "npi.g11.v8-claim-contract-audit.v1"

EXPECTED_PRIMARY_TASKS = {"terminal", "discrete_barrier"}
EXPECTED_CONTRIBUTION_IDS = {
    "exact_defensive_mixture_conditional_path_integration",
    "model_level_strict_improvement_or_mesh_rate_theorem",
    "frozen_strong_baseline_total_work_evidence",
}
EXPECTED_COMPARATOR_ROLES = {
    "mechanism": "fixed_raw_defensive",
    "adaptive_work": "task_tuned_pure_cem",
    "closest_published_method": "numerical_smoothing_rqmc",
}
REQUIRED_SECONDARY_COMPARATORS = {
    "crude_antithetic_mc",
    "task_tuned_defensive_cem",
    "large_deviation_adaptive_is",
    "exact_likelihood_flow_is",
}
EXPECTED_COST_CATEGORIES = {
    "proposal_training",
    "hyperparameter_search",
    "failed_retry",
    "screening",
    "allocation_pilot",
    "final_sampling",
    "payoff_evaluation",
    "likelihood_evaluation",
    "conditional_integration",
}
EXPECTED_HETEROGENEOUS_METRICS = {
    "algorithmic_work_units",
    "standardized_hardware_wall_time",
    "actual_compute_cost",
}
REQUIRED_SCOPE_LANGUAGE = {
    "finite_grid",
    "training_inclusive",
    "predeclared_comparator",
}
REQUIRED_PROHIBITED_CLAIMS = {
    "unbiased_continuous_barrier",
    "universal_rbergomi_optimality",
    "unconditional_rbergomi_mlmc_complexity",
    "superiority_over_all_importance_samplers",
    "neural_architecture_contribution",
    "quantum_or_feynman_path_integral",
    "successful_hybrid_router",
    "independent_physical_reproduction_already_complete",
    "exact_percentile_bootstrap_rmse_coverage",
}

EXPECTED_KEYS = {
    "root": {
        "schema",
        "protocol_id",
        "date",
        "phase",
        "frozen",
        "paper",
        "estimand",
        "comparators",
        "statistics",
        "cost_accounting",
        "flow_extension",
        "claim_boundaries",
        "phase_policy",
    },
    "paper": {"route", "working_title", "primary_claim", "contributions"},
    "estimand": {
        "underlying_model",
        "target",
        "primary_grid_steps",
        "continuous_monitoring",
        "primary_tasks",
        "excluded_primary_tasks",
        "nominal_probability_minimum",
        "nominal_probability_maximum",
    },
    "comparators": {
        "outcome_selected_primary_comparator",
        "roles",
        "mandatory_secondary",
    },
    "comparator_roles": set(EXPECTED_COMPARATOR_ROLES),
    "comparator_role": {"id", "required", "training_inclusive"},
    "statistics": {
        "inference_unit",
        "equal_cell_weight_within_cluster",
        "simultaneous_intervals_for_multiple_primary_comparators",
        "reference_uncertainty_in_accuracy",
        "provisional_gates",
    },
    "provisional_gates": {
        "minimum_probe_variance_ratio_lower",
        "minimum_execution_variance_ratio_lower",
        "minimum_fixed_raw_final_work_ratio_lower",
        "minimum_external_training_inclusive_work_ratio_lower",
        "minimum_exact_attainment_lower",
        "maximum_simultaneous_rmse_ratio",
        "maximum_floor_binding_fraction",
        "maximum_resource_censoring_count",
        "maximum_cross_phase_seed_intersection",
    },
    "cost_accounting": {
        "training_inclusive",
        "failed_attempts_charged",
        "hyperparameter_search_charged",
        "categories",
        "heterogeneous_hardware_metrics",
    },
    "flow_extension": {
        "full_path_flow_is_baseline_only",
        "residual_flow_dcs_requires_exact_density",
        "residual_flow_dcs_requires_tractable_conditional_integral",
    },
    "claim_boundaries": {"required_scope_language", "prohibited_claims"},
    "phase_policy": {
        "commits_per_phase",
        "commit_only_after_theory_audit",
        "commit_only_after_technical_audit",
        "commit_only_after_tests",
        "intermediate_commits_prohibited",
    },
}


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contract root must be a mapping")
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected contract schema: {payload.get('schema')!r}")
    return payload, hashlib.sha256(raw).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _keys_exact(value: dict[str, Any], expected: set[str]) -> bool:
    return set(value) == expected


def audit_contract(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    """Audit the claim contract and return a deterministic JSON-ready report."""

    checks: dict[str, bool] = {}

    def record(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    paper = _mapping(config.get("paper"))
    contributions = paper.get("contributions")
    estimand = _mapping(config.get("estimand"))
    comparators = _mapping(config.get("comparators"))
    comparator_roles = _mapping(comparators.get("roles"))
    statistics = _mapping(config.get("statistics"))
    gates = _mapping(statistics.get("provisional_gates"))
    cost_accounting = _mapping(config.get("cost_accounting"))
    flow = _mapping(config.get("flow_extension"))
    claim_boundaries = _mapping(config.get("claim_boundaries"))
    commit_policy = _mapping(config.get("phase_policy"))

    record("schema_exact", config.get("schema") == EXPECTED_SCHEMA)
    record("root_keys_exact", _keys_exact(config, EXPECTED_KEYS["root"]))
    record("paper_keys_exact", _keys_exact(paper, EXPECTED_KEYS["paper"]))
    record("estimand_keys_exact", _keys_exact(estimand, EXPECTED_KEYS["estimand"]))
    record(
        "comparator_keys_exact",
        _keys_exact(comparators, EXPECTED_KEYS["comparators"]),
    )
    record(
        "comparator_role_keys_exact",
        _keys_exact(comparator_roles, EXPECTED_KEYS["comparator_roles"])
        and all(
            _keys_exact(
                _mapping(comparator_roles.get(role)),
                EXPECTED_KEYS["comparator_role"],
            )
            for role in EXPECTED_COMPARATOR_ROLES
        ),
    )
    record(
        "statistics_keys_exact",
        _keys_exact(statistics, EXPECTED_KEYS["statistics"]),
    )
    record(
        "provisional_gate_keys_exact",
        _keys_exact(gates, EXPECTED_KEYS["provisional_gates"]),
    )
    record(
        "cost_accounting_keys_exact",
        _keys_exact(cost_accounting, EXPECTED_KEYS["cost_accounting"]),
    )
    record(
        "flow_extension_keys_exact",
        _keys_exact(flow, EXPECTED_KEYS["flow_extension"]),
    )
    record(
        "claim_boundary_keys_exact",
        _keys_exact(claim_boundaries, EXPECTED_KEYS["claim_boundaries"]),
    )
    record(
        "phase_policy_keys_exact",
        _keys_exact(commit_policy, EXPECTED_KEYS["phase_policy"]),
    )
    record(
        "protocol_is_v8",
        config.get("protocol_id") == "g11-v8-top-journal-claim-contract-v1",
    )
    record("contract_date_exact", config.get("date") == "2026-07-25")
    record("phase_is_development", config.get("phase") == "development")
    record("not_prematurely_frozen", config.get("frozen") is False)
    record(
        "paper_route_is_exact",
        paper.get("route") == "theory_plus_computation",
    )
    record("working_title_present", isinstance(paper.get("working_title"), str))
    record(
        "primary_claim_is_scoped",
        isinstance(paper.get("primary_claim"), str)
        and "finite-grid" in paper["primary_claim"]
        and (
            "rBergomi" in paper["primary_claim"]
            or "rough Bergomi" in paper["primary_claim"]
        )
        and "training-inclusive" in paper["primary_claim"],
    )

    contribution_ids = _string_set(contributions)
    record(
        "contributions_are_strings",
        isinstance(contributions, list)
        and all(isinstance(item, str) for item in contributions),
    )
    record(
        "exactly_three_predeclared_contributions",
        isinstance(contributions, list)
        and len(contributions) == 3
        and contribution_ids == EXPECTED_CONTRIBUTION_IDS,
    )

    primary_tasks = _string_set(estimand.get("primary_tasks"))
    excluded_tasks = _string_set(estimand.get("excluded_primary_tasks"))
    p_min = _number(estimand.get("nominal_probability_minimum"))
    p_max = _number(estimand.get("nominal_probability_maximum"))
    record(
        "model_is_rough_bergomi",
        estimand.get("underlying_model") == "rough_bergomi",
    )
    record(
        "target_is_finite_grid_probability",
        estimand.get("target") == "finite_grid_probability",
    )
    record("reference_grid_is_128", estimand.get("primary_grid_steps") == 128)
    record(
        "continuous_monitoring_not_claimed",
        estimand.get("continuous_monitoring") is False,
    )
    record("primary_tasks_exact", primary_tasks == EXPECTED_PRIMARY_TASKS)
    record(
        "occupation_excluded_from_primary",
        "hit_plus_occupation" in excluded_tasks,
    )
    record(
        "probability_range_valid",
        p_min is not None and p_max is not None and 0.0 < p_min < p_max < 1.0,
    )

    record(
        "comparator_selection_predeclared",
        comparators.get("outcome_selected_primary_comparator") is False,
    )
    role_ids = {
        role: _mapping(comparator_roles.get(role)).get("id")
        for role in EXPECTED_COMPARATOR_ROLES
    }
    record(
        "primary_comparator_roles_exact",
        role_ids == EXPECTED_COMPARATOR_ROLES,
    )
    record(
        "primary_comparators_required",
        all(
            _mapping(comparator_roles.get(role)).get("required") is True
            for role in EXPECTED_COMPARATOR_ROLES
        ),
    )
    record(
        "primary_comparators_training_inclusive",
        all(
            _mapping(comparator_roles.get(role)).get("training_inclusive") is True
            for role in EXPECTED_COMPARATOR_ROLES
        ),
    )
    record(
        "secondary_comparators_complete",
        REQUIRED_SECONDARY_COMPARATORS.issubset(
            _string_set(comparators.get("mandatory_secondary"))
        ),
    )

    record(
        "inference_unit_is_seed_cluster",
        statistics.get("inference_unit") == "independent_seed_cluster",
    )
    record(
        "cell_weight_dependence_preserved",
        statistics.get("equal_cell_weight_within_cluster") is True,
    )
    record(
        "simultaneous_inference_required",
        statistics.get("simultaneous_intervals_for_multiple_primary_comparators")
        is True,
    )
    record(
        "reference_uncertainty_propagated",
        statistics.get("reference_uncertainty_in_accuracy") is True,
    )

    gate_expectations = {
        "minimum_probe_variance_ratio_lower": 2.0,
        "minimum_execution_variance_ratio_lower": 2.0,
        "minimum_fixed_raw_final_work_ratio_lower": 1.5,
        "minimum_external_training_inclusive_work_ratio_lower": 1.2,
        "minimum_exact_attainment_lower": 0.8,
        "maximum_simultaneous_rmse_ratio": 1.0,
        "maximum_floor_binding_fraction": 0.05,
        "maximum_resource_censoring_count": 0.0,
        "maximum_cross_phase_seed_intersection": 0.0,
    }
    for name, expected in gate_expectations.items():
        actual = _number(gates.get(name))
        record(f"gate_{name}", actual is not None and actual == expected)

    record(
        "all_training_costs_counted",
        cost_accounting.get("training_inclusive") is True
        and cost_accounting.get("failed_attempts_charged") is True
        and cost_accounting.get("hyperparameter_search_charged") is True,
    )
    record(
        "cost_categories_complete",
        EXPECTED_COST_CATEGORIES.issubset(
            _string_set(cost_accounting.get("categories"))
        ),
    )
    record(
        "heterogeneous_metrics_exact",
        _string_set(cost_accounting.get("heterogeneous_hardware_metrics"))
        == EXPECTED_HETEROGENEOUS_METRICS,
    )

    record(
        "full_path_flow_is_baseline_only",
        flow.get("full_path_flow_is_baseline_only") is True,
    )
    record(
        "residual_flow_requires_exact_density",
        flow.get("residual_flow_dcs_requires_exact_density") is True,
    )
    record(
        "residual_flow_requires_tractable_conditional_integral",
        flow.get("residual_flow_dcs_requires_tractable_conditional_integral")
        is True,
    )

    record(
        "scope_language_complete",
        REQUIRED_SCOPE_LANGUAGE.issubset(
            _string_set(claim_boundaries.get("required_scope_language"))
        ),
    )
    record(
        "prohibited_claims_complete",
        REQUIRED_PROHIBITED_CLAIMS.issubset(
            _string_set(config.get("prohibited_claims"))
            | _string_set(claim_boundaries.get("prohibited_claims"))
        ),
    )

    record("one_commit_per_phase", commit_policy.get("commits_per_phase") == 1)
    record(
        "commit_requires_theory_audit",
        commit_policy.get("commit_only_after_theory_audit") is True,
    )
    record(
        "commit_requires_technical_audit",
        commit_policy.get("commit_only_after_technical_audit") is True,
    )
    record(
        "commit_requires_full_tests",
        commit_policy.get("commit_only_after_tests") is True,
    )
    record(
        "intermediate_commits_prohibited",
        commit_policy.get("intermediate_commits_prohibited") is True,
    )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "contract_schema": config.get("schema"),
        "config_sha256": config_sha256,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    config, config_sha256 = _load_config(args.config)
    report = audit_contract(config, config_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
