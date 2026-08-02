"""Frozen V9 terminal-only efficiency metrics and gated aggregation.

The module contains no simulator calls.  Both the executor and the independent
auditor use these pure transformations so that total-work arithmetic and the
regime-selection rule have one explicit mathematical definition.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any

from scipy.stats import t as student_t


def geometric_mean(values: list[float]) -> float:
    """Return a strict geometric mean of positive finite values."""

    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def work_to_target(
    *,
    unit_variance: float,
    unit_work: float,
    one_time_work: float,
    query_count: int,
    target_estimator_variance: float,
) -> dict[str, float | int]:
    """Cost an ordinary mean at a frozen target variance.

    ``unit_variance`` and ``unit_work`` refer to one independent inferential
    unit (one path for IID methods or one randomization for RQMC).  Training,
    allocation-pilot, and diagnostic work are one-time costs and are amortized
    over exactly ``query_count`` repeated estimates.  At least two units are
    retained so a variance remains estimable.
    """

    numeric = (unit_variance, unit_work, one_time_work, target_estimator_variance)
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("work-to-target inputs must be finite")
    if unit_variance < 0.0 or unit_work <= 0.0 or one_time_work < 0.0:
        raise ValueError("invalid work-to-target variance or work")
    if target_estimator_variance <= 0.0:
        raise ValueError("target estimator variance must be positive")
    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count < 1:
        raise ValueError("query count must be a positive integer")
    required_units = max(2, math.ceil(unit_variance / target_estimator_variance))
    evaluation_work = required_units * unit_work
    amortized = one_time_work / query_count
    return {
        "required_units": required_units,
        "unit_variance": unit_variance,
        "unit_work": unit_work,
        "one_time_work": one_time_work,
        "amortized_one_time_work": amortized,
        "evaluation_work": evaluation_work,
        "total_work": evaluation_work + amortized,
    }


def attach_work_to_target(
    *,
    config: dict[str, Any],
    paired_records: list[dict[str, Any]],
    external_records: list[dict[str, Any]],
    bank: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return records with reproducible target-work and comparison fields."""

    paired = deepcopy(paired_records)
    external = deepcopy(external_records)
    queries = [int(value) for value in config["amortization_query_counts"]]
    relative_rmse = float(config["relative_rmse_target"])
    entries = {str(entry["cell_id"]): entry for entry in bank["entries"]}
    external_index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in external:
        external_index[(str(record["cell_id"]), int(record["cluster"]))].append(record)
        units = int(record["allocation"]["planned_units"])
        if units < 2:
            raise ValueError("external final allocation must contain at least two units")
        final = record["estimate"]["final_cost"]
        final_work = float(final["algorithmic_work_units"])
        unit_work = final_work / units
        unit_variance = float(record["estimate"]["variance"]) * units
        total = float(record["total_algorithmic_work_units_including_diagnostic"])
        one_time = total - final_work
        if one_time < -1e-9:
            raise ValueError("external total work omits final work")
        one_time = max(0.0, one_time)
        reference = float(record["reference_estimate"])
        target = (relative_rmse * reference) ** 2
        record["work_to_target"] = {
            str(query): work_to_target(
                unit_variance=unit_variance,
                unit_work=unit_work,
                one_time_work=one_time,
                query_count=query,
                target_estimator_variance=target,
            )
            for query in queries
        }

    primary = tuple(str(value) for value in config["external_methods"]["primary"])
    for record in paired:
        cell_id = str(record["cell_id"])
        count = int(record["sample_count"])
        if count < 2:
            raise ValueError("paired record must contain at least two paths")
        reference = float(record["reference_estimate"])
        target = (relative_rmse * reference) ** 2
        training = float(entries[cell_id]["training_cost"]["algorithmic_work_units"])
        record["work_to_target"] = {}
        for query in queries:
            record["work_to_target"][str(query)] = {}
            for method in ("raw", "dcs"):
                unit_work = float(record[method]["cost"]["algorithmic_work_units"]) / count
                record["work_to_target"][str(query)][method] = work_to_target(
                    unit_variance=float(record[method]["variance"]),
                    unit_work=unit_work,
                    one_time_work=training,
                    query_count=query,
                    target_estimator_variance=target,
                )

        matches = external_index[(cell_id, int(record["cluster"]))]
        methods = {str(item["method"]): item for item in matches}
        if set(methods) != set(primary):
            raise ValueError("paired record lacks one unique primary external comparator")
        record["comparison"] = {}
        for query in queries:
            key = str(query)
            dcs_work = float(record["work_to_target"][key]["dcs"]["total_work"])
            raw_work = float(record["work_to_target"][key]["raw"]["total_work"])
            ratios = {
                method: float(methods[method]["work_to_target"][key]["total_work"])
                / dcs_work
                for method in primary
            }
            record["comparison"][key] = {
                "dcs_vs_raw_total_work_ratio": raw_work / dcs_work,
                "dcs_vs_external_total_work_ratios": ratios,
                "dcs_vs_best_primary_total_work_ratio": min(ratios.values()),
                "best_primary_method": min(ratios, key=ratios.__getitem__),
            }
    return paired, external


def _cluster_geometric_summary(
    values: list[tuple[int, float]], *, confidence: float, multiplicity: int
) -> dict[str, Any]:
    by_cluster: dict[int, list[float]] = defaultdict(list)
    for cluster, value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("efficiency ratios must be positive and finite")
        by_cluster[cluster].append(math.log(value))
    cluster_logs = [sum(items) / len(items) for _, items in sorted(by_cluster.items())]
    if len(cluster_logs) < 2:
        raise ValueError("at least two independent clusters are required")
    mean_log = sum(cluster_logs) / len(cluster_logs)
    variance = sum((value - mean_log) ** 2 for value in cluster_logs) / (
        len(cluster_logs) - 1
    )
    standard_error = math.sqrt(variance / len(cluster_logs))
    alpha = (1.0 - confidence) / multiplicity
    critical = float(student_t.ppf(1.0 - alpha, df=len(cluster_logs) - 1))
    lower = math.exp(mean_log - critical * standard_error)
    return {
        "record_count": len(values),
        "cluster_count": len(cluster_logs),
        "cluster_geometric_ratios": [math.exp(value) for value in cluster_logs],
        "geometric_ratio": math.exp(mean_log),
        "log_standard_error": standard_error,
        "one_sided_confidence": 1.0 - alpha,
        "multiplicity": multiplicity,
        "lower_confidence_bound": lower,
    }


def aggregate_terminal_benchmark(
    *,
    config: dict[str, Any],
    paired_records: list[dict[str, Any]],
    external_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the complete V9 correctness and regime-conditional gate."""

    if not paired_records or not external_records:
        raise ValueError("V9 aggregation requires paired and external records")
    gate = config["gate"]
    maximum_error = float(config["maximum_exactness_error"])
    exactness = all(
        max(float(value) for value in record["exactness"].values()) <= maximum_error
        for record in paired_records
    ) and all(record["lifecycle_audit_passed"] is True for record in external_records)
    finite = all(
        int(record["likelihood_diagnostics"]["nonfinite_weight_count"]) == 0
        and record["proposal"]["exact_likelihood"] is True
        and record["proposal"]["self_normalized"] is False
        for record in external_records
    )
    normalization_limit = float(config["maximum_likelihood_normalization_absolute_z"])
    normalization_checks = [
        math.isfinite(float(record["likelihood"]["normalization_z"]))
        and abs(float(record["likelihood"]["normalization_z"])) <= normalization_limit
        for record in paired_records
    ]
    normalization_fraction = sum(normalization_checks) / len(normalization_checks)
    normalization = normalization_fraction >= float(
        gate["minimum_likelihood_normalization_pass_fraction"]
    )
    dcs_accuracy_max = max(
        float(record["dcs"]["combined_reference_z"]) for record in paired_records
    )
    difference_z_max = max(
        float(record["mechanism"]["difference_z"]) for record in paired_records
    )
    external_accuracy_max = max(
        float(record["estimate"]["combined_reference_z"]) for record in external_records
    )
    accuracy_limit = float(gate["maximum_combined_reference_z"])
    difference_limit = float(gate["maximum_paired_difference_z"])
    accuracy = dcs_accuracy_max <= accuracy_limit and external_accuracy_max <= accuracy_limit
    paired_identity = difference_z_max <= difference_limit
    resource_censoring = sum(
        bool(record["allocation"]["resource_censored"]) for record in external_records
    )
    resource = resource_censoring == 0
    correctness = exactness and finite and normalization and accuracy and paired_identity and resource

    stage = str(config["stage"])
    if stage not in {"development", "qualification"}:
        raise ValueError("V9 stage must be development or qualification")
    primary_query = str(config["primary_query_count"])
    hursts = sorted({float(record["hurst"]) for record in paired_records})
    confidence = float(gate[stage]["confidence"])
    multiplicity = len(hursts) if stage == "qualification" else 1
    group_summaries: list[dict[str, Any]] = []
    selected: list[float] = []
    for hurst in hursts:
        group = [record for record in paired_records if float(record["hurst"]) == hurst]
        raw_values = [
            (
                int(record["cluster"]),
                float(record["comparison"][primary_query]["dcs_vs_raw_total_work_ratio"]),
            )
            for record in group
        ]
        best_values = [
            (
                int(record["cluster"]),
                float(
                    record["comparison"][primary_query][
                        "dcs_vs_best_primary_total_work_ratio"
                    ]
                ),
            )
            for record in group
        ]
        raw_summary = _cluster_geometric_summary(
            raw_values, confidence=confidence, multiplicity=multiplicity
        )
        best_summary = _cluster_geometric_summary(
            best_values, confidence=confidence, multiplicity=multiplicity
        )
        thresholds = gate[stage]
        efficiency_pass = (
            raw_summary["geometric_ratio"]
            >= float(thresholds["minimum_dcs_vs_raw_geometric_ratio"])
            and raw_summary["lower_confidence_bound"]
            >= float(thresholds["minimum_dcs_vs_raw_lower_bound"])
            and best_summary["geometric_ratio"]
            >= float(thresholds["minimum_dcs_vs_best_geometric_ratio"])
            and best_summary["lower_confidence_bound"]
            >= float(thresholds["minimum_dcs_vs_best_lower_bound"])
        )
        passed = correctness and efficiency_pass
        if passed:
            selected.append(hurst)
        group_summaries.append(
            {
                "hurst": hurst,
                "cell_ids": sorted({str(record["cell_id"]) for record in group}),
                "dcs_vs_raw": raw_summary,
                "dcs_vs_best_primary": best_summary,
                "efficiency_pass": efficiency_pass,
                "all_correctness_gates_pass": correctness,
                "group_pass": passed,
            }
        )

    minimum_groups = int(gate[stage]["minimum_passing_hurst_groups"])
    stage_pass = correctness and len(selected) >= minimum_groups
    if stage == "qualification":
        stage_pass = stage_pass and len(selected) == len(hursts)
    blockers = [
        *([] if exactness else ["exactness_failure"]),
        *([] if finite else ["likelihood_or_mean_contract_failure"]),
        *([] if normalization else ["likelihood_normalization_failure"]),
        *([] if accuracy else ["reference_accuracy_failure"]),
        *([] if paired_identity else ["paired_identity_failure"]),
        *([] if resource else ["resource_censoring"]),
        *([] if len(selected) >= minimum_groups else ["insufficient_passing_hurst_groups"]),
        *(
            []
            if stage != "qualification" or len(selected) == len(hursts)
            else ["qualification_group_failure"]
        ),
    ]
    return {
        "stage": stage,
        "paired_record_count": len(paired_records),
        "external_record_count": len(external_records),
        "maximum_exactness_error": max(
            max(float(value) for value in record["exactness"].values())
            for record in paired_records
        ),
        "exactness_pass": exactness,
        "likelihood_and_ordinary_mean_contract_pass": finite,
        "likelihood_normalization_pass_fraction": normalization_fraction,
        "likelihood_normalization_pass": normalization,
        "dcs_accuracy_maximum_combined_z": dcs_accuracy_max,
        "external_accuracy_maximum_combined_z": external_accuracy_max,
        "reference_accuracy_pass": accuracy,
        "paired_difference_maximum_z": difference_z_max,
        "paired_identity_pass": paired_identity,
        "primary_resource_censoring_count": resource_censoring,
        "resource_gate_pass": resource,
        "all_correctness_gates_pass": correctness,
        "primary_query_count": int(primary_query),
        "group_summaries": group_summaries,
        "selected_hurst_groups": selected,
        "stage_pass": stage_pass,
        "blockers": blockers,
    }
