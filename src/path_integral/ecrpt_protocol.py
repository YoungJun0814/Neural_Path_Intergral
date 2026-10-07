"""Pure aggregation and fail-closed gates for the V12 ECRPT micro-study."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from scipy.stats import t as student_t


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def one_sided_geometric_lower(ratios: list[float], *, confidence_level: float = 0.95) -> float:
    if len(ratios) < 2:
        raise ValueError("one-sided cluster inference requires at least two ratios")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    logs = [math.log(value) for value in ratios]
    mean = sum(logs) / len(logs)
    variance = sum((value - mean) ** 2 for value in logs) / (len(logs) - 1)
    standard_error = math.sqrt(variance / len(logs))
    critical = float(student_t.ppf(confidence_level, df=len(logs) - 1))
    return math.exp(mean - critical * standard_error)


def aggregate_ecrpt_microstudy(
    *, config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute all V12 development decisions from record-level evidence."""

    cells = config["cells"]
    clusters = int(config["clusters"])
    expected = len(cells) * clusters
    if len(records) != expected:
        raise ValueError("micro-study record roster is incomplete")
    identities = {(str(record["cell_id"]), int(record["cluster"])) for record in records}
    expected_identities = {
        (str(cell["cell_id"]), cluster) for cell in cells for cluster in range(clusters)
    }
    if identities != expected_identities:
        raise ValueError("micro-study record identities differ from the config")
    gate = config["gate"]
    exact_limit = float(gate["maximum_exactness_error"])
    reference_limit = float(gate["maximum_combined_reference_z"])
    difference_limit = float(gate["maximum_paired_difference_z"])
    normalization_limit = float(gate["maximum_likelihood_normalization_z"])
    primary = tuple(str(name) for name in config["primary_comparators"])

    exactness_pass = all(
        max(float(value) for value in record["candidate"]["exactness"].values()) <= exact_limit
        for record in records
    )
    paired_identity_pass = all(
        float(record["candidate"]["paired_difference_z"]) <= difference_limit for record in records
    )
    likelihood_pass = all(
        abs(float(record["candidate"]["likelihood_normalization_z"])) <= normalization_limit
        for record in records
    )
    candidate_accuracy_pass = all(
        float(record["candidate"]["combined_reference_z"]) <= reference_limit for record in records
    )
    primary_accuracy_pass = all(
        float(record["comparators"][name]["combined_reference_z"]) <= reference_limit
        for record in records
        for name in primary
    )
    primary_censored = sum(
        bool(record["comparators"][name]["tail_safe_forecast"]["resource_censored"])
        for record in records
        for name in primary
    )
    candidate_censored = sum(
        bool(record["candidate"]["tail_safe_forecast"]["resource_censored"]) for record in records
    )
    mechanism_pass = all(
        float(record["candidate"]["raw_over_ecrpt_variance_ratio"])
        >= float(gate["minimum_raw_over_ecrpt_variance_ratio"])
        for record in records
    )

    ratios: list[float] = []
    by_cell: dict[str, list[float]] = defaultdict(list)
    best_methods: dict[str, int] = defaultdict(int)
    for record in records:
        candidate_work = float(record["candidate"]["plugin_work_to_target"]["total_work"])
        comparator_work = {
            name: float(record["comparators"][name]["plugin_work_to_target"]["total_work"])
            for name in primary
        }
        best_name = min(comparator_work, key=comparator_work.__getitem__)
        ratio = comparator_work[best_name] / candidate_work
        ratios.append(ratio)
        by_cell[str(record["cell_id"])].append(ratio)
        best_methods[best_name] += 1
    geometric_ratio = geometric_mean(ratios)
    lower_ratio = one_sided_geometric_lower(
        ratios, confidence_level=float(gate["confidence_level"])
    )
    cell_ratios = {cell: geometric_mean(values) for cell, values in by_cell.items()}
    cells_favoring = sum(value > 1.0 for value in cell_ratios.values())
    no_censoring = primary_censored == 0 and candidate_censored == 0
    performance_pass = (
        no_censoring
        and geometric_ratio > float(gate["minimum_best_primary_over_ecrpt_geometric_ratio"])
        and lower_ratio > float(gate["minimum_one_sided_lower_ratio"])
        and cells_favoring >= int(gate["minimum_cells_favoring_ecrpt"])
    )
    correctness_pass = (
        exactness_pass
        and paired_identity_pass
        and likelihood_pass
        and candidate_accuracy_pass
        and primary_accuracy_pass
        and mechanism_pass
    )
    blockers: list[str] = []
    for name, passed in (
        ("exactness", exactness_pass),
        ("paired_identity", paired_identity_pass),
        ("likelihood_normalization", likelihood_pass),
        ("candidate_accuracy", candidate_accuracy_pass),
        ("primary_accuracy", primary_accuracy_pass),
        ("paired_variance", mechanism_pass),
        ("tail_safe_uncensored", no_censoring),
        ("performance", performance_pass),
    ):
        if not passed:
            blockers.append(name)
    return {
        "record_count": len(records),
        "exactness_pass": exactness_pass,
        "paired_identity_pass": paired_identity_pass,
        "likelihood_normalization_pass": likelihood_pass,
        "candidate_accuracy_pass": candidate_accuracy_pass,
        "primary_accuracy_pass": primary_accuracy_pass,
        "mechanism_pass": mechanism_pass,
        "primary_resource_censored_records": primary_censored,
        "candidate_resource_censored_records": candidate_censored,
        "no_resource_censoring": no_censoring,
        "descriptive_best_primary_over_ecrpt_geometric_ratio": geometric_ratio,
        "descriptive_one_sided_lower_ratio": lower_ratio,
        "descriptive_cell_ratios": cell_ratios,
        "cells_favoring_ecrpt": cells_favoring,
        "best_primary_method_counts": dict(best_methods),
        "correctness_pass": correctness_pass,
        "performance_pass": performance_pass,
        "stage_pass": correctness_pass and performance_pass,
        "blockers": blockers,
    }


def aggregate_structured_ecrpt_development(
    *, config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """V13 gates with the exact Rao--Blackwell mechanism separated from noise.

    The observed raw/conditional variance ratio remains a performance diagnostic.
    It is not a valid finite-sample test of the population variance identity.
    """

    aggregate = aggregate_ecrpt_microstudy(config=config, records=records)
    tolerance = float(config["gate"]["maximum_exactness_error"])
    mechanism = all(
        float(record["candidate"]["rao_blackwell_gap_minimum"]) >= -tolerance
        and float(record["candidate"]["rao_blackwell_gap_estimate"]) > 0.0
        for record in records
    )
    aggregate["mechanism_pass"] = mechanism
    correctness = all(
        bool(aggregate[key])
        for key in (
            "exactness_pass",
            "paired_identity_pass",
            "likelihood_normalization_pass",
            "candidate_accuracy_pass",
            "primary_accuracy_pass",
            "mechanism_pass",
        )
    )
    aggregate["correctness_pass"] = correctness
    aggregate["stage_pass"] = correctness and bool(aggregate["performance_pass"])
    blockers = [str(blocker) for blocker in aggregate["blockers"] if blocker != "paired_variance"]
    if not mechanism:
        blockers.append("rao_blackwell_mechanism")
    aggregate["blockers"] = blockers
    return aggregate


def aggregate_local_volterra_development(
    *, config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """V14 empirical development gate with tail certification kept orthogonal.

    Distribution-free bounded-range certification is still reported. The frozen
    V14 claim contract does not make its resource feasibility a gate for the
    replicated empirical efficiency claim.
    """

    aggregate = aggregate_structured_ecrpt_development(config=config, records=records)
    gate = config["gate"]
    minimum_qualified = gate.get("minimum_accuracy_qualified_primary_per_record")
    if minimum_qualified is not None:
        primary = tuple(str(name) for name in config["primary_comparators"])
        reference_limit = float(gate["maximum_combined_reference_z"])
        ratios: list[float] = []
        by_cell: dict[str, list[float]] = defaultdict(list)
        best_methods: dict[str, int] = defaultdict(int)
        qualified_counts: list[int] = []
        for record in records:
            qualified = {
                name: record["comparators"][name]
                for name in primary
                if float(record["comparators"][name]["combined_reference_z"]) <= reference_limit
            }
            qualified_counts.append(len(qualified))
            if not qualified:
                continue
            best_name = min(
                qualified,
                key=lambda name: float(qualified[name]["plugin_work_to_target"]["total_work"]),
            )
            candidate_work = float(record["candidate"]["plugin_work_to_target"]["total_work"])
            ratio = (
                float(qualified[best_name]["plugin_work_to_target"]["total_work"]) / candidate_work
            )
            ratios.append(ratio)
            by_cell[str(record["cell_id"])].append(ratio)
            best_methods[best_name] += 1
        roster_pass = len(ratios) == len(records) and all(
            count >= int(minimum_qualified) for count in qualified_counts
        )
        aggregate["all_primary_accuracy_pass"] = bool(aggregate["primary_accuracy_pass"])
        aggregate["primary_accuracy_pass"] = roster_pass
        aggregate["accuracy_qualified_primary_counts"] = qualified_counts
        aggregate["minimum_accuracy_qualified_primary_per_record"] = int(minimum_qualified)
        if roster_pass:
            aggregate["descriptive_best_primary_over_ecrpt_geometric_ratio"] = geometric_mean(
                ratios
            )
            aggregate["descriptive_one_sided_lower_ratio"] = one_sided_geometric_lower(
                ratios, confidence_level=float(gate["confidence_level"])
            )
            cell_ratios = {cell: geometric_mean(values) for cell, values in by_cell.items()}
            aggregate["descriptive_cell_ratios"] = cell_ratios
            aggregate["cells_favoring_ecrpt"] = sum(value > 1.0 for value in cell_ratios.values())
            aggregate["best_primary_method_counts"] = dict(best_methods)
        aggregate["correctness_pass"] = all(
            bool(aggregate[key])
            for key in (
                "exactness_pass",
                "paired_identity_pass",
                "likelihood_normalization_pass",
                "candidate_accuracy_pass",
                "primary_accuracy_pass",
                "mechanism_pass",
            )
        )
    empirical_performance = (
        float(aggregate["descriptive_best_primary_over_ecrpt_geometric_ratio"])
        > float(gate["minimum_best_primary_over_ecrpt_geometric_ratio"])
        and float(aggregate["descriptive_one_sided_lower_ratio"])
        > float(gate["minimum_one_sided_lower_ratio"])
        and int(aggregate["cells_favoring_ecrpt"]) >= int(gate["minimum_cells_favoring_ecrpt"])
    )
    aggregate["distribution_free_tail_certificate_pass"] = bool(aggregate["no_resource_censoring"])
    aggregate["distribution_free_tail_certificate_is_claim_gate"] = bool(
        gate["distribution_free_tail_certificate_is_claim_gate"]
    )
    aggregate["performance_pass"] = empirical_performance
    aggregate["stage_pass"] = bool(aggregate["correctness_pass"]) and empirical_performance
    blockers = [
        str(blocker)
        for blocker in aggregate["blockers"]
        if blocker not in {"tail_safe_uncensored", "performance"}
    ]
    if bool(aggregate["primary_accuracy_pass"]):
        blockers = [blocker for blocker in blockers if blocker != "primary_accuracy"]
    if not empirical_performance:
        blockers.append("empirical_performance")
    if bool(gate["distribution_free_tail_certificate_is_claim_gate"]) and not bool(
        aggregate["distribution_free_tail_certificate_pass"]
    ):
        blockers.append("distribution_free_tail_certificate")
        aggregate["stage_pass"] = False
    aggregate["blockers"] = blockers
    return aggregate
