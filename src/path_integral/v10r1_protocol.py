"""Pure work accounting and gated aggregation for corrected V10R1 results."""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any

from src.path_integral.v9_terminal_protocol import (
    aggregate_terminal_benchmark,
    work_to_target,
)


def attach_v10r1_work(
    *,
    config: dict[str, Any],
    paired_records: list[dict[str, Any]],
    external_records: list[dict[str, Any]],
    bank: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct training-inclusive target work for every query count."""

    paired = deepcopy(paired_records)
    external = deepcopy(external_records)
    queries = [int(value) for value in config["amortization_query_counts"]]
    relative_rmse = float(config["relative_rmse_target"])
    entries = {
        (str(entry["cell_id"]), int(entry["replicate"])): entry
        for entry in bank["entries"]
    }
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
        target = (relative_rmse * float(record["reference_estimate"])) ** 2
        record["work_to_target"] = {
            str(query): work_to_target(
                unit_variance=unit_variance,
                unit_work=unit_work,
                one_time_work=max(0.0, one_time),
                query_count=query,
                target_estimator_variance=target,
            )
            for query in queries
        }

    primary = tuple(str(value) for value in config["external_methods"]["primary"])
    for record in paired:
        key = (str(record["cell_id"]), int(record["proposal_replicate"]))
        if key not in entries:
            raise ValueError("paired record lacks its exact proposal-bank replicate")
        entry = entries[key]
        count = int(record["sample_count"])
        if count < 2:
            raise ValueError("paired record must contain at least two paths")
        training = float(entry["proposal"]["training_cost"]["algorithmic_work_units"])
        target = (relative_rmse * float(record["reference_estimate"])) ** 2
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

        matches = external_index[(str(record["cell_id"]), int(record["cluster"]))]
        methods = {str(item["method"]): item for item in matches}
        if set(methods) != set(primary):
            raise ValueError("paired record lacks one unique primary comparator per method")
        record["comparison"] = {}
        for query in queries:
            query_key = str(query)
            dcs_work = float(record["work_to_target"][query_key]["dcs"]["total_work"])
            raw_work = float(record["work_to_target"][query_key]["raw"]["total_work"])
            ratios = {
                method: float(methods[method]["work_to_target"][query_key]["total_work"])
                / dcs_work
                for method in primary
            }
            record["comparison"][query_key] = {
                "dcs_vs_raw_total_work_ratio": raw_work / dcs_work,
                "dcs_vs_external_total_work_ratios": ratios,
                "dcs_vs_best_primary_total_work_ratio": min(ratios.values()),
                "best_primary_method": min(ratios, key=ratios.__getitem__),
            }
    return paired, external


def aggregate_v10r1(
    *,
    config: dict[str, Any],
    paired_records: list[dict[str, Any]],
    external_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use the frozen V9 cluster-level gate and add training-replicate checks."""

    aggregate = aggregate_terminal_benchmark(
        config=config,
        paired_records=paired_records,
        external_records=external_records,
    )
    clusters = int(config["clusters"])
    by_cell: dict[str, set[int]] = defaultdict(set)
    for record in paired_records:
        by_cell[str(record["cell_id"])].add(int(record["proposal_replicate"]))
    training_replicates_complete = bool(by_cell) and all(
        values == set(range(clusters)) for values in by_cell.values()
    )
    aggregate["training_replicates_complete"] = training_replicates_complete
    if not training_replicates_complete:
        aggregate["stage_pass"] = False
        aggregate["selected_hurst_groups"] = []
        aggregate["blockers"] = [
            *aggregate["blockers"],
            "training_replicate_inference_failure",
        ]
    return aggregate


def semantic_equal(left: Any, right: Any) -> bool:
    """Compare nested result structures with only ulp-scale float tolerance."""

    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            semantic_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            semantic_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)
    return type(left) is type(right) and left == right
