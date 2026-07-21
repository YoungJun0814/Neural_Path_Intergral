"""Independent integrity and statistical audit for a completed G11 M7 artifact.

This module intentionally does not call ``g11_m7_confirmatory.summarize``.  It
reconstructs the cell matrix, derived work ratios, seed-evidence hash, clustered
uncertainty, and frozen gates from the serialized artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import scipy.stats
import yaml

from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.seed_ledger import SeedKey, SeedLedger

METHODS = ("raw_defensive", "dcs_mgi")


def _strict_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    payload = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("M7 result root must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported M7 config")
    return config, hashlib.sha256(raw).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob(repository: Path, revision: str, relative_path: str) -> bytes:
    normalized_path = relative_path.replace("\\", "/")
    completed = subprocess.run(
        ["git", "show", f"{revision}:{normalized_path}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot read {relative_path} at Git revision {revision}")
    return completed.stdout


def frozen_source_failures(config: dict[str, Any], repository: Path) -> list[str]:
    """Verify the declared core blobs at their frozen source commit and tag ancestry."""

    failures: list[str] = []
    core_commit = str(config["core_source_commit"])
    tag = str(config["required_git_tag"])
    for declaration in config["core_source_manifest"]:
        relative_path = str(declaration["path"])
        try:
            blob = _git_blob(repository, core_commit, relative_path)
        except ValueError as error:
            failures.append(str(error))
            continue
        if hashlib.sha256(blob).hexdigest() != str(declaration["sha256"]):
            failures.append(f"core source hash mismatch at {core_commit}: {relative_path}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", core_commit, tag],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        failures.append("core source commit is not an ancestor of the frozen tag")
    return failures


def _task_id(family: str, probability: float) -> str:
    exponent = int(round(-math.log10(probability)))
    if not math.isclose(probability, 10.0**-exponent, rel_tol=1e-12):
        raise ValueError(f"M7 probability is not a power of ten: {probability}")
    return f"{family}_1e{exponent}"


def expected_cell_keys(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    repetitions = int(config["sampling"]["repetitions"])
    for replicate in range(repetitions):
        for regime in config["regimes"]:
            for family in regime["included_tasks"]:
                for probability in regime["target_probabilities"]:
                    keys.add(
                        (
                            str(regime["name"]),
                            _task_id(str(family), float(probability)),
                            replicate,
                        )
                    )
    return keys


def _expected_cell_contracts(
    config: dict[str, Any],
) -> dict[tuple[str, str, int], tuple[float, float]]:
    contracts: dict[tuple[str, str, int], tuple[float, float]] = {}
    repetitions = int(config["sampling"]["repetitions"])
    relative_rmse = float(config["sampling"]["relative_rmse_target"])
    for replicate in range(repetitions):
        for regime in config["regimes"]:
            for family in regime["included_tasks"]:
                for probability_value in regime["target_probabilities"]:
                    probability = float(probability_value)
                    key = (
                        str(regime["name"]),
                        _task_id(str(family), probability),
                        replicate,
                    )
                    contracts[key] = (probability, probability * relative_rmse)
    return contracts


def _close(left: Any, right: Any, *, rel_tol: float = 2e-11) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=1e-15)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, rel_tol=rel_tol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close(left[key], right[key], rel_tol=rel_tol) for key in left
        )
    return left == right


def _cell_failures(cell: dict[str, Any], raw_cap: int) -> list[str]:
    identity = f"{cell.get('regime')}/{cell.get('task')}/rep={cell.get('replicate')}"
    failures: list[str] = []
    methods = cell.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(METHODS):
        return [f"{identity}: methods are incomplete or unexpected"]

    rmse = float(cell["rmse_target"])
    for method_name in METHODS:
        method = methods[method_name]
        prefix = f"{identity}/{method_name}"
        if not isinstance(method, dict):
            failures.append(f"{prefix}: method result is not an object")
            continue
        pilot_censored = method.get("pilot_resource_censored") is True
        if pilot_censored:
            valid_pilot_censor = method_name == "raw_defensive"
            valid_pilot_censor &= method.get("resource_censored") is True
            valid_pilot_censor &= method.get("target_attained") is False
            valid_pilot_censor &= method.get("censor_stage") == "pilot"
            valid_pilot_censor &= method.get("estimate") is None
            valid_pilot_censor &= (
                method.get("seed_evidence_kind") == "attempted_batch_seed_manifest"
            )
            if not valid_pilot_censor:
                failures.append(f"{prefix}: invalid pilot-censoring semantics")
        else:
            variance = method.get("empirical_sampling_variance")
            if not isinstance(variance, (int, float)) or float(variance) < 0.0:
                failures.append(f"{prefix}: invalid empirical sampling variance")
                continue
            target_attained = float(variance) <= rmse**2
            if method.get("target_attained") is not target_attained:
                failures.append(f"{prefix}: target-attainment flag is inconsistent")
            standard_error = method.get("standard_error")
            if not isinstance(standard_error, (int, float)) or not _close(
                float(standard_error) ** 2,
                float(variance),
                rel_tol=1e-9,
            ):
                failures.append(f"{prefix}: standard error is inconsistent")
            estimate = method.get("estimate")
            interval = method.get("confidence_interval_95")
            if (
                not isinstance(estimate, (int, float))
                or not isinstance(interval, list)
                or not isinstance(standard_error, (int, float))
            ):
                failures.append(f"{prefix}: estimate or confidence interval is missing")
            else:
                radius = 1.959963984540054 * float(standard_error)
                expected_interval = [float(estimate) - radius, float(estimate) + radius]
                if not _close(interval, expected_interval, rel_tol=1e-9):
                    failures.append(f"{prefix}: confidence interval is inconsistent")
            if method_name == "dcs_mgi" and method.get("resource_censored") is not False:
                failures.append(f"{prefix}: DCS must not be resource-censored")
            if method_name == "raw_defensive":
                expected_censor = bool(method.get("allocation_capped")) and not target_attained
                if method.get("resource_censored") is not expected_censor:
                    failures.append(f"{prefix}: final-censoring flag is inconsistent")
                allocations = method.get("allocations")
                if isinstance(allocations, list) and any(
                    int(item["final_count"]) > raw_cap for item in allocations
                ):
                    failures.append(f"{prefix}: capped raw allocation exceeds its ceiling")
        evidence = method.get("seed_evidence_sha256")
        if not isinstance(evidence, str) or len(evidence) != 64:
            failures.append(f"{prefix}: invalid seed-evidence hash")
        for field in ("total_work_units", "total_wall_seconds", "process_cpu_seconds"):
            value = method.get(field)
            if not isinstance(value, (int, float)) or float(value) < 0.0:
                failures.append(f"{prefix}: invalid {field}")

    raw = methods["raw_defensive"]
    dcs = methods["dcs_mgi"]
    ratio = float(raw["total_work_units"]) / float(dcs["total_work_units"])
    expected_fields = {
        "allocated_work_ratio_raw_over_dcs": ratio,
        "matched_work_ratio_raw_over_dcs": (
            ratio if raw["target_attained"] and dcs["target_attained"] else None
        ),
        "censored_work_ratio_lower_bound": (
            ratio if raw["resource_censored"] and dcs["target_attained"] else None
        ),
        "wall_ratio_raw_over_dcs": (
            float(raw["total_wall_seconds"]) / float(dcs["total_wall_seconds"])
        ),
        "paired_estimate_difference": (
            float(dcs["estimate"]) - float(raw["estimate"])
            if dcs["estimate"] is not None and raw["estimate"] is not None
            else None
        ),
    }
    for field, expected_value in expected_fields.items():
        if not _close(cell.get(field), expected_value):
            failures.append(f"{identity}: inconsistent derived field {field}")
    return failures


def _independent_complete_seed_hash(
    config: dict[str, Any], cell: dict[str, Any], method: dict[str, Any]
) -> str:
    """Rebuild a complete method ledger from serialized allocation metadata."""

    ledger = SeedLedger()
    streams = ("proposal", "labels")
    protocol = (
        f"{config['protocol_id']}:regime={cell['regime']}:"
        f"task={cell['task']}:rep={cell['replicate']}"
    )

    pilot_batch = int(config["sampling"]["pilot_samples"])
    pilot = method.get("pilot")
    allocations = method.get("allocations")
    if not isinstance(pilot, list) or not isinstance(allocations, list):
        raise ValueError("complete method has no pilot or allocation metadata")
    for item in pilot:
        samples = int(item["count"])
        if samples < pilot_batch or samples % pilot_batch:
            raise ValueError("pilot sample count cannot reconstruct batch seeds")
        for replicate in range(samples // pilot_batch):
            for stream in streams:
                ledger.allocate(
                    SeedKey(
                        protocol,
                        "pilot",
                        str(cell["regime"]),
                        str(cell["task"]),
                        int(item["level"]),
                        replicate,
                        stream,
                    )
                )

    chunk_size = int(config["sampling"]["chunk_size"])
    for item in allocations:
        count = int(item["final_count"])
        if count < 1:
            raise ValueError("final allocation must be positive")
        for replicate in range(math.ceil(count / chunk_size)):
            for stream in streams:
                ledger.allocate(
                    SeedKey(
                        protocol,
                        "final",
                        str(cell["regime"]),
                        str(cell["task"]),
                        int(item["level"]),
                        replicate,
                        stream,
                    )
                )
    return ledger.sha256


def _cluster_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_replicate: dict[int, list[float]] = {}
    for cell in cells:
        raw = cell["methods"]["raw_defensive"]
        dcs = cell["methods"]["dcs_mgi"]
        if raw["target_attained"] and dcs["target_attained"]:
            by_replicate.setdefault(int(cell["replicate"]), []).append(
                float(cell["matched_work_ratio_raw_over_dcs"])
            )
    ratios = [
        statistics.geometric_mean(values) for _, values in sorted(by_replicate.items()) if values
    ]
    lower: float | None = None
    if len(ratios) >= 2:
        logs = [math.log(value) for value in ratios]
        standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
        critical = float(scipy.stats.t.ppf(0.95, len(logs) - 1))
        lower = math.exp(statistics.mean(logs) - critical * standard_error)
    return {
        "cluster_count": len(ratios),
        "cluster_geometric_ratios": ratios,
        "one_sided_95_lower_bound": lower,
    }


def independent_summary(
    config: dict[str, Any], cells: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    matched = [
        cell
        for cell in cells
        if cell["methods"]["raw_defensive"]["target_attained"]
        and cell["methods"]["dcs_mgi"]["target_attained"]
    ]
    censored = [
        cell
        for cell in cells
        if cell["methods"]["dcs_mgi"]["target_attained"]
        and cell["methods"]["raw_defensive"]["resource_censored"]
    ]
    dcs_fraction = statistics.fmean(
        float(cell["methods"]["dcs_mgi"]["target_attained"]) for cell in cells
    )
    raw_fraction = statistics.fmean(
        float(cell["methods"]["raw_defensive"]["target_attained"]) for cell in cells
    )
    matched_ratio = (
        statistics.geometric_mean(
            float(cell["matched_work_ratio_raw_over_dcs"]) for cell in matched
        )
        if matched
        else None
    )
    censored_ratio = (
        statistics.geometric_mean(
            float(cell["censored_work_ratio_lower_bound"]) for cell in censored
        )
        if censored
        else None
    )
    cluster = _cluster_summary(cells)
    limits = config["gates"]
    expected = len(expected_cell_keys(config))
    gates: dict[str, Any] = {
        "protocol_complete": len(cells) == expected,
        "no_unexpected_failures": not failures,
        "dcs_target_attainment_fraction": dcs_fraction,
        "dcs_target_attainment_at_least_threshold": dcs_fraction
        >= float(limits["minimum_dcs_target_attainment_fraction"]),
        "raw_target_attainment_fraction": raw_fraction,
        "matched_target_cell_count": len(matched),
        "matched_target_cells_at_least_minimum": len(matched)
        >= int(limits["minimum_matched_target_cells"]),
        "resource_censored_cell_count": len(censored),
        "matched_geometric_work_ratio": matched_ratio,
        "matched_geometric_work_ratio_above_threshold": matched_ratio is not None
        and matched_ratio > float(limits["minimum_geometric_work_ratio"]),
        "seed_cluster_count": cluster["cluster_count"],
        "seed_clusters_at_least_minimum": cluster["cluster_count"]
        >= int(limits["minimum_seed_clusters_for_uncertainty"]),
        "one_sided_95_cluster_lower_bound": cluster["one_sided_95_lower_bound"],
        "one_sided_95_cluster_lower_bound_above_one": cluster["one_sided_95_lower_bound"]
        is not None
        and float(cluster["one_sided_95_lower_bound"]) > 1.0,
    }
    gates["performance_headline_passed"] = all(
        gates[name]
        for name in (
            "protocol_complete",
            "no_unexpected_failures",
            "dcs_target_attainment_at_least_threshold",
            "matched_target_cells_at_least_minimum",
            "matched_geometric_work_ratio_above_threshold",
            "seed_clusters_at_least_minimum",
            "one_sided_95_cluster_lower_bound_above_one",
        )
    )
    summary = {
        "expected_cell_count": expected,
        "complete_cell_count": len(cells),
        "matched_cell_count": len(matched),
        "resource_censored_cell_count": len(censored),
        "dcs_target_attainment_fraction": dcs_fraction,
        "raw_target_attainment_fraction": raw_fraction,
        "matched_geometric_work_ratio": matched_ratio,
        "censored_geometric_work_ratio_lower_bound": censored_ratio,
        "cluster_uncertainty": cluster,
    }
    return summary, gates


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _variance_exponent(method: dict[str, Any]) -> float | None:
    levels = method.get("levels")
    if not isinstance(levels, list):
        return None
    points = [
        (float(item["level"]), math.log2(float(item["variance"])))
        for item in levels
        if int(item["level"]) > 0 and float(item["variance"]) > 0.0
    ]
    if len(points) < 3:
        return None
    mean_level = statistics.fmean(point[0] for point in points)
    mean_log_variance = statistics.fmean(point[1] for point in points)
    denominator = math.fsum((point[0] - mean_level) ** 2 for point in points)
    if denominator <= 0.0:
        return None
    slope = (
        math.fsum(
            (level - mean_level) * (log_variance - mean_log_variance)
            for level, log_variance in points
        )
        / denominator
    )
    return -slope


def _diagnostic_group(cells: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [
        cell
        for cell in cells
        if cell["methods"]["raw_defensive"]["target_attained"]
        and cell["methods"]["dcs_mgi"]["target_attained"]
    ]
    ratios = [float(cell["allocated_work_ratio_raw_over_dcs"]) for cell in cells]
    matched_ratios = [float(cell["matched_work_ratio_raw_over_dcs"]) for cell in matched]
    dcs_exponents = [
        exponent
        for cell in cells
        if (exponent := _variance_exponent(cell["methods"]["dcs_mgi"])) is not None
    ]
    raw_exponents = [
        exponent
        for cell in cells
        if (exponent := _variance_exponent(cell["methods"]["raw_defensive"])) is not None
    ]
    allocation_counts = {
        method_name: [
            float(allocation["final_count"])
            for cell in cells
            for allocation in (cell["methods"][method_name].get("allocations") or [])
        ]
        for method_name in METHODS
    }
    uncapped_allocation_counts = {
        method_name: [
            float(count)
            for cell in cells
            for count in (cell["methods"][method_name].get("uncapped_final_counts") or [])
        ]
        for method_name in METHODS
    }
    return {
        "cell_count": len(cells),
        "dcs_target_attainment_fraction": statistics.fmean(
            float(cell["methods"]["dcs_mgi"]["target_attained"]) for cell in cells
        ),
        "raw_target_attainment_fraction": statistics.fmean(
            float(cell["methods"]["raw_defensive"]["target_attained"]) for cell in cells
        ),
        "matched_count": len(matched),
        "raw_pilot_censored_count": sum(
            cell["methods"]["raw_defensive"]["pilot_resource_censored"] is True for cell in cells
        ),
        "raw_resource_censored_count": sum(
            cell["methods"]["raw_defensive"]["resource_censored"] is True for cell in cells
        ),
        "raw_allocation_capped_count": sum(
            cell["methods"]["raw_defensive"]["allocation_capped"] is True for cell in cells
        ),
        "allocated_work_ratio_geometric_mean": statistics.geometric_mean(ratios),
        "matched_work_ratio_geometric_mean": (
            statistics.geometric_mean(matched_ratios) if matched_ratios else None
        ),
        "allocated_work_ratio_quantiles": {
            "p50": _quantile(ratios, 0.50),
            "p90": _quantile(ratios, 0.90),
            "p99": _quantile(ratios, 0.99),
        },
        "dcs_correction_variance_exponent": {
            "count": len(dcs_exponents),
            "median": _quantile(dcs_exponents, 0.50),
            "p10": _quantile(dcs_exponents, 0.10),
            "p90": _quantile(dcs_exponents, 0.90),
        },
        "raw_correction_variance_exponent": {
            "count": len(raw_exponents),
            "median": _quantile(raw_exponents, 0.50),
            "p10": _quantile(raw_exponents, 0.10),
            "p90": _quantile(raw_exponents, 0.90),
        },
        "allocation_final_count_quantiles": {
            method_name: {
                "p50": _quantile(values, 0.50),
                "p90": _quantile(values, 0.90),
                "p99": _quantile(values, 0.99),
                "maximum": max(values) if values else None,
            }
            for method_name, values in allocation_counts.items()
        },
        "uncapped_allocation_final_count_quantiles": {
            method_name: {
                "p50": _quantile(values, 0.50),
                "p90": _quantile(values, 0.90),
                "p99": _quantile(values, 0.99),
                "maximum": max(values) if values else None,
            }
            for method_name, values in uncapped_allocation_counts.items()
        },
        "resumed_method_count": sum(
            method.get("resumed_from_checkpoint") is True
            for cell in cells
            for method in cell["methods"].values()
        ),
        "recovery_pilot_work_units": math.fsum(
            float(method.get("recovery_pilot_work_units", 0.0))
            for cell in cells
            for method in cell["methods"].values()
        ),
    }


def diagnostic_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "regime": lambda cell: str(cell["regime"]),
        "task_family": lambda cell: str(cell["task"]).split("_1e", 1)[0],
        "target_probability": lambda cell: f"{float(cell['target_probability']):.0e}",
    }
    grouped: dict[str, dict[str, Any]] = {}
    for dimension, key_function in dimensions.items():
        keys = sorted({key_function(cell) for cell in cells})
        grouped[dimension] = {
            key: _diagnostic_group([cell for cell in cells if key_function(cell) == key])
            for key in keys
        }
    return {"overall": _diagnostic_group(cells), "by": grouped}


def reference_consistency(
    config: dict[str, Any], cells: list[dict[str, Any]], repository: Path
) -> dict[str, Any]:
    references: dict[tuple[str, str], tuple[float, float]] = {}
    for regime in config["regimes"]:
        config_path = str(regime["calibration_config"]["path"])
        config_payload = yaml.safe_load(
            _git_blob(
                repository,
                str(config["required_git_tag"]),
                config_path,
            )
        )
        if not isinstance(config_payload, dict):
            raise ValueError(f"frozen calibration config is invalid: {config_path}")
        if _canonical_sha256(config_payload) != str(
            regime["calibration_config"]["canonical_sha256"]
        ):
            raise ValueError(f"frozen calibration config hash mismatch: {config_path}")
        relative_path = str(regime["calibration_result"]["path"]).replace("\\", "/")
        payload = json.loads(
            _git_blob(
                repository,
                str(config["required_git_tag"]),
                relative_path,
            )
        )
        if not isinstance(payload, dict):
            raise ValueError(f"frozen calibration result is invalid: {relative_path}")
        if _canonical_sha256(payload) != str(regime["calibration_result"]["canonical_sha256"]):
            raise ValueError(f"frozen calibration hash mismatch: {relative_path}")
        if payload.get("config_sha256") != str(regime["calibration_config"]["generation_sha256"]):
            raise ValueError(
                f"frozen calibration result/config generation mismatch: {relative_path}"
            )
        for reference in payload.get("cells", []):
            probability = float(reference["target_probability"])
            key = (
                str(regime["name"]),
                _task_id(str(reference["task"]), probability),
            )
            references[key] = (
                float(reference["validation_estimate"]),
                float(reference["validation_standard_error"]),
            )

    by_method: dict[str, Any] = {}
    for method_name in METHODS:
        scores: list[float] = []
        signed_differences: list[float] = []
        for cell in cells:
            method = cell["methods"][method_name]
            estimate = method.get("estimate")
            variance = method.get("empirical_sampling_variance")
            reference = references.get((str(cell["regime"]), str(cell["task"])))
            if (
                not isinstance(estimate, (int, float))
                or not isinstance(variance, (int, float))
                or reference is None
            ):
                continue
            difference = float(estimate) - reference[0]
            combined_standard_error = math.sqrt(float(variance) + reference[1] ** 2)
            if combined_standard_error <= 0.0:
                continue
            signed_differences.append(difference)
            scores.append(difference / combined_standard_error)
        absolute_scores = [abs(value) for value in scores]
        by_method[method_name] = {
            "comparison_count": len(scores),
            "within_independent_95_interval_fraction": (
                statistics.fmean(float(value <= 1.959963984540054) for value in absolute_scores)
                if absolute_scores
                else None
            ),
            "median_absolute_z": _quantile(absolute_scores, 0.50),
            "p95_absolute_z": _quantile(absolute_scores, 0.95),
            "maximum_absolute_z": max(absolute_scores) if absolute_scores else None,
            "mean_signed_z": statistics.fmean(scores) if scores else None,
            "mean_signed_estimate_difference": (
                statistics.fmean(signed_differences) if signed_differences else None
            ),
        }
    return {"reference_count": len(references), "by_method": by_method}


def _aggregate_seed_hash(cells: list[dict[str, Any]]) -> str:
    manifest = [
        {
            "regime": cell["regime"],
            "task": cell["task"],
            "replicate": cell["replicate"],
            "method": method,
            "seed_evidence_kind": cell["methods"][method]["seed_evidence_kind"],
            "seed_evidence_sha256": cell["methods"][method]["seed_evidence_sha256"],
        }
        for cell in cells
        for method in sorted(METHODS)
    ]
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def _tag_commit(tag: str, cwd: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _recovered_operational_incident(
    incident: dict[str, Any], cells_by_key: dict[tuple[str, str, int], dict[str, Any]]
) -> bool:
    """Recognize a checkpoint I/O incident followed by a complete resumed method."""

    if incident.get("type") not in {"PermissionError", "OSError"}:
        return False
    message = str(incident.get("message", "")).lower()
    if "checkpoint" not in message:
        return False
    try:
        key = (
            str(incident["regime"]),
            str(incident["task"]),
            int(incident["replicate"]),
        )
        method = cells_by_key[key]["methods"][str(incident["method"])]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        method.get("resumed_from_checkpoint") is True
        and method.get("seed_evidence_kind") == "complete_mlmc_seed_ledger"
        and method.get("target_attained") in {True, False}
        and method.get("estimate") is not None
    )


def run(config_path: Path, result_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load_config(config_path)
    result, result_hash = _strict_json(result_path)
    failures: list[str] = []
    repository = config_path.resolve().parents[1]
    failures.extend(frozen_source_failures(config, repository))
    expected_keys = expected_cell_keys(config)
    expected_contracts = _expected_cell_contracts(config)
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise ValueError("M7 result has no cell list")
    observed_keys = [
        (str(cell.get("regime")), str(cell.get("task")), int(cell.get("replicate")))
        for cell in cells
    ]
    if len(observed_keys) != len(set(observed_keys)):
        failures.append("duplicate cell identity")
    if set(observed_keys) != expected_keys:
        failures.append("observed cell matrix differs from the frozen config")
    for cell in cells:
        identity = (
            str(cell.get("regime")),
            str(cell.get("task")),
            int(cell.get("replicate")),
        )
        contract = expected_contracts.get(identity)
        if contract is not None and (
            not _close(cell.get("target_probability"), contract[0])
            or not _close(cell.get("rmse_target"), contract[1])
        ):
            failures.append(
                f"{identity[0]}/{identity[1]}/rep={identity[2]}: "
                "probability or RMSE contract mismatch"
            )
        failures.extend(
            _cell_failures(
                cell,
                int(config["resource_limits"]["raw_max_final_samples_per_level"]),
            )
        )
        for method_name in METHODS:
            method = cell.get("methods", {}).get(method_name)
            if not isinstance(method, dict):
                continue
            if method.get("seed_evidence_kind") != "complete_mlmc_seed_ledger":
                continue
            try:
                reconstructed = _independent_complete_seed_hash(config, cell, method)
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    f"{cell.get('regime')}/{cell.get('task')}/"
                    f"rep={cell.get('replicate')}/{method_name}: "
                    f"seed-ledger reconstruction failed: {error}"
                )
                continue
            if method.get("seed_ledger_sha256") != reconstructed:
                failures.append(
                    f"{cell.get('regime')}/{cell.get('task')}/"
                    f"rep={cell.get('replicate')}/{method_name}: "
                    "independent seed-ledger hash mismatch"
                )
            if method.get("seed_evidence_sha256") != reconstructed:
                failures.append(
                    f"{cell.get('regime')}/{cell.get('task')}/"
                    f"rep={cell.get('replicate')}/{method_name}: "
                    "seed-evidence hash differs from reconstructed ledger"
                )

    serialized_failures = result.get("failures")
    if not isinstance(serialized_failures, list):
        failures.append("result failures field is not a list")
        serialized_failures = []
    cells_by_key = {
        (str(cell["regime"]), str(cell["task"]), int(cell["replicate"])): cell for cell in cells
    }
    recovered_incidents = [
        incident
        for incident in serialized_failures
        if isinstance(incident, dict) and _recovered_operational_incident(incident, cells_by_key)
    ]
    unresolved_incidents = [
        incident for incident in serialized_failures if incident not in recovered_incidents
    ]
    if unresolved_incidents:
        failures.append("result contains unresolved execution failures")
    if result.get("budget_exhausted") is not False:
        failures.append("CPU budget was exhausted")

    if set(observed_keys) == expected_keys and len(observed_keys) == len(expected_keys):
        summary, gates = independent_summary(config, cells, serialized_failures)
        if not _close(result.get("summary"), summary, rel_tol=1e-9):
            failures.append("serialized summary differs from independent reconstruction")
        if not _close(result.get("gates"), gates, rel_tol=1e-9):
            failures.append("serialized gates differ from independent reconstruction")
    else:
        summary, gates = {}, {}

    expected_seed_hash = _aggregate_seed_hash(cells)
    if result.get("seed_evidence_sha256") != expected_seed_hash:
        failures.append("aggregate seed-evidence hash mismatch")
    if result.get("seed_manifest_entries") != 2 * len(cells):
        failures.append("seed-manifest entry count mismatch")
    all_complete_ledgers = all(
        method["seed_evidence_kind"] == "complete_mlmc_seed_ledger"
        for cell in cells
        for method in cell.get("methods", {}).values()
    )
    expected_ledger_hash = expected_seed_hash if all_complete_ledgers else None
    if result.get("seed_ledger_sha256") != expected_ledger_hash:
        failures.append("aggregate complete-ledger hash mismatch")

    expected_wall = math.fsum(
        float(method["total_wall_seconds"])
        for cell in cells
        for method in cell.get("methods", {}).values()
    )
    expected_cpu = math.fsum(
        float(method["process_cpu_seconds"])
        for cell in cells
        for method in cell.get("methods", {}).values()
    )
    work = result.get("work_ledger", {})
    if not _close(work.get("measured_method_wall_seconds"), expected_wall, rel_tol=1e-9):
        failures.append("method wall-time ledger mismatch")
    if not _close(work.get("measured_process_cpu_seconds"), expected_cpu, rel_tol=1e-9):
        failures.append("process CPU ledger mismatch")

    fixed_fields = {
        "schema": "npi.g11.m7-confirmatory.v1",
        "protocol_id": config["protocol_id"],
        "run_class": config["run_class"],
        "frozen": True,
        "config_sha256": config_hash,
        "core_source_commit": config["core_source_commit"],
        "estimand": "fixed finest finite-grid probability",
        "continuous_time_claim": False,
        "self_normalized": False,
        "dirty_worktree": False,
        "resource_limits": config["resource_limits"],
        "core_source_manifest": [
            {
                "path": str(Path(str(item["path"]))),
                "sha256": str(item["sha256"]),
            }
            for item in config["core_source_manifest"]
        ],
    }
    for field, expected in fixed_fields.items():
        if result.get(field) != expected:
            failures.append(f"invalid frozen field {field}")
    environment = result.get("environment")
    if not isinstance(environment, dict):
        failures.append("missing environment provenance")
    else:
        if int(environment.get("torch_threads", -1)) != int(
            config["resource_limits"]["torch_threads"]
        ):
            failures.append("recorded torch thread count differs from config")
        if environment.get("dtype") != "torch.float64":
            failures.append("recorded dtype is not torch.float64")
    tag_commit = _tag_commit(str(config["required_git_tag"]), config_path.resolve().parents[1])
    if tag_commit is None or result.get("source_commit") != tag_commit:
        failures.append("result source commit does not match the required freeze tag")

    expected_inputs: list[dict[str, str]] = []
    for regime in config["regimes"]:
        expected_inputs.extend(
            (
                {
                    "path": str(regime["calibration_config"]["path"]),
                    "canonical_sha256": str(regime["calibration_config"]["canonical_sha256"]),
                    "generation_sha256": str(regime["calibration_config"]["generation_sha256"]),
                },
                {
                    "path": str(regime["calibration_result"]["path"]),
                    "canonical_sha256": str(regime["calibration_result"]["canonical_sha256"]),
                },
            )
        )
    if result.get("input_artifacts") != expected_inputs:
        failures.append("result input-artifact manifest differs from frozen config")

    relative_config = config_path.resolve().relative_to(repository).as_posix()
    tagged_config = subprocess.run(
        ["git", "show", f"{config['required_git_tag']}:{relative_config}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if (
        tagged_config.returncode != 0
        or hashlib.sha256(tagged_config.stdout).hexdigest() != config_hash
    ):
        failures.append("audited config bytes differ from the frozen Git tag")

    pilot_censored_evidence = sum(
        method.get("seed_evidence_kind") == "attempted_batch_seed_manifest"
        for cell in cells
        for method in cell.get("methods", {}).values()
    )
    resumed_methods = sum(
        method.get("resumed_from_checkpoint") is True
        for cell in cells
        for method in cell.get("methods", {}).values()
    )
    limitations: list[str] = []
    if pilot_censored_evidence:
        limitations.append(
            "pilot-censored attempted-batch seed hashes are aggregate evidence only; "
            "their individual call manifests were not serialized by V3"
        )
    if resumed_methods:
        limitations.append(
            "V3 operation work and sampler wall time survive checkpoints, but process CPU "
            "and orchestration time spent after the last successful checkpoint are not "
            "fully recoverable across an external interruption"
        )

    return {
        "schema": "npi.g11.m7-result-audit.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "result": str(result_path),
        "result_sha256": result_hash,
        "expected_cell_count": len(expected_keys),
        "observed_cell_count": len(cells),
        "independent_summary": summary,
        "independent_gates": gates,
        "diagnostics": diagnostic_summary(cells) if cells else {},
        "reference_consistency": (
            reference_consistency(config, cells, repository) if cells else {}
        ),
        "scientific_gate_passed": gates.get("performance_headline_passed") is True,
        "performance_subgates_excluding_recovered_operational_events_passed": (
            bool(recovered_incidents)
            and not unresolved_incidents
            and all(
                gates.get(name) is True
                for name in (
                    "protocol_complete",
                    "dcs_target_attainment_at_least_threshold",
                    "matched_target_cells_at_least_minimum",
                    "matched_geometric_work_ratio_above_threshold",
                    "seed_clusters_at_least_minimum",
                    "one_sided_95_cluster_lower_bound_above_one",
                )
            )
        ),
        "execution_incidents": serialized_failures,
        "recovered_operational_incident_count": len(recovered_incidents),
        "unresolved_execution_incident_count": len(unresolved_incidents),
        "integrity_failures": failures,
        "integrity_passed": not failures,
        "audit_limitations": limitations,
        "work_ledger": {"audit_seconds": time.perf_counter() - started},
        "environment": runtime_provenance(dtype="not_applicable"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit = run(arguments.config, arguments.result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "integrity_passed": audit["integrity_passed"],
                "scientific_gate_passed": audit["scientific_gate_passed"],
                "integrity_failures": audit["integrity_failures"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
