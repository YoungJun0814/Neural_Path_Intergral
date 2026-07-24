"""Joint mechanism and fair-work analysis for the V7 raw/DCS study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import scipy.stats
import yaml

from src.path_integral.provenance import runtime_provenance, source_provenance

_SCHEMA = "npi.g11.v7-mechanism-analysis.config.v1"


def _load_json(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"unsupported artifact schema; expected {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V7 mechanism-analysis config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "matrix",
        "statistics",
        "development_thresholds",
        "requirements",
    }:
        raise ValueError("malformed V7 mechanism-analysis config fields")
    if payload["phase"] not in {"development", "qualification", "confirmation"}:
        raise ValueError("unsupported V7 mechanism-analysis phase")
    if payload["phase"] != "development" and payload["frozen"] is not True:
        raise ValueError("formal V7 mechanism analyses must be frozen")
    matrix = payload["matrix"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "expected_cells",
        "expected_clusters",
        "methods",
    }:
        raise ValueError("malformed V7 mechanism-analysis matrix")
    _positive_integer(matrix["expected_cells"], "expected_cells")
    _positive_integer(matrix["expected_clusters"], "expected_clusters")
    if matrix["methods"] != ["fixed_dcs_slis", "fixed_raw_defensive"]:
        raise ValueError("V7 analysis requires the frozen raw/DCS method pair")
    statistics = payload["statistics"]
    if not isinstance(statistics, dict) or set(statistics) != {"confidence_level"}:
        raise ValueError("malformed V7 mechanism-analysis statistics")
    confidence = _positive_real(statistics["confidence_level"], "confidence_level")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    thresholds = payload["development_thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "minimum_probe_variance_ratio_lower",
        "minimum_execution_variance_ratio_lower",
        "minimum_final_work_ratio_lower",
        "maximum_floor_fraction_per_method",
    }:
        raise ValueError("malformed V7 mechanism-analysis thresholds")
    for field in (
        "minimum_probe_variance_ratio_lower",
        "minimum_execution_variance_ratio_lower",
        "minimum_final_work_ratio_lower",
    ):
        if _positive_real(thresholds[field], field) <= 1.0:
            raise ValueError(f"{field} must exceed one")
    floor_fraction = _positive_real(
        thresholds["maximum_floor_fraction_per_method"],
        "maximum_floor_fraction_per_method",
    )
    if floor_fraction >= 1.0:
        raise ValueError("maximum floor fraction must be below one")
    requirements = payload["requirements"]
    expected_requirements = {
        "identical_manifest",
        "identical_reference",
        "identical_proposal_training_source",
        "disjoint_probe_and_execution_seeds",
        "raw_fast_path_required",
    }
    if (
        not isinstance(requirements, dict)
        or set(requirements) != expected_requirements
        or any(requirements[field] is not True for field in expected_requirements)
    ):
        raise ValueError("every V7 mechanism-analysis requirement must be true")
    return payload, hashlib.sha256(raw).hexdigest()


def _one_sided_ratio(
    cluster_log_ratios: list[float],
    *,
    confidence_level: float,
) -> dict[str, float | int]:
    count = len(cluster_log_ratios)
    if count < 2:
        raise ValueError("paired ratio inference requires at least two clusters")
    mean = math.fsum(cluster_log_ratios) / count
    variance = math.fsum((value - mean) ** 2 for value in cluster_log_ratios) / (
        count - 1
    )
    standard_error = math.sqrt(variance / count)
    if standard_error == 0.0:
        lower = mean
        p_value = 0.0 if mean > 0.0 else 1.0
    else:
        critical = float(scipy.stats.t.ppf(confidence_level, df=count - 1))
        lower = mean - critical * standard_error
        p_value = float(scipy.stats.t.sf(mean / standard_error, df=count - 1))
    return {
        "cluster_count": count,
        "mean_log_ratio": mean,
        "standard_error": standard_error,
        "geometric_mean_ratio": math.exp(mean),
        "one_sided_lower_geometric_mean_ratio": math.exp(lower),
        "p_value_against_ratio_one": p_value,
    }


def _work(record: dict[str, Any], *roles: str) -> float:
    entries = record["result"]["core"]["work"]["entries"]
    selected = [
        float(entry["work_units"])
        for entry in entries
        if not roles or str(entry["role"]) in roles
    ]
    if roles and not selected:
        raise ValueError(f"record lacks required work roles {roles}")
    result = math.fsum(selected)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("record work must be finite and positive")
    return result


def _wall(record: dict[str, Any], role: str) -> float:
    values = [
        float(entry["wall_seconds"])
        for entry in record["result"]["core"]["work"]["entries"]
        if entry["role"] == role
    ]
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0.0:
        raise ValueError(f"record requires one positive {role} wall-time entry")
    return values[0]


def _seed_values(payload: dict[str, Any]) -> set[int]:
    ledger = payload.get("seed_ledger")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("records"), list):
        raise ValueError("artifact lacks a seed ledger")
    seeds = [record.get("seed") for record in ledger["records"]]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("artifact seed ledger is malformed")
    result = set(seeds)
    if len(result) != len(seeds):
        raise ValueError("artifact seed ledger contains duplicate seed values")
    return result


def run(
    config_path: Path,
    probe_path: Path,
    fixed_estimators_path: Path,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    probe, probe_hash = _load_json(
        probe_path,
        "npi.g11.v7-mechanism-probe.v1",
    )
    fixed, fixed_hash = _load_json(
        fixed_estimators_path,
        "npi.g11.v6-secondary-baselines.v1",
    )
    matrix = config["matrix"]
    expected_cells = int(matrix["expected_cells"])
    expected_clusters = int(matrix["expected_clusters"])
    methods = list(matrix["methods"])
    fixed_records = list(fixed["records"])
    fixed_keys = [
        (
            str(record["cell_id"]),
            int(record["cluster"]),
            str(record["method"]),
        )
        for record in fixed_records
    ]
    if len(set(fixed_keys)) != len(fixed_keys):
        raise ValueError("fixed-estimator artifact contains duplicate records")
    cells = sorted({key[0] for key in fixed_keys})
    clusters = sorted({key[1] for key in fixed_keys})
    expected_keys = {
        (cell, cluster, method)
        for cell in cells
        for cluster in clusters
        for method in methods
    }
    if (
        len(cells) != expected_cells
        or len(clusters) != expected_clusters
        or set(fixed_keys) != expected_keys
    ):
        raise ValueError("fixed-estimator matrix violates the V7 analysis contract")
    probe_records = list(probe["records"])
    probe_keys = [
        (str(record["cell_id"]), int(record["cluster"]))
        for record in probe_records
    ]
    expected_probe_keys = {
        (cell, cluster) for cell in cells for cluster in clusters
    }
    if (
        len(set(probe_keys)) != len(probe_keys)
        or set(probe_keys) != expected_probe_keys
    ):
        raise ValueError("mechanism-probe matrix does not match fixed estimators")
    records_by_key = {
        (str(record["cell_id"]), int(record["cluster"]), str(record["method"])): record
        for record in fixed_records
    }
    confidence = float(config["statistics"]["confidence_level"])
    metric_logs: dict[str, list[float]] = {
        "execution_variance": [],
        "final_work": [],
        "nontraining_work": [],
        "training_inclusive_work": [],
        "final_wall": [],
    }
    paired_records = []
    floor_counts = {method: 0 for method in methods}
    for cluster in clusters:
        within: dict[str, list[float]] = {
            metric: [] for metric in metric_logs
        }
        for cell in cells:
            dcs = records_by_key[(cell, cluster, "fixed_dcs_slis")]
            raw = records_by_key[(cell, cluster, "fixed_raw_defensive")]
            dcs_variance = float(dcs["result"]["core"]["terms"][0]["variance"])
            raw_variance = float(raw["result"]["core"]["terms"][0]["variance"])
            ratios = {
                "execution_variance": raw_variance / dcs_variance,
                "final_work": _work(raw, "final") / _work(dcs, "final"),
                "nontraining_work": _work(
                    raw,
                    "allocation_pilot",
                    "final",
                )
                / _work(dcs, "allocation_pilot", "final"),
                "training_inclusive_work": _work(raw) / _work(dcs),
                "final_wall": _wall(raw, "final") / _wall(dcs, "final"),
            }
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in ratios.values()
            ):
                raise ValueError("paired V7 ratio is nonfinite or nonpositive")
            for metric, ratio in ratios.items():
                within[metric].append(math.log(ratio))
            for method, record in (
                ("fixed_dcs_slis", dcs),
                ("fixed_raw_defensive", raw),
            ):
                final_count = int(
                    record["result"]["core"]["allocations"][0]["final_count"]
                )
                floor = int(record["preparation"]["minimum_final_samples"])
                if final_count == floor:
                    floor_counts[method] += 1
                elif final_count < floor:
                    raise ValueError("executed final count is below its common floor")
            paired_records.append(
                {
                    "cell_id": cell,
                    "cluster": cluster,
                    **{f"raw_over_dcs_{key}_ratio": value for key, value in ratios.items()},
                }
            )
        for metric in metric_logs:
            metric_logs[metric].append(math.fsum(within[metric]) / len(cells))
    effects = {
        metric: _one_sided_ratio(values, confidence_level=confidence)
        for metric, values in metric_logs.items()
    }
    floor_fractions = {
        method: floor_counts[method] / (len(cells) * len(clusters))
        for method in methods
    }
    probe_effect = probe["paired_cluster_effect"]
    thresholds = config["development_thresholds"]
    same_source_commit = (
        isinstance(probe.get("source_commit"), str)
        and probe["source_commit"] == fixed.get("source_commit")
    )
    probe_seeds = _seed_values(probe)
    fixed_seeds = _seed_values(fixed)
    gates = {
        "probe_integrity": bool(probe.get("development_mechanism_passed")),
        "fixed_estimator_integrity": all(
            bool(value) for value in fixed.get("qualification_gates", {}).values()
        ),
        "same_manifest": probe.get("manifest_sha256") == fixed.get("manifest_sha256"),
        "same_reference": probe.get("reference_artifact_sha256")
        == fixed.get("reference_artifact_sha256"),
        "same_proposal_training_source": probe.get(
            "proposal_training_audit", {}
        ).get("source_artifact_sha256")
        == fixed.get("proposal_training_audit", {}).get("source_artifact_sha256"),
        "same_execution_source": same_source_commit,
        "disjoint_probe_and_execution_seeds": probe_seeds.isdisjoint(fixed_seeds),
        "probe_variance_ratio_lower": float(
            probe_effect["one_sided_lower_raw_over_dcs_variance_ratio"]
        )
        >= float(thresholds["minimum_probe_variance_ratio_lower"]),
        "execution_variance_ratio_lower": float(
            effects["execution_variance"][
                "one_sided_lower_geometric_mean_ratio"
            ]
        )
        >= float(thresholds["minimum_execution_variance_ratio_lower"]),
        "final_work_ratio_lower": float(
            effects["final_work"]["one_sided_lower_geometric_mean_ratio"]
        )
        >= float(thresholds["minimum_final_work_ratio_lower"]),
        "floor_not_binding": all(
            fraction
            <= float(thresholds["maximum_floor_fraction_per_method"])
            for fraction in floor_fractions.values()
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_analysis": bool(config["frozen"]),
        "frozen_probe": bool(probe.get("formal_readiness", {}).get("frozen_config")),
        "frozen_fixed_estimators": bool(fixed.get("formal_readiness", {}).get("frozen_config")),
        "clean_analysis_source": not bool(provenance["dirty_worktree"]),
        "clean_input_sources": not bool(probe.get("dirty_worktree"))
        and not bool(fixed.get("dirty_worktree")),
        "non_smoke_inputs": not bool(probe.get("smoke"))
        and not bool(fixed.get("smoke")),
        "raw_fast_path_source_contract": bool(
            config["requirements"]["raw_fast_path_required"]
        )
        and same_source_commit,
    }
    return {
        "schema": "npi.g11.v7-mechanism-analysis.v1",
        "protocol_id": config["protocol_id"],
        "phase": config["phase"],
        "config_sha256": config_hash,
        "source_artifact_sha256": {
            "mechanism_probe": probe_hash,
            "fixed_estimators": fixed_hash,
        },
        "cell_count": len(cells),
        "cluster_count": len(clusters),
        "paired_records": paired_records,
        "effects": effects,
        "probe_effect": probe_effect,
        "floor_counts": floor_counts,
        "floor_fractions": floor_fractions,
        "seed_counts": {
            "mechanism_probe": len(probe_seeds),
            "fixed_estimators": len(fixed_seeds),
        },
        "gates": gates,
        "formal_readiness": formal,
        "development_mechanism_qualified": all(gates.values()),
        "formal_mechanism_qualified": all(gates.values()) and all(formal.values()),
        "interpretation": (
            "development mechanism evidence only; paired-probe work is diagnostic "
            "and is excluded from either production estimator's work ratio"
        ),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--fixed-estimators", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.probe,
        arguments.fixed_estimators,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "development_mechanism_qualified": result[
                    "development_mechanism_qualified"
                ],
                **result["gates"],
            }
        )
    )


if __name__ == "__main__":
    main()
