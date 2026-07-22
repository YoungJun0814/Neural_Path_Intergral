"""Equal-cell paired power and resource planning from V6 achieved-RMSE artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral import paired_power_forecast
from src.path_integral.provenance import runtime_provenance, source_provenance

_SCHEMA = "npi.g11.v6-power-analysis.config.v1"


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 power-analysis config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "frozen",
        "statistics",
        "conservatism",
        "design",
        "resources",
    }:
        raise ValueError("malformed V6 power-analysis config fields")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_artifact(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema} artifact")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return payload, hashlib.sha256(canonical).hexdigest()


def _total_work(record: dict[str, Any]) -> float:
    records = record["result"]["total_work"]["records"]
    value = math.fsum(float(item["work_units"]) for item in records)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("qualification work must be finite and positive")
    return value


def _quantile(values: list[float], probability: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability))


def run(config_path: Path, baseline_path: Path, policy_path: Path) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    baseline, baseline_hash = _load_artifact(
        baseline_path, "npi.g11.v6-baseline-qualification.v1"
    )
    policy, policy_hash = _load_artifact(policy_path, "npi.g11.v6-routed-policy.v1")
    baseline_records = {
        (str(record["cell_id"]), int(record["cluster"])): record
        for record in baseline["records"]
        if record["method"] == "pure_cem"
    }
    policy_records = {
        (str(record["cell_id"]), int(record["cluster"])): record
        for record in policy["records"]
    }
    if not baseline_records or set(baseline_records) != set(policy_records):
        raise ValueError("baseline and policy must contain the identical complete pair set")
    pairs = []
    for key in sorted(policy_records):
        baseline_record = baseline_records[key]
        policy_record = policy_records[key]
        if (
            not baseline_record["result"]["core"]["complete"]
            or baseline_record["result"]["core"]["resource_censored"]
            or not policy_record["result"]["core"]["complete"]
            or policy_record["result"]["core"]["resource_censored"]
        ):
            raise ValueError("censored or incomplete pairs cannot be dropped from power planning")
        cem_work = _total_work(baseline_record)
        policy_work = _total_work(policy_record)
        pairs.append(
            {
                "cell_id": key[0],
                "cluster": key[1],
                "cem_work": cem_work,
                "policy_work": policy_work,
                "log_ratio": math.log(cem_work / policy_work),
            }
        )
    cells = sorted({item["cell_id"] for item in pairs})
    clusters = sorted({item["cluster"] for item in pairs})
    cluster_log_ratios = []
    for cluster in clusters:
        cluster_pairs = [item for item in pairs if item["cluster"] == cluster]
        if {item["cell_id"] for item in cluster_pairs} != set(cells):
            raise ValueError("each cluster must contain every primary cell")
        cluster_log_ratios.append(
            math.fsum(item["log_ratio"] for item in cluster_pairs) / len(cells)
        )
    if len(cluster_log_ratios) < 3:
        raise ValueError("at least three complete paired clusters are required")
    sample = torch.tensor(cluster_log_ratios, dtype=torch.float64)
    observed_mean = float(torch.mean(sample))
    observed_sd = float(torch.std(sample, unbiased=True))
    practical_effect = math.log(float(config["statistics"]["practical_geometric_mean_ratio"]))
    shrunken_effect = float(config["conservatism"]["effect_shrinkage"]) * observed_mean
    planning_effect = min(shrunken_effect, practical_effect) if shrunken_effect > 0.0 else None
    planning_sd = float(config["conservatism"]["sd_inflation"]) * observed_sd
    forecast = None
    if planning_effect is not None and planning_sd > 0.0:
        value = paired_power_forecast(
            mean_log_effect=planning_effect,
            standard_deviation=planning_sd,
            alpha=float(config["statistics"]["alpha"]),
            power=float(config["statistics"]["power"]),
        )
        forecast = {
            "mean_log_effect": value.mean_log_effect,
            "standard_deviation": value.standard_deviation,
            "alpha": value.alpha,
            "power": value.power,
            "required_clusters_normal_approximation": value.required_clusters_normal_approximation,
        }
    planned_clusters = int(config["design"]["planned_clusters"])
    paired_total_work = [item["cem_work"] + item["policy_work"] for item in pairs]
    work_p90 = _quantile(paired_total_work, 0.90)
    projected_total = work_p90 * len(cells) * planned_clusters
    projected_hours = projected_total / float(
        config["resources"]["conservative_work_units_per_second"]
    ) / 3600.0
    gates = {
        "observed_direction_favors_policy": observed_mean > 0.0,
        "power_forecast_available": forecast is not None,
        "planned_clusters_reach_power": forecast is not None
        and planned_clusters >= int(forecast["required_clusters_normal_approximation"]),
        "wall_time_within_budget": projected_hours
        <= float(config["resources"]["maximum_wall_hours"]),
        "no_pairs_excluded": len(pairs) == len(cells) * len(clusters),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "qualified_baseline_source": bool(baseline.get("baseline_qualified")),
        "qualified_policy_source": bool(policy.get("policy_qualified")),
        "clean_source": not bool(provenance["dirty_worktree"]),
    }
    return {
        "schema": "npi.g11.v6-power-analysis.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "baseline_artifact_sha256": baseline_hash,
        "policy_artifact_sha256": policy_hash,
        "cell_count": len(cells),
        "qualification_cluster_count": len(clusters),
        "equal_cell_cluster_log_ratios": cluster_log_ratios,
        "observed_mean_log_ratio": observed_mean,
        "observed_geometric_mean_ratio": math.exp(observed_mean),
        "observed_standard_deviation": observed_sd,
        "planning_effect": planning_effect,
        "planning_standard_deviation": planning_sd,
        "forecast": forecast,
        "planned_clusters": planned_clusters,
        "paired_total_work_p90_per_cell_cluster": work_p90,
        "projected_total_work_units": projected_total,
        "projected_wall_hours": projected_hours,
        "gates": gates,
        "formal_readiness": formal,
        "development_power_ready": all(gates.values()),
        "freeze_power_ready": all(gates.values()) and all(formal.values()),
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/power_analysis_development.yaml"),
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.baseline, arguments.policy)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"power_ready": result["freeze_power_ready"], **result["gates"]}))


if __name__ == "__main__":
    main()

