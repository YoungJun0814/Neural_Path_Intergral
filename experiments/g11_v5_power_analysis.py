"""Pre-freeze paired work-ratio power and resource forecast for G11 V5."""

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


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != (
        "npi.g11.v5-power-analysis.config.v1"
    ):
        raise ValueError("unsupported V5 power-analysis config")
    return payload, hashlib.sha256(raw).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability))


def run(config_path: Path) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    source_path = Path(config["qualification_source"]["path"])
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != config["qualification_source"]["sha256"]:
        raise ValueError("qualification source hash mismatch")
    source = json.loads(source_bytes)
    if source.get("schema") != "npi.g11.v4-crossover-qualification.v1" or not source.get(
        "gates", {}
    ).get("qualification_passed"):
        raise ValueError("power analysis requires the qualified V4 source artifact")

    forecasts: dict[str, Any] = {}
    for target in source["relative_rmse_targets"]:
        target_key = f"{float(target):.2f}"
        log_ratios: list[float] = []
        selected_work: list[float] = []
        excluded = 0
        for cell in source["cells"]:
            for replicate in cell["runs"]:
                decision = replicate["decisions"][target_key]
                cem = decision["cem_slis_total_work"]
                selected = decision["selected_total_work"]
                if cem is None or selected is None or cem <= 0.0 or selected <= 0.0:
                    excluded += 1
                    continue
                log_ratios.append(math.log(float(cem) / float(selected)))
                selected_work.append(float(selected))
        if len(log_ratios) < 3:
            raise ValueError("too few paired qualification observations for power")
        tensor = torch.tensor(log_ratios, dtype=torch.float64)
        observed_mean = float(torch.mean(tensor))
        observed_sd = float(torch.std(tensor, unbiased=True))
        planning_mean = float(config["conservatism"]["effect_shrinkage"]) * observed_mean
        planning_sd = float(config["conservatism"]["sd_inflation"]) * observed_sd
        forecast = paired_power_forecast(
            mean_log_effect=planning_mean,
            standard_deviation=planning_sd,
            alpha=float(config["statistics"]["alpha"]),
            power=float(config["statistics"]["power"]),
        )
        work_p90 = _quantile(selected_work, 0.90)
        clusters = int(config["design"]["clusters"])
        cells = int(config["design"]["primary_cells"])
        total_work = work_p90 * clusters * cells
        throughput = float(config["resources"]["conservative_work_units_per_second"])
        forecasts[target_key] = {
            "paired_observations": len(log_ratios),
            "excluded_unavailable_cem_pairs": excluded,
            "observed_mean_log_effect": observed_mean,
            "observed_standard_deviation": observed_sd,
            "planning_mean_log_effect": planning_mean,
            "planning_standard_deviation": planning_sd,
            "forecast": {
                "alpha": forecast.alpha,
                "power": forecast.power,
                "required_clusters_normal_approximation": (
                    forecast.required_clusters_normal_approximation
                ),
            },
            "planned_clusters": clusters,
            "power_gate": clusters >= forecast.required_clusters_normal_approximation,
            "selected_work_p90_per_cell_cluster": work_p90,
            "projected_total_work_units": total_work,
            "projected_wall_hours": total_work / throughput / 3600.0,
        }
    gates = {
        "at_least_80_percent_power_all_targets": all(
            item["power_gate"] for item in forecasts.values()
        ),
        "wall_time_within_budget": max(item["projected_wall_hours"] for item in forecasts.values())
        <= float(config["resources"]["maximum_wall_hours"]),
    }
    provenance = source_provenance()
    formal_readiness = {
        "frozen_config": bool(config.get("frozen")),
        "clean_source": not bool(provenance["dirty_worktree"]),
    }
    return {
        "schema": "npi.g11.v5-power-analysis.v1",
        "config_sha256": config_hash,
        "qualification_source": {
            "path": str(source_path),
            "sha256": source_hash,
        },
        "scope_warning": (
            "V4 profile work is a planning source, not achieved-RMSE V5 evidence; "
            "effect shrinkage and variance inflation are predeclared."
        ),
        "forecasts": forecasts,
        "gates": gates,
        "formal_readiness": formal_readiness,
        "development_power_ready": all(gates.values()),
        "freeze_power_ready": all(gates.values()) and all(formal_readiness.values()),
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g11_v5_power_analysis.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
