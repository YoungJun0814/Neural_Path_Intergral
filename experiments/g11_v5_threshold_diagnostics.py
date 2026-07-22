"""V5 falsification diagnostics for terminal and discrete-barrier thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral import (
    DiscreteBarrierHitTask,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    evaluate_rbergomi_dcs_adjacent,
    evaluate_rbergomi_threshold_coupling,
    simulate_coupled_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != (
        "npi.g11.v5-threshold-diagnostics.config.v1"
    ):
        raise ValueError("unsupported V5 threshold-diagnostics config")
    if payload.get("estimand") != "finite_grid":
        raise ValueError("threshold diagnostics require an explicit finite-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _task(specification: dict[str, Any]):
    if specification.get("kind") == "terminal":
        return TerminalThresholdTask(float(specification["level"]))
    if specification.get("kind") == "barrier":
        return DiscreteBarrierHitTask(float(specification["barrier"]))
    raise ValueError("V5 threshold diagnostics support terminal and barrier tasks")


def _moments(values: torch.Tensor) -> dict[str, float | None]:
    sample = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if sample.numel() == 0:
        return {
            "mean": None,
            "absolute_first": None,
            "second": None,
            "fourth": None,
            "kurtosis": None,
        }
    mean = float(torch.mean(sample))
    centered = sample - mean
    second_centered = float(torch.mean(centered.square()))
    fourth_centered = float(torch.mean(centered.pow(4)))
    return {
        "mean": mean,
        "absolute_first": float(torch.mean(torch.abs(sample))),
        "second": float(torch.mean(sample.square())),
        "fourth": float(torch.mean(sample.pow(4))),
        "kurtosis": (fourth_centered / second_centered**2 if second_centered > 0.0 else None),
    }


def _positive_rate(level_values: list[tuple[float, float]]) -> float | None:
    usable = [(step, value) for step, value in level_values if step > 0.0 and value > 0.0]
    if len(usable) < 2:
        return None
    x = torch.log(torch.tensor([item[0] for item in usable], dtype=torch.float64))
    y = torch.log(torch.tensor([item[1] for item in usable], dtype=torch.float64))
    centered = x - torch.mean(x)
    denominator = float(torch.sum(centered.square()))
    if denominator == 0.0:
        return None
    return float(torch.sum(centered * (y - torch.mean(y))) / denominator)


def _cell_metrics(
    evaluation,
    diagnostics,
    kappas: tuple[float, ...],
    maturity: float,
    early_active_fraction: float,
):
    finite = diagnostics.finite_threshold
    path_index = torch.arange(finite.numel(), device=finite.device)
    active_slope = evaluation.fine.log_spot_slope[path_index, diagnostics.fine_active_index]
    finite_active_slope = active_slope[finite]
    negative_slope_moments: dict[str, float | None] = {}
    for power in (1, 2):
        negative_slope_moments[str(power)] = (
            float(torch.mean(finite_active_slope.pow(-power)))
            if finite_active_slope.numel()
            else None
        )
    bad_event_probability = {
        format(kappa, ".12g"): (
            float(torch.mean((finite_active_slope < kappa).to(torch.float64)))
            if finite_active_slope.numel()
            else None
        )
        for kappa in kappas
    }
    early_active = (
        float(
            torch.mean(
                (diagnostics.fine_active_time[finite] <= early_active_fraction * maturity).to(
                    torch.float64
                )
            )
        )
        if finite.any()
        else None
    )
    return {
        "finite_threshold_fraction": float(torch.mean(finite.to(torch.float64))),
        "initially_hit_fraction": float(torch.mean(diagnostics.initially_hit.to(torch.float64))),
        "numerator_error": _moments(diagnostics.numerator_error[finite]),
        "denominator_error": _moments(diagnostics.denominator_error[finite]),
        "common_candidate_error": _moments(diagnostics.common_candidate_error[finite]),
        "mesh_enrichment_defect": _moments(diagnostics.mesh_enrichment_defect[finite]),
        "threshold_error": _moments(diagnostics.threshold_error[finite]),
        "negative_active_slope_moments": negative_slope_moments,
        "active_slope_bad_event_probability": bad_event_probability,
        "early_active_probability": early_active,
        "raw_correction": _moments(evaluation.raw_correction),
        "dcs_correction": _moments(evaluation.marginalized_correction),
        "maximum_good_event_bound_violation": (diagnostics.maximum_good_event_bound_violation),
        "maximum_exact_decomposition_violation": (
            diagnostics.maximum_exact_decomposition_violation
        ),
        "maximum_path_reconstruction_error": max(
            evaluation.fine.maximum_path_reconstruction_error,
            evaluation.coarse.maximum_path_reconstruction_error,
        ),
        "maximum_coordinate_mismatch": evaluation.maximum_coordinate_mismatch,
        "maximum_full_likelihood": float(torch.amax(evaluation.fine.density.full_likelihood)),
        "maximum_residual_likelihood": float(
            torch.amax(evaluation.fine.density.residual_likelihood)
        ),
        "maximum_density_reconstruction_error": max(
            evaluation.fine.density.maximum_component_reconstruction_error,
            evaluation.fine.density.maximum_mixture_reconstruction_error,
        ),
        "maximum_likelihood_bound_violation": max(
            evaluation.fine.density.maximum_full_bound_violation,
            evaluation.fine.density.maximum_residual_bound_violation,
        ),
    }


def _rate_summary(records: list[dict[str, Any]], minimum_level: int) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if int(record["level"]) >= minimum_level:
            grouped.setdefault(int(record["level"]), []).append(record)
    metric_paths = {
        "numerator_l1": ("numerator_error", "absolute_first"),
        "numerator_l2": ("numerator_error", "second"),
        "denominator_l1": ("denominator_error", "absolute_first"),
        "denominator_l2": ("denominator_error", "second"),
        "mesh_l1": ("mesh_enrichment_defect", "absolute_first"),
        "mesh_l2": ("mesh_enrichment_defect", "second"),
        "threshold_l1": ("threshold_error", "absolute_first"),
        "threshold_l2": ("threshold_error", "second"),
        "raw_second": ("raw_correction", "second"),
        "dcs_second": ("dcs_correction", "second"),
    }
    level_means: dict[str, list[tuple[float, float]]] = {name: [] for name in metric_paths}
    for level in sorted(grouped):
        items = grouped[level]
        step_dt = float(items[0]["fine_step_dt"])
        for name, (group, metric) in metric_paths.items():
            values = [item["metrics"][group][metric] for item in items]
            finite_values = [float(value) for value in values if value is not None]
            if finite_values:
                level_means[name].append((step_dt, math.fsum(finite_values) / len(finite_values)))
    return {
        "minimum_level": minimum_level,
        "levels_used": sorted(grouped),
        "descriptive_exponents": {
            name: _positive_rate(values) for name, values in level_means.items()
        },
        "level_means": {
            name: [{"fine_step_dt": step, "value": value} for step, value in values]
            for name, values in level_means.items()
        },
        "claim_scope": "falsification diagnostic only; not a model-level proof",
    }


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    hierarchy = config["hierarchy"]
    sampling = config["sampling"]
    levels = tuple(int(item) for item in hierarchy["adjacent_levels"])
    if levels != tuple(range(levels[0], levels[-1] + 1)) or levels[0] < 1:
        raise ValueError("adjacent levels must be consecutive and positive")
    if smoke:
        levels = levels[: min(3, len(levels))]
    replicates = 2 if smoke else int(sampling["replicates"])
    paths = 256 if smoke else int(sampling["paths_per_level"])
    maturity = float(config["model_common"]["maturity"])
    controls = tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(value) for value in segment) for segment in schedule),
            maturity=maturity,
        )
        for schedule in config["proposal"]["controls"]
    )
    weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
    tasks = {name: _task(spec) for name, spec in config["tasks"].items()}
    kappas = tuple(float(value) for value in config["analysis"]["kappa_grid"])
    denominator_floor = float(config["analysis"]["denominator_floor"])
    early_active_fraction = float(config["analysis"]["early_active_fraction"])
    if not 0.0 < early_active_fraction < 1.0:
        raise ValueError("early_active_fraction must lie in (0, 1)")
    ledger = SeedLedger()
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        model["id"]: {task_name: [] for task_name in tasks} for model in config["models"]
    }
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for model_spec in config["models"]:
        model_id = str(model_spec["id"])
        simulator = RBergomiSimulator(
            H=float(model_spec["H"]),
            eta=float(config["model_common"]["eta"]),
            xi=float(config["model_common"]["xi"]),
            rho=float(config["model_common"]["rho"]),
            device="cpu",
        )
        for replicate in range(replicates):
            for level in levels:
                try:
                    seeds = {
                        stream: ledger.allocate(
                            SeedKey(
                                config["protocol_id"],
                                "diagnostic",
                                model_id,
                                "shared-terminal-barrier",
                                level,
                                replicate,
                                stream,
                            )
                        )
                        for stream in ("proposal", "labels")
                    }
                    torch.manual_seed(seeds["proposal"])
                    fine_steps = int(hierarchy["coarsest_steps"]) * 2**level
                    sample = simulate_coupled_rbergomi_mixture(
                        simulator,
                        controls,
                        weights,
                        spot=float(config["model_common"]["spot"]),
                        maturity=maturity,
                        fine_steps=fine_steps,
                        num_paths=paths,
                        label_generator=torch.Generator().manual_seed(seeds["labels"]),
                        engine=str(sampling["engine"]),
                    )
                    for task_name, task in tasks.items():
                        evaluation = evaluate_rbergomi_dcs_adjacent(
                            sample, task=task, rho=simulator.rho
                        )
                        diagnostics = evaluate_rbergomi_threshold_coupling(
                            evaluation.fine.log_spot_intercept,
                            evaluation.fine.log_spot_slope,
                            evaluation.coarse.log_spot_intercept,
                            evaluation.coarse.log_spot_slope,
                            fine_step_dt=sample.paths.fine.step_dt,
                            coarse_step_dt=sample.paths.coarse.step_dt,
                            task=task,
                            denominator_floor=denominator_floor,
                        )
                        records[model_id][task_name].append(
                            {
                                "model_id": model_id,
                                "H": float(model_spec["H"]),
                                "task": task_name,
                                "level": level,
                                "replicate": replicate,
                                "fine_steps": fine_steps,
                                "fine_step_dt": sample.paths.fine.step_dt,
                                "paths": paths,
                                "metrics": _cell_metrics(
                                    evaluation,
                                    diagnostics,
                                    kappas,
                                    maturity,
                                    early_active_fraction,
                                ),
                            }
                        )
                except Exception as error:
                    failures.append(
                        {
                            "model_id": model_id,
                            "level": level,
                            "replicate": replicate,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
    minimum_level = int(config["analysis"]["rate_minimum_level"])
    rates = {
        model_id: {
            task_name: _rate_summary(task_records, minimum_level)
            for task_name, task_records in task_map.items()
        }
        for model_id, task_map in records.items()
    }
    all_metrics = [
        record["metrics"]
        for task_map in records.values()
        for task_records in task_map.values()
        for record in task_records
    ]
    exact_tolerance = float(config["gates"]["pathwise_tolerance"])
    exactness_passed = bool(all_metrics) and all(
        metric["maximum_good_event_bound_violation"] <= exact_tolerance
        and metric["maximum_exact_decomposition_violation"] <= exact_tolerance
        and metric["maximum_path_reconstruction_error"] <= exact_tolerance
        and metric["maximum_coordinate_mismatch"] <= exact_tolerance
        and metric["maximum_density_reconstruction_error"] <= exact_tolerance
        and metric["maximum_likelihood_bound_violation"] <= exact_tolerance
        for metric in all_metrics
    )
    return {
        "schema": "npi.g11.v5-threshold-diagnostics.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "estimand": "fixed finest-grid event probability",
        "continuous_time_claim": False,
        "records": records,
        "rate_summaries": rates,
        "failures": failures,
        "gates": {
            "no_failures": not failures,
            "pathwise_exactness": exactness_passed,
            "rate_results_are_descriptive_only": True,
        },
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v5_threshold_diagnostics_development.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
