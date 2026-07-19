"""Disjoint-data finite-grid rarity calibration for G11 confirmatory tasks."""

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
    DownsideExcursionTask,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    evaluate_rbergomi_dcs_level,
    simulate_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected a rarity-calibration schema-version-1 config")
    if config.get("estimand") != "finite_grid" or config.get("frozen") is not False:
        raise ValueError("rarity calibration must be unfrozen and finite-grid explicit")
    return config, hashlib.sha256(raw).hexdigest()


def _threshold_for_target(
    score: torch.Tensor, likelihood: torch.Tensor, target: float
) -> float:
    finite = torch.isfinite(score)
    ordered_score, order = torch.sort(score[finite])
    ordered_likelihood = likelihood[finite][order]
    cumulative = torch.cumsum(ordered_likelihood, dim=0) / score.numel()
    index = int(torch.searchsorted(cumulative, target, right=False))
    if index >= ordered_score.numel():
        raise ValueError(f"proposal sample does not reach target mass {target}")
    return float(ordered_score[index])


def _task(task_name: str, specification: dict[str, Any], threshold: float):
    if specification["kind"] == "terminal":
        return TerminalThresholdTask(threshold)
    if specification["kind"] == "barrier":
        return DiscreteBarrierHitTask(threshold)
    if specification["kind"] == "hit_plus_occupation":
        return DownsideExcursionTask(
            hit_barrier=threshold,
            stress_level=float(specification["stress_level"]),
            minimum_occupation=float(specification["minimum_occupation"]),
            hit_scale=float(specification["hit_scale"]),
            occupation_scale=float(specification["occupation_scale"]),
        )
    raise ValueError(f"unsupported task {task_name}")


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load(config_path)
    model = config["model"]
    sampling = config["sampling"]
    proposal = config["proposal"]
    calibration_paths = 8192 if smoke else int(sampling["calibration_paths"])
    validation_paths = 8192 if smoke else int(sampling["validation_paths"])
    batch_size = int(sampling["batch_size"])
    targets = [float(value) for value in config["target_probabilities"]]
    if smoke:
        targets = targets[:2]
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    maturity = float(model["maturity"])
    controls = tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(value) for value in segment) for segment in schedule),
            maturity=maturity,
        )
        for schedule in proposal["controls"]
    )
    weights = torch.tensor(proposal["weights"], dtype=torch.float64)
    steps = int(config["finest_steps"])
    ledger = SeedLedger()

    scores: dict[str, list[torch.Tensor]] = {
        name: [] for name in config["tasks"]
    }
    likelihoods: list[torch.Tensor] = []
    normalization_values: list[torch.Tensor] = []
    for offset in range(0, calibration_paths, batch_size):
        count = min(batch_size, calibration_paths - offset)
        replicate = offset // batch_size
        proposal_seed = ledger.allocate(
            SeedKey(
                config["protocol_id"],
                "calibration",
                "primary",
                "shared_tasks",
                0,
                replicate,
                "proposal",
            )
        )
        label_seed = ledger.allocate(
            SeedKey(
                config["protocol_id"],
                "calibration",
                "primary",
                "shared_tasks",
                0,
                replicate,
                "labels",
            )
        )
        torch.manual_seed(proposal_seed)
        sample = simulate_rbergomi_mixture(
            simulator,
            controls,
            weights,
            spot=float(model["spot"]),
            maturity=maturity,
            dt=maturity / steps,
            num_paths=count,
            dtype=torch.float64,
            label_generator=torch.Generator().manual_seed(label_seed),
            engine=str(sampling["engine"]),
        )
        spot = sample.paths.spot
        likelihood = torch.exp(sample.mixture_log_likelihood).detach().cpu()
        likelihoods.append(likelihood)
        normalization_values.append(likelihood)
        scores["terminal"].append(spot[:, -1].detach().cpu())
        scores["barrier"].append(torch.amin(spot, dim=1).detach().cpu())
        excursion_spec = config["tasks"]["excursion"]
        occupation = torch.sum(
            (spot[:, 1:] <= float(excursion_spec["stress_level"])).to(
                torch.float64
            ),
            dim=1,
        ) * (maturity / steps)
        eligible = occupation + 1e-15 >= float(
            excursion_spec["minimum_occupation"]
        )
        excursion_score = torch.where(
            eligible,
            torch.amin(spot, dim=1),
            torch.full_like(occupation, math.inf),
        )
        scores["excursion"].append(excursion_score.detach().cpu())
    joined_likelihood = torch.cat(likelihoods)
    calibrated: dict[str, dict[str, float]] = {}
    for task_name, parts in scores.items():
        joined_score = torch.cat(parts)
        calibrated[task_name] = {
            f"{target:.0e}": _threshold_for_target(
                joined_score, joined_likelihood, target
            )
            for target in targets
        }

    accumulators = {
        (task_name, target): {
            "count": 0,
            "sum": 0.0,
            "sum_square": 0.0,
            "barrier_sum": 0.0,
            "occupation_excluded_sum": 0.0,
        }
        for task_name in config["tasks"]
        for target in targets
    }
    validation_normalization: list[torch.Tensor] = []
    for offset in range(0, validation_paths, batch_size):
        count = min(batch_size, validation_paths - offset)
        replicate = offset // batch_size
        proposal_seed = ledger.allocate(
            SeedKey(
                config["protocol_id"],
                "validation",
                "primary",
                "shared_tasks",
                0,
                replicate,
                "proposal",
            )
        )
        label_seed = ledger.allocate(
            SeedKey(
                config["protocol_id"],
                "validation",
                "primary",
                "shared_tasks",
                0,
                replicate,
                "labels",
            )
        )
        torch.manual_seed(proposal_seed)
        sample = simulate_rbergomi_mixture(
            simulator,
            controls,
            weights,
            spot=float(model["spot"]),
            maturity=maturity,
            dt=maturity / steps,
            num_paths=count,
            dtype=torch.float64,
            label_generator=torch.Generator().manual_seed(label_seed),
            engine=str(sampling["engine"]),
        )
        validation_normalization.append(
            torch.exp(sample.mixture_log_likelihood).detach().cpu()
        )
        for task_name, specification in config["tasks"].items():
            for target in targets:
                threshold = calibrated[task_name][f"{target:.0e}"]
                task = _task(task_name, specification, threshold)
                values = evaluate_rbergomi_dcs_level(
                    sample, task=task, rho=simulator.rho
                ).marginalized_contribution
                accumulator = accumulators[(task_name, target)]
                accumulator["count"] += values.numel()
                accumulator["sum"] += float(torch.sum(values))
                accumulator["sum_square"] += float(torch.sum(values**2))
                if task_name == "excursion":
                    spot = sample.paths.spot
                    barrier_event = DiscreteBarrierHitTask(threshold).hard_event(
                        spot, sample.paths.step_dt
                    )
                    excursion_event = task.hard_event(spot, sample.paths.step_dt)
                    full_likelihood = torch.exp(sample.mixture_log_likelihood)
                    accumulator["barrier_sum"] += float(
                        torch.sum(barrier_event.to(torch.float64) * full_likelihood)
                    )
                    accumulator["occupation_excluded_sum"] += float(
                        torch.sum(
                            (barrier_event & ~excursion_event).to(torch.float64)
                            * full_likelihood
                        )
                    )

    cells: list[dict[str, Any]] = []
    lower_factor = float(config["gates"]["probability_band_lower_factor"])
    upper_factor = float(config["gates"]["probability_band_upper_factor"])
    maximum_relative_se = float(
        config["gates"]["maximum_relative_standard_error"]
    )
    for (task_name, target), accumulator in accumulators.items():
        count = accumulator["count"]
        mean = accumulator["sum"] / count
        variance = (
            accumulator["sum_square"] - count * mean * mean
        ) / (count - 1)
        standard_error = math.sqrt(max(0.0, variance) / count)
        relative_se = standard_error / mean if mean > 0.0 else math.inf
        exclusion_fraction = (
            accumulator["occupation_excluded_sum"] / accumulator["barrier_sum"]
            if task_name == "excursion" and accumulator["barrier_sum"] > 0.0
            else None
        )
        cells.append(
            {
                "task": task_name,
                "target_probability": target,
                "calibrated_threshold": calibrated[task_name][f"{target:.0e}"],
                "validation_estimate": mean,
                "validation_standard_error": standard_error,
                "relative_standard_error": relative_se,
                "probability_band_passed": lower_factor * target
                <= mean
                <= upper_factor * target,
                "precision_passed": relative_se <= maximum_relative_se,
                "occupation_exclusion_fraction": exclusion_fraction,
            }
        )
    normalization = torch.cat(validation_normalization)
    normalization_se = float(torch.std(normalization, unbiased=True)) / math.sqrt(
        normalization.numel()
    )
    normalization_z = (float(torch.mean(normalization)) - 1.0) / normalization_se
    gates = {
        "all_probability_bands": all(
            cell["probability_band_passed"] for cell in cells
        ),
        "all_relative_standard_errors": all(
            cell["precision_passed"] for cell in cells
        ),
        "likelihood_normalization": abs(normalization_z)
        <= float(config["gates"]["maximum_normalization_z"]),
        "calibration_validation_seed_roles_disjoint": True,
        "excursion_is_non_degenerate": all(
            cell["occupation_exclusion_fraction"]
            >= float(config["gates"]["minimum_excursion_exclusion_fraction"])
            for cell in cells
            if cell["task"] == "excursion"
        ),
    }
    elapsed = time.perf_counter() - started
    return {
        "schema": "npi.g11.rarity-calibration.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "estimand": "fixed 128-step finite-grid probability",
        "continuous_time_claim": False,
        "calibration_paths": calibration_paths,
        "validation_paths": validation_paths,
        "cells": cells,
        "normalization_z": normalization_z,
        "seed_ledger_sha256": ledger.sha256,
        "seed_count": len(ledger),
        "gates": gates,
        "passed": all(gates.values()),
        "work_ledger": {
            "warmup_seconds": 0.0,
            "compilation_seconds": 0.0,
            "calibration_and_validation_seconds": elapsed,
            "audit_seconds": 0.0,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g11_rarity_calibration.yaml")
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
    print(json.dumps({"passed": result["passed"], **result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
