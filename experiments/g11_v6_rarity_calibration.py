"""V6 multi-H terminal/barrier rarity calibration on one fixed finite grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import torch
import yaml

from src.path_integral import (
    V6_CELL_MANIFEST_SCHEMA,
    DiscreteBarrierHitTask,
    OnlineMoments,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    V6CellManifest,
    V6RBergomiCell,
    evaluate_rbergomi_dcs_level,
    simulate_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

_SCHEMA = "npi.g11.v6-rarity-calibration.config.v1"


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 rarity-calibration config")
    required = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "model_common",
        "models",
        "grid",
        "tasks",
        "target_probabilities",
        "proposal",
        "sampling",
        "gates",
    }
    if set(payload) != required:
        raise ValueError("malformed V6 rarity-calibration config fields")
    if payload["phase"] != "development" or payload["frozen"] is not False:
        raise ValueError("rarity calibration must be an unfrozen development protocol")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("V6 calibration must declare a fixed-finest-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _controls(config: dict[str, Any], maturity: float):
    proposal = config["proposal"]
    return tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(value) for value in segment) for segment in schedule),
            maturity=maturity,
        )
        for schedule in proposal["controls"]
    )


def _weighted_threshold(score: torch.Tensor, likelihood: torch.Tensor, target: float) -> float:
    if score.ndim != 1 or likelihood.shape != score.shape:
        raise ValueError("score and likelihood must be matching vectors")
    if not torch.isfinite(score).all() or not torch.isfinite(likelihood).all():
        raise ValueError("calibration sample must be finite")
    ordered_score, order = torch.sort(score)
    cumulative = torch.cumsum(likelihood[order], dim=0) / score.numel()
    index = int(torch.searchsorted(cumulative, target, right=False))
    if index >= score.numel():
        raise ValueError(f"calibration proposal does not reach target mass {target}")
    return float(ordered_score[index])


def _draw(
    *,
    simulator: RBergomiSimulator,
    controls,
    weights: torch.Tensor,
    model: dict[str, Any],
    steps: int,
    count: int,
    proposal_seed: int,
    label_seed: int,
    engine: str,
):
    torch.manual_seed(proposal_seed)
    return simulate_rbergomi_mixture(
        simulator,
        controls,
        weights,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        dt=float(model["maturity"]) / steps,
        num_paths=count,
        dtype=torch.float64,
        label_generator=torch.Generator().manual_seed(label_seed),
        engine=engine,
    )


def _seeds(
    ledger: SeedLedger,
    protocol: str,
    role: str,
    model_id: str,
    replicate: int,
) -> tuple[int, int]:
    proposal_seed = ledger.allocate(
        SeedKey(protocol, role, model_id, "shared_tasks", 0, replicate, "proposal")
    )
    label_seed = ledger.allocate(
        SeedKey(protocol, role, model_id, "shared_tasks", 0, replicate, "labels")
    )
    return proposal_seed, label_seed


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load(config_path)
    common = config["model_common"]
    maturity = float(common["maturity"])
    controls = _controls(config, maturity)
    weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
    if weights.ndim != 1 or len(controls) != weights.numel():
        raise ValueError("proposal controls and weights must have matching lengths")
    if abs(float(weights.sum()) - 1.0) > 1e-12 or bool((weights <= 0.0).any()):
        raise ValueError("proposal weights must be positive and normalized")
    steps = int(config["grid"]["steps"])
    sampling = config["sampling"]
    calibration_paths = 2048 if smoke else int(sampling["calibration_paths"])
    validation_paths = 2048 if smoke else int(sampling["validation_paths"])
    batch_size = min(int(sampling["batch_size"]), calibration_paths, validation_paths)
    models = config["models"][:1] if smoke else config["models"]
    targets = tuple(float(value) for value in config["target_probabilities"])
    tasks = config["tasks"]
    expected_cells = len(models) * len(tasks) * len(targets)
    familywise_alpha = float(config["gates"]["familywise_alpha"])
    critical = NormalDist().inv_cdf(1.0 - familywise_alpha / (2.0 * expected_cells))
    ledger = SeedLedger()
    result_cells: list[dict[str, Any]] = []
    manifest_cells: list[V6RBergomiCell] = []

    for model_spec in models:
        model = {**common, **model_spec}
        model_id = str(model_spec["id"])
        simulator = RBergomiSimulator(
            H=float(model["H"]),
            eta=float(model["eta"]),
            xi=float(model["xi"]),
            rho=float(model["rho"]),
            device="cpu",
        )
        score_parts = {task_name: [] for task_name in tasks}
        likelihood_parts: list[torch.Tensor] = []
        for offset in range(0, calibration_paths, batch_size):
            count = min(batch_size, calibration_paths - offset)
            proposal_seed, label_seed = _seeds(
                ledger,
                str(config["protocol_id"]),
                "calibration",
                model_id,
                offset // batch_size,
            )
            sample = _draw(
                simulator=simulator,
                controls=controls,
                weights=weights,
                model=model,
                steps=steps,
                count=count,
                proposal_seed=proposal_seed,
                label_seed=label_seed,
                engine=str(sampling["engine"]),
            )
            likelihood_parts.append(torch.exp(sample.mixture_log_likelihood).detach().cpu())
            score_parts["terminal"].append(sample.paths.spot[:, -1].detach().cpu())
            score_parts["barrier"].append(torch.amin(sample.paths.spot, dim=1).detach().cpu())
        likelihood = torch.cat(likelihood_parts)
        thresholds = {
            (task_name, target): _weighted_threshold(
                torch.cat(score_parts[task_name]), likelihood, target
            )
            for task_name in tasks
            for target in targets
        }
        moments = {(task_name, target): OnlineMoments() for task_name in tasks for target in targets}
        normalization = OnlineMoments()
        for offset in range(0, validation_paths, batch_size):
            count = min(batch_size, validation_paths - offset)
            proposal_seed, label_seed = _seeds(
                ledger,
                str(config["protocol_id"]),
                "calibration-validation",
                model_id,
                offset // batch_size,
            )
            sample = _draw(
                simulator=simulator,
                controls=controls,
                weights=weights,
                model=model,
                steps=steps,
                count=count,
                proposal_seed=proposal_seed,
                label_seed=label_seed,
                engine=str(sampling["engine"]),
            )
            normalization.update(torch.exp(sample.mixture_log_likelihood))
            for task_name, task_spec in tasks.items():
                for target in targets:
                    threshold = thresholds[(task_name, target)]
                    task = (
                        TerminalThresholdTask(threshold)
                        if task_spec["kind"] == "terminal"
                        else DiscreteBarrierHitTask(threshold)
                    )
                    values = evaluate_rbergomi_dcs_level(
                        sample, task=task, rho=simulator.rho
                    ).marginalized_contribution
                    moments[(task_name, target)].update(values)

        normalization_se = math.sqrt(normalization.variance / normalization.count)
        normalization_z = (
            (normalization.mean - 1.0) / normalization_se
            if normalization_se > 0.0
            else math.inf
        )
        for task_name in tasks:
            for target in targets:
                summary = moments[(task_name, target)]
                standard_error = math.sqrt(summary.variance / summary.count)
                interval = (
                    max(0.0, summary.mean - critical * standard_error),
                    min(1.0, summary.mean + critical * standard_error),
                )
                lower = float(config["gates"]["probability_band_lower_factor"]) * target
                upper = float(config["gates"]["probability_band_upper_factor"]) * target
                relative_se = standard_error / summary.mean if summary.mean > 0.0 else math.inf
                threshold = thresholds[(task_name, target)]
                cell_id = f"{model_id}-{task_name}-p{target:.0e}".replace("+", "")
                result_cells.append(
                    {
                        "cell_id": cell_id,
                        "model_id": model_id,
                        "task": task_name,
                        "target_probability": target,
                        "threshold": threshold,
                        "validation_estimate": summary.mean,
                        "validation_variance": summary.variance,
                        "validation_standard_error": standard_error,
                        "simultaneous_asymptotic_interval": list(interval),
                        "relative_standard_error": relative_se,
                        "probability_band": [lower, upper],
                        "point_band_passed": lower <= summary.mean <= upper,
                        "interval_band_passed": lower <= interval[0] and interval[1] <= upper,
                        "precision_passed": relative_se
                        <= float(config["gates"]["maximum_relative_standard_error"]),
                        "normalization_z": normalization_z,
                    }
                )
                manifest_cells.append(
                    V6RBergomiCell(
                        cell_id=cell_id,
                        hurst=float(model["H"]),
                        eta=float(model["eta"]),
                        xi=float(model["xi"]),
                        rho=float(model["rho"]),
                        spot=float(model["spot"]),
                        maturity=maturity,
                        finest_steps=steps,
                        task=(
                            "terminal_left_tail"
                            if task_name == "terminal"
                            else "discrete_lower_barrier"
                        ),
                        event_threshold=threshold,
                        nominal_probability=target,
                        probability_band=(lower, upper),
                    )
                )

    provenance = source_provenance()
    manifest = V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol=str(config["protocol_id"]),
        phase="development",
        frozen=False,
        source_commit=(
            str(provenance["source_commit"])
            if isinstance(provenance["source_commit"], str)
            and len(str(provenance["source_commit"])) == 40
            else "uncommitted"
        ),
        dirty_tree=bool(provenance["dirty_worktree"]),
        config_sha256=config_hash,
        smoke=smoke,
        cells=tuple(manifest_cells),
    )
    gates = {
        "complete_matrix": len(result_cells) == expected_cells,
        "all_point_probability_bands": all(cell["point_band_passed"] for cell in result_cells),
        "all_simultaneous_interval_bands": all(
            cell["interval_band_passed"] for cell in result_cells
        ),
        "all_relative_standard_errors": all(cell["precision_passed"] for cell in result_cells),
        "all_likelihood_normalizations": all(
            abs(float(cell["normalization_z"]))
            <= float(config["gates"]["maximum_normalization_z"])
            for cell in result_cells
        ),
        "calibration_validation_seed_roles_disjoint": True,
    }
    return {
        "schema": "npi.g11.v6-rarity-calibration.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "estimand": "fixed finest-grid probability",
        "continuous_time_claim": False,
        "cells": result_cells,
        "candidate_manifest": manifest.to_dict(),
        "candidate_manifest_sha256": manifest.sha256,
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "gates": gates,
        "passed": all(gates.values()),
        "formal_readiness": False,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/rarity_calibration_development.yaml"),
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
