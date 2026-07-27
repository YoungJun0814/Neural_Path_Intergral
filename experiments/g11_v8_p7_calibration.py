"""V8 P7 disjoint-data finite-grid threshold calibration.

This creates a *candidate* threshold manifest only.  A calibrated threshold becomes
part of the later finite-grid estimand after it is hash-bound; validation samples are
independent of calibration samples and never enter a final estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, cast

import torch
import yaml

from src.path_integral import (
    DiscreteBarrierHitTask,
    OnlineMoments,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    evaluate_rbergomi_dcs_level,
    simulate_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-p7-development-calibration.v1"
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "p5_reference_matrix_design_sha256",
    "p6_statistical_design_sha256",
    "seed_namespace",
    "estimand",
    "model",
    "grid",
    "tasks",
    "nominal_probabilities",
    "proposal",
    "sampling",
    "gates",
    "decision",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected P7 calibration schema")
    if set(config) != ROOT_KEYS:
        raise ValueError("malformed P7 calibration root fields")
    if config.get("phase") != "p7_development" or config.get("outcome_data_used") is not False:
        raise ValueError("P7 calibration must be outcome-blind development")
    if config.get("estimand") != "fixed_finest_grid_probability":
        raise ValueError("P7 calibration must declare a fixed finite-grid estimand")
    p5 = ROOT / "configs/g11_v8/p5_reference_matrix_design_v1.yaml"
    p6 = ROOT / "configs/g11_v8/p6_statistical_design_v1.yaml"
    if config.get("p5_reference_matrix_design_sha256") != _sha(p5):
        raise ValueError("P7 calibration is not bound to the P5 matrix")
    if config.get("p6_statistical_design_sha256") != _sha(p6):
        raise ValueError("P7 calibration is not bound to P6 statistics")
    model = config.get("model")
    tasks = config.get("tasks")
    probabilities = config.get("nominal_probabilities")
    if not isinstance(model, dict) or model.get("hurst_values") != [0.05, 0.12, 0.20]:
        raise ValueError("P7 calibration H matrix must match P5")
    if not isinstance(tasks, dict) or list(tasks) != [
        "terminal_left_tail",
        "discrete_lower_barrier",
    ]:
        raise ValueError("P7 calibration task matrix must match P5")
    if probabilities != [0.01, 0.001, 0.0001, 0.00001]:
        raise ValueError("P7 calibration probability matrix must match P5")
    if config.get("seed_namespace") != "v8-p7-development":
        raise ValueError("P7 calibration seed namespace must match P6")
    return config, hashlib.sha256(raw).hexdigest()


def weighted_threshold(score: torch.Tensor, likelihood: torch.Tensor, target: float) -> float:
    """Return the weighted empirical lower-tail quantile on independent calibration data."""

    if score.ndim != 1 or likelihood.shape != score.shape or score.numel() < 1:
        raise ValueError("score and likelihood must be nonempty matching vectors")
    if not torch.isfinite(score).all() or not torch.isfinite(likelihood).all():
        raise ValueError("calibration score and likelihood must be finite")
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target probability must lie in (0, 1)")
    if bool((likelihood < 0.0).any()):
        raise ValueError("calibration likelihood must be nonnegative")
    ordered_score, order = torch.sort(score)
    cumulative = torch.cumsum(likelihood[order], dim=0) / score.numel()
    index = int(torch.searchsorted(cumulative, target, right=False))
    if index >= ordered_score.numel():
        raise ValueError("calibration proposal does not reach the requested target mass")
    threshold = float(ordered_score[index])
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("calibration produced a nonpositive or nonfinite threshold")
    return threshold


def _controls(config: dict[str, Any], task: str) -> tuple[TimePiecewiseTwoDriverControl, ...]:
    maturity = float(config["model"]["maturity"])
    schedules = cast(list[list[list[float]]], config["proposal"]["task_controls"][task])
    return tuple(
        TimePiecewiseTwoDriverControl(
            tuple((float(segment[0]), float(segment[1])) for segment in schedule),
            maturity=maturity,
        )
        for schedule in schedules
    )


def _draw(
    *,
    simulator: RBergomiSimulator,
    controls: tuple[TimePiecewiseTwoDriverControl, ...],
    weights: torch.Tensor,
    model: dict[str, Any],
    steps: int,
    count: int,
    proposal_seed: int,
    label_seed: int,
    engine: Literal["fft", "reference"],
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


def _require_finite_path_sample(sample: Any, *, stage: str, hurst: float, task: str, offset: int) -> None:
    spot = sample.paths.spot
    variance = sample.paths.variance
    valid_spot = torch.isfinite(spot) & (spot > 0.0)
    valid_variance = torch.isfinite(variance) & (variance > 0.0)
    if bool(valid_spot.all() and valid_variance.all()):
        return
    invalid_spot = int((~valid_spot).sum())
    invalid_variance = int((~valid_variance).sum())
    raise FloatingPointError(
        "nonfinite-or-nonpositive rBergomi path "
        f"stage={stage} H={hurst:.2f} task={task} offset={offset} "
        f"spot={invalid_spot} variance={invalid_variance}"
    )


def _seeds(
    ledger: SeedLedger, *, protocol_id: str, role: str, hurst: float, replicate: int
) -> tuple[int, int]:
    model_id = f"h{hurst:.2f}"
    proposal_seed = ledger.allocate(
        SeedKey(protocol_id, role, model_id, "shared_tasks", 0, replicate, "proposal")
    )
    label_seed = ledger.allocate(
        SeedKey(protocol_id, role, model_id, "shared_tasks", 0, replicate, "labels")
    )
    return proposal_seed, label_seed


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = load_config(config_path)
    sampling = config["sampling"]
    calibration_paths = int(
        sampling["smoke_calibration_paths"] if smoke else sampling["calibration_paths"]
    )
    validation_paths = int(
        sampling["smoke_validation_paths"] if smoke else sampling["validation_paths"]
    )
    batch_size = min(int(sampling["batch_size"]), calibration_paths, validation_paths)
    hursts = config["model"]["hurst_values"][: int(sampling["smoke_hurst_count"])] if smoke else config["model"]["hurst_values"]
    probabilities = (
        config["nominal_probabilities"][: int(sampling["smoke_probability_count"])]
        if smoke
        else config["nominal_probabilities"]
    )
    tasks = config["tasks"]
    weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
    if bool((weights <= 0.0).any()) or not math.isclose(float(weights.sum()), 1.0):
        raise ValueError("P7 proposal weights must be positive and normalized")
    if any(len(_controls(config, task)) != weights.numel() for task in tasks):
        raise ValueError("P7 proposal controls and mixture weights disagree")
    steps = int(config["grid"]["steps"])
    expected_cells = len(hursts) * len(tasks) * len(probabilities)
    critical = NormalDist().inv_cdf(
        1.0 - float(config["gates"]["familywise_alpha"]) / (2.0 * expected_cells)
    )
    ledger = SeedLedger()
    cells: list[dict[str, Any]] = []
    manifest_cells: list[dict[str, Any]] = []
    engine = cast(Literal["fft", "reference"], sampling["engine"])
    if engine not in ("fft", "reference"):
        raise ValueError("P7 calibration engine must be 'fft' or 'reference'")

    for hurst_value in hursts:
        model = {key: value for key, value in config["model"].items() if key != "hurst_values"}
        model["H"] = float(hurst_value)
        simulator = RBergomiSimulator(
            H=float(model["H"]),
            eta=float(model["eta"]),
            xi=float(model["xi"]),
            rho=float(model["rho"]),
            device="cpu",
        )
        calibration_scores: dict[str, list[torch.Tensor]] = {task: [] for task in tasks}
        calibration_likelihoods: dict[str, list[torch.Tensor]] = {
            task: [] for task in tasks
        }
        for offset in range(0, calibration_paths, batch_size):
            count = min(batch_size, calibration_paths - offset)
            for task in tasks:
                proposal_seed, label_seed = _seeds(
                    ledger,
                    protocol_id=str(config["protocol_id"]),
                    role=f"p7-calibration-{task}",
                    hurst=float(hurst_value),
                    replicate=offset // batch_size,
                )
                sample = _draw(
                    simulator=simulator,
                    controls=_controls(config, task),
                    weights=weights,
                    model=model,
                    steps=steps,
                    count=count,
                    proposal_seed=proposal_seed,
                    label_seed=label_seed,
                    engine=engine,
                )
                _require_finite_path_sample(
                    sample,
                    stage="calibration",
                    hurst=float(hurst_value),
                    task=task,
                    offset=offset,
                )
                if task == "terminal_left_tail":
                    calibration_scores[task].append(sample.paths.spot[:, -1].detach().cpu())
                else:
                    calibration_scores[task].append(torch.amin(sample.paths.spot, dim=1).detach().cpu())
                calibration_likelihoods[task].append(
                    torch.exp(sample.mixture_log_likelihood).detach().cpu()
                )
        thresholds = {
            (task, float(probability)): weighted_threshold(
                torch.cat(calibration_scores[task]),
                torch.cat(calibration_likelihoods[task]),
                float(probability),
            )
            for task in tasks
            for probability in probabilities
        }
        moments = {(task, float(probability)): OnlineMoments() for task in tasks for probability in probabilities}
        normalizations = {task: OnlineMoments() for task in tasks}
        for offset in range(0, validation_paths, batch_size):
            count = min(batch_size, validation_paths - offset)
            for task_name, specification in tasks.items():
                proposal_seed, label_seed = _seeds(
                    ledger,
                    protocol_id=str(config["protocol_id"]),
                    role=f"p7-validation-{task_name}",
                    hurst=float(hurst_value),
                    replicate=offset // batch_size,
                )
                sample = _draw(
                    simulator=simulator,
                    controls=_controls(config, task_name),
                    weights=weights,
                    model=model,
                    steps=steps,
                    count=count,
                    proposal_seed=proposal_seed,
                    label_seed=label_seed,
                    engine=engine,
                )
                _require_finite_path_sample(
                    sample,
                    stage="validation",
                    hurst=float(hurst_value),
                    task=task_name,
                    offset=offset,
                )
                normalizations[task_name].update(torch.exp(sample.mixture_log_likelihood))
                for probability in probabilities:
                    threshold = thresholds[(task_name, float(probability))]
                    task = (
                        TerminalThresholdTask(threshold)
                        if specification["kind"] == "terminal"
                        else DiscreteBarrierHitTask(threshold)
                    )
                    moments[(task_name, float(probability))].update(
                        evaluate_rbergomi_dcs_level(
                            sample, task=task, rho=simulator.rho
                        ).marginalized_contribution
                    )
                    _require_finite_path_sample(
                        sample,
                        stage="validation-after-dcs",
                        hurst=float(hurst_value),
                        task=task_name,
                        offset=offset,
                    )
        normalization_z: dict[str, float] = {}
        for task_name, normalization in normalizations.items():
            normalization_se = math.sqrt(normalization.variance / normalization.count)
            normalization_z[task_name] = (
                (normalization.mean - 1.0) / normalization_se
                if normalization_se > 0.0
                else (0.0 if normalization.mean == 1.0 else math.inf)
            )
        for task_name in tasks:
            for probability in probabilities:
                moment = moments[(task_name, float(probability))]
                standard_error = math.sqrt(moment.variance / moment.count)
                interval = (
                    max(0.0, moment.mean - critical * standard_error),
                    min(1.0, moment.mean + critical * standard_error),
                )
                lower = float(config["gates"]["probability_band_lower_factor"]) * float(probability)
                upper = float(config["gates"]["probability_band_upper_factor"]) * float(probability)
                relative_se = standard_error / moment.mean if moment.mean > 0.0 else math.inf
                threshold = thresholds[(task_name, float(probability))]
                cell_id = f"h{float(hurst_value):.2f}-{task_name}-p{float(probability):.0e}".replace("+", "")
                cells.append(
                    {
                        "cell_id": cell_id,
                        "hurst": float(hurst_value),
                        "task": task_name,
                        "nominal_probability": float(probability),
                        "calibrated_threshold": threshold,
                        "validation_estimate": moment.mean,
                        "validation_standard_error": standard_error,
                        "simultaneous_asymptotic_interval": list(interval),
                        "relative_standard_error": relative_se,
                        "point_band_passed": lower <= moment.mean <= upper,
                        "interval_band_passed": lower <= interval[0] and interval[1] <= upper,
                        "precision_passed": relative_se <= float(config["gates"]["maximum_relative_standard_error"]),
                        "normalization_z": normalization_z[task_name],
                    }
                )
                manifest_cells.append(
                    {
                        "cell_id": cell_id,
                        "hurst": float(hurst_value),
                        "eta": float(model["eta"]),
                        "xi": float(model["xi"]),
                        "rho": float(model["rho"]),
                        "spot": float(model["spot"]),
                        "maturity": float(model["maturity"]),
                        "finest_steps": steps,
                        "task": task_name,
                        "event_threshold": threshold,
                        "nominal_probability": float(probability),
                        "probability_band": [lower, upper],
                    }
                )
    provenance = source_provenance()
    manifest = {
        "schema": "npi.g11.v8-p7-candidate-threshold-manifest.v1",
        "protocol_id": str(config["protocol_id"]),
        "phase": "development",
        "frozen": False,
        "source_commit": (
            str(provenance["source_commit"])
            if isinstance(provenance["source_commit"], str)
            and len(str(provenance["source_commit"])) == 40
            else "uncommitted"
        ),
        "dirty_tree": bool(provenance["dirty_worktree"]),
        "config_sha256": config_hash,
        "smoke": smoke,
        "cells": manifest_cells,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    gates = {
        "complete_requested_matrix": len(cells) == expected_cells,
        "all_thresholds_positive": all(float(cell["calibrated_threshold"]) > 0.0 for cell in cells),
        "all_point_probability_bands": all(bool(cell["point_band_passed"]) for cell in cells),
        "all_simultaneous_interval_bands": all(bool(cell["interval_band_passed"]) for cell in cells),
        "all_relative_standard_errors": all(bool(cell["precision_passed"]) for cell in cells),
        "likelihood_normalization": all(
            abs(float(cell["normalization_z"])) <= float(config["gates"]["maximum_normalization_z"])
            for cell in cells
        ),
        "calibration_validation_seed_roles_disjoint": all(
            not record.key.role.startswith("p7-calibration")
            or all(
                other.key.role != record.key.role.replace("p7-calibration", "p7-validation")
                or other.seed != record.seed
                for other in ledger.records
            )
            for record in ledger.records
        ),
    }
    return {
        "schema": "npi.g11.v8-p7-development-calibration.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "estimand": "fixed 128-step finite-grid probability conditional on the candidate threshold manifest",
        "continuous_time_claim": False,
        "cells": cells,
        "candidate_manifest": manifest,
        "candidate_manifest_sha256": manifest_sha256,
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "gates": gates,
        "passed": all(gates.values()),
        "thresholds_hash_bound": False,
        "reference_execution_complete": False,
        "performance_claim_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing calibration: {args.output}")
    try:
        result = run(args.config, smoke=args.smoke)
    except (FloatingPointError, ValueError) as error:
        config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
        result = {
            "schema": "npi.g11.v8-p7-development-calibration-failure.v1",
            "config_sha256": config_hash,
            "smoke": args.smoke,
            "passed": False,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "performance_claim_authorized": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], **result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
