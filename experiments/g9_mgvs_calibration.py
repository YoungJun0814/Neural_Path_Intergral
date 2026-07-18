"""Pre-validation CEM calibration for the frozen G9 multi-regime protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from src.path_integral.controllers import TimePiecewiseTwoDriverControl
from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_fft import simulate_rbergomi_fft
from src.path_integral.rbergomi_smoothing import evaluate_smoothed_rbergomi_sample
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_piecewise_cem import (
    PiecewiseValues,
    fit_rbergomi_piecewise_cem,
)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G9 calibration schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("calibration protocol must be frozen before execution")
    return payload, hashlib.sha256(raw).hexdigest()


def _task(values: dict[str, Any]) -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=float(values["hit_barrier"]),
        stress_level=float(values["stress_level"]),
        minimum_occupation=float(values["minimum_occupation"]),
        hit_scale=float(values["hit_scale"]),
        occupation_scale=float(values["occupation_scale"]),
    )


def _likelihood_z(likelihood: torch.Tensor) -> float:
    standard_error = float(likelihood.std(unbiased=True)) / math.sqrt(likelihood.numel())
    return abs(float(likelihood.mean()) - 1.0) / standard_error if standard_error > 0.0 else 0.0


def _validate_control(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    control: TimePiecewiseTwoDriverControl,
    *,
    spot: float,
    maturity: float,
    steps: int,
    paths: int,
    chunk_size: int,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    raw_values: list[torch.Tensor] = []
    smoothed_values: list[torch.Tensor] = []
    events: list[torch.Tensor] = []
    likelihoods: list[torch.Tensor] = []
    maximum_error = 0.0
    completed = 0
    while completed < paths:
        current = min(chunk_size, paths - completed)
        sample = simulate_rbergomi_fft(
            simulator,
            S0=spot,
            T=maturity,
            dt=maturity / steps,
            num_paths=current,
            control_fn=control,
        )
        smoothed = evaluate_smoothed_rbergomi_sample(
            sample,
            task=task,
            rho=simulator.rho,
            declared_deterministic_control=True,
        )
        raw_values.append(smoothed.level.raw_contribution.cpu())
        smoothed_values.append(smoothed.level.smoothed_contribution.cpu())
        events.append(smoothed.level.hard_event.cpu())
        likelihoods.append(torch.exp(sample.log_likelihood).cpu())
        maximum_error = max(
            maximum_error,
            smoothed.maximum_likelihood_reconstruction_error,
            smoothed.maximum_path_reconstruction_error,
            smoothed.maximum_residual_projection,
        )
        completed += current
    raw = torch.cat(raw_values)
    smoothed = torch.cat(smoothed_values)
    event = torch.cat(events)
    likelihood = torch.cat(likelihoods)
    difference = smoothed - raw
    paired_se = math.sqrt(float(difference.var(unbiased=True)) / paths)
    return {
        "raw_estimate": float(raw.mean()),
        "smoothed_estimate": float(smoothed.mean()),
        "smoothed_standard_error": math.sqrt(float(smoothed.var(unbiased=True)) / paths),
        "smoothed_over_raw_variance": float(smoothed.var(unbiased=True) / raw.var(unbiased=True)),
        "event_fraction": float(event.double().mean()),
        "likelihood_normalization_z": _likelihood_z(likelihood),
        "paired_mean_difference_z": (
            float(difference.mean()) / paired_se if paired_se > 0.0 else 0.0
        ),
        "maximum_exactness_error": maximum_error,
    }


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    common = config["common"]
    cem = config["cem"]
    validation = config["calibration_validation"]
    task_items = list(config["tasks"].items())
    model_items = list(config["models"].items())
    if smoke:
        task_items = task_items[:1]
        model_items = model_items[:1]
    regimes: list[dict[str, Any]] = []
    regime_index = 0
    for model_name, model in model_items:
        for task_name, task_values in task_items:
            simulator = RBergomiSimulator(
                H=float(model["H"]),
                eta=float(model["eta"]),
                xi=float(common["xi"]),
                rho=float(model["rho"]),
                device="cpu",
            )
            task = _task(task_values)
            segments = int(cem["segments"])
            initial = tuple(float(value) for value in cem["initial_segment"])
            initial_control = cast(
                PiecewiseValues,
                tuple((initial[0], initial[1]) for _ in range(segments)),
            )
            training_paths = 2_000 if smoke else int(cem["paths_per_iteration"])
            iterations = 2 if smoke else int(cem["max_iterations"])
            training_seed = int(config["seeds"]["training_base"]) + regime_index
            start = time.perf_counter()
            fit = fit_rbergomi_piecewise_cem(
                simulator,
                task,
                spot=float(common["spot"]),
                maturity=float(common["maturity"]),
                dt=float(common["maturity"]) / int(common["training_steps"]),
                initial_control=initial_control,
                num_paths=training_paths,
                seed=training_seed,
                max_iterations=iterations,
                elite_quantile=float(cem["elite_quantile"]),
                smoothing=float(cem["smoothing"]),
                min_elite_paths=32 if smoke else int(cem["min_elite_paths"]),
                control_bound=float(cem["control_bound"]),
                target_level_repetitions=(1 if smoke else int(cem["target_level_repetitions"])),
            )
            training_seconds = time.perf_counter() - start
            control = TimePiecewiseTwoDriverControl(fit.control, maturity=float(common["maturity"]))
            validation_paths = 2_000 if smoke else int(validation["paths"])
            diagnostics = _validate_control(
                simulator,
                task,
                control,
                spot=float(common["spot"]),
                maturity=float(common["maturity"]),
                steps=int(common["validation_steps"]),
                paths=validation_paths,
                chunk_size=min(validation_paths, int(validation["chunk_size"])),
                seed=int(config["seeds"]["validation_base"]) + regime_index,
            )
            gates = {
                "finite_positive_estimate": diagnostics["smoothed_estimate"] > 0.0,
                "event_support": diagnostics["event_fraction"]
                >= float(validation["minimum_event_fraction"]),
                "likelihood_normalization": diagnostics["likelihood_normalization_z"]
                <= float(validation["maximum_likelihood_normalization_z"]),
                "exactness": diagnostics["maximum_exactness_error"] <= 1e-11,
                "mean_consistency": abs(diagnostics["paired_mean_difference_z"]) <= 3.0,
            }
            regimes.append(
                {
                    "regime_id": f"{model_name}__{task_name}",
                    "model": {
                        "spot": float(common["spot"]),
                        "maturity": float(common["maturity"]),
                        "H": float(model["H"]),
                        "eta": float(model["eta"]),
                        "xi": float(common["xi"]),
                        "rho": float(model["rho"]),
                    },
                    "task": task_values,
                    "training_seed": training_seed,
                    "validation_seed": int(config["seeds"]["validation_base"]) + regime_index,
                    "control": fit.control,
                    "converged": fit.converged,
                    "training_seconds": training_seconds,
                    "training_history": [asdict(value) for value in fit.history],
                    "calibration_diagnostics": diagnostics,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )
            regime_index += 1
    passed_regimes = sum(bool(regime["passed"]) for regime in regimes)
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "smoke": smoke,
        "regimes": regimes,
        "passed_regimes": passed_regimes,
        "total_regimes": len(regimes),
        "passed": passed_regimes == len(regimes),
        "usage": "controls may be copied into a separately hashed frozen validation config",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g9_mgvs_calibration.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_regimes": result["passed_regimes"],
                "total_regimes": result["total_regimes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
