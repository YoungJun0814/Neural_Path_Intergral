"""Calibration-only selection of deterministic positive MGVS time directions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g9_mgvs_frozen import _control, _simulator, _task, _verified_json
from src.path_integral.gaussian_smoothing import positive_exponential_direction
from src.path_integral.rbergomi_fft import (
    simulate_coupled_rbergomi_adjacent_fft,
    simulate_rbergomi_fft,
)
from src.path_integral.rbergomi_smoothing import (
    evaluate_smoothed_adjacent_rbergomi_sample,
    evaluate_smoothed_rbergomi_sample,
)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G9 direction schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("direction calibration protocol must be frozen")
    candidates = [float(value) for value in payload["direction_family"]["decay_candidates"]]
    if 0.0 not in candidates or len(set(candidates)) != len(candidates):
        raise ValueError("direction candidates must be unique and contain the flat decay 0")
    return payload, hashlib.sha256(raw).hexdigest()


def _candidate_metrics(values: dict[float, list[torch.Tensor]]) -> dict[float, dict[str, float]]:
    result: dict[float, dict[str, float]] = {}
    for decay, chunks in values.items():
        contribution = torch.cat(chunks)
        variance = float(contribution.var(unbiased=True))
        result[decay] = {
            "estimate": float(contribution.mean()),
            "variance": variance,
            "standard_error": math.sqrt(variance / contribution.numel()),
        }
    return result


def _select(metrics: dict[float, dict[str, float]]) -> float:
    return min(
        metrics,
        key=lambda decay: (metrics[decay]["variance"], abs(decay), decay),
    )


def _evaluate_grid_target(
    regime: dict[str, Any],
    *,
    fine_steps: int,
    level: int,
    candidates: list[float],
    paths: int,
    chunk_size: int,
    seed: int,
) -> tuple[dict[float, dict[str, float]], float]:
    simulator = _simulator(regime["model"])
    task = _task(regime["task"])
    control = _control(regime)
    torch.manual_seed(seed)
    contributions: dict[float, list[torch.Tensor]] = {value: [] for value in candidates}
    maximum_error = 0.0
    completed = 0
    while completed < paths:
        current = min(chunk_size, paths - completed)
        if level == 0:
            sample = simulate_rbergomi_fft(
                simulator,
                S0=float(regime["model"]["spot"]),
                T=float(regime["model"]["maturity"]),
                dt=float(regime["model"]["maturity"]) / fine_steps,
                num_paths=current,
                control_fn=control,
            )
        else:
            sample = simulate_coupled_rbergomi_adjacent_fft(
                simulator,
                S0=float(regime["model"]["spot"]),
                T=float(regime["model"]["maturity"]),
                fine_steps=fine_steps,
                num_paths=current,
                control_fn=control,
            )
        for decay in candidates:
            direction = positive_exponential_direction(
                fine_steps,
                decay=decay,
                device="cpu",
                dtype=torch.float64,
            )
            if level == 0:
                result = evaluate_smoothed_rbergomi_sample(
                    sample,
                    task=task,
                    rho=simulator.rho,
                    direction=direction,
                    declared_deterministic_control=True,
                )
                contribution = result.level.smoothed_contribution
                path_error = result.maximum_path_reconstruction_error
            else:
                result = evaluate_smoothed_adjacent_rbergomi_sample(
                    sample,
                    task=task,
                    rho=simulator.rho,
                    direction=direction,
                    declared_deterministic_control=True,
                )
                contribution = result.smoothed_correction
                path_error = max(
                    result.maximum_fine_path_reconstruction_error,
                    result.maximum_coarse_path_reconstruction_error,
                )
            contributions[decay].append(contribution.detach().cpu())
            maximum_error = max(
                maximum_error,
                result.maximum_likelihood_reconstruction_error,
                result.maximum_residual_projection,
                path_error,
            )
        completed += current
    return _candidate_metrics(contributions), maximum_error


def _evaluate_single_finest(
    regime: dict[str, Any],
    *,
    steps: int,
    candidates: list[float],
    paths: int,
    chunk_size: int,
    seed: int,
) -> tuple[dict[float, dict[str, float]], float]:
    return _evaluate_grid_target(
        regime,
        fine_steps=steps,
        level=0,
        candidates=candidates,
        paths=paths,
        chunk_size=chunk_size,
        seed=seed,
    )


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    regimes: list[tuple[str, dict[str, Any]]] = []
    source_hashes: dict[str, str] = {}
    for source in config["calibration_sources"]:
        payload = _verified_json(source)
        source_hashes[str(source["path"])] = str(source["canonical_json_sha256"])
        regimes.extend((str(source["group"]), regime) for regime in payload["regimes"])
    if smoke:
        regimes = regimes[:1]
    levels = [int(value) for value in config["hierarchy"]["fine_steps"]]
    if smoke:
        levels = levels[:2]
    candidates = [float(value) for value in config["direction_family"]["decay_candidates"]]
    paths = 1_000 if smoke else int(config["evaluation"]["paths"])
    chunk_size = min(paths, int(config["evaluation"]["chunk_size"]))
    torch.set_num_threads(int(config["evaluation"]["thread_count"]))
    outputs: list[dict[str, Any]] = []
    for regime_index, (group, regime) in enumerate(regimes):
        selections: dict[str, Any] = {}
        maximum_error = 0.0
        for level, fine_steps in enumerate(levels):
            metrics, error = _evaluate_grid_target(
                regime,
                fine_steps=fine_steps,
                level=level,
                candidates=candidates,
                paths=paths,
                chunk_size=chunk_size,
                seed=int(config["evaluation"]["seed_base"]) + 1_000 * regime_index + level,
            )
            selected = _select(metrics)
            maximum_error = max(maximum_error, error)
            selections[f"level_{level}_{fine_steps}"] = {
                "selected_decay": selected,
                "selected_over_flat_variance": (
                    metrics[selected]["variance"] / metrics[0.0]["variance"]
                ),
                "candidates": {str(key): value for key, value in metrics.items()},
            }
        finest = levels[-1]
        single_metrics, error = _evaluate_single_finest(
            regime,
            steps=finest,
            candidates=candidates,
            paths=paths,
            chunk_size=chunk_size,
            seed=int(config["evaluation"]["seed_base"]) + 1_000 * regime_index + 900,
        )
        single_selected = _select(single_metrics)
        maximum_error = max(maximum_error, error)
        selections[f"single_{finest}"] = {
            "selected_decay": single_selected,
            "selected_over_flat_variance": (
                single_metrics[single_selected]["variance"] / single_metrics[0.0]["variance"]
            ),
            "candidates": {str(key): value for key, value in single_metrics.items()},
        }
        maximum_selected_ratio = max(
            float(value["selected_over_flat_variance"]) for value in selections.values()
        )
        gates = {
            "selection_noninferiority": maximum_selected_ratio
            <= float(config["gates"]["maximum_selected_over_flat_variance"]),
            "exactness": maximum_error <= float(config["gates"]["maximum_exactness_error"]),
        }
        outputs.append(
            {
                "regime_id": regime["regime_id"],
                "group": group,
                "selections": selections,
                "maximum_exactness_error": maximum_error,
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "smoke": smoke,
        "source_hashes": source_hashes,
        "direction_formula": config["direction_family"]["formula"],
        "regimes": outputs,
        "passed_regimes": sum(bool(value["passed"]) for value in outputs),
        "total_regimes": len(outputs),
        "passed": all(bool(value["passed"]) for value in outputs),
        "restriction": "selection used calibration seeds only; validation paths may not retune decay",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g9_direction_calibration.yaml")
    )
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
