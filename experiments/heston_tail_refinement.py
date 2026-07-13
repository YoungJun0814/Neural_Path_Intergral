"""Time-step refinement of likelihood-weighted Heston left-tail estimates."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.g2_heston_benchmark import (
    _reference_params,
    _sample_is,
    _simulator,
)
from src.evaluation.heston_reference import heston_left_tail_quantile, heston_terminal_cdf
from src.evaluation.protocol import load_frozen_protocol
from src.evaluation.statistics import repeated_estimate_report
from src.training.cem import HestonTerminalLossSampler, fit_constant_control_cem


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0.0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_refinement(
    protocol_path: Path,
    *,
    probability: float,
    time_steps: tuple[float, ...],
    smoke: bool,
) -> dict[str, Any]:
    protocol = load_frozen_protocol(protocol_path)
    base_model = protocol.payload["model"]
    reference_params = _reference_params(base_model)
    threshold = heston_left_tail_quantile(
        probability,
        spot=float(base_model["spot"]),
        maturity=float(base_model["maturity"]),
        params=reference_params,
    )
    truth = heston_terminal_cdf(
        terminal_spot=threshold,
        spot=float(base_model["spot"]),
        maturity=float(base_model["maturity"]),
        params=reference_params,
    )
    cem_config = protocol.payload["cem"]
    evaluation_seeds = protocol.seeds.validation[:3] if smoke else protocol.seeds.validation
    train_paths = 5_000 if smoke else int(cem_config["paths_per_iteration"])
    evaluation_paths = 20_000 if smoke else int(protocol.payload["evaluation"]["paths_per_seed"])
    max_iterations = 5 if smoke else int(cem_config["max_iterations"])

    rows: list[dict[str, Any]] = []
    for dt in time_steps:
        if dt <= 0.0:
            raise ValueError("all time steps must be positive")
        model = dict(base_model)
        model["dt"] = float(dt)
        simulator = _simulator(model)
        sampler = HestonTerminalLossSampler(
            simulator,
            spot=float(model["spot"]),
            variance=float(model["variance"]),
            maturity=float(model["maturity"]),
            dt=float(dt),
        )
        torch.manual_seed(protocol.seeds.train[0])
        training_start = time.perf_counter()
        fit = fit_constant_control_cem(
            sampler,
            initial_control=float(cem_config["initial_control"]),
            target_score=-threshold,
            num_paths=train_paths,
            max_iterations=max_iterations,
            elite_quantile=float(cem_config["elite_quantile"]),
            smoothing=float(cem_config["smoothing"]),
        )
        training_seconds = time.perf_counter() - training_start
        runs = [
            _sample_is(
                simulator,
                model,
                threshold=threshold,
                control=fit.control,
                num_paths=evaluation_paths,
                seed=seed,
            )
            for seed in evaluation_seeds
        ]
        report = repeated_estimate_report(
            np.array([run["estimate"] for run in runs]),
            np.array([run["standard_error"] for run in runs]),
            truth=truth,
        )
        rows.append(
            {
                "dt_requested": dt,
                "steps": math.ceil(float(model["maturity"]) / dt),
                "dt_effective": float(model["maturity"]) / math.ceil(float(model["maturity"]) / dt),
                "control": fit.control,
                "cem_converged": fit.converged,
                "training_seconds": training_seconds,
                "report": asdict(report),
                "runs": runs,
            }
        )
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "mode": "smoke_non_publication" if smoke else "full_refinement",
        "target_probability": probability,
        "reference_probability": truth,
        "threshold": threshold,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "g2_heston_benchmark.yaml",
    )
    parser.add_argument("--probability", type=float, default=1e-4)
    parser.add_argument(
        "--time-steps",
        type=float,
        nargs="+",
        default=(1 / 64, 1 / 128, 1 / 256, 1 / 512),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_refinement(
        args.protocol,
        probability=args.probability,
        time_steps=tuple(args.time_steps),
        smoke=args.smoke,
    )
    rendered = json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
