"""Fresh-training CEM, defensive-mixture, and crude baseline qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
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
    exact_binomial_probability_interval,
    heavy_tail_diagnostics,
    simulate_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.rbergomi_fft import simulate_rbergomi_fft
from src.physics_engine import RBergomiSimulator
from src.training import fit_rbergomi_piecewise_cem


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != (
        "npi.g11.v5-baseline-qualification.config.v1"
    ):
        raise ValueError("unsupported V5 baseline qualification config")
    if payload.get("estimand") != "finite_grid":
        raise ValueError("baseline qualification requires a finite-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _task(specification: dict[str, Any]):
    if specification["kind"] == "terminal":
        return TerminalThresholdTask(float(specification["level"]))
    if specification["kind"] == "barrier":
        return DiscreteBarrierHitTask(float(specification["barrier"]))
    raise ValueError("baseline qualification supports terminal and barrier tasks")


def _hash_control(control) -> str:
    payload = json.dumps(control, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _summary(values: torch.Tensor, work_units: float) -> dict[str, Any]:
    count = values.numel()
    variance = float(torch.var(values, unbiased=True))
    standard_error = math.sqrt(variance / count)
    tails = heavy_tail_diagnostics(values)
    return {
        "samples": count,
        "estimate": float(torch.mean(values)),
        "variance": variance,
        "standard_error": standard_error,
        "asymptotic_confidence_interval_95": [
            float(torch.mean(values)) - 1.959963984540054 * standard_error,
            float(torch.mean(values)) + 1.959963984540054 * standard_error,
        ],
        "work_units": work_units,
        "tail_diagnostics": asdict(tails),
    }


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    sampling = config["sampling"]
    training = config["training"]
    common = config["model_common"]
    clusters = 2 if smoke else int(sampling["clusters"])
    training_paths = 512 if smoke else int(training["paths_per_iteration"])
    evaluation_paths = 1024 if smoke else int(sampling["evaluation_paths"])
    maximum_iterations = 3 if smoke else int(training["maximum_iterations"])
    steps = int(config["grid"]["steps"])
    maturity = float(common["maturity"])
    natural = TimePiecewiseTwoDriverControl(
        tuple((0.0, 0.0) for _ in range(int(training["segments"]))),
        maturity=maturity,
    )
    ledger = SeedLedger()
    records: list[dict[str, Any]] = []
    for model_spec in config["models"]:
        model = {**common, **model_spec}
        simulator = RBergomiSimulator(
            H=float(model["H"]),
            eta=float(model["eta"]),
            xi=float(model["xi"]),
            rho=float(model["rho"]),
            device="cpu",
        )
        model_id = str(model_spec["id"])
        for task_name, task_spec in config["tasks"].items():
            task = _task(task_spec)
            for cluster in range(clusters):
                train_seed = ledger.allocate(
                    SeedKey(
                        config["protocol_id"],
                        "cem-training",
                        model_id,
                        task_name,
                        0,
                        cluster,
                        "proposal",
                    )
                )
                fit = fit_rbergomi_piecewise_cem(
                    simulator,
                    task,
                    spot=float(model["spot"]),
                    maturity=maturity,
                    dt=maturity / steps,
                    initial_control=tuple(
                        tuple(float(value) for value in segment)
                        for segment in training["initial_control"]
                    ),
                    num_paths=training_paths,
                    seed=train_seed,
                    max_iterations=maximum_iterations,
                    elite_quantile=float(training["elite_quantile"]),
                    smoothing=float(training["smoothing"]),
                    min_elite_paths=min(int(training["minimum_elite_paths"]), training_paths),
                    control_bound=float(training["control_bound"]),
                    target_level_repetitions=int(training["target_level_repetitions"]),
                )
                fitted = TimePiecewiseTwoDriverControl(fit.control, maturity=maturity)
                control_hash = _hash_control(fit.control)
                pure_seed = ledger.allocate(
                    SeedKey(
                        config["protocol_id"],
                        "cem-evaluation",
                        model_id,
                        task_name,
                        0,
                        cluster,
                        "proposal",
                    )
                )
                torch.manual_seed(pure_seed)
                pure_paths = simulate_rbergomi_fft(
                    simulator,
                    S0=float(model["spot"]),
                    T=maturity,
                    dt=maturity / steps,
                    num_paths=evaluation_paths,
                    control_fn=fitted,
                    dtype=torch.float64,
                )
                pure_event = task.hard_event(pure_paths.spot, pure_paths.step_dt)
                pure_values = pure_event.to(torch.float64) * torch.exp(pure_paths.log_likelihood)

                mixture_seeds = {
                    stream: ledger.allocate(
                        SeedKey(
                            config["protocol_id"],
                            "defensive-cem-evaluation",
                            model_id,
                            task_name,
                            0,
                            cluster,
                            stream,
                        )
                    )
                    for stream in ("proposal", "labels")
                }
                torch.manual_seed(mixture_seeds["proposal"])
                defensive_weight = float(config["defensive_mixture"]["natural_weight"])
                mixture = simulate_rbergomi_mixture(
                    simulator,
                    (natural, fitted),
                    torch.tensor(
                        [defensive_weight, 1.0 - defensive_weight],
                        dtype=torch.float64,
                    ),
                    spot=float(model["spot"]),
                    maturity=maturity,
                    dt=maturity / steps,
                    num_paths=evaluation_paths,
                    label_generator=torch.Generator().manual_seed(mixture_seeds["labels"]),
                    engine=str(sampling["engine"]),
                )
                defensive_event = task.hard_event(mixture.paths.spot, mixture.paths.step_dt)
                defensive_likelihood = torch.exp(mixture.mixture_log_likelihood)
                defensive_values = defensive_event.to(torch.float64) * defensive_likelihood
                maximum_defensive_bound_violation = float(
                    torch.amax(
                        torch.clamp(
                            defensive_likelihood - 1.0 / defensive_weight,
                            min=0.0,
                        )
                    )
                )

                crude_seed = ledger.allocate(
                    SeedKey(
                        config["protocol_id"],
                        "crude-evaluation",
                        model_id,
                        task_name,
                        0,
                        cluster,
                        "proposal",
                    )
                )
                torch.manual_seed(crude_seed)
                crude_paths = simulate_rbergomi_fft(
                    simulator,
                    S0=float(model["spot"]),
                    T=maturity,
                    dt=maturity / steps,
                    num_paths=evaluation_paths,
                    control_fn=natural,
                    dtype=torch.float64,
                )
                crude_event = task.hard_event(crude_paths.spot, crude_paths.step_dt)
                crude_values = crude_event.to(torch.float64)
                hits = int(torch.count_nonzero(crude_event))
                exact_interval = exact_binomial_probability_interval(
                    hits,
                    evaluation_paths,
                    confidence_level=float(sampling["confidence_level"]),
                )
                operation_scale = steps * max(1.0, math.log2(steps))
                training_work = training_paths * len(fit.history) * operation_scale
                records.append(
                    {
                        "model_id": model_id,
                        "task": task_name,
                        "cluster": cluster,
                        "training_seed": train_seed,
                        "evaluation_seed_roles_are_disjoint": True,
                        "cem": {
                            "converged": fit.converged,
                            "iterations": len(fit.history),
                            "history": [asdict(item) for item in fit.history],
                            "frozen_control": fit.control,
                            "control_sha256": control_hash,
                            "training_work_units": training_work,
                        },
                        "pure_cem_slis": {
                            **_summary(
                                pure_values,
                                evaluation_paths * operation_scale + training_work,
                            ),
                            "interval_type": "asymptotic_unbounded_likelihood",
                        },
                        "defensive_cem_mixture": {
                            **_summary(
                                defensive_values,
                                evaluation_paths * operation_scale + training_work,
                            ),
                            "natural_weight": defensive_weight,
                            "interval_type": "asymptotic; bounded interval deferred to final allocator",
                            "maximum_full_likelihood_bound_violation": maximum_defensive_bound_violation,
                        },
                        "crude_mc": {
                            **_summary(crude_values, evaluation_paths * operation_scale),
                            "hits": hits,
                            "exact_binomial_interval": asdict(exact_interval),
                            "zero_hit_censored": hits == 0,
                        },
                    }
                )
    finite = all(
        math.isfinite(record[method]["estimate"])
        and math.isfinite(record[method]["standard_error"])
        for record in records
        for method in ("pure_cem_slis", "defensive_cem_mixture", "crude_mc")
    )
    gates = {
        "all_estimators_finite": finite,
        "fresh_training_per_cluster": len({record["training_seed"] for record in records})
        == len(records),
        "training_and_evaluation_seeds_disjoint": all(
            record["evaluation_seed_roles_are_disjoint"] for record in records
        ),
        "defensive_likelihood_bounds_hold": all(
            record["defensive_cem_mixture"]["maximum_full_likelihood_bound_violation"] <= 1e-12
            for record in records
        ),
        "zero_hit_cases_use_exact_intervals": all(
            not record["crude_mc"]["zero_hit_censored"]
            or record["crude_mc"]["exact_binomial_interval"]["upper"] > 0.0
            for record in records
        ),
    }
    provenance = source_provenance()
    formal_readiness = {
        "frozen_config": bool(config.get("frozen")),
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
    }
    return {
        "schema": "npi.g11.v5-baseline-qualification.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "records": records,
        "gates": gates,
        "formal_readiness": formal_readiness,
        "baseline_qualified": all(gates.values()) and all(formal_readiness.values()),
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v5_baseline_qualification.yaml"),
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
