"""Development runner for the Plan-v3 G1 two-driver Heston gate.

The runner uses only development roots declared in its config.  It performs
oracle distillation, soft PI, feedback PICE, hard J2 refinement, and a
time-step validation.  It does not touch the sealed v3 evaluation seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.evaluation.heston_reference import (
    HestonReferenceParams,
    heston_left_tail_quantile,
    heston_terminal_cdf,
)
from src.path_integral.heston_oracle import HestonOracleNumerics
from src.physics_engine import MarketSimulator
from src.training.cem import HestonTerminalLossSampler, fit_constant_control_cem
from src.training.heston_feedback import (
    HestonOracleDataset,
    TwoDriverHestonControl,
    build_heston_oracle_dataset,
    fit_heston_oracle_distillation,
    oracle_alignment,
    save_two_driver_control_checkpoint,
    train_feedback_pice_stage,
    train_hard_j2_stage,
    train_soft_pi_stage,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G1 schema-version-1 config")
    evaluation_roots = set(range(11101, 11121))
    declared: set[int] = set()
    for value in payload["seeds"].values():
        if isinstance(value, list):
            declared.update(int(item) for item in value)
        else:
            declared.add(int(value))
    if declared & evaluation_roots:
        raise ValueError("G1 development config must not access sealed v3 evaluation roots")
    return payload


def _reference_params(model: dict[str, Any]) -> HestonReferenceParams:
    return HestonReferenceParams(
        v0=float(model["variance"]),
        kappa=float(model["kappa"]),
        theta=float(model["theta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        r=float(model["rate"]),
        q=float(model["dividend_yield"]),
    )


def _simulator(model: dict[str, Any]) -> MarketSimulator:
    return MarketSimulator(
        mu=float(model["rate"]) - float(model["dividend_yield"]),
        kappa=float(model["kappa"]),
        theta=float(model["theta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )


def _oracle_grid(
    *,
    config: dict[str, Any],
    model: dict[str, Any],
    barrier: float,
    validation: bool,
    smoke: bool,
):
    oracle = config["oracle"]
    prefix = "validation" if validation else "train"
    time_fractions = list(oracle[f"{prefix}_time_fractions"])
    spot_multipliers = list(oracle[f"{prefix}_spot_multipliers"])
    variance_multipliers = list(oracle[f"{prefix}_variance_multipliers"])
    if smoke:
        time_fractions = time_fractions[:2]
        spot_multipliers = spot_multipliers[:2]
        variance_multipliers = variance_multipliers[:1]
    maturity = float(model["maturity"])
    # A Cartesian time x spot grid includes numerically meaningless states
    # such as S≈S0 immediately before an extreme left-tail maturity.  Center
    # each time slice on a deterministic bridge from S0 to the event barrier;
    # multipliers then cover the locally relevant neighborhood without
    # silently flooring an under-resolved desirability.
    datasets: list[HestonOracleDataset] = []
    initial_spot = float(model["spot"])
    numerics = HestonOracleNumerics(
        variance_relative_step=float(oracle["variance_relative_step"])
    )
    for fraction in time_fractions:
        fraction = float(fraction)
        bridge_center = (1.0 - fraction) * initial_spot + fraction * barrier
        datasets.append(
            build_heston_oracle_dataset(
                times=(maturity * fraction,),
                spots=tuple(bridge_center * float(value) for value in spot_multipliers),
                variances=tuple(
                    float(model["variance"]) * float(value)
                    for value in variance_multipliers
                ),
                maturity=maturity,
                barrier=barrier,
                temperature=float(config["event"]["soft_temperature"]),
                params=_reference_params(model),
                numerics=numerics,
            )
        )
    combined = HestonOracleDataset(
        time=torch.cat([dataset.time for dataset in datasets]),
        spot=torch.cat([dataset.spot for dataset in datasets]),
        variance=torch.cat([dataset.variance for dataset in datasets]),
        control=torch.cat([dataset.control for dataset in datasets]),
        maximum_gradient_discrepancy=max(
            dataset.maximum_gradient_discrepancy for dataset in datasets
        ),
    )
    combined.validate()
    return combined


def _evaluate(
    simulator: MarketSimulator,
    control_fn: Callable[[float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    | None,
    model: dict[str, Any],
    *,
    barrier: float,
    reference_probability: float,
    dt: float,
    num_paths: int,
    seed: int,
    method: str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float | str]:
    torch.manual_seed(seed)
    if isinstance(control_fn, TwoDriverHestonControl):
        control_fn.eval()
    start = time.perf_counter()
    with torch.no_grad():
        paths = simulator.simulate_controlled_two_driver(
            S0=float(model["spot"]),
            v0=float(model["variance"]),
            T=float(model["maturity"]),
            dt=dt,
            num_paths=num_paths,
            control_fn=control_fn,
            record_brownian=False,
            dtype=dtype,
        )
    elapsed = time.perf_counter() - start
    event = (paths.spot[:, -1] <= barrier).double()
    contribution = event * torch.exp(paths.log_likelihood)
    estimate = float(contribution.mean())
    standard_error = float(contribution.std(unbiased=True) / math.sqrt(num_paths))
    contribution_ess = float(
        contribution.sum().square() / contribution.square().sum().clamp_min(1e-300)
    )
    return {
        "method": method,
        "seed": float(seed),
        "dt": dt,
        "estimate": estimate,
        "standard_error": standard_error,
        "reported_bias_z": (estimate - reference_probability) / max(standard_error, 1e-300),
        "event_fraction_under_proposal": float(event.mean()),
        "single_path_variance": float(contribution.var(unbiased=True)),
        "contribution_ess": contribution_ess,
        "elapsed_seconds": elapsed,
        "cost_per_path": elapsed / num_paths,
    }


def _constant_two_driver(control_1: float):
    def control_fn(
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _average: torch.Tensor,
    ) -> torch.Tensor:
        first = torch.full_like(spot, control_1)
        return torch.stack((first, torch.zeros_like(first)), dim=-1)

    return control_fn


def _one_driver_ablation(control: TwoDriverHestonControl):
    def control_fn(
        time: float,
        spot: torch.Tensor,
        variance: torch.Tensor,
        average: torch.Tensor,
    ) -> torch.Tensor:
        full = control(time, spot, variance, average)
        return torch.stack((full[:, 0], torch.zeros_like(full[:, 0])), dim=-1)

    return control_fn


def run(
    config_path: Path,
    *,
    smoke: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    protocol_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    model = config["model"]
    event = config["event"]
    oracle_config = config["oracle"]
    training = config["training"]
    validation = config["validation"]
    seeds = config["seeds"]

    params = _reference_params(model)
    target_probability = float(event["target_probability"])
    barrier = heston_left_tail_quantile(
        target_probability,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        params=params,
        lower_spot=1e-4,
        upper_spot=float(model["spot"]) * 2.0,
    )
    reference_probability = heston_terminal_cdf(
        terminal_spot=barrier,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        params=params,
    )

    train_oracle = _oracle_grid(
        config=config, model=model, barrier=barrier, validation=False, smoke=smoke
    )
    validation_oracle = _oracle_grid(
        config=config, model=model, barrier=barrier, validation=True, smoke=smoke
    )
    controller = config["controller"]
    torch.manual_seed(int(seeds["oracle_initialization"]))
    control = TwoDriverHestonControl(
        barrier=barrier,
        maturity=float(model["maturity"]),
        variance_scale=float(model["variance"]),
        architecture=str(controller["architecture"]),  # type: ignore[arg-type]
        hidden_dim=int(controller["hidden_dim"]),
        n_layers=int(controller["n_layers"]),
        control_bound=tuple(float(value) for value in controller["control_bound"]),
        initial_control=(0.0, 0.0),
    )

    distillation_epochs = 75 if smoke else int(oracle_config["distillation_epochs"])
    start = time.perf_counter()
    distillation_history = fit_heston_oracle_distillation(
        control,
        train_oracle,
        epochs=distillation_epochs,
        learning_rate=float(oracle_config["learning_rate"]),
    )
    distillation_seconds = time.perf_counter() - start
    alignment_after_oracle = oracle_alignment(control, validation_oracle)
    oracle_snapshot = control.frozen_copy()

    paths_per_batch = 512 if smoke else int(training["paths_per_batch"])
    updates = 2 if smoke else None
    common = {
        "spot": float(model["spot"]),
        "variance": float(model["variance"]),
        "maturity": float(model["maturity"]),
        "dt": 1.0 / 32.0 if smoke else float(model["dt"]),
        "barrier": barrier,
        "num_paths": paths_per_batch,
    }
    simulator = _simulator(model)
    cem_config = config["cem"]
    torch.manual_seed(int(seeds["cem"]))
    stage_start = time.perf_counter()
    cem_fit = fit_constant_control_cem(
        HestonTerminalLossSampler(
            simulator,
            spot=float(model["spot"]),
            variance=float(model["variance"]),
            maturity=float(model["maturity"]),
            dt=float(common["dt"]),
        ),
        initial_control=float(cem_config["initial_control"]),
        target_score=-barrier,
        num_paths=2_000 if smoke else int(cem_config["paths_per_iteration"]),
        max_iterations=2 if smoke else int(cem_config["max_iterations"]),
        elite_quantile=float(cem_config["elite_quantile"]),
        smoothing=float(cem_config["smoothing"]),
    )
    cem_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    pi_records = train_soft_pi_stage(
        simulator,
        control,
        updates=updates or int(training["pi_updates"]),
        learning_rate=float(training["pi_learning_rate"]),
        seed=int(seeds["pi"]),
        gradient_clip=float(training["gradient_clip"]),
        temperature=float(event["soft_temperature"]),
        **common,
    )
    pi_seconds = time.perf_counter() - stage_start
    alignment_after_pi = oracle_alignment(control, validation_oracle)
    pi_snapshot = control.frozen_copy()
    stage_start = time.perf_counter()
    pice_records = train_feedback_pice_stage(
        simulator,
        control,
        updates=updates or int(training["pice_updates"]),
        learning_rate=float(training["pice_learning_rate"]),
        seed=int(seeds["pice"]),
        gradient_clip=float(training["gradient_clip"]),
        behavior_refresh=int(training["behavior_refresh"]),
        temperature=float(event["soft_temperature"]),
        **common,
    )
    pice_seconds = time.perf_counter() - stage_start
    alignment_after_pice = oracle_alignment(control, validation_oracle)
    pice_snapshot = control.frozen_copy()
    stage_start = time.perf_counter()
    j2_records = train_hard_j2_stage(
        simulator,
        control,
        updates=updates or int(training["j2_updates"]),
        learning_rate=float(training["j2_learning_rate"]),
        seed=int(seeds["j2"]),
        gradient_clip=float(training["gradient_clip"]),
        **common,
    )
    j2_seconds = time.perf_counter() - stage_start
    alignment_after_j2 = oracle_alignment(control, validation_oracle)
    j2_snapshot = control.frozen_copy()

    validation_paths = 3_000 if smoke else int(validation["paths_per_seed"])
    validation_seeds = list(seeds["validation"][:2] if smoke else seeds["validation"])
    dt_grid = [1.0 / 32.0, 1.0 / 64.0] if smoke else list(validation["dt_grid"])
    refinement_runs = [
        _evaluate(
            simulator,
            j2_snapshot,
            model,
            barrier=barrier,
            reference_probability=reference_probability,
            dt=float(dt),
            num_paths=validation_paths,
            seed=int(seed),
            method="j2",
        )
        for dt in dt_grid
        for seed in validation_seeds
    ]
    mean_by_dt = {
        str(dt): sum(
            float(run["estimate"]) for run in refinement_runs if run["dt"] == dt
        )
        / len(validation_seeds)
        for dt in dt_grid
    }
    finest = float(dt_grid[-1])
    previous = float(dt_grid[-2])
    methods = {
        "natural": None,
        "cem_constant": _constant_two_driver(cem_fit.control),
        "oracle_distilled": oracle_snapshot,
        "pi": pi_snapshot,
        "pice": pice_snapshot,
        "j2": j2_snapshot,
        "j2_one_driver": _one_driver_ablation(j2_snapshot),
    }
    method_runs = [
        _evaluate(
            simulator,
            method_control,
            model,
            barrier=barrier,
            reference_probability=reference_probability,
            dt=finest,
            num_paths=validation_paths,
            seed=int(seed),
            method=method_name,
        )
        for method_name, method_control in methods.items()
        for seed in validation_seeds
    ]
    method_summary = {
        method_name: {
            "mean_estimate": sum(
                float(run["estimate"])
                for run in method_runs
                if run["method"] == method_name
            )
            / len(validation_seeds),
            "mean_single_path_variance": sum(
                float(run["single_path_variance"])
                for run in method_runs
                if run["method"] == method_name
            )
            / len(validation_seeds),
            "mean_cost_per_path": sum(
                float(run["cost_per_path"])
                for run in method_runs
                if run["method"] == method_name
            )
            / len(validation_seeds),
        }
        for method_name in methods
    }
    for summary in method_summary.values():
        summary["online_work_proxy"] = (
            summary["mean_single_path_variance"] * summary["mean_cost_per_path"]
        )
    cem_work = method_summary["cem_constant"]["online_work_proxy"]
    for summary in method_summary.values():
        summary["online_work_vrf_vs_cem"] = cem_work / max(
            summary["online_work_proxy"], 1e-300
        )
    relative_refinement_change = abs(mean_by_dt[str(finest)] - mean_by_dt[str(previous)]) / max(
        abs(mean_by_dt[str(finest)]), 1e-300
    )
    oracle_pass = (
        alignment_after_oracle.normalized_rmse
        <= float(oracle_config["maximum_normalized_rmse"])
        and alignment_after_oracle.mean_cosine
        >= float(oracle_config["minimum_mean_cosine"])
        and alignment_after_oracle.sign_agreement
        >= float(oracle_config["minimum_sign_agreement"])
        and max(
            train_oracle.maximum_gradient_discrepancy,
            validation_oracle.maximum_gradient_discrepancy,
        )
        <= float(oracle_config["maximum_gradient_discrepancy"])
    )
    bias_pass = all(
        abs(run["reported_bias_z"]) <= float(validation["maximum_absolute_bias_z"])
        for run in refinement_runs
        if run["dt"] == finest
    )
    refinement_pass = relative_refinement_change <= float(
        validation["maximum_relative_refinement_change"]
    )
    gate_pass = oracle_pass and bias_pass and refinement_pass
    checkpoint: dict[str, str] | None = None
    if checkpoint_path is not None:
        state_hash = save_two_driver_control_checkpoint(
            checkpoint_path,
            j2_snapshot,
            metadata={
                "protocol_id": config["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "smoke": smoke,
                "gate_pass": gate_pass,
            },
        )
        checkpoint = {"path": str(checkpoint_path), "state_sha256": state_hash}

    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "protocol_frozen": bool(config.get("frozen", False)),
        "smoke": smoke,
        "sealed_v3_evaluation_seeds_accessed": False,
        "barrier": barrier,
        "reference_probability": reference_probability,
        "checkpoint": checkpoint,
        "oracle": {
            "train_samples": int(train_oracle.time.shape[0]),
            "validation_samples": int(validation_oracle.time.shape[0]),
            "maximum_gradient_discrepancy": max(
                train_oracle.maximum_gradient_discrepancy,
                validation_oracle.maximum_gradient_discrepancy,
            ),
            "distillation_initial_loss": distillation_history[0],
            "distillation_final_loss": distillation_history[-1],
            "distillation_seconds": distillation_seconds,
            "alignment_after_oracle": asdict(alignment_after_oracle),
            "alignment_after_pi": asdict(alignment_after_pi),
            "alignment_after_pice": asdict(alignment_after_pice),
            "alignment_after_j2": asdict(alignment_after_j2),
        },
        "training": {
            "cem": {
                "control": cem_fit.control,
                "converged": cem_fit.converged,
                "seconds": cem_seconds,
                "history": [asdict(item) for item in cem_fit.history],
            },
            "pi": [asdict(record) for record in pi_records],
            "pice": [asdict(record) for record in pice_records],
            "j2": [asdict(record) for record in j2_records],
            "seconds": {
                "oracle_distillation": distillation_seconds,
                "cem": cem_seconds,
                "pi": pi_seconds,
                "pice": pice_seconds,
                "j2": j2_seconds,
                "feedback_total": distillation_seconds
                + pi_seconds
                + pice_seconds
                + j2_seconds,
            },
        },
        "validation": {
            "refinement_runs": refinement_runs,
            "mean_by_dt": mean_by_dt,
            "relative_refinement_change": relative_refinement_change,
            "method_runs_at_finest_dt": method_runs,
            "method_summary": method_summary,
        },
        "gates": {
            "oracle_alignment": oracle_pass,
            "reported_bias": bias_pass,
            "time_step_refinement": refinement_pass,
            "g1_gate_pass": gate_pass,
            "confirmatory_pass": bool(config.get("frozen", False)) and not smoke and gate_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g1_heston_feedback.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke, checkpoint_path=args.checkpoint)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
