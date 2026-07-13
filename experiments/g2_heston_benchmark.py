"""Frozen G2 Heston terminal-left-tail benchmark.

This pipeline calibrates event thresholds from the semi-analytic Heston CDF,
fits CEM constant controls using training seeds, selects them on validation
seeds, and reports results exactly once on held-out evaluation seeds.

``--smoke`` reduces seeds and paths for code-path verification. Smoke output is
explicitly marked non-publication and must never be used in a paper table.
"""

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

from src.evaluation.heston_reference import (
    HestonReferenceParams,
    heston_left_tail_quantile,
    heston_terminal_cdf,
)
from src.evaluation.likelihood import likelihood_diagnostics
from src.evaluation.protocol import FrozenExperimentProtocol, load_frozen_protocol
from src.evaluation.statistics import repeated_estimate_report
from src.physics_engine import MarketSimulator
from src.training.cem import HestonTerminalLossSampler, fit_constant_control_cem
from src.training.markov_control import (
    MarkovianHestonControl,
    markov_control_state_sha256,
    save_markovian_control_checkpoint,
    train_markovian_control,
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


def _constant_control(control: float):
    def control_fn(
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _average_spot: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full_like(spot, control)

    return control_fn


def _sample_is(
    simulator: MarketSimulator,
    model: dict[str, Any],
    *,
    threshold: float,
    control: float,
    num_paths: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    start = time.perf_counter()
    paths, _variance, log_weight, _barrier, _average = simulator.simulate_controlled(
        S0=float(model["spot"]),
        v0=float(model["variance"]),
        T=float(model["maturity"]),
        dt=float(model["dt"]),
        num_paths=num_paths,
        control_fn=_constant_control(control),
    )
    elapsed = time.perf_counter() - start
    event = (paths[:, -1] <= threshold).to(log_weight.dtype)
    contributions = event * torch.exp(log_weight)
    diagnostics = likelihood_diagnostics(
        log_weight.detach().cpu().numpy(), event.detach().cpu().numpy()
    )
    return {
        "estimate": float(contributions.mean()),
        "standard_error": float(contributions.std(unbiased=True) / math.sqrt(num_paths)),
        "single_path_variance": float(contributions.var(unbiased=True)),
        "event_fraction_under_proposal": float(event.mean()),
        "elapsed_seconds": elapsed,
        "diagnostics": asdict(diagnostics),
    }


def _sample_feedback_is(
    simulator: MarketSimulator,
    model: dict[str, Any],
    *,
    threshold: float,
    control: MarkovianHestonControl,
    num_paths: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    control.eval()
    start = time.perf_counter()
    with torch.no_grad():
        paths, _variance, log_weight, _barrier, _average = simulator.simulate_controlled(
            S0=float(model["spot"]),
            v0=float(model["variance"]),
            T=float(model["maturity"]),
            dt=float(model["dt"]),
            num_paths=num_paths,
            control_fn=control,
        )
    elapsed = time.perf_counter() - start
    event = (paths[:, -1] <= threshold).to(log_weight.dtype)
    contributions = event * torch.exp(log_weight)
    diagnostics = likelihood_diagnostics(
        log_weight.detach().cpu().numpy(), event.detach().cpu().numpy()
    )
    return {
        "estimate": float(contributions.mean()),
        "standard_error": float(contributions.std(unbiased=True) / math.sqrt(num_paths)),
        "single_path_variance": float(contributions.var(unbiased=True)),
        "event_fraction_under_proposal": float(event.mean()),
        "elapsed_seconds": elapsed,
        "diagnostics": asdict(diagnostics),
    }


def _sample_mc(
    simulator: MarketSimulator,
    model: dict[str, Any],
    *,
    threshold: float,
    num_paths: int,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    start = time.perf_counter()
    paths, _variance = simulator.simulate(
        S0=float(model["spot"]),
        v0=float(model["variance"]),
        T=float(model["maturity"]),
        dt=float(model["dt"]),
        num_paths=num_paths,
    )
    elapsed = time.perf_counter() - start
    event = (paths[:, -1] <= threshold).float()
    return {
        "estimate": float(event.mean()),
        "standard_error": float(event.std(unbiased=True) / math.sqrt(num_paths)),
        "single_path_variance": float(event.var(unbiased=True)),
        "elapsed_seconds": elapsed,
    }


def _fit_and_select_control(
    protocol: FrozenExperimentProtocol,
    simulator: MarketSimulator,
    model: dict[str, Any],
    *,
    threshold: float,
    smoke: bool,
) -> tuple[float, float, list[dict[str, Any]]]:
    cem = protocol.payload["cem"]
    train_seeds = protocol.seeds.train[:2] if smoke else protocol.seeds.train
    validation_seeds = protocol.seeds.validation[:2] if smoke else protocol.seeds.validation
    train_paths = (
        min(int(cem["paths_per_iteration"]), 5_000) if smoke else int(cem["paths_per_iteration"])
    )
    validation_paths = 5_000 if smoke else int(protocol.payload["evaluation"]["paths_per_seed"])
    max_iterations = min(int(cem["max_iterations"]), 5) if smoke else int(cem["max_iterations"])
    sampler = HestonTerminalLossSampler(
        simulator,
        spot=float(model["spot"]),
        variance=float(model["variance"]),
        maturity=float(model["maturity"]),
        dt=float(model["dt"]),
    )

    training_start = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        torch.manual_seed(train_seed)
        fit = fit_constant_control_cem(
            sampler,
            initial_control=float(cem["initial_control"]),
            target_score=-threshold,
            num_paths=train_paths,
            max_iterations=max_iterations,
            elite_quantile=float(cem["elite_quantile"]),
            smoothing=float(cem["smoothing"]),
        )
        validation_second_moments: list[float] = []
        for validation_seed in validation_seeds:
            sample = _sample_is(
                simulator,
                model,
                threshold=threshold,
                control=fit.control,
                num_paths=validation_paths,
                seed=validation_seed,
            )
            validation_second_moments.append(
                sample["single_path_variance"] + sample["estimate"] ** 2
            )
        candidates.append(
            {
                "train_seed": train_seed,
                "control": fit.control,
                "converged": fit.converged,
                "iterations": len(fit.history),
                "validation_second_moment": float(np.mean(validation_second_moments)),
                "history": [asdict(item) for item in fit.history],
            }
        )
    training_seconds = time.perf_counter() - training_start
    selected = min(candidates, key=lambda candidate: candidate["validation_second_moment"])
    return float(selected["control"]), training_seconds, candidates


def run_benchmark(
    protocol_path: Path,
    *,
    smoke: bool,
    include_neural: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    protocol = load_frozen_protocol(protocol_path)
    model = protocol.payload["model"]
    reference_params = _reference_params(model)
    simulator = _simulator(model)
    target_probabilities = [
        float(value) for value in protocol.payload["events"]["target_probabilities"]
    ]
    evaluation_seeds = protocol.seeds.evaluation[:3] if smoke else protocol.seeds.evaluation
    evaluation_paths = (
        min(int(protocol.payload["evaluation"]["paths_per_seed"]), 10_000)
        if smoke
        else int(protocol.payload["evaluation"]["paths_per_seed"])
    )

    event_results: list[dict[str, Any]] = []
    for target_probability in target_probabilities:
        threshold = heston_left_tail_quantile(
            target_probability,
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            params=reference_params,
        )
        reference_probability = heston_terminal_cdf(
            terminal_spot=threshold,
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            params=reference_params,
        )
        control, training_seconds, candidates = _fit_and_select_control(
            protocol, simulator, model, threshold=threshold, smoke=smoke
        )
        mc_runs = [
            _sample_mc(
                simulator,
                model,
                threshold=threshold,
                num_paths=evaluation_paths,
                seed=seed,
            )
            for seed in evaluation_seeds
        ]
        is_runs = [
            _sample_is(
                simulator,
                model,
                threshold=threshold,
                control=control,
                num_paths=evaluation_paths,
                seed=seed,
            )
            for seed in evaluation_seeds
        ]
        mc_report = repeated_estimate_report(
            np.array([run["estimate"] for run in mc_runs]),
            np.array([run["standard_error"] for run in mc_runs]),
            truth=reference_probability,
            confidence_level=float(protocol.payload["evaluation"]["confidence_level"]),
        )
        is_report = repeated_estimate_report(
            np.array([run["estimate"] for run in is_runs]),
            np.array([run["standard_error"] for run in is_runs]),
            truth=reference_probability,
            confidence_level=float(protocol.payload["evaluation"]["confidence_level"]),
        )
        # At 1e-6 a finite MC batch commonly observes no events and reports a
        # misleading zero sample variance. The Heston CDF gives the exact
        # Bernoulli variance for the work-normalized comparison.
        mc_variance = reference_probability * (1.0 - reference_probability)
        is_variance = float(np.mean([run["single_path_variance"] for run in is_runs]))
        mc_cost = float(np.mean([run["elapsed_seconds"] for run in mc_runs])) / evaluation_paths
        is_cost = float(np.mean([run["elapsed_seconds"] for run in is_runs])) / evaluation_paths
        online_vrf = (mc_variance * mc_cost) / max(is_variance * is_cost, 1e-300)
        amortized_training_cost = training_seconds / (len(evaluation_seeds) * evaluation_paths)
        end_to_end_vrf = (mc_variance * mc_cost) / max(
            is_variance * (is_cost + amortized_training_cost), 1e-300
        )
        neural_ablations: list[dict[str, Any]] = []
        if include_neural:
            neural = protocol.payload["neural"]
            train_seeds = protocol.seeds.train[:2] if smoke else protocol.seeds.train
            validation_seeds = protocol.seeds.validation[:2] if smoke else protocol.seeds.validation
            neural_epochs = min(int(neural["epochs"]), 5) if smoke else int(neural["epochs"])
            neural_paths = (
                min(int(neural["paths_per_batch"]), 1_500)
                if smoke
                else int(neural["paths_per_batch"])
            )
            neural_validation_paths = (
                min(int(neural["validation_paths"]), 3_000)
                if smoke
                else int(neural["validation_paths"])
            )
            neural_validate_every = (
                min(int(neural["validate_every"]), neural_epochs)
                if smoke
                else int(neural["validate_every"])
            )
            for objective in neural["objectives"]:
                torch.manual_seed(train_seeds[0])
                feedback_control = MarkovianHestonControl(
                    initial_spot=float(model["spot"]),
                    barrier=threshold,
                    maturity=float(model["maturity"]),
                    variance_scale=float(model["theta"]),
                    hidden_dim=int(neural["hidden_dim"]),
                    n_layers=int(neural["n_layers"]),
                    control_bound=float(neural["control_bound"]),
                    initial_constant=control,
                )
                neural_start = time.perf_counter()
                training = train_markovian_control(
                    simulator,
                    feedback_control,
                    spot=float(model["spot"]),
                    variance=float(model["variance"]),
                    maturity=float(model["maturity"]),
                    dt=float(model["dt"]),
                    barrier=threshold,
                    reference_probability=reference_probability,
                    objective=objective,
                    train_seeds=train_seeds,
                    validation_seeds=validation_seeds,
                    epochs=neural_epochs,
                    paths_per_batch=neural_paths,
                    validation_paths=neural_validation_paths,
                    learning_rate=float(neural["learning_rate"]),
                    validate_every=neural_validate_every,
                    gradient_clip=float(neural["gradient_clip"]),
                )
                neural_training_seconds = time.perf_counter() - neural_start
                neural_runs = [
                    _sample_feedback_is(
                        simulator,
                        model,
                        threshold=threshold,
                        control=feedback_control,
                        num_paths=evaluation_paths,
                        seed=seed,
                    )
                    for seed in evaluation_seeds
                ]
                neural_report = repeated_estimate_report(
                    np.array([run["estimate"] for run in neural_runs]),
                    np.array([run["standard_error"] for run in neural_runs]),
                    truth=reference_probability,
                    confidence_level=float(protocol.payload["evaluation"]["confidence_level"]),
                )
                neural_variance = float(
                    np.mean([run["single_path_variance"] for run in neural_runs])
                )
                neural_cost = (
                    float(np.mean([run["elapsed_seconds"] for run in neural_runs]))
                    / evaluation_paths
                )
                neural_online_vrf = (mc_variance * mc_cost) / max(
                    neural_variance * neural_cost, 1e-300
                )
                total_training_seconds = training_seconds + neural_training_seconds
                neural_amortized_training_cost = total_training_seconds / (
                    len(evaluation_seeds) * evaluation_paths
                )
                neural_end_to_end_vrf = (mc_variance * mc_cost) / max(
                    neural_variance * (neural_cost + neural_amortized_training_cost),
                    1e-300,
                )
                state_hash = markov_control_state_sha256(feedback_control)
                checkpoint_path: str | None = None
                if checkpoint_dir is not None:
                    probability_tag = f"{target_probability:.0e}".replace("-", "m")
                    destination = checkpoint_dir / f"{probability_tag}_{objective}.pt"
                    save_markovian_control_checkpoint(
                        destination,
                        feedback_control,
                        metadata={
                            "protocol_id": protocol.protocol_id,
                            "protocol_sha256": protocol.sha256,
                            "target_probability": target_probability,
                            "threshold": threshold,
                            "objective": objective,
                            "mode": "smoke_non_publication" if smoke else "frozen_full_evaluation",
                        },
                    )
                    checkpoint_path = str(destination)
                neural_ablations.append(
                    {
                        "objective": objective,
                        "cem_warm_start": control,
                        "neural_training_seconds": neural_training_seconds,
                        "total_training_seconds_including_cem": total_training_seconds,
                        "control_state_sha256": state_hash,
                        "checkpoint_path": checkpoint_path,
                        "best_validation_log_second_moment": (
                            training.best_validation_log_second_moment
                        ),
                        "training_history": [asdict(item) for item in training.history],
                        "report": asdict(neural_report),
                        "online_work_normalized_vrf": neural_online_vrf,
                        "end_to_end_work_normalized_vrf_at_evaluation_budget": (
                            neural_end_to_end_vrf
                        ),
                        "runs": neural_runs,
                    }
                )
        event_results.append(
            {
                "target_probability": target_probability,
                "reference_probability": reference_probability,
                "threshold": threshold,
                "selected_control": control,
                "training_seconds": training_seconds,
                "candidates": candidates,
                "mc_report": asdict(mc_report),
                "is_report": asdict(is_report),
                "mc_variance_source_for_vrf": "analytic_bernoulli_reference",
                "online_work_normalized_vrf": online_vrf,
                "end_to_end_work_normalized_vrf_at_evaluation_budget": end_to_end_vrf,
                "mc_runs": mc_runs,
                "is_runs": is_runs,
                "neural_ablations": neural_ablations,
            }
        )

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "mode": "smoke_non_publication" if smoke else "frozen_full_evaluation",
        "device": "cpu",
        "dtype": "simulator_float32_likelihood_diagnostics_float64",
        "events": event_results,
    }


def _json_safe(value: Any) -> Any:
    """Represent diagnostic infinities explicitly without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "g2_heston_benchmark.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()

    result = run_benchmark(
        args.protocol,
        smoke=args.smoke,
        include_neural=not args.skip_neural,
        checkpoint_dir=args.checkpoint_dir,
    )
    rendered = json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
