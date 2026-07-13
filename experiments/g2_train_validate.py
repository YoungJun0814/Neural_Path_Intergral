"""Train/validation-only G2 model selection with sealed evaluation seeds.

This command never reads or simulates the protocol's evaluation seeds. It
compares affine and MLP Markov feedback controls against the CEM constant using
validation second moment and measured online work. A separate final command is
required to unseal evaluation after the method and acceptance rule are frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from experiments.g2_heston_benchmark import (
    _fit_and_select_control,
    _json_safe,
    _reference_params,
    _sample_feedback_is,
    _sample_is,
    _simulator,
)
from src.evaluation.heston_reference import heston_left_tail_quantile, heston_terminal_cdf
from src.evaluation.protocol import load_frozen_protocol
from src.evaluation.statistics import repeated_estimate_report
from src.training.markov_control import (
    ArchitectureName,
    FeatureMapName,
    MarkovianHestonControl,
    ObjectiveName,
    save_markovian_control_checkpoint,
    train_markovian_control,
)


def _mean_work(runs: list[dict[str, Any]], num_paths: int) -> tuple[float, float, float]:
    variance = float(np.mean([run["single_path_variance"] for run in runs]))
    cost_per_path = float(np.mean([run["elapsed_seconds"] for run in runs])) / num_paths
    return variance, cost_per_path, variance * cost_per_path


def run_selection(
    protocol_path: Path,
    *,
    probabilities: tuple[float, ...] | None,
    architectures: tuple[str, ...] | None,
    objectives: tuple[str, ...] | None,
    learning_rates: tuple[float, ...] | None,
    epoch_override: int | None,
    smoke: bool,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    protocol = load_frozen_protocol(protocol_path)
    if protocol.payload.get("evaluation", {}).get("sealed") is not True:
        raise ValueError("selection requires a protocol with sealed evaluation")

    model = protocol.payload["model"]
    selection = protocol.payload["selection"]
    reference_params = _reference_params(model)
    simulator = _simulator(model)
    configured_probabilities = tuple(
        float(value) for value in protocol.payload["events"]["target_probabilities"]
    )
    targets = probabilities or configured_probabilities
    if not set(targets).issubset(set(configured_probabilities)):
        raise ValueError("all requested probabilities must be frozen in the protocol")

    train_seeds = protocol.seeds.train[:2] if smoke else protocol.seeds.train
    selection_seed_count = int(selection["validation_selection_seed_count"])
    if not 0 < selection_seed_count < len(protocol.seeds.validation):
        raise ValueError("validation selection count must leave at least one audit seed")
    validation_seeds = protocol.seeds.validation[:selection_seed_count]
    audit_seeds = protocol.seeds.validation[selection_seed_count:]
    architecture_grid = architectures or tuple(selection["architectures"])
    objective_grid = objectives or tuple(selection["objectives"])
    learning_rate_grid = learning_rates or tuple(
        float(value) for value in selection["learning_rates"]
    )
    if not set(architecture_grid).issubset(set(selection["architectures"])):
        raise ValueError("requested architectures must be frozen in the protocol")
    if not set(objective_grid).issubset(set(selection["objectives"])):
        raise ValueError("requested objectives must be frozen in the protocol")
    frozen_learning_rates = {float(value) for value in selection["learning_rates"]}
    if not set(learning_rate_grid).issubset(frozen_learning_rates):
        raise ValueError("requested learning rates must be frozen in the protocol")
    epochs = epoch_override or (
        min(int(selection["epochs"]), 20) if smoke else int(selection["epochs"])
    )
    if epochs <= 0 or epochs > int(selection["epochs"]):
        raise ValueError("epochs must lie within the frozen maximum")
    paths_per_batch = (
        min(int(selection["paths_per_batch"]), 2_000)
        if smoke
        else int(selection["paths_per_batch"])
    )
    validation_paths = (
        min(int(selection["validation_paths"]), 5_000)
        if smoke
        else int(selection["validation_paths"])
    )
    validate_every = (
        min(int(selection["validate_every"]), epochs) if smoke else int(selection["validate_every"])
    )
    minimum_vrf = float(selection["minimum_online_vrf_vs_cem"])
    maximum_absolute_bias_z = float(selection["maximum_absolute_bias_z"])
    minimum_coverage = float(selection["minimum_validation_ci_coverage"])
    audit_paths = int(selection["audit_paths_per_seed"])

    events: list[dict[str, Any]] = []
    for probability in targets:
        threshold = heston_left_tail_quantile(
            probability,
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            params=reference_params,
        )
        truth = heston_terminal_cdf(
            terminal_spot=threshold,
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            params=reference_params,
        )
        cem_control, cem_training_seconds, cem_candidates = _fit_and_select_control(
            protocol,
            simulator,
            model,
            threshold=threshold,
            smoke=smoke,
        )
        cem_runs = [
            _sample_is(
                simulator,
                model,
                threshold=threshold,
                control=cem_control,
                num_paths=validation_paths,
                seed=seed,
            )
            for seed in validation_seeds
        ]
        cem_variance, cem_cost, cem_work = _mean_work(cem_runs, validation_paths)
        cem_report = repeated_estimate_report(
            np.array([run["estimate"] for run in cem_runs]),
            np.array([run["standard_error"] for run in cem_runs]),
            truth=truth,
        )

        candidates: list[dict[str, Any]] = []
        trained_controls: list[MarkovianHestonControl] = []
        for architecture in architecture_grid:
            for objective in objective_grid:
                for learning_rate in learning_rate_grid:
                    architecture_name = cast(ArchitectureName, architecture)
                    objective_name = cast(ObjectiveName, objective)
                    feature_map = cast(
                        FeatureMapName,
                        selection["affine_feature_map"]
                        if architecture == "affine"
                        else selection["mlp_feature_map"],
                    )
                    torch.manual_seed(train_seeds[0])
                    control = MarkovianHestonControl(
                        initial_spot=float(model["spot"]),
                        barrier=threshold,
                        maturity=float(model["maturity"]),
                        variance_scale=float(model["theta"]),
                        hidden_dim=int(selection["hidden_dim"]),
                        n_layers=int(selection["n_layers"]),
                        architecture=architecture_name,
                        feature_map=feature_map,
                        control_bound=float(selection["control_bound"]),
                        initial_constant=cem_control,
                    )
                    start = time.perf_counter()
                    training = train_markovian_control(
                        simulator,
                        control,
                        spot=float(model["spot"]),
                        variance=float(model["variance"]),
                        maturity=float(model["maturity"]),
                        dt=float(model["dt"]),
                        barrier=threshold,
                        reference_probability=truth,
                        objective=objective_name,
                        train_seeds=train_seeds,
                        validation_seeds=validation_seeds,
                        epochs=epochs,
                        paths_per_batch=paths_per_batch,
                        validation_paths=validation_paths,
                        learning_rate=float(learning_rate),
                        validate_every=validate_every,
                        gradient_clip=float(selection["gradient_clip"]),
                    )
                    training_seconds = time.perf_counter() - start
                    runs = [
                        _sample_feedback_is(
                            simulator,
                            model,
                            threshold=threshold,
                            control=control,
                            num_paths=validation_paths,
                            seed=seed,
                        )
                        for seed in validation_seeds
                    ]
                    report = repeated_estimate_report(
                        np.array([run["estimate"] for run in runs]),
                        np.array([run["standard_error"] for run in runs]),
                        truth=truth,
                    )
                    variance, cost, work = _mean_work(runs, validation_paths)
                    online_vrf_vs_cem = cem_work / max(work, 1e-300)
                    passes_work = online_vrf_vs_cem >= minimum_vrf
                    passes_bias = abs(report.reported_bias_z_score) <= maximum_absolute_bias_z
                    passes_coverage = report.ci_coverage >= minimum_coverage
                    candidates.append(
                        {
                            "architecture": architecture,
                            "feature_map": control.feature_map,
                            "objective": objective,
                            "learning_rate": float(learning_rate),
                            "epochs": epochs,
                            "training_seconds": training_seconds,
                            "best_validation_log_second_moment": (
                                training.best_validation_log_second_moment
                            ),
                            "validation_report": asdict(report),
                            "validation_single_path_variance": variance,
                            "validation_cost_per_path": cost,
                            "validation_work": work,
                            "online_vrf_vs_cem": online_vrf_vs_cem,
                            "passes_work_gate": passes_work,
                            "passes_bias_gate": passes_bias,
                            "passes_coverage_gate": passes_coverage,
                            "eligible_for_evaluation": (
                                passes_work and passes_bias and passes_coverage
                            ),
                            "training_history": [asdict(item) for item in training.history],
                        }
                    )
                    trained_controls.append(control)

        best_index = min(
            range(len(candidates)), key=lambda index: candidates[index]["validation_work"]
        )
        best = candidates[best_index]
        best_control = trained_controls[best_index]
        audit_cem_runs = [
            _sample_is(
                simulator,
                model,
                threshold=threshold,
                control=cem_control,
                num_paths=audit_paths,
                seed=seed,
            )
            for seed in audit_seeds
        ]
        audit_control_runs = [
            _sample_feedback_is(
                simulator,
                model,
                threshold=threshold,
                control=best_control,
                num_paths=audit_paths,
                seed=seed,
            )
            for seed in audit_seeds
        ]
        audit_cem_report = repeated_estimate_report(
            np.array([run["estimate"] for run in audit_cem_runs]),
            np.array([run["standard_error"] for run in audit_cem_runs]),
            truth=truth,
        )
        audit_control_report = repeated_estimate_report(
            np.array([run["estimate"] for run in audit_control_runs]),
            np.array([run["standard_error"] for run in audit_control_runs]),
            truth=truth,
        )
        audit_cem_variance, audit_cem_cost, audit_cem_work = _mean_work(audit_cem_runs, audit_paths)
        audit_control_variance, audit_control_cost, audit_control_work = _mean_work(
            audit_control_runs, audit_paths
        )
        audit_vrf_vs_cem = audit_cem_work / max(audit_control_work, 1e-300)
        audit_passes_work = audit_vrf_vs_cem >= minimum_vrf
        audit_passes_bias = (
            abs(audit_control_report.reported_bias_z_score) <= maximum_absolute_bias_z
        )
        audit_passes_coverage = audit_control_report.ci_coverage >= minimum_coverage
        passes_audit = audit_passes_work and audit_passes_bias and audit_passes_coverage
        checkpoint_path: str | None = None
        if checkpoint_dir is not None:
            probability_tag = f"{probability:.0e}".replace("-", "m")
            destination = checkpoint_dir / f"{probability_tag}_selected.pt"
            save_markovian_control_checkpoint(
                destination,
                best_control,
                metadata={
                    "protocol_id": protocol.protocol_id,
                    "protocol_sha256": protocol.sha256,
                    "selection_only": True,
                    "evaluation_seeds_accessed": False,
                    "probability": probability,
                    "threshold": threshold,
                    "candidate": {
                        "architecture": best["architecture"],
                        "feature_map": best["feature_map"],
                        "objective": best["objective"],
                        "learning_rate": best["learning_rate"],
                    },
                },
            )
            checkpoint_path = str(destination)
        events.append(
            {
                "target_probability": probability,
                "reference_probability": truth,
                "threshold": threshold,
                "cem": {
                    "control": cem_control,
                    "training_seconds": cem_training_seconds,
                    "validation_single_path_variance": cem_variance,
                    "validation_cost_per_path": cem_cost,
                    "validation_work": cem_work,
                    "validation_report": asdict(cem_report),
                    "candidates": cem_candidates,
                },
                "candidates": candidates,
                "selected_candidate_index": best_index,
                "selected_candidate": best,
                "selected_checkpoint": checkpoint_path,
                "audit": {
                    "seeds_accessed": list(audit_seeds),
                    "paths_per_seed": audit_paths,
                    "cem_report": asdict(audit_cem_report),
                    "control_report": asdict(audit_control_report),
                    "cem_single_path_variance": audit_cem_variance,
                    "control_single_path_variance": audit_control_variance,
                    "cem_cost_per_path": audit_cem_cost,
                    "control_cost_per_path": audit_control_cost,
                    "cem_work": audit_cem_work,
                    "control_work": audit_control_work,
                    "online_vrf_vs_cem": audit_vrf_vs_cem,
                    "passes_work_gate": audit_passes_work,
                    "passes_bias_gate": audit_passes_bias,
                    "passes_coverage_gate": audit_passes_coverage,
                    "cem_runs": audit_cem_runs,
                    "control_runs": audit_control_runs,
                },
                "passes_evaluation_gate": passes_audit,
            }
        )

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "mode": "smoke_train_validation_only" if smoke else "full_train_validation_only",
        "evaluation_seeds_accessed": False,
        "evaluation_remains_sealed": True,
        "selection_rule": "minimum validation variance_times_measured_cost",
        "minimum_online_vrf_vs_cem": minimum_vrf,
        "maximum_absolute_bias_z": maximum_absolute_bias_z,
        "minimum_validation_ci_coverage": minimum_coverage,
        "events": events,
        "all_events_pass_evaluation_gate": all(event["passes_evaluation_gate"] for event in events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "g2_heston_benchmark_v2.yaml",
    )
    parser.add_argument("--probabilities", type=float, nargs="+")
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--objectives", nargs="+")
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_selection(
        args.protocol,
        probabilities=tuple(args.probabilities) if args.probabilities else None,
        architectures=tuple(args.architectures) if args.architectures else None,
        objectives=tuple(args.objectives) if args.objectives else None,
        learning_rates=tuple(args.learning_rates) if args.learning_rates else None,
        epoch_override=args.epochs,
        smoke=args.smoke,
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
