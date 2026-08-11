"""Replicated operator-initializer confirmation with a shuffled-teacher control."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models.volterra_transport_operator import (
    VolterraTransportOperator,
    VolterraTransportOperatorConfig,
    correct_and_build_operator_transport,
    encode_rbergomi_transport_task,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.cameron_martin_modes import (
    ActionOptimizationResult,
    ActionSolverConfig,
    ModeSearchConfig,
    find_rbergomi_conditional_modes,
    optimize_action,
)
from src.path_integral.finite_rank_gaussian_transport import CurvatureTransportConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import evaluate_rbergomi_cm_transport
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance
from src.path_integral.volterra_action import (
    RBergomiConditionalAction,
    evaluate_action_derivatives,
)
from src.training.volterra_transport_operator import (
    VolterraOperatorTeachers,
    VolterraOperatorTrainingConfig,
    train_volterra_transport_operator,
)

ROOT = Path(__file__).resolve().parents[1]


def _problem(task_id: str, raw: dict[str, Any], *, steps: int) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id=task_id,
        task=TerminalThresholdTask(level=100.0 * float(raw["strike_ratio"])),
        spot=100.0,
        maturity=float(raw["maturity"]),
        steps=steps,
        hurst=float(raw["hurst"]),
        eta=float(raw["eta"]),
        xi=float(raw["xi"]),
        rho=float(raw["rho"]),
    )


def _random_teacher_tasks(config: dict[str, Any]) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(int(config["teacher_seed"]))
    ranges = config["teacher_ranges"]
    names = ("hurst", "eta", "rho", "xi", "maturity", "strike_ratio")
    tasks = []
    for _ in range(int(config["teacher_count"])):
        values = {}
        for name in names:
            lower, upper = (float(value) for value in ranges[name])
            values[name] = lower + (upper - lower) * float(
                torch.rand((), generator=generator)
            )
        tasks.append(values)
    return tasks


def _action(problem: RBergomiBaselineProblem) -> RBergomiConditionalAction:
    basis = build_mesh_compatible_blp_hybrid_basis(
        steps=problem.steps,
        maturity=problem.maturity,
        hurst=problem.hurst,
        drift_modes=8,
        bridge_modes=3,
    )
    return RBergomiConditionalAction(problem=problem, basis=basis)


def _training_config(config: dict[str, Any]) -> VolterraOperatorTrainingConfig:
    values = config["operator"]
    return VolterraOperatorTrainingConfig(
        epochs=int(values["epochs"]),
        learning_rate=float(values["learning_rate"]),
    )


def _new_model(config: dict[str, Any], rank: int) -> VolterraTransportOperator:
    return VolterraTransportOperator(
        VolterraTransportOperatorConfig(
            feature_dimension=7,
            modes=1,
            rank=rank,
            hidden_features=int(config["operator"]["hidden_features"]),
        )
    )


def _certificate_evaluations(certificate: Any) -> int:
    if certificate.corrected_modes is None:
        return 0
    return sum(
        attempt.function_evaluations
        for attempt in certificate.corrected_modes.raw.attempts
    )


def _cold_solve(
    action: RBergomiConditionalAction,
    rank: int,
    solver: ActionSolverConfig,
) -> tuple[ActionOptimizationResult, float]:
    started = time.perf_counter()
    result = optimize_action(
        action,
        torch.zeros(rank, dtype=torch.float64),
        config=solver,
    )
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    solver_values = config["mode_solver"]
    solver = ActionSolverConfig(
        maximum_iterations=int(solver_values["maximum_iterations"]),
        gradient_tolerance=float(solver_values["gradient_tolerance"]),
    )

    features = []
    coefficients = []
    precisions = []
    teacher_function_evaluations = 0
    teacher_started = time.perf_counter()
    for index, raw in enumerate(_random_teacher_tasks(config)):
        problem = _problem(f"teacher-{index}", raw, steps=int(config["steps"]))
        action = _action(problem)
        modes = find_rbergomi_conditional_modes(
            action,
            config=ModeSearchConfig(
                methods=("lbfgs",),
                random_starts=1,
                random_seed=int(config["teacher_seed"]) + 1009 * index,
                start_scale=2.0,
                solver=solver,
            ),
        )
        if not modes.modes:
            raise RuntimeError(f"teacher {index} has no certified mode")
        mode = modes.modes[0]
        derivatives = evaluate_action_derivatives(
            action,
            mode.coefficients,
            include_hessian=True,
        )
        if (
            derivatives.hessian is None
            or derivatives.hessian_eigenvalues is None
            or float(torch.min(derivatives.hessian_eigenvalues)) <= 0.0
        ):
            raise RuntimeError(f"teacher {index} is not a positive-curvature mode")
        features.append(encode_rbergomi_transport_task(problem, epsilon=1.0))
        coefficients.append(mode.coefficients)
        precisions.append(derivatives.hessian)
        teacher_function_evaluations += sum(
            attempt.function_evaluations for attempt in modes.raw.attempts
        )
    teacher_seconds = time.perf_counter() - teacher_started
    feature_tensor = torch.stack(features)
    coefficient_tensor = torch.stack(coefficients).unsqueeze(1)
    precision_tensor = torch.stack(precisions).unsqueeze(1)
    teacher_count = feature_tensor.shape[0]
    rank = coefficient_tensor.shape[2]
    teachers = VolterraOperatorTeachers(
        features=feature_tensor,
        coefficients=coefficient_tensor,
        mode_weights=torch.ones((teacher_count, 1), dtype=torch.float64),
        precision_matrices=precision_tensor,
        mode_mask=torch.ones((teacher_count, 1), dtype=torch.bool),
    )
    permutation_generator = torch.Generator().manual_seed(int(config["teacher_seed"]) + 99)
    permutation = torch.randperm(teacher_count, generator=permutation_generator)
    shuffled_teachers = VolterraOperatorTeachers(
        features=feature_tensor,
        coefficients=coefficient_tensor[permutation],
        mode_weights=torch.ones((teacher_count, 1), dtype=torch.float64),
        precision_matrices=precision_tensor[permutation],
        mode_mask=torch.ones((teacher_count, 1), dtype=torch.bool),
    )

    holdouts = []
    for raw in config["holdouts"]:
        problem = _problem(str(raw["id"]), raw, steps=int(config["steps"]))
        action = _action(problem)
        cold, cold_seconds = _cold_solve(action, rank, solver)
        if not cold.success:
            raise RuntimeError(f"{problem.task_id}: cold oracle did not converge")
        holdouts.append((str(raw["group"]), problem, action, cold, cold_seconds))

    failures = []
    replications = []
    transport_config = CurvatureTransportConfig(
        defensive_mass=0.15,
        asymptotic_safety_mass=0.02,
        safety_spectrum_decay=2.0,
        safety_spectrum_scale=4.0,
    )
    corrector_config = ModeSearchConfig(
        methods=("lbfgs",),
        random_starts=0,
        include_zero_start=False,
        solver=solver,
    )
    gates = config["gates"]
    for replication, training_seed in enumerate(config["operator"]["training_seeds"]):
        learned = _new_model(config, rank)
        shuffled = _new_model(config, rank)
        training_started = time.perf_counter()
        learned_training = train_volterra_transport_operator(
            learned,
            teachers,
            seed=int(training_seed),
            config=_training_config(config),
        )
        learned_training_seconds = time.perf_counter() - training_started
        training_started = time.perf_counter()
        shuffled_training = train_volterra_transport_operator(
            shuffled,
            shuffled_teachers,
            seed=int(training_seed),
            config=_training_config(config),
        )
        shuffled_training_seconds = time.perf_counter() - training_started
        records = []
        speedups = []
        learned_distances = []
        shuffled_distances = []
        fallbacks = 0
        for index, (group, problem, action, cold, cold_seconds) in enumerate(holdouts):
            feature = encode_rbergomi_transport_task(problem, epsilon=1.0).unsqueeze(0)
            learned_prediction = learned(feature)
            shuffled_prediction = shuffled(feature)
            learned_start = learned_prediction.coefficients[0, 0].detach()
            shuffled_start = shuffled_prediction.coefficients[0, 0].detach()
            learned_distance = float(
                torch.linalg.vector_norm(learned_start - cold.coefficients)
            )
            shuffled_distance = float(
                torch.linalg.vector_norm(shuffled_start - cold.coefficients)
            )
            learned_distances.append(learned_distance)
            shuffled_distances.append(shuffled_distance)
            correction_started = time.perf_counter()
            certificate = correct_and_build_operator_transport(
                action,
                learned_prediction,
                mode_search=corrector_config,
                transport_config=transport_config,
            )
            correction_seconds = time.perf_counter() - correction_started
            correction_evaluations = _certificate_evaluations(certificate)
            if certificate.used_natural_fallback or certificate.corrected_modes is None:
                fallbacks += 1
                action_gap = None
                gradient_norm = None
                failures.append(
                    f"seed-{training_seed}/{problem.task_id}: correction fallback"
                )
            else:
                corrected_mode = certificate.corrected_modes.modes[0]
                action_gap = abs(corrected_mode.action_value - cold.action_value)
                gradient_norm = corrected_mode.gradient_norm
                if action_gap > float(gates["maximum_action_gap"]):
                    failures.append(
                        f"seed-{training_seed}/{problem.task_id}: action gate failed"
                    )
                if gradient_norm > float(gates["maximum_gradient_norm"]):
                    failures.append(
                        f"seed-{training_seed}/{problem.task_id}: gradient gate failed"
                    )
            evaluated = evaluate_rbergomi_cm_transport(
                problem,
                certificate.proposal,
                sample_count=int(config["evaluation_samples"]),
                path_seed=int(config["teacher_seed"]) + 1_000_003 + 1009 * index + replication,
                label_seed=int(config["teacher_seed"]) + 2_000_003 + 1009 * index + replication,
            )
            if evaluated.maximum_likelihood_bound_violation > float(
                gates["maximum_likelihood_bound_violation"]
            ):
                failures.append(
                    f"seed-{training_seed}/{problem.task_id}: likelihood bound failed"
                )
            speedup = cold.function_evaluations / max(correction_evaluations, 1)
            speedups.append(speedup)
            records.append(
                {
                    "task_id": problem.task_id,
                    "group": group,
                    "cold_action": cold.action_value,
                    "action_gap": action_gap,
                    "gradient_norm": gradient_norm,
                    "cold_function_evaluations": cold.function_evaluations,
                    "correction_function_evaluations": correction_evaluations,
                    "function_evaluation_speedup": speedup,
                    "cold_seconds": cold_seconds,
                    "correction_seconds": correction_seconds,
                    "learned_start_distance": learned_distance,
                    "shuffled_start_distance": shuffled_distance,
                    "used_natural_fallback": certificate.used_natural_fallback,
                    "maximum_likelihood_bound_violation": (
                        evaluated.maximum_likelihood_bound_violation
                    ),
                }
            )
        median_speedup = float(torch.median(torch.tensor(speedups, dtype=torch.float64)))
        distance_ratio = float(
            torch.median(torch.tensor(shuffled_distances, dtype=torch.float64))
            / torch.median(torch.tensor(learned_distances, dtype=torch.float64))
        )
        if median_speedup < float(
            gates["minimum_seed_median_function_evaluation_speedup"]
        ):
            failures.append(f"seed-{training_seed}: median speedup gate failed")
        if distance_ratio < float(
            gates["minimum_shuffled_over_learned_distance_ratio"]
        ):
            failures.append(f"seed-{training_seed}: shuffled-control gate failed")
        if fallbacks > int(gates["maximum_fallbacks"]):
            failures.append(f"seed-{training_seed}: fallback-count gate failed")
        replications.append(
            {
                "training_seed": int(training_seed),
                "learned_training": {
                    "initial_loss": learned_training.initial_loss,
                    "final_loss": learned_training.final_loss,
                    "seconds": learned_training_seconds,
                },
                "shuffled_training": {
                    "initial_loss": shuffled_training.initial_loss,
                    "final_loss": shuffled_training.final_loss,
                    "seconds": shuffled_training_seconds,
                },
                "median_function_evaluation_speedup": median_speedup,
                "shuffled_over_learned_median_distance_ratio": distance_ratio,
                "fallbacks": fallbacks,
                "holdouts": records,
            }
        )

    operator_values = config["operator"]
    output_features = rank + rank * (rank + 1) // 2 + 1
    forward_multiply_adds = (
        7 * int(operator_values["hidden_features"])
        + int(operator_values["hidden_features"]) ** 2
        + int(operator_values["hidden_features"]) * output_features
    )
    payload = {
        "schema": "npi.g11.v16-operator-confirmation.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": git_source_provenance(ROOT),
        "passed": not failures,
        "failures": failures,
        "teacher_bank": {
            "count": teacher_count,
            "rank": rank,
            "function_evaluations": teacher_function_evaluations,
            "seconds": teacher_seconds,
        },
        "operator_forward_multiply_adds_per_query": forward_multiply_adds,
        "replications": replications,
        "claim_boundary": {
            "operator_is_initializer_only": True,
            "operator_only_correction_measured": True,
            "deterministic_correction_required": True,
            "natural_fallback_exact": True,
            "exactness_independent_of_prediction": True,
            "teacher_and_training_cost_reported_separately": True,
            "global_amortized_cost_advantage_proved": False,
        },
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": not failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
