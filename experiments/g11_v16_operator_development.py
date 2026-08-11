"""Teacher-bank and OOD corrector study for the regular-regime V16 operator."""

from __future__ import annotations

import argparse
import json
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


def _problem(task_id: str, values: dict[str, float], *, steps: int) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id=task_id,
        task=TerminalThresholdTask(level=100.0 * values["strike_ratio"]),
        spot=100.0,
        maturity=values["maturity"],
        steps=steps,
        hurst=values["hurst"],
        eta=values["eta"],
        xi=values["xi"],
        rho=values["rho"],
    )


def _random_tasks(config: dict[str, Any]) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(int(config["seed"]))
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
    for index, values in enumerate(_random_tasks(config)):
        problem = _problem(f"teacher-{index}", values, steps=int(config["steps"]))
        action = _action(problem)
        modes = find_rbergomi_conditional_modes(
            action,
            config=ModeSearchConfig(
                methods=("lbfgs",),
                random_starts=1,
                random_seed=int(config["seed"]) + 1009 * index,
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
    feature_tensor = torch.stack(features)
    coefficient_tensor = torch.stack(coefficients).unsqueeze(1)
    precision_tensor = torch.stack(precisions).unsqueeze(1)
    teacher_count = len(features)
    rank = coefficient_tensor.shape[2]
    teachers = VolterraOperatorTeachers(
        features=feature_tensor,
        coefficients=coefficient_tensor,
        mode_weights=torch.ones((teacher_count, 1), dtype=torch.float64),
        precision_matrices=precision_tensor,
        mode_mask=torch.ones((teacher_count, 1), dtype=torch.bool),
    )
    operator_values = config["operator"]
    model = VolterraTransportOperator(
        VolterraTransportOperatorConfig(
            feature_dimension=7,
            modes=1,
            rank=rank,
            hidden_features=int(operator_values["hidden_features"]),
        )
    )
    training = train_volterra_transport_operator(
        model,
        teachers,
        seed=int(config["seed"]) + 1,
        config=VolterraOperatorTrainingConfig(
            epochs=int(operator_values["epochs"]),
            learning_rate=float(operator_values["learning_rate"]),
        ),
    )
    holdout_records = []
    failures = []
    speedups = []
    transport_config = CurvatureTransportConfig(
        defensive_mass=0.15,
        asymptotic_safety_mass=0.02,
        safety_spectrum_decay=2.0,
        safety_spectrum_scale=4.0,
    )
    for index, raw_values in enumerate(config["holdouts"]):
        values = {key: float(value) for key, value in raw_values.items() if key != "id"}
        task_id = str(raw_values["id"])
        problem = _problem(task_id, values, steps=int(config["steps"]))
        action = _action(problem)
        feature = encode_rbergomi_transport_task(problem, epsilon=1.0)
        prediction = model(feature.unsqueeze(0))
        predicted_start = prediction.coefficients[0, 0].detach()
        zero = optimize_action(action, torch.zeros(rank, dtype=torch.float64), config=solver)
        predicted = optimize_action(action, predicted_start, config=solver)
        action_gap = abs(predicted.action_value - zero.action_value)
        speedup = zero.function_evaluations / max(predicted.function_evaluations, 1)
        speedups.append(speedup)
        certificate = correct_and_build_operator_transport(
            action,
            prediction,
            mode_search=ModeSearchConfig(
                methods=("lbfgs",),
                random_starts=0,
                include_zero_start=False,
                random_seed=int(config["seed"]) + 100_003 + 1009 * index,
                start_scale=2.0,
                solver=solver,
            ),
            transport_config=transport_config,
        )
        if certificate.used_natural_fallback:
            failures.append(f"{task_id}: operator correction used fallback")
        evaluated = evaluate_rbergomi_cm_transport(
            problem,
            certificate.proposal,
            sample_count=int(config["evaluation_samples"]),
            path_seed=int(config["seed"]) + 200_003 + 2 * index,
            label_seed=int(config["seed"]) + 200_004 + 2 * index,
        )
        if action_gap > float(config["gates"]["maximum_action_gap"]):
            failures.append(f"{task_id}: corrected action differs from zero-start oracle")
        if predicted.gradient_norm > float(config["gates"]["maximum_gradient_norm"]):
            failures.append(f"{task_id}: corrected gradient gate failed")
        if evaluated.maximum_likelihood_bound_violation > float(
            config["gates"]["maximum_likelihood_bound_violation"]
        ):
            failures.append(f"{task_id}: likelihood bound failed")
        holdout_records.append(
            {
                "task_id": task_id,
                "features": [float(value) for value in feature],
                "predicted_initial_action": float(action(predicted_start)),
                "zero_initial_action": float(action(torch.zeros(rank, dtype=torch.float64))),
                "corrected_action": predicted.action_value,
                "zero_start_action": zero.action_value,
                "action_gap": action_gap,
                "predicted_function_evaluations": predicted.function_evaluations,
                "zero_function_evaluations": zero.function_evaluations,
                "function_evaluation_speedup": speedup,
                "predicted_gradient_norm": predicted.gradient_norm,
                "used_natural_fallback": certificate.used_natural_fallback,
                "maximum_likelihood_bound_violation": (
                    evaluated.maximum_likelihood_bound_violation
                ),
            }
        )
    median_speedup = float(torch.median(torch.tensor(speedups, dtype=torch.float64)))
    if median_speedup < float(
        config["gates"]["minimum_median_function_evaluation_speedup"]
    ):
        failures.append("median function-evaluation speedup gate failed")
    operator_training_work = (
        int(operator_values["epochs"])
        * teacher_count
        * (
            7 * int(operator_values["hidden_features"])
            + int(operator_values["hidden_features"])
            * (rank + rank * (rank + 1) // 2 + 1)
        )
    )
    payload = {
        "schema": "npi.g11.v16-operator-development.v1",
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
        },
        "operator_training": {
            "initial_loss": training.initial_loss,
            "final_loss": training.final_loss,
            "work_units": operator_training_work,
        },
        "holdouts": holdout_records,
        "median_function_evaluation_speedup": median_speedup,
        "claim_boundary": {
            "operator_is_initializer_only": True,
            "deterministic_correction_required": True,
            "exactness_independent_of_prediction": True,
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
