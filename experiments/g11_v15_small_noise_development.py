"""Empirical small-noise diagnostic; never promotes an observed slope to a theorem."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_drift_basis,
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.cameron_martin_modes import ActionSolverConfig, ModeSearchConfig
from src.path_integral.finite_rank_gaussian_transport import CurvatureTransportConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import (
    evaluate_rbergomi_cm_transport,
    train_rbergomi_cm_transport,
)
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    source_provenance = git_source_provenance(ROOT)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    problem = RBergomiBaselineProblem(
        task_id=str(config["task_id"]),
        task=TerminalThresholdTask(level=float(config["threshold"])),
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        steps=int(model["steps"]),
        hurst=float(model["hurst"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
    )
    records = []
    mode_basis_kind = str(config.get("mode_basis", "channel_dct"))
    if mode_basis_kind == "mesh_compatible_drift":
        mode_basis = build_mesh_compatible_blp_drift_basis(
            steps=problem.steps,
            maturity=problem.maturity,
            hurst=problem.hurst,
            modes=int(config["modes"]),
        )
    elif mode_basis_kind == "mesh_compatible_hybrid":
        mode_basis = build_mesh_compatible_blp_hybrid_basis(
            steps=problem.steps,
            maturity=problem.maturity,
            hurst=problem.hurst,
            drift_modes=int(config["modes"]),
            bridge_modes=int(config["bridge_modes"]),
        )
    elif mode_basis_kind == "channel_dct":
        mode_basis = None
    else:
        raise ValueError(
            "mode_basis must be channel_dct, mesh_compatible_drift, or mesh_compatible_hybrid"
        )
    for index, epsilon_value in enumerate(config["epsilons"]):
        epsilon = float(epsilon_value)
        trained = train_rbergomi_cm_transport(
            problem,
            epsilon=epsilon,
            modes_per_driver=int(config["modes_per_driver"]),
            basis=mode_basis,
            mode_search=ModeSearchConfig(
                methods=("lbfgs", "trust-ncg"),
                random_starts=int(config["random_starts"]),
                random_seed=int(config["seed"]) + 101 * index,
                start_scale=float(config["start_scale"]),
                solver=ActionSolverConfig(
                    maximum_iterations=int(config["maximum_iterations"]),
                    gradient_tolerance=float(config["gradient_tolerance"]),
                ),
            ),
            transport_config=CurvatureTransportConfig(
                defensive_mass=float(config["defensive_mass"]),
                asymptotic_safety_mass=float(
                    config.get("asymptotic_safety_mass", 0.0)
                ),
                safety_spectrum_decay=(
                    float(config["safety_spectrum_decay"])
                    if "safety_spectrum_decay" in config
                    else None
                ),
                safety_spectrum_scale=float(config.get("safety_spectrum_scale", 1.0)),
                safety_complement_decay=float(
                    config.get("safety_complement_decay", 2.0)
                ),
            ),
        )
        candidate = evaluate_rbergomi_cm_transport(
            problem,
            trained.proposal,
            sample_count=int(config["sample_count"]),
            path_seed=int(config["seed"]) + 101 * index + 1,
            label_seed=int(config["seed"]) + 101 * index + 2,
            epsilon=epsilon,
        )
        natural_latent = torch.randn(
            (int(config["reference_count"]), problem.local_dimension),
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(int(config["seed"]) + 101 * index + 3),
        )
        natural = evaluate_rbergomi_conditional_terminal(
            problem,
            natural_latent,
            epsilon=epsilon,
        ).payoffs.left_probability
        natural_probability = float(torch.mean(natural))
        natural_se = math.sqrt(float(torch.var(natural, unbiased=True)) / natural.numel())
        if "transport_reference_count" in config:
            transport_reference = evaluate_rbergomi_cm_transport(
                problem,
                trained.proposal,
                sample_count=int(config["transport_reference_count"]),
                path_seed=int(config["seed"]) + 101 * index + 4,
                label_seed=int(config["seed"]) + 101 * index + 5,
                epsilon=epsilon,
            )
            probability = float(torch.mean(transport_reference.contribution))
            reference_se = math.sqrt(
                float(torch.var(transport_reference.contribution, unbiased=True))
                / transport_reference.contribution.numel()
            )
            reference_method = "independent_frozen_v15_ordinary_is"
        else:
            probability = natural_probability
            reference_se = natural_se
            reference_method = "natural_conditional_mc"
        candidate_mean = float(torch.mean(candidate.contribution))
        candidate_variance = float(torch.var(candidate.contribution, unbiased=True))
        candidate_se = math.sqrt(candidate_variance / candidate.contribution.numel())
        second_moment = candidate_variance + candidate_mean**2
        squared_contribution = candidate.contribution.square()
        total_squared_contribution = float(torch.sum(squared_contribution))
        component_diagnostics = []
        for component_index, component_weight in enumerate(trained.proposal.weights):
            selected = candidate.component_labels == component_index
            selected_count = int(torch.sum(selected))
            component_diagnostics.append(
                {
                    "component_index": component_index,
                    "configured_weight": float(component_weight),
                    "sample_count": selected_count,
                    "sample_fraction": selected_count / candidate.contribution.numel(),
                    "second_moment_share": (
                        float(torch.sum(squared_contribution[selected]))
                        / max(total_squared_contribution, 1e-300)
                    ),
                    "mean_conditional_probability": float(
                        torch.mean(candidate.conditional_probability[selected])
                    ),
                    "mean_likelihood": float(torch.mean(candidate.likelihood[selected])),
                }
            )
        records.append(
            {
                "epsilon": epsilon,
                "reference_probability": probability,
                "reference_standard_error": reference_se,
                "reference_method": reference_method,
                "natural_conditional_estimate": natural_probability,
                "natural_conditional_standard_error": natural_se,
                "candidate_estimate": candidate_mean,
                "candidate_standard_error": candidate_se,
                "combined_accuracy_z": abs(candidate_mean - probability)
                / max(math.sqrt(reference_se**2 + candidate_se**2), 1e-300),
                "relative_variance": candidate_variance / max(candidate_mean**2, 1e-300),
                "negative_epsilon_log_probability": -epsilon * math.log(probability),
                "second_moment_exponent_ratio": math.log(second_moment)
                / (2.0 * math.log(probability)),
                "best_action": trained.modes.modes[0].action_value,
                "mode_count": len(trained.modes.modes),
                "asymptotic_safety_mass": float(
                    config.get("asymptotic_safety_mass", 0.0)
                ),
                "safety_spectrum_decay": (
                    float(config["safety_spectrum_decay"])
                    if "safety_spectrum_decay" in config
                    else None
                ),
                "safety_complement_decay": float(
                    config.get("safety_complement_decay", 2.0)
                ),
                "proposal_sha256": trained.proposal_sha256,
                "component_diagnostics": component_diagnostics,
                "maximum_likelihood_bound_violation": (
                    candidate.maximum_likelihood_bound_violation
                ),
                "training_cost": asdict(trained.training_cost),
                "evaluation_cost": asdict(candidate.evaluation_cost),
            }
        )
    payload = {
        "schema": "npi.g11.v15-small-noise-diagnostic.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": source_provenance,
        "claim_boundary": {
            "empirical_diagnostic_only": True,
            "fixed_grid_probability_exponent_proved": True,
            "fixed_grid_asymptotic_efficiency_proved": bool(
                float(config.get("asymptotic_safety_mass", 0.0)) > 0.0
            ),
            "continuous_terminal_probability_exponent_proved_under_T16_4_scope": True,
            "continuous_trace_safety_proposal_efficiency_proved_under_T16_5_scope": bool(
                float(config.get("asymptotic_safety_mass", 0.0)) > 0.0
                and "safety_spectrum_decay" in config
            ),
            "joint_mesh_noise_asymptotic_efficiency_proved_under_T16_9_scope": bool(
                float(config.get("asymptotic_safety_mass", 0.0)) > 0.0
                and "safety_spectrum_decay" in config
            ),
            "joint_mesh_noise_work_complexity_proved": False,
        },
        "mode_basis": {
            "kind": mode_basis_kind,
            "rank": int(mode_basis.rank) if mode_basis is not None else 2 * int(config["modes_per_driver"]),
        },
        "records": records,
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
