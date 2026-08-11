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
    for index, epsilon_value in enumerate(config["epsilons"]):
        epsilon = float(epsilon_value)
        trained = train_rbergomi_cm_transport(
            problem,
            epsilon=epsilon,
            modes_per_driver=int(config["modes_per_driver"]),
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
                "proposal_sha256": trained.proposal_sha256,
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
            "continuous_large_deviation_principle_proved": False,
            "mesh_uniform_asymptotic_efficiency_proved": False,
        },
        "records": records,
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
