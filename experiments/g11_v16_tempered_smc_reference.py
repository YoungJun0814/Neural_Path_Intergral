"""Proposal-independent tempered-SMC reference for V16 terminal probabilities."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.tempered_conditional_smc import (
    TemperedSMCConfig,
    estimate_tempered_normalizer,
)
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
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
    stages = int(config["temperature_stages"])
    power = float(config["temperature_power"])
    temperatures = tuple((index / stages) ** power for index in range(stages + 1))
    records = []
    for index, epsilon_value in enumerate(config["epsilons"]):
        epsilon = float(epsilon_value)

        def log_potential(
            points: torch.Tensor,
            *,
            resolved_epsilon: float = epsilon,
        ) -> torch.Tensor:
            return evaluate_rbergomi_conditional_terminal(
                problem,
                points,
                epsilon=resolved_epsilon,
            ).payoffs.log_left_probability

        result = estimate_tempered_normalizer(
            log_potential,
            dimension=problem.local_dimension,
            config=TemperedSMCConfig(
                particles=int(config["particles"]),
                temperatures=temperatures,
                mutation_steps=int(config["mutation_steps"]),
                pcn_scale=float(config["pcn_scale"]),
                replicates=int(config["replicates"]),
                seed=int(config["seed"]) + 10_007 * index,
            ),
        )
        records.append(
            {
                "epsilon": epsilon,
                "estimate": result.mean,
                "standard_error": result.standard_error,
                "negative_epsilon_log_probability": -epsilon * math.log(result.mean),
                "mutation_acceptance_rate": result.mutation_acceptance_rate,
                "potential_evaluations": result.potential_evaluations,
                "replicate_estimates": [float(value) for value in result.replicate_estimates],
            }
        )
    payload = {
        "schema": "npi.g11.v16-tempered-smc-reference.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": git_source_provenance(ROOT),
        "method_boundary": {
            "proposal_family": "tempered_smc_with_pcn_mutation",
            "uses_v16_gaussian_transport": False,
            "deterministic_temperature_schedule": True,
            "normalizing_constant_estimator_unbiased": True,
            "replicates_independent": True,
        },
        "records": records,
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
