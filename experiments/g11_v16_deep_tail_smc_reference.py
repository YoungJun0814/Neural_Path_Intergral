"""Independent tempered-SMC references for the V16 deep-tail cells."""

from __future__ import annotations

import argparse
import json
import time
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
    stages = int(config["temperature_stages"])
    power = float(config["temperature_power"])
    temperatures = tuple((index / stages) ** power for index in range(stages + 1))
    gates = config["gates"]
    failures = []
    records = []
    for index, cell in enumerate(config["cells"]):
        problem = RBergomiBaselineProblem(
            task_id=str(cell["cell_id"]),
            task=TerminalThresholdTask(level=float(cell["threshold"])),
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            steps=int(model["steps"]),
            hurst=float(cell.get("hurst", model["hurst"])),
            eta=float(cell.get("eta", model["eta"])),
            xi=float(cell.get("xi", model["xi"])),
            rho=float(cell.get("rho", model["rho"])),
        )

        def log_potential(
            points: torch.Tensor,
            *,
            resolved_problem: RBergomiBaselineProblem = problem,
        ) -> torch.Tensor:
            return evaluate_rbergomi_conditional_terminal(
                resolved_problem,
                points,
            ).payoffs.log_left_probability

        started = time.perf_counter()
        result = estimate_tempered_normalizer(
            log_potential,
            dimension=problem.local_dimension,
            config=TemperedSMCConfig(
                particles=int(config["particles"]),
                temperatures=temperatures,
                mutation_steps=int(config["mutation_steps"]),
                pcn_scale=float(config["pcn_scale"]),
                replicates=int(config["replicates"]),
                seed=int(config["seed"]) + 100_003 * index,
            ),
        )
        seconds = time.perf_counter() - started
        signal_to_noise = result.mean / result.standard_error
        if signal_to_noise < float(gates["minimum_reference_signal_to_noise"]):
            failures.append(f"{problem.task_id}: reference signal-to-noise gate failed")
        if result.minimum_incremental_ess_fraction < float(
            gates["minimum_incremental_ess_fraction"]
        ):
            failures.append(f"{problem.task_id}: incremental ESS gate failed")
        if not (
            float(gates["minimum_mutation_acceptance_rate"])
            <= result.mutation_acceptance_rate
            <= float(gates["maximum_mutation_acceptance_rate"])
        ):
            failures.append(f"{problem.task_id}: mutation acceptance gate failed")
        records.append(
            {
                "cell_id": problem.task_id,
                "threshold": float(cell["threshold"]),
                "estimate": result.mean,
                "standard_error": result.standard_error,
                "signal_to_noise": signal_to_noise,
                "mutation_acceptance_rate": result.mutation_acceptance_rate,
                "minimum_incremental_ess_fraction": (
                    result.minimum_incremental_ess_fraction
                ),
                "potential_evaluations": result.potential_evaluations,
                "seconds": seconds,
                "replicate_estimates": [
                    float(value) for value in result.replicate_estimates
                ],
            }
        )
    payload = {
        "schema": "npi.g11.v16-deep-tail-smc-reference.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": git_source_provenance(ROOT),
        "passed": not failures,
        "failures": failures,
        "method_boundary": {
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
    print(json.dumps({"output": str(output), "passed": not failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
