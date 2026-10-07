"""Run the V15 conditional adjacent-mesh falsification study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_mesh import run_conditional_mesh_study
from src.path_integral.rbergomi_cm_mlmc import MLMCLevelPilot, allocate_pilot_frozen_mlmc
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance

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
        steps=int(config["steps"][0]),
        hurst=float(model["hurst"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
    )
    study = run_conditional_mesh_study(
        problem,
        steps=tuple(int(value) for value in config["steps"]),
        sample_count=int(config["sample_count"]),
        seed=int(config["seed"]),
        batch_size=int(config.get("batch_size", config["sample_count"])),
    )
    level_zero = study.levels[0]
    pilots = [
        MLMCLevelPilot(
            level=0,
            mean=level_zero.mean,
            variance=level_zero.variance,
            work_per_sample=float(study.steps[0]),
        )
    ]
    pilots.extend(
        MLMCLevelPilot(
            level=index,
            mean=pair.correction.mean,
            variance=pair.correction.variance,
            work_per_sample=float(pair.fine_steps + pair.coarse_steps),
        )
        for index, pair in enumerate(study.adjacent, start=1)
    )
    allocation = allocate_pilot_frozen_mlmc(
        tuple(pilots),
        target_rmse=float(config["target_rmse"]),
        bias_multiplier=float(config["bias_multiplier"]),
    )
    payload = {
        "schema": "npi.g11.v15-mesh-study.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": source_provenance,
        "claim_boundary": {
            "empirical_mesh_diagnostic": True,
            "continuous_rate_proved": False,
            "transport_mesh_convergence_proved": False,
        },
        "study": asdict(study),
        "pilot_frozen_mlmc_allocation": asdict(allocation),
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "bias_gate_pass": allocation.bias_gate_pass}, indent=2))


if __name__ == "__main__":
    main()
