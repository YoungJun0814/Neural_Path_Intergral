"""Execute the outcome-free B1 production-code implementation smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from src.path_integral.baselines import (
    CEMTrainingConfig,
    FlowTrainingConfig,
    LargeDeviationTrainingConfig,
    RBergomiBaselineProblem,
    freeze_conditional_rbergomi_proposal,
    freeze_crude_or_antithetic_proposal,
    freeze_smoothing_rqmc_proposal,
    train_cem_proposal,
    train_coupling_flow_proposal,
    train_large_deviation_proposal,
)
from src.path_integral.benchmark_executor import (
    BaselineExecutionRequest,
    execute_baseline_lifecycle,
)
from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    TerminalThresholdTask,
)
from src.path_integral.provenance import runtime_provenance, source_provenance

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-b1-baseline-implementation.v1"
RESULT_SCHEMA = "npi.g11.v8-b1-baseline-implementation-result.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected B1 implementation config schema")
    return config, hashlib.sha256(raw).hexdigest()


def _validate_bindings(config: dict[str, Any]) -> None:
    for key in ("completion_status_ledger_v2", "matrix_design", "threshold_calibration"):
        binding = config.get(key)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid {key} binding")
        relative = binding["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or "\\" in relative:
            raise ValueError(f"nonportable {key} path")
        target = ROOT / relative
        if not target.is_file() or binding["sha256"] != _sha256(target):
            raise ValueError(f"{key} binding hash mismatch")


def _problems(config: dict[str, Any]) -> list[RBergomiBaselineProblem]:
    model = config["model"]
    calibration = json.loads((ROOT / config["threshold_calibration"]["path"]).read_text())
    thresholds = {item["cell_id"]: item["calibrated_threshold"] for item in calibration["cells"]}
    result: list[RBergomiBaselineProblem] = []
    for cell in config["cells"]:
        cell_id = cell["cell_id"]
        threshold = float(cell["threshold"])
        if cell_id not in thresholds or threshold != float(thresholds[cell_id]):
            raise ValueError(f"threshold is not calibration-bound for {cell_id}")
        task: TerminalThresholdTask | DiscreteBarrierHitTask
        if cell["task"] == "terminal_left_tail":
            task = TerminalThresholdTask(threshold)
        elif cell["task"] == "discrete_lower_barrier":
            task = DiscreteBarrierHitTask(threshold)
        else:
            raise ValueError("unsupported B1 task")
        result.append(
            RBergomiBaselineProblem(
                task_id=cell_id,
                task=task,
                spot=float(model["spot"]),
                maturity=float(model["maturity"]),
                steps=int(model["steps"]),
                hurst=float(model["hurst"]),
                eta=float(model["eta"]),
                xi=float(model["xi"]),
                rho=float(model["rho"]),
            )
        )
    return result


def _proposal(problem: RBergomiBaselineProblem, method: str, seed: int, config: dict[str, Any]):
    training = config["training"]
    if method in {"crude_mc", "antithetic_mc"}:
        return freeze_crude_or_antithetic_proposal(
            problem,
            method=cast(Literal["crude_mc", "antithetic_mc"], method),
            training_seed=seed,
        )
    if method == "conditional_rbergomi":
        return freeze_conditional_rbergomi_proposal(problem, training_seed=seed)
    if method == "smoothing_rqmc":
        return freeze_smoothing_rqmc_proposal(problem, training_seed=seed)
    if method in {"pure_cem", "defensive_cem"}:
        return train_cem_proposal(
            problem,
            method=cast(Literal["pure_cem", "defensive_cem"], method),
            training_seed=seed,
            config=CEMTrainingConfig(**training["cem"]),
        )
    if method == "ld_subspace_is":
        return train_large_deviation_proposal(
            problem,
            training_seed=seed,
            config=LargeDeviationTrainingConfig(**training["large_deviation"]),
        )
    if method == "flow_is":
        return train_coupling_flow_proposal(
            problem,
            training_seed=seed,
            config=FlowTrainingConfig(**training["flow"]),
        )
    raise ValueError(f"unsupported B1 method: {method}")


def run(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    _validate_bindings(config)
    methods = config["methods"]
    execution = config["execution"]
    base_seed = int(execution["base_seed"])
    records: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    conditional_barrier_rejected = False
    run_index = 0
    for problem in _problems(config):
        barrier = isinstance(problem.task, DiscreteBarrierHitTask)
        for method in methods:
            if barrier and method == "conditional_rbergomi":
                try:
                    freeze_conditional_rbergomi_proposal(problem, training_seed=base_seed)
                except ValueError:
                    conditional_barrier_rejected = True
                else:
                    raise AssertionError("conditional rBergomi accepted a barrier task")
                continue
            training_seed = base_seed + 3 * run_index
            pilot_seed = training_seed + 1
            final_seed = training_seed + 2
            seeds = {training_seed, pilot_seed, final_seed}
            if len(seeds) != 3 or used_seeds & seeds:
                raise AssertionError("B1 global seed collision")
            used_seeds.update(seeds)
            proposal = _proposal(problem, method, training_seed, config)
            rqmc_points = (
                int(execution["rqmc_points_per_randomization"]) if method == "smoothing_rqmc" else 1
            )
            artifact = execute_baseline_lifecycle(
                problem,
                proposal,
                BaselineExecutionRequest(
                    pilot_units=int(execution["pilot_units"]),
                    target_estimator_variance=float(execution["target_estimator_variance"]),
                    pilot_seed=pilot_seed,
                    final_seed=final_seed,
                    minimum_final_units=int(execution["minimum_final_units"]),
                    maximum_final_units=int(execution["maximum_final_units"]),
                    rqmc_points_per_randomization=rqmc_points,
                ),
            )
            records.append(
                {
                    "cell_id": problem.task_id,
                    "method": method,
                    "artifact": asdict(artifact),
                }
            )
            run_index += 1

    gate = config["gate"]
    checks = {
        "run_count_exact": len(records) == int(gate["expected_run_count"]),
        "global_seeds_disjoint": len(used_seeds) == 3 * len(records),
        "conditional_barrier_rejected": conditional_barrier_rejected,
        "all_lifecycle_audits_pass": all(
            item["artifact"]["audit"]["passed"] is True for item in records
        ),
        "all_exact_likelihoods": all(
            item["artifact"]["proposal"]["exact_likelihood"] is True
            and item["artifact"]["proposal"]["self_normalized"] is False
            for item in records
        ),
        "all_estimates_finite_nonnegative": all(
            math.isfinite(item["artifact"]["estimate"]["estimate"])
            and item["artifact"]["estimate"]["estimate"] >= 0.0
            for item in records
        ),
        "all_final_costs_positive": all(
            item["artifact"]["estimate"]["final_cost"]["final_samples"] > 0
            and item["artifact"]["estimate"]["final_cost"]["algorithmic_work_units"] > 0.0
            and item["artifact"]["estimate"]["final_cost"]["wall_seconds"] > 0.0
            and item["artifact"]["estimate"]["final_cost"]["peak_memory_bytes"] > 0
            for item in records
        ),
        "performance_claim_refused": config.get("performance_claim_authorized") is False
        and gate.get("performance_claim_authorized") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "records": records,
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "b1_implementation_complete": not failures,
            "d1_falsification_authorized": not failures,
            "performance_claim_authorized": False,
            "p8_qualification_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config, digest = _load_config(args.config)
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], **result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
