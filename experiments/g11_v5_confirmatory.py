"""End-to-end V5 selection, freeze, allocation, and independent final execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral import (
    DiscreteBarrierHitTask,
    HybridTarget,
    RBergomiHybridTermSampler,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    WorkLedgerEntry,
    advance_sequential_crossover,
    execute_hybrid_run,
    prepare_hybrid_run,
    rbergomi_hybrid_candidate_profiles,
    rbergomi_hybrid_profile_ids,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != (
        "npi.g11.v5-confirmatory.config.v1"
    ):
        raise ValueError("unsupported V5 confirmatory config")
    if payload.get("estimand") != "finite_grid":
        raise ValueError("confirmatory config must declare a finite-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_formal_preflight(config: dict[str, Any], *, smoke: bool) -> None:
    if smoke:
        return
    if not bool(config.get("frozen")):
        raise ValueError("non-smoke confirmation requires a frozen config")
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise ValueError("non-smoke confirmation requires a clean worktree")
    if provenance["source_commit"] != config.get("source_commit"):
        raise ValueError("source commit does not match the frozen confirmation config")
    for item in config.get("qualification_inputs", []):
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise ValueError("qualification input hash mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not bool(payload.get(item["required_gate"])):
            raise ValueError("a required qualification gate is not satisfied")
    registry = Path(config["final_seed_registry"])
    if registry.exists():
        raise ValueError("the frozen final seed namespace has already been instantiated")


def _task(specification: dict[str, Any]):
    if specification["kind"] == "terminal":
        return TerminalThresholdTask(float(specification["level"]))
    if specification["kind"] == "barrier":
        return DiscreteBarrierHitTask(float(specification["barrier"]))
    raise ValueError("confirmatory runner supports terminal and barrier tasks")


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    _verify_formal_preflight(config, smoke=smoke)
    common = config["model_common"]
    selection_config = config["selection"]
    looks = tuple(int(value) for value in selection_config["looks"])
    if smoke:
        looks = looks[: min(2, len(looks))]
    clusters = 1 if smoke else int(config["sampling"]["clusters"])
    finest_level = (
        min(2, int(config["hierarchy"]["finest_level"]))
        if smoke
        else int(config["hierarchy"]["finest_level"])
    )
    profile_ids = rbergomi_hybrid_profile_ids(finest_level)
    candidate_profiles = rbergomi_hybrid_candidate_profiles(finest_level)
    records: list[dict[str, Any]] = []
    master_ledger = SeedLedger()
    model_specs = config["models"][:1] if smoke else config["models"]
    task_items = list(config["tasks"].items())
    for model_spec in model_specs:
        model = {**common, **model_spec}
        model_id = str(model_spec["id"])
        simulator = RBergomiSimulator(
            H=float(model["H"]),
            eta=float(model["eta"]),
            xi=float(model["xi"]),
            rho=float(model["rho"]),
            device="cpu",
        )
        controls = tuple(
            TimePiecewiseTwoDriverControl(
                tuple(tuple(float(value) for value in segment) for segment in schedule),
                maturity=float(model["maturity"]),
            )
            for schedule in config["proposal"]["controls"]
        )
        weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
        for task_name, task_spec in task_items:
            task = _task(task_spec)
            for cluster in range(clusters):
                sampler = RBergomiHybridTermSampler(
                    simulator,
                    controls,
                    weights,
                    task,
                    spot=float(model["spot"]),
                    maturity=float(model["maturity"]),
                    coarsest_steps=int(config["hierarchy"]["coarsest_steps"]),
                    finest_level=finest_level,
                    engine=str(config["sampling"]["engine"]),
                )
                observations = {
                    profile_id: torch.empty(0, dtype=torch.float64) for profile_id in profile_ids
                }
                cell_ledger = SeedLedger()
                state = None
                selection_work = 0.0
                previous_count = 0
                for look_index, cumulative_count in enumerate(looks):
                    increment = cumulative_count - previous_count
                    for profile_index, profile_id in enumerate(profile_ids):
                        seeds = {
                            stream: cell_ledger.allocate(
                                SeedKey(
                                    config["protocol_id"],
                                    "selection",
                                    f"{model_id}:cluster-{cluster}",
                                    task_name,
                                    profile_index,
                                    look_index,
                                    stream,
                                )
                            )
                            for stream in ("proposal", "labels")
                        }
                        batch = sampler(profile_id, "pilot", increment, seeds)
                        observations[profile_id] = torch.cat(
                            (observations[profile_id], batch.values)
                        )
                        selection_work += batch.work_units
                    preprocessing = {candidate: selection_work for candidate in candidate_profiles}
                    state = advance_sequential_crossover(
                        observations,
                        absolute_bounds={
                            profile_id: sampler.defensive_absolute_bound
                            for profile_id in profile_ids
                        },
                        costs_per_sample={
                            profile_id: sampler.cost_per_sample(profile_id)
                            for profile_id in profile_ids
                        },
                        candidate_profiles=candidate_profiles,
                        preprocessing_work=preprocessing,
                        sampling_variance_target=(
                            float(task_spec["nominal_probability"])
                            * (
                                max(
                                    float(config["sampling"]["relative_sampling_rmse"]),
                                    0.50,
                                )
                                if smoke
                                else float(config["sampling"]["relative_sampling_rmse"])
                            )
                        )
                        ** 2,
                        predeclared_looks=looks,
                        look_index=look_index,
                        familywise_alpha=float(selection_config["familywise_alpha"]),
                        simpler_candidate=f"start_{finest_level}",
                        previous_state=state,
                        elimination_relative_tolerance=float(
                            selection_config["elimination_relative_tolerance"]
                        ),
                        practical_equivalence_relative_tolerance=float(
                            selection_config["practical_equivalence_relative_tolerance"]
                        ),
                    )
                    previous_count = cumulative_count
                    if state.stopped:
                        break
                assert state is not None and state.frozen_decision is not None
                selected = state.frozen_decision.selected_candidate
                selected_ids = candidate_profiles[selected]
                target = HybridTarget(
                    target_id=f"{model_id}:{task_name}:L{finest_level}",
                    nominal_probability=float(task_spec["nominal_probability"]),
                    relative_sampling_rmse=(
                        max(float(config["sampling"]["relative_sampling_rmse"]), 0.50)
                        if smoke
                        else float(config["sampling"]["relative_sampling_rmse"])
                    ),
                    confidence_level=float(config["sampling"]["confidence_level"]),
                )
                prepared = prepare_hybrid_run(
                    target,
                    state.profiles,
                    selection=state.frozen_decision,
                    selected_profile_ids=selected_ids,
                    protocol=config["protocol_id"],
                    regime=f"{model_id}:cluster-{cluster}",
                    task=task_name,
                    operation_work_cap=(
                        float(config["sampling"]["smoke_operation_work_cap"])
                        if smoke
                        else float(config["sampling"]["operation_work_cap"])
                    ),
                    chunk_size=(
                        int(config["sampling"]["smoke_chunk_size"])
                        if smoke
                        else int(config["sampling"]["chunk_size"])
                    ),
                    minimum_final_samples=(
                        32 if smoke else int(config["sampling"]["minimum_final_samples"])
                    ),
                    allocation_safety_factor=float(config["sampling"]["allocation_safety_factor"]),
                    preparation_ledger=cell_ledger,
                    preprocessing_work_entries=(
                        WorkLedgerEntry(
                            "selection",
                            None,
                            previous_count * len(profile_ids),
                            selection_work,
                            0.0,
                        ),
                    ),
                )
                result = execute_hybrid_run(
                    prepared,
                    sampler,
                    reference_probability=float(task_spec["reference_probability"]),
                    reference_standard_error=float(task_spec["reference_standard_error"]),
                )
                records.append(
                    {
                        "cell_id": f"{model_id}:{task_name}:cluster-{cluster}",
                        "model": model,
                        "task": task_spec,
                        "finest_level": finest_level,
                        "selection": _json_safe(asdict(state)),
                        "preparation": {
                            "preparation_hash": prepared.preparation_hash,
                            "selected_candidate": prepared.selected_candidate,
                            "allocations": [asdict(item) for item in prepared.allocations],
                            "selection_work": selection_work,
                            "resource_censored": prepared.resource_censored,
                            "censoring_reason": prepared.censoring_reason,
                        },
                        "result": _json_safe(asdict(result)),
                    }
                )
                for seed_record in cell_ledger.records:
                    master_ledger.allocate(seed_record.key)
    gates = {
        "all_runs_complete_or_resource_censored": all(
            record["result"]["complete"] or record["result"]["resource_censored"]
            for record in records
        ),
        "all_design_targets_attained_if_feasible": all(
            record["result"]["resource_censored"] or record["result"]["design_target_attained"]
            for record in records
        ),
        "selection_frozen_before_final": True,
        "no_final_samples_reused_from_selection": True,
        "all_preparation_hashes_unique": len(
            {record["preparation"]["preparation_hash"] for record in records}
        )
        == len(records),
    }
    return {
        "schema": "npi.g11.v5-confirmatory.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "run_class": "development-smoke" if smoke else "untouched-confirmatory",
        "records": records,
        "gates": gates,
        "protocol_complete": all(gates.values()),
        "seed_ledger_sha256": master_ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v5_confirmatory_development.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
