"""End-to-end V5 selection, freeze, allocation, and independent final execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
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


def _canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("qualification input must contain a JSON object")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload, hashlib.sha256(canonical).hexdigest()


def _verify_formal_preflight(config: dict[str, Any], *, smoke: bool) -> dict[str, bool]:
    if smoke:
        return {"development_smoke": True}
    run_class = config.get("run_class")
    if run_class not in {"qualification", "untouched-confirmatory"}:
        raise ValueError("non-smoke run_class must be qualification or untouched-confirmatory")
    if not bool(config.get("frozen")):
        raise ValueError("non-smoke confirmation requires a frozen config")
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise ValueError("non-smoke confirmation requires a clean worktree")
    if provenance["source_commit"] != config.get("source_commit"):
        raise ValueError("source commit does not match the frozen confirmation config")
    for item in config.get("qualification_inputs", []):
        path = Path(item["path"])
        payload, canonical_hash = _canonical_json(path)
        if canonical_hash != item["canonical_json_sha256"]:
            raise ValueError("qualification input hash mismatch")
        if not bool(payload.get(item["required_gate"])):
            raise ValueError("a required qualification gate is not satisfied")
    if run_class == "untouched-confirmatory":
        registry_value = config.get("final_seed_registry")
        if not registry_value:
            raise ValueError("untouched confirmation requires a final seed registry")
        registry = Path(registry_value)
        if registry.exists():
            raise ValueError("the frozen final seed namespace has already been instantiated")
    elif config.get("final_seed_registry"):
        raise ValueError("qualification must not instantiate the final seed namespace")
    return {
        "frozen_config": True,
        "clean_source": True,
        "source_commit_match": True,
        "qualification_inputs_passed": True,
        "run_class_valid": True,
    }


def _cell_reference(
    config: dict[str, Any],
    *,
    model_id: str,
    task_name: str,
    task_spec: dict[str, Any],
    smoke: bool,
) -> dict[str, float]:
    """Resolve a model/task-specific reference; formal runs forbid task fallbacks."""

    references = config.get("references", {})
    model_references = references.get(model_id, {}) if isinstance(references, dict) else {}
    reference = (
        model_references.get(task_name, {}) if isinstance(model_references, dict) else {}
    )
    if isinstance(reference, dict) and {
        "probability",
        "standard_error",
    }.issubset(reference):
        probability = float(reference["probability"])
        standard_error = float(reference["standard_error"])
    elif smoke and {"reference_probability", "reference_standard_error"}.issubset(task_spec):
        probability = float(task_spec["reference_probability"])
        standard_error = float(task_spec["reference_standard_error"])
    else:
        raise ValueError(f"missing qualified reference for {model_id}:{task_name}")
    if not (0.0 < probability < 1.0) or not (
        math.isfinite(standard_error) and standard_error >= 0.0
    ):
        raise ValueError(f"invalid qualified reference for {model_id}:{task_name}")
    return {"probability": probability, "standard_error": standard_error}


def _linear_quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (position - lower) * (
        sorted_values[upper] - sorted_values[lower]
    )


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute across-cluster accuracy, coverage, work, and selection summaries."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model_id"], record["task_name"]), []).append(record)
    summaries: list[dict[str, Any]] = []
    critical = NormalDist().inv_cdf(0.975)
    for (model_id, task_name), group in sorted(grouped.items()):
        completed = [
            record
            for record in group
            if record["result"]["complete"] and not record["result"]["resource_censored"]
        ]
        reference = group[0]["reference"]
        probability = float(reference["probability"])
        reference_se = float(reference["standard_error"])
        estimates = [float(record["result"]["estimate"]) for record in completed]
        squared_errors = [(estimate - probability) ** 2 for estimate in estimates]
        empirical_rmse = math.sqrt(math.fsum(squared_errors) / len(squared_errors)) if (
            squared_errors
        ) else None
        requested_relative = float(group[0]["result"]["requested_relative_sampling_rmse"])
        coverage = []
        bounded_coverage = []
        work = []
        selected_counts: dict[str, int] = {}
        empirical_attainment = []
        for record in completed:
            result = record["result"]
            combined_radius = critical * math.sqrt(
                float(result["empirical_sampling_variance"]) + reference_se**2
            )
            coverage.append(abs(float(result["estimate"]) - probability) <= combined_radius)
            bounded = result["bounded_confidence_interval"]
            bounded_coverage.append(float(bounded[0]) <= probability <= float(bounded[1]))
            work.append(
                math.fsum(float(item["work_units"]) for item in result["work"]["entries"])
            )
            candidate = str(result["selected_candidate"])
            selected_counts[candidate] = selected_counts.get(candidate, 0) + 1
            empirical_attainment.append(bool(result["empirical_target_attained"]))
        work.sort()

        summaries.append(
            {
                "model_id": model_id,
                "task_name": task_name,
                "clusters_planned": len(group),
                "clusters_complete": len(completed),
                "resource_censored": len(group) - len(completed),
                "reference": reference,
                "requested_relative_sampling_rmse": requested_relative,
                "empirical_rmse_against_reference": empirical_rmse,
                "empirical_relative_rmse_against_reference": (
                    empirical_rmse / probability if empirical_rmse is not None else None
                ),
                "empirical_target_attainment_fraction": (
                    sum(empirical_attainment) / len(empirical_attainment)
                    if empirical_attainment
                    else None
                ),
                "combined_asymptotic_95_coverage": (
                    sum(coverage) / len(coverage) if coverage else None
                ),
                "bounded_interval_coverage": (
                    sum(bounded_coverage) / len(bounded_coverage)
                    if bounded_coverage
                    else None
                ),
                "work_units_median": _linear_quantile(work, 0.5),
                "work_units_p90": _linear_quantile(work, 0.9),
                "selected_candidate_counts": selected_counts,
            }
        )
    return summaries


def _qualification_gates(
    records: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    *,
    expected_records: int,
    expected_cells: int,
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    """Evaluate full-run gates without allowing censoring or vacuous truth."""

    complete_matrix = len(records) == expected_records and expected_records > 0
    complete_aggregates = len(aggregates) == expected_cells and expected_cells > 0
    no_censoring = complete_matrix and all(
        not record["result"]["resource_censored"] for record in records
    )
    all_complete = complete_matrix and all(record["result"]["complete"] for record in records)
    return {
        "complete_cluster_matrix": complete_matrix and complete_aggregates,
        "no_resource_censoring": no_censoring,
        "all_runs_complete": all_complete,
        "all_design_targets_attained": all_complete
        and all(record["result"]["design_target_attained"] for record in records),
        "minimum_empirical_target_attainment": complete_aggregates
        and all(
            isinstance(summary["empirical_target_attainment_fraction"], (int, float))
            and summary["empirical_target_attainment_fraction"]
            >= float(thresholds["minimum_empirical_target_attainment"])
            for summary in aggregates
        ),
        "across_cluster_relative_rmse": complete_aggregates
        and all(
            isinstance(summary["empirical_relative_rmse_against_reference"], (int, float))
            and summary["empirical_relative_rmse_against_reference"]
            <= summary["requested_relative_sampling_rmse"]
            * float(thresholds["maximum_relative_rmse_ratio"])
            for summary in aggregates
        ),
        "minimum_combined_asymptotic_coverage": complete_aggregates
        and all(
            isinstance(summary["combined_asymptotic_95_coverage"], (int, float))
            and summary["combined_asymptotic_95_coverage"]
            >= float(thresholds["minimum_combined_asymptotic_coverage"])
            for summary in aggregates
        ),
        "selection_frozen_before_final": complete_matrix
        and all(
            record["selection"]["stopped"]
            and record["selection"]["frozen_decision"] is not None
            for record in records
        ),
        "no_final_samples_reused_from_selection": complete_matrix
        and all(record["seed_role_audit"]["selection_final_disjoint"] for record in records),
        "all_preparation_hashes_unique": complete_matrix
        and len({record["preparation"]["preparation_hash"] for record in records})
        == len(records),
    }


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
    preflight = _verify_formal_preflight(config, smoke=smoke)
    run_class = "development-smoke" if smoke else str(config["run_class"])
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
            reference = _cell_reference(
                config,
                model_id=model_id,
                task_name=task_name,
                task_spec=task_spec,
                smoke=smoke,
            )
            nominal_probability = float(reference["probability"])
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
                            nominal_probability
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
                    nominal_probability=nominal_probability,
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
                    reference_probability=nominal_probability,
                    reference_standard_error=float(reference["standard_error"]),
                )
                selection_seeds = {
                    record.seed for record in cell_ledger.records if record.key.role == "selection"
                }
                final_seeds = {
                    record.seed for record in cell_ledger.records if record.key.role == "final"
                }
                unexpected_seed_roles = sorted(
                    {record.key.role for record in cell_ledger.records}
                    - {"selection", "final"}
                )
                serialized_task = {
                    **task_spec,
                    "nominal_probability": nominal_probability,
                    "reference_probability": nominal_probability,
                    "reference_standard_error": float(reference["standard_error"]),
                }
                records.append(
                    {
                        "cell_id": f"{model_id}:{task_name}:cluster-{cluster}",
                        "model_id": model_id,
                        "task_name": task_name,
                        "model": model,
                        "task": serialized_task,
                        "reference": reference,
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
                        "seed_role_audit": {
                            "selection_seed_count": len(selection_seeds),
                            "final_seed_count": len(final_seeds),
                            "selection_final_disjoint": selection_seeds.isdisjoint(final_seeds)
                            and not unexpected_seed_roles,
                            "unexpected_roles": unexpected_seed_roles,
                        },
                    }
                )
                for seed_record in cell_ledger.records:
                    master_ledger.allocate(seed_record.key)
    aggregates = _aggregate_records(records)
    thresholds = config.get(
        "qualification_gates",
        {
            "minimum_empirical_target_attainment": 0.0,
            "maximum_relative_rmse_ratio": 10.0,
            "minimum_combined_asymptotic_coverage": 0.0,
        },
    )
    gates = _qualification_gates(
        records,
        aggregates,
        expected_records=len(model_specs) * len(task_items) * clusters,
        expected_cells=len(model_specs) * len(task_items),
        thresholds=thresholds,
    )
    provenance = source_provenance()
    formal_readiness = {
        "frozen_config": bool(config.get("frozen")) and not smoke,
        "clean_source": not bool(provenance["dirty_worktree"]),
        "source_commit_match": smoke
        or provenance["source_commit"] == config.get("source_commit"),
        "qualification_inputs_passed": bool(preflight),
        "non_smoke": not smoke,
    }
    protocol_complete = all(gates.values()) and (smoke or all(formal_readiness.values()))
    return {
        "schema": "npi.g11.v5-confirmatory.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "run_class": run_class,
        "records": records,
        "aggregates": aggregates,
        "gates": gates,
        "formal_readiness": formal_readiness,
        "protocol_complete": protocol_complete,
        "qualification_passed": run_class == "qualification" and protocol_complete,
        "seed_ledger_sha256": master_ledger.sha256,
        "seed_ledger": master_ledger.to_dict(),
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
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
