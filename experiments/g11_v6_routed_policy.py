"""End-to-end V6 rarity routing, capped selection, and achieved-RMSE execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from experiments.g11_v6_baseline_qualification import (
    _design_from_pilot,
    _DirectRBergomiSampler,
    _load_references,
    _smoke_cells,
    _task,
    _work_record,
)
from experiments.g11_v6_reference import _load_manifest
from src.path_integral import (
    FrozenCrossoverDecision,
    HybridProfileOpportunity,
    HybridTarget,
    RarityRouterConfig,
    RBergomiHybridTermSampler,
    RoutingWorkInterval,
    SeedKey,
    SeedLedger,
    SingleTermDesign,
    TimePiecewiseTwoDriverControl,
    V6ProgressJournal,
    V6WorkLedger,
    advance_sequential_crossover,
    audit_v6_policy,
    conservative_bernoulli_variance_upper,
    exact_binomial_probability_interval,
    execute_v6_policy,
    execute_v6_policy_durable,
    freeze_rarity_route,
    load_v6_progress,
    prepare_v6_direct_policy,
    prepare_v6_hybrid_policy,
    rbergomi_hybrid_candidate_profiles,
    rbergomi_hybrid_profile_ids,
    save_v6_progress,
    update_profile_intervals,
    v6_policy_preparation_to_dict,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

_SCHEMA_V1 = "npi.g11.v6-routed-policy.config.v1"
_SCHEMA_V2 = "npi.g11.v6-routed-policy.config.v2"
_SCHEMA_V3 = "npi.g11.v6-routed-policy.config.v3"


def _record_checkpoint_path(directory: Path, *, cell_id: str, cluster: int) -> Path:
    """Return a path-safe, deterministic checkpoint name for one final run."""

    identity = json.dumps(
        {"cell_id": cell_id, "cluster": cluster},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return directory / "records" / f"{digest}.json"


def _clear_record_checkpoint(checkpoint: Path) -> None:
    """Remove a completed record's replay checkpoint after its journal is durable."""

    checkpoint.unlink(missing_ok=True)
    checkpoint.with_suffix(checkpoint.suffix + ".v6.json").unlink(missing_ok=True)


def _task_conditioned_training_source_summary(source_path: Path) -> dict[str, Any]:
    """Derive the deterministic V3 proposal bank from a pure-CEM artifact."""

    raw = source_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    source = json.loads(raw)
    if not isinstance(source, dict) or source.get("schema") not in {
        "npi.g11.v6-baseline-qualification.v1",
        "npi.g11.v6-proposal-training.v1",
    }:
        raise ValueError("unsupported V6 proposal-training source artifact")
    source_schema = str(source["schema"])
    dedicated_source = source_schema == "npi.g11.v6-proposal-training.v1"
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("proposal training source must contain records")

    source_contract_verified = False
    if dedicated_source:
        gates = source.get("gates")
        formal = source.get("formal_readiness")
        if (
            source.get("proposal_training_qualified") is not True
            or not isinstance(gates, dict)
            or not gates
            or not all(value is True for value in gates.values())
            or not isinstance(formal, dict)
            or not formal
            or not all(value is True for value in formal.values())
        ):
            raise ValueError("dedicated proposal-training source is not qualified")
        seed_payload = source.get("seed_ledger")
        work_payload = source.get("work_ledger")
        if not isinstance(seed_payload, dict) or not isinstance(work_payload, dict):
            raise ValueError("proposal-training source lacks a strict ledger")
        seed_ledger = SeedLedger.from_dict(seed_payload)
        work_ledger = V6WorkLedger.from_dict(work_payload)
        if (
            seed_ledger.sha256 != source.get("seed_ledger_sha256")
            or work_ledger.sha256 != source.get("work_ledger_sha256")
            or len(seed_ledger) != len(records)
            or len(work_ledger.records) != len(records)
        ):
            raise ValueError("proposal-training source ledger hash or count mismatch")
        record_seed_pairs = {
            (
                json.dumps(
                    record.get("seed_key"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                record.get("seed"),
            )
            for record in records
        }
        ledger_seed_pairs = {
            (
                json.dumps(
                    asdict(seed_record.key),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                seed_record.seed,
            )
            for seed_record in seed_ledger.records
        }
        if record_seed_pairs != ledger_seed_pairs:
            raise ValueError("proposal-training record seeds do not match the ledger")
        if [record.get("training_work_record") for record in records] != list(
            work_payload["records"]
        ):
            raise ValueError("proposal-training record work does not match the ledger")
        source_contract_verified = True

    grouped_controls: dict[str, list[list[list[float]]]] = {}
    total_samples = 0
    total_work = 0.0
    total_wall = 0.0
    total_cpu = 0.0
    for record in records:
        if record.get("method") != "pure_cem":
            raise ValueError("proposal training source may contain only pure-CEM records")
        if dedicated_source:
            if record.get("cem_fit", {}).get("converged") is not True:
                raise ValueError("proposal training source contains a nonconverged CEM fit")
            task = str(record.get("task"))
            entry = record.get("training_work_record")
        else:
            if not record["result"]["core"]["complete"]:
                raise ValueError("proposal training source contains an incomplete record")
            task = str(record["preparation"]["core"]["task"])
            training_records = [
                entry
                for entry in record["result"]["total_work"]["records"]
                if entry["category"] == "proposal_training"
            ]
            if len(training_records) != 1:
                raise ValueError(
                    "each proposal training record must have one training ledger entry"
                )
            entry = training_records[0]
        control = record["cem_fit"]["control"]
        if (
            task not in {"terminal_left_tail", "discrete_lower_barrier"}
            or not isinstance(control, list)
            or not control
            or any(
                not isinstance(segment, list)
                or len(segment) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in segment
                )
                for segment in control
            )
        ):
            raise ValueError("proposal training source contains a malformed CEM control")
        if (
            not isinstance(entry, dict)
            or entry.get("category") != "proposal_training"
            or entry.get("successful", True) is not True
        ):
            raise ValueError("proposal training source contains invalid charged work")
        grouped_controls.setdefault(task, []).append(control)
        total_samples += int(entry["samples"])
        total_work += float(entry["work_units"])
        total_wall += float(entry["wall_seconds"])
        total_cpu += float(entry["cpu_seconds"])

    expected_tasks = {"terminal_left_tail", "discrete_lower_barrier"}
    if set(grouped_controls) != expected_tasks:
        raise ValueError("proposal training source must cover both task families")
    derived: dict[str, list[list[list[float]]]] = {}
    for task, controls in grouped_controls.items():
        segments = len(controls[0])
        if any(len(control) != segments for control in controls):
            raise ValueError("proposal training controls have inconsistent segment counts")
        median_control = [
            [statistics.median(control[segment][driver] for control in controls) for driver in range(2)]
            for segment in range(segments)
        ]
        zero = [[0.0, 0.0] for _ in range(segments)]
        half = [[0.5 * value for value in segment] for segment in median_control]
        derived[task] = [zero, half, median_control]

    return {
        "verified": True,
        "source_artifact_sha256": raw_sha256,
        "source_commit": source.get("source_commit"),
        "source_dirty_worktree": source.get("dirty_worktree"),
        "source_smoke": source.get("smoke"),
        "source_schema": source_schema,
        "source_contract_verified": source_contract_verified,
        "formal_training_source_readiness": (
            dedicated_source
            and source_contract_verified
            and source.get("dirty_worktree") is False
            and source.get("smoke") is False
            and isinstance(source.get("source_commit"), str)
            and len(source["source_commit"]) == 40
            and all(
                character in "0123456789abcdef"
                for character in source["source_commit"]
            )
            and source["source_commit"] != "uncommitted"
        ),
        "source_record_count": len(records),
        "derivation": "componentwise_median_pure_cem_then_zero_half_full_bank",
        "task_controls": derived,
        "total_samples": total_samples,
        "total_work_units": total_work,
        "total_wall_seconds": total_wall,
        "total_cpu_seconds": total_cpu,
    }


def _task_conditioned_training_source_audit(
    proposal: dict[str, Any], source_path: Path
) -> dict[str, Any]:
    """Verify a V3 proposal bank against its pure-CEM training artifact."""

    summary = _task_conditioned_training_source_summary(source_path)
    if (
        summary["source_artifact_sha256"]
        != proposal["training_source_artifact_sha256"]
    ):
        raise ValueError("proposal training source hash does not match the config")
    configured = proposal["task_controls"]
    derived = summary["task_controls"]
    expected_tasks = {"terminal_left_tail", "discrete_lower_barrier"}
    for task in sorted(expected_tasks):
        configured_tensor = torch.tensor(configured[task], dtype=torch.float64)
        derived_tensor = torch.tensor(derived[task], dtype=torch.float64)
        if configured_tensor.shape != derived_tensor.shape or not torch.allclose(
            configured_tensor, derived_tensor, rtol=0.0, atol=1e-9
        ):
            raise ValueError("configured proposal controls do not match the declared derivation")

    expected_totals = {
        "training_source_record_count": summary["source_record_count"],
        "training_total_samples": summary["total_samples"],
        "training_total_work_units": summary["total_work_units"],
        "training_total_wall_seconds": summary["total_wall_seconds"],
        "training_total_cpu_seconds": summary["total_cpu_seconds"],
    }
    for key, observed in expected_totals.items():
        declared = proposal[key]
        if isinstance(observed, int):
            if int(declared) != observed:
                raise ValueError(f"proposal {key} does not match the source ledger")
        elif not math.isclose(float(declared), observed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"proposal {key} does not match the source ledger")
    return {key: value for key, value in summary.items() if key != "task_controls"}


def _apportion_shared_training(
    proposal: dict[str, Any],
    record_count: int,
    *,
    enforce_declared_count: bool = True,
) -> tuple[tuple[dict[str, float | int], ...], dict[str, Any]]:
    """Allocate one shared proposal-training ledger exactly across final records.

    Integer sample counts use quotient/remainder apportionment. Floating work,
    wall-time, and CPU-time totals use equal shares with a final residual.  The
    ordering contract is manifest order followed by cluster index, so resume state
    cannot change which record receives a remainder.
    """

    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 1:
        raise ValueError("proposal training apportionment requires a positive record count")
    declared_count = int(proposal["training_amortization_record_count"])
    if enforce_declared_count and declared_count != record_count:
        raise ValueError(
            "proposal training amortization count must equal the executed cell-cluster matrix"
        )
    total_samples = int(proposal["training_total_samples"])
    quotient, remainder = divmod(total_samples, record_count)
    if quotient < 1:
        raise ValueError("proposal training sample total is smaller than the execution matrix")

    sample_allocations = tuple(
        quotient + (1 if index < remainder else 0) for index in range(record_count)
    )

    def floating_allocations(field: str) -> tuple[float, ...]:
        total = float(proposal[field])
        share = total / record_count
        values = [share] * record_count
        values[-1] = total - math.fsum(values[:-1])
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"proposal {field} cannot be apportioned safely")
        if not math.isclose(math.fsum(values), total, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"proposal {field} apportionment does not conserve its total")
        return tuple(values)

    work = floating_allocations("training_total_work_units")
    wall = floating_allocations("training_total_wall_seconds")
    cpu = floating_allocations("training_total_cpu_seconds")
    allocations = tuple(
        {
            "samples": sample_allocations[index],
            "work_units": work[index],
            "wall_seconds": wall[index],
            "cpu_seconds": cpu[index],
        }
        for index in range(record_count)
    )
    contract = {
        "rule": "manifest_order_then_cluster_quotient_remainder_v1",
        "record_count": record_count,
        "declared_record_count": declared_count,
        "declared_count_enforced": enforce_declared_count,
        "integer_sample_quotient": quotient,
        "integer_sample_remainder": remainder,
        "totals": {
            "samples": total_samples,
            "work_units": float(proposal["training_total_work_units"]),
            "wall_seconds": float(proposal["training_total_wall_seconds"]),
            "cpu_seconds": float(proposal["training_total_cpu_seconds"]),
        },
    }
    return allocations, contract


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") not in (
        _SCHEMA_V1,
        _SCHEMA_V2,
        _SCHEMA_V3,
    ):
        raise ValueError("unsupported V6 routed-policy config")
    expected = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "hierarchy",
        "proposal",
        "router",
        "selector",
        "sampling",
        "gates",
    }
    if set(payload) != expected:
        raise ValueError("malformed V6 routed-policy config fields")
    if payload["phase"] not in ("development", "qualification", "confirmation"):
        raise ValueError("unsupported V6 routed-policy phase")
    if payload["phase"] != "development" and payload["frozen"] is not True:
        raise ValueError("qualification and confirmation policy configs must be frozen")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("routed policy must declare a fixed-grid estimand")
    if payload["schema"] in (_SCHEMA_V2, _SCHEMA_V3):
        selector = payload["selector"]
        required = {
            "decision_mode",
            "planning_replicates",
            "samples_per_replicate",
            "planning_variance_statistic",
            "familywise_alpha",
            "practical_equivalence_relative_tolerance",
        }
        if not isinstance(selector, dict) or set(selector) != required:
            raise ValueError("malformed V2 routed-policy selector fields")
        if selector["decision_mode"] != "replicated_planning":
            raise ValueError("unsupported V2 routed-policy decision mode")
        if selector["planning_variance_statistic"] not in {
            "median_replicate_variance",
            "mean_replicate_variance",
        }:
            raise ValueError("unsupported V2 routed-policy planning statistic")
        if int(selector["planning_replicates"]) < 3:
            raise ValueError("replicated planning requires at least three replicates")
        proposal = payload["proposal"]
        if not isinstance(proposal, dict):
            raise ValueError("V2 routed-policy proposal must be an object")
        if "task_controls" in proposal:
            required_proposal = (
                {
                    "weights",
                    "task_controls",
                    "training_source_artifact_sha256",
                    "amortized_training_samples_per_record",
                    "amortized_training_work_per_record",
                    "amortized_training_wall_seconds_per_record",
                    "amortized_training_cpu_seconds_per_record",
                }
                if payload["schema"] == _SCHEMA_V2
                else {
                    "weights",
                    "task_controls",
                    "training_source_artifact_sha256",
                    "training_derivation",
                    "training_source_record_count",
                    "training_total_samples",
                    "training_total_work_units",
                    "training_total_wall_seconds",
                    "training_total_cpu_seconds",
                    "training_amortization_record_count",
                }
            )
            if set(proposal) != required_proposal:
                raise ValueError("malformed task-conditioned proposal fields")
            controls = proposal["task_controls"]
            if not isinstance(controls, dict) or set(controls) != {
                "terminal_left_tail",
                "discrete_lower_barrier",
            }:
                raise ValueError("task-conditioned proposal must cover both task families")
            digest = proposal["training_source_artifact_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("proposal training-source hash must be lowercase SHA-256")
            if payload["schema"] == _SCHEMA_V3:
                if proposal["training_derivation"] != (
                    "componentwise_median_pure_cem_then_zero_half_full_bank"
                ):
                    raise ValueError("unsupported proposal training derivation")
                positive_integer_fields = (
                    "training_source_record_count",
                    "training_total_samples",
                    "training_amortization_record_count",
                )
                if any(
                    isinstance(proposal[field], bool)
                    or not isinstance(proposal[field], int)
                    or proposal[field] < 1
                    for field in positive_integer_fields
                ):
                    raise ValueError("proposal training counts must be positive integers")
                positive_float_fields = (
                    "training_total_work_units",
                    "training_total_wall_seconds",
                    "training_total_cpu_seconds",
                )
                if any(
                    isinstance(proposal[field], bool)
                    or not isinstance(proposal[field], (int, float))
                    or not math.isfinite(float(proposal[field]))
                    or float(proposal[field]) <= 0.0
                    for field in positive_float_fields
                ):
                    raise ValueError("proposal training totals must be finite and positive")
    if (
        payload["phase"] != "development"
        and "task_controls" in payload["proposal"]
        and payload["schema"] != _SCHEMA_V3
    ):
        raise ValueError("formal task-conditioned policies require V3 training provenance")
    return payload, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ReplicatedPlanningSelection:
    schema: str
    planning_replicates: int
    samples_per_replicate: int
    variance_statistic: str
    profile_replicate_variances: dict[str, tuple[float, ...]]
    profile_planning_variances: dict[str, float]
    candidate_point_work: dict[str, float]
    cumulative_sample_count: int
    cumulative_profile_work: float
    finite_sample_work_certificate: bool
    profiles: tuple[Any, ...]
    frozen_decision: FrozenCrossoverDecision


def _fixed_hybrid_work(
    variances: tuple[float, ...],
    costs: tuple[float, ...],
    *,
    target: float,
    minimum_final_samples: int,
) -> float:
    root_sum = math.fsum(
        math.sqrt(variance * cost)
        for variance, cost in zip(variances, costs, strict=True)
    )
    counts = [
        max(
            minimum_final_samples,
            math.ceil(root_sum * math.sqrt(variance / cost) / target)
            if variance > 0.0
            else minimum_final_samples,
        )
        for variance, cost in zip(variances, costs, strict=True)
    ]
    while (
        math.fsum(
            variance / count
            for variance, count in zip(variances, counts, strict=True)
        )
        > target * (1.0 + 1e-14)
    ):
        best = max(
            range(len(counts)),
            key=lambda index: (
                (
                    variances[index] / counts[index]
                    - variances[index] / (counts[index] + 1)
                )
                / costs[index],
                -index,
            ),
        )
        counts[best] += 1
    return math.fsum(count * cost for count, cost in zip(counts, costs, strict=True))


def _router_config(payload: dict[str, Any], *, smoke: bool) -> RarityRouterConfig:
    maximum = int(payload["maximum_screening_trials"])
    initial = int(payload["initial_screening_trials"])
    if smoke:
        maximum = min(maximum, 256)
        initial = min(initial, maximum)
    fallback = str(payload["ambiguous_fallback"])
    if fallback not in ("crude", "dcs_slis"):
        raise ValueError("unsupported ambiguous router fallback")
    return RarityRouterConfig(
        probability_cutoff=float(payload["probability_cutoff"]),
        confidence_level=float(payload["confidence_level"]),
        initial_screening_trials=initial,
        maximum_screening_trials=maximum,
        minimum_certified_relative_saving=float(payload["minimum_certified_relative_saving"]),
        maximum_hybrid_profile_work=(
            1.0 if smoke else float(payload["maximum_hybrid_profile_work"])
        ),
        maximum_profile_fraction=float(payload["maximum_profile_fraction"]),
        ambiguous_fallback=cast(Literal["crude", "dcs_slis"], fallback),
    )


def _append_screening(
    *,
    sampler,
    ledger: SeedLedger,
    protocol: str,
    cell_id: str,
    cluster: int,
    look: int,
    count: int,
) -> tuple[torch.Tensor, float, float, float]:
    seed = ledger.allocate(
        SeedKey(
            protocol,
            "router-screening",
            f"{cell_id}:cluster-{cluster}",
            "crude",
            0,
            look,
            "proposal",
        )
    )
    cpu_started = time.process_time()
    batch = sampler("crude", "pilot", count, {"proposal": seed})
    cpu = time.process_time() - cpu_started
    return batch.values, batch.work_units, batch.wall_seconds, cpu


def _work_interval_from_profile(
    profile,
    *,
    preprocessing_work: float,
    target: float,
    minimum_final_samples: int,
    point_variance_floor: float = 0.0,
):
    lower, upper = profile.moments.variance_interval
    point = profile.moments.sample_variance
    cost = profile.cost_per_sample
    return RoutingWorkInterval(
        "dcs_slis",
        preprocessing_work + max(minimum_final_samples, lower / target) * cost,
        preprocessing_work
        + max(minimum_final_samples, max(point, point_variance_floor) / target) * cost,
        preprocessing_work + max(minimum_final_samples, upper / target) * cost,
    )


def _crude_work_interval(
    values: torch.Tensor,
    *,
    cost: float,
    preprocessing_work: float,
    target: float,
    confidence_level: float,
    minimum_final_samples: int,
    point_probability: float | None = None,
) -> RoutingWorkInterval:
    hits = int(torch.count_nonzero(values))
    interval = exact_binomial_probability_interval(
        hits, values.numel(), confidence_level=confidence_level
    )
    endpoint_variances = (
        interval.lower * (1.0 - interval.lower),
        interval.upper * (1.0 - interval.upper),
    )
    lower = min(endpoint_variances)
    upper = conservative_bernoulli_variance_upper(interval)
    point = (
        float(torch.var(values, unbiased=True))
        if point_probability is None
        else point_probability * (1.0 - point_probability)
    )
    return RoutingWorkInterval(
        "crude",
        preprocessing_work + max(minimum_final_samples, lower / target) * cost,
        preprocessing_work + max(minimum_final_samples, point / target) * cost,
        preprocessing_work + max(minimum_final_samples, upper / target) * cost,
    )


def _linear_quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _replicated_planning_selection(
    *,
    config: dict[str, Any],
    dcs_sampler: RBergomiHybridTermSampler,
    ledger: SeedLedger,
    cell,
    cluster: int,
    profile_ids: tuple[str, ...],
    candidate_profiles: dict[str, tuple[str, ...]],
    preprocessing_work: float,
    sampling_variance_target: float,
    minimum_final_samples: int,
    smoke: bool,
) -> tuple[ReplicatedPlanningSelection, float, float, float]:
    selector = config["selector"]
    replicates = 3 if smoke else int(selector["planning_replicates"])
    samples_per_replicate = 64 if smoke else int(selector["samples_per_replicate"])
    observations = {profile_id: [] for profile_id in profile_ids}
    replicate_variances = {profile_id: [] for profile_id in profile_ids}
    selector_work = 0.0
    selector_wall = 0.0
    selector_cpu = 0.0
    for replicate in range(replicates):
        for profile_index, profile_id in enumerate(profile_ids):
            seeds = {
                stream: ledger.allocate(
                    SeedKey(
                        str(config["protocol_id"]),
                        "selector-planning",
                        f"{cell.cell_id}:cluster-{cluster}",
                        profile_id,
                        profile_index,
                        replicate,
                        stream,
                    )
                )
                for stream in ("proposal", "labels")
            }
            cpu_started = time.process_time()
            batch = dcs_sampler(profile_id, "pilot", samples_per_replicate, seeds)
            selector_cpu += time.process_time() - cpu_started
            selector_work += batch.work_units
            selector_wall += batch.wall_seconds
            observations[profile_id].append(batch.values)
            replicate_variances[profile_id].append(
                float(torch.var(batch.values, unbiased=True))
            )
    combined = {
        profile_id: torch.cat(batches) for profile_id, batches in observations.items()
    }
    profiles = update_profile_intervals(
        combined,
        absolute_bounds={
            profile_id: dcs_sampler.defensive_absolute_bound for profile_id in profile_ids
        },
        costs_per_sample={
            profile_id: dcs_sampler.cost_per_sample(profile_id) for profile_id in profile_ids
        },
        familywise_alpha=float(selector["familywise_alpha"]),
        total_predeclared_looks=1,
    )
    statistic = str(selector["planning_variance_statistic"])
    planning_variances = {
        profile_id: (
            statistics.median(values)
            if statistic == "median_replicate_variance"
            else statistics.fmean(values)
        )
        for profile_id, values in replicate_variances.items()
    }
    safety = float(config["sampling"]["allocation_safety_factor"])
    candidate_work: dict[str, float] = {}
    for candidate, term_ids in candidate_profiles.items():
        variances = tuple(safety * planning_variances[term_id] for term_id in term_ids)
        costs = tuple(dcs_sampler.cost_per_sample(term_id) for term_id in term_ids)
        candidate_work[candidate] = preprocessing_work + selector_work + _fixed_hybrid_work(
            variances,
            costs,
            target=sampling_variance_target,
            minimum_final_samples=minimum_final_samples,
        )
    best = min(candidate_work, key=lambda item: (candidate_work[item], item))
    simpler = f"start_{max(int(item.split('_')[1]) for item in candidate_profiles)}"
    tolerance = float(selector["practical_equivalence_relative_tolerance"])
    selected = (
        simpler
        if candidate_work[simpler] <= (1.0 + tolerance) * candidate_work[best]
        else best
    )
    selected_work = candidate_work[selected]
    decision = FrozenCrossoverDecision(
        selected_candidate=selected,
        look_index=replicates - 1,
        reason=(
            "replicated planning point-work selection; no finite-sample work-regret "
            "certificate"
        ),
        surviving_candidates=tuple(sorted(candidate_profiles)),
        selected_work_interval=(selected_work, selected_work),
        selected_point_work=selected_work,
        worst_case_interval_regret_bound=selected_work / candidate_work[best],
    )
    state = ReplicatedPlanningSelection(
        schema="npi.g11.v6-replicated-planning-selection.v1",
        planning_replicates=replicates,
        samples_per_replicate=samples_per_replicate,
        variance_statistic=statistic,
        profile_replicate_variances={
            profile_id: tuple(values) for profile_id, values in replicate_variances.items()
        },
        profile_planning_variances=planning_variances,
        candidate_point_work=candidate_work,
        cumulative_sample_count=replicates * samples_per_replicate,
        cumulative_profile_work=selector_work,
        finite_sample_work_certificate=False,
        profiles=profiles,
        frozen_decision=decision,
    )
    return state, selector_work, selector_wall, selector_cpu


def run(
    config_path: Path,
    manifest_path: Path,
    reference_path: Path,
    *,
    smoke: bool = False,
    checkpoint_directory: Path | None = None,
    resume: bool = False,
    proposal_training_source_path: Path | None = None,
):
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    references, reference_hash = _load_references(reference_path)
    if resume and checkpoint_directory is None:
        raise ValueError("routed-policy resume requires a checkpoint directory")
    if not smoke and config["phase"] != "development":
        if (
            manifest.phase != config["phase"]
            or not manifest.frozen
            or manifest.smoke
        ):
            raise ValueError("formal routed policy requires a same-phase frozen manifest")
    router_config = _router_config(config["router"], smoke=smoke)
    sampling = config["sampling"]
    hierarchy = config["hierarchy"]
    finest_level = int(hierarchy["finest_level"])
    if int(hierarchy["coarsest_steps"]) * 2**finest_level != manifest.cells[0].finest_steps:
        raise ValueError("routed-policy hierarchy does not match the manifest grid")
    cells = _smoke_cells(manifest.cells) if smoke else manifest.cells
    clusters = 1 if smoke else int(sampling["clusters"])
    relative_rmse = max(0.50, float(sampling["relative_sampling_rmse"])) if smoke else float(
        sampling["relative_sampling_rmse"]
    )
    minimum_final_samples = 128 if smoke else int(sampling["minimum_final_samples"])
    proposal = config["proposal"]
    proposal_weights = torch.tensor(proposal["weights"], dtype=torch.float64)
    proposal_training_audit: dict[str, Any] | None = None
    proposal_training_allocations: tuple[dict[str, float | int], ...] | None = None
    proposal_training_allocation_contract: dict[str, Any] | None = None
    if config["schema"] == _SCHEMA_V3:
        expected_count = len(cells) * clusters
        (
            proposal_training_allocations,
            proposal_training_allocation_contract,
        ) = _apportion_shared_training(
            proposal, expected_count, enforce_declared_count=not smoke
        )
        if proposal_training_source_path is not None:
            proposal_training_audit = _task_conditioned_training_source_audit(
                proposal, proposal_training_source_path
            )
        elif config["phase"] != "development":
            raise ValueError("formal V3 policy execution requires its proposal training source")
        else:
            proposal_training_audit = {
                "verified": False,
                "source_artifact_sha256": proposal["training_source_artifact_sha256"],
                "reason": "development execution did not receive the source artifact",
            }
    profile_ids = rbergomi_hybrid_profile_ids(finest_level)
    candidate_profiles = rbergomi_hybrid_candidate_profiles(finest_level)
    progress_identities: dict[str, object] = {
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_sha256": reference_hash,
        "smoke": smoke,
    }
    progress_path = (
        None if checkpoint_directory is None else checkpoint_directory / "progress.json"
    )
    if progress_path is not None and resume:
        records = list(
            load_v6_progress(
                progress_path,
                experiment="g11_v6_routed_policy",
                identities=progress_identities,
            ).records
        )
    else:
        if progress_path is not None and progress_path.exists():
            raise FileExistsError("fresh routed-policy execution refuses existing progress")
        records = []
    completed = {
        (str(record["cell_id"]), int(record["cluster"])) for record in records
    }
    master_ledger = SeedLedger()
    for record in records:
        restored = SeedLedger.from_dict(record["result"]["core"]["seed_ledger_payload"])
        for seed_record in restored.records:
            master_ledger.allocate(seed_record.key)
    if progress_path is not None and not progress_path.exists():
        save_v6_progress(
            progress_path,
            V6ProgressJournal(
                "g11_v6_routed_policy", progress_identities, tuple(records)
            ),
        )
    for cell_index, cell in enumerate(cells):
        if cell.cell_id not in references:
            raise ValueError(f"reference artifact lacks cell {cell.cell_id}")
        reference_probability, reference_se, reference_cell = references[cell.cell_id]
        if reference_cell != cell.to_dict():
            raise ValueError(f"reference estimand drift for cell {cell.cell_id}")
        task = _task(cell)
        simulator = RBergomiSimulator(
            H=cell.hurst, eta=cell.eta, xi=cell.xi, rho=cell.rho, device="cpu"
        )
        schedules = (
            proposal["task_controls"][cell.task]
            if "task_controls" in proposal
            else proposal["controls"]
        )
        controls = tuple(
            TimePiecewiseTwoDriverControl(
                tuple(tuple(float(value) for value in segment) for segment in schedule),
                maturity=cell.maturity,
            )
            for schedule in schedules
        )
        if len(controls) != proposal_weights.numel():
            raise ValueError("task-conditioned proposal controls do not match mixture weights")
        natural = controls[0]
        for cluster in range(clusters):
            if (cell.cell_id, cluster) in completed:
                continue
            ledger = SeedLedger()
            work = V6WorkLedger()
            if proposal_training_allocations is not None:
                training_allocation = proposal_training_allocations[
                    cell_index * clusters + cluster
                ]
                work = work.append(
                    _work_record(
                        "proposal_training",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=int(training_allocation["samples"]),
                        work_units=float(training_allocation["work_units"]),
                        wall_seconds=float(training_allocation["wall_seconds"]),
                        cpu_seconds=float(training_allocation["cpu_seconds"]),
                    )
                )
            elif "amortized_training_work_per_record" in proposal:
                work = work.append(
                    _work_record(
                        "proposal_training",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=int(proposal["amortized_training_samples_per_record"]),
                        work_units=float(proposal["amortized_training_work_per_record"]),
                        wall_seconds=float(
                            proposal["amortized_training_wall_seconds_per_record"]
                        ),
                        cpu_seconds=float(
                            proposal["amortized_training_cpu_seconds_per_record"]
                        ),
                    )
                )
            crude_interval = None
            dcs_interval = None
            hybrid_opportunity = None
            crude_sampler = _DirectRBergomiSampler(
                method="crude",
                simulator=simulator,
                task=task,
                natural=natural,
                fitted=None,
                defensive_weight=float(proposal_weights[0]),
                cell=cell,
                engine=str(sampling["engine"]),
            )
            screening, units, wall, cpu = _append_screening(
                sampler=crude_sampler,
                ledger=ledger,
                protocol=str(config["protocol_id"]),
                cell_id=cell.cell_id,
                cluster=cluster,
                look=0,
                count=router_config.initial_screening_trials,
            )
            work = work.append(
                _work_record(
                    "screening",
                    method="v6_policy",
                    cell_id=cell.cell_id,
                    samples=screening.numel(),
                    work_units=units,
                    wall_seconds=wall,
                    cpu_seconds=cpu,
                )
            )
            route = freeze_rarity_route(
                successes=int(torch.count_nonzero(screening)),
                trials=int(screening.numel()),
                screening_work=work.category_work("screening"),
                config=router_config,
            )
            if route.action == "continue_screening":
                increment = router_config.maximum_screening_trials - screening.numel()
                extra, units, wall, cpu = _append_screening(
                    sampler=crude_sampler,
                    ledger=ledger,
                    protocol=str(config["protocol_id"]),
                    cell_id=cell.cell_id,
                    cluster=cluster,
                    look=1,
                    count=increment,
                )
                screening = torch.cat((screening, extra))
                work = work.append(
                    _work_record(
                        "screening",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=extra.numel(),
                        work_units=units,
                        wall_seconds=wall,
                        cpu_seconds=cpu,
                    )
                )
                route = freeze_rarity_route(
                    successes=int(torch.count_nonzero(screening)),
                    trials=int(screening.numel()),
                    screening_work=work.category_work("screening"),
                    config=router_config,
                )

            dcs_sampler = RBergomiHybridTermSampler(
                simulator,
                controls,
                proposal_weights,
                task,
                spot=cell.spot,
                maturity=cell.maturity,
                coarsest_steps=int(hierarchy["coarsest_steps"]),
                finest_level=finest_level,
                engine=str(sampling["engine"]),
            )
            dcs_profile = None
            if route.action != "crude":
                dcs_count = 128 if smoke else int(sampling["dcs_pilot_samples"])
                dcs_seeds = {
                    stream: ledger.allocate(
                        SeedKey(
                            str(config["protocol_id"]),
                            "allocation-pilot",
                            f"{cell.cell_id}:cluster-{cluster}",
                            "dcs-slis",
                            finest_level,
                            0,
                            stream,
                        )
                    )
                    for stream in ("proposal", "labels")
                }
                dcs_cpu_started = time.process_time()
                dcs_batch = dcs_sampler(f"single_{finest_level}", "pilot", dcs_count, dcs_seeds)
                dcs_cpu = time.process_time() - dcs_cpu_started
                work = work.append(
                    _work_record(
                        "allocation_pilot",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=dcs_count,
                        work_units=dcs_batch.work_units,
                        wall_seconds=dcs_batch.wall_seconds,
                        cpu_seconds=dcs_cpu,
                    )
                )
                dcs_profile = update_profile_intervals(
                    {f"single_{finest_level}": dcs_batch.values},
                    absolute_bounds={
                        f"single_{finest_level}": dcs_sampler.defensive_absolute_bound
                    },
                    costs_per_sample={
                        f"single_{finest_level}": dcs_sampler.cost_per_sample(
                            f"single_{finest_level}"
                        )
                    },
                    familywise_alpha=float(config["selector"]["familywise_alpha"]),
                    total_predeclared_looks=1,
                )[0]
                target_variance = (cell.nominal_probability * relative_rmse) ** 2
                common_prework = work.total_work_units
                crude_interval = _crude_work_interval(
                    screening,
                    cost=crude_sampler.cost,
                    preprocessing_work=common_prework,
                    target=target_variance,
                    confidence_level=router_config.confidence_level,
                    minimum_final_samples=minimum_final_samples,
                    point_probability=(
                        cell.nominal_probability
                        if config["schema"] in (_SCHEMA_V2, _SCHEMA_V3)
                        else None
                    ),
                )
                dcs_interval = _work_interval_from_profile(
                    dcs_profile,
                    preprocessing_work=common_prework,
                    target=target_variance,
                    minimum_final_samples=minimum_final_samples,
                    point_variance_floor=(
                        cell.nominal_probability * (1.0 - cell.nominal_probability)
                        if config["schema"] in (_SCHEMA_V2, _SCHEMA_V3)
                        else 0.0
                    ),
                )
                if config["schema"] in (_SCHEMA_V2, _SCHEMA_V3):
                    planning_replicates = (
                        3 if smoke else int(config["selector"]["planning_replicates"])
                    )
                    planning_samples = (
                        64 if smoke else int(config["selector"]["samples_per_replicate"])
                    )
                    first_look = planning_replicates * planning_samples
                else:
                    first_look = 32 if smoke else int(config["selector"]["looks"][0])
                minimum_profile_work = math.fsum(
                    first_look * dcs_sampler.cost_per_sample(profile_id)
                    for profile_id in profile_ids
                )
                hybrid_opportunity = HybridProfileOpportunity(
                    minimum_profile_work=minimum_profile_work,
                    optimistic_total_work=common_prework + minimum_profile_work,
                    external_profile_work_cap=float(
                        config["router"]["maximum_hybrid_profile_work"]
                    ),
                )
                route = freeze_rarity_route(
                    successes=int(torch.count_nonzero(screening)),
                    trials=int(screening.numel()),
                    screening_work=work.category_work("screening"),
                    config=router_config,
                    crude_work=crude_interval,
                    dcs_work=dcs_interval,
                    hybrid_opportunity=hybrid_opportunity,
                )
            work = work.append(
                _work_record(
                    "routing",
                    method="v6_policy",
                    cell_id=cell.cell_id,
                    samples=0,
                    work_units=0.0,
                    wall_seconds=0.0,
                    cpu_seconds=0.0,
                )
            )

            selection_state = None
            if route.action == "profile_hybrid" and config["schema"] in (
                _SCHEMA_V2,
                _SCHEMA_V3,
            ):
                selection_state, selector_work, selector_wall, selector_cpu_total = (
                    _replicated_planning_selection(
                        config=config,
                        dcs_sampler=dcs_sampler,
                        ledger=ledger,
                        cell=cell,
                        cluster=cluster,
                        profile_ids=profile_ids,
                        candidate_profiles=candidate_profiles,
                        preprocessing_work=work.total_work_units,
                        sampling_variance_target=(
                            cell.nominal_probability * relative_rmse
                        )
                        ** 2,
                        minimum_final_samples=minimum_final_samples,
                        smoke=smoke,
                    )
                )
                work = work.append(
                    _work_record(
                        "selector_profile",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=(
                            selection_state.cumulative_sample_count * len(profile_ids)
                        ),
                        work_units=selector_work,
                        wall_seconds=selector_wall,
                        cpu_seconds=selector_cpu_total,
                    )
                )
                selected = selection_state.frozen_decision.selected_candidate
                selected_profile_ids = candidate_profiles[selected]
                prepared = prepare_v6_hybrid_policy(
                    HybridTarget(
                        f"{cell.cell_id}:v6-policy",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    selection_state.profiles,
                    policy_name="v6_policy",
                    cell_id=cell.cell_id,
                    route=route,
                    selection=selection_state.frozen_decision,
                    selected_profile_ids=selected_profile_ids,
                    protocol=f"{config['protocol_id']}:cluster-{cluster}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=minimum_final_samples,
                    allocation_safety_factor=float(sampling["allocation_safety_factor"]),
                    design_variance_overrides={
                        profile_id: selection_state.profile_planning_variances[profile_id]
                        for profile_id in selected_profile_ids
                    },
                    preparation_seed_ledger=ledger,
                )
                final_sampler = dcs_sampler
            elif route.action == "profile_hybrid":
                looks = tuple(int(value) for value in config["selector"]["looks"])
                if smoke:
                    looks = (32, 64)
                observations = {
                    profile_id: torch.empty(0, dtype=torch.float64) for profile_id in profile_ids
                }
                previous = 0
                selector_work = 0.0
                selector_wall = 0.0
                selector_cpu_total = 0.0
                for look_index, cumulative in enumerate(looks):
                    increment = cumulative - previous
                    for profile_index, profile_id in enumerate(profile_ids):
                        seeds = {
                            stream: ledger.allocate(
                                SeedKey(
                                    str(config["protocol_id"]),
                                    "selector-profile",
                                    f"{cell.cell_id}:cluster-{cluster}",
                                    profile_id,
                                    profile_index,
                                    look_index,
                                    stream,
                                )
                            )
                            for stream in ("proposal", "labels")
                        }
                        selector_cpu_started = time.process_time()
                        batch = dcs_sampler(profile_id, "pilot", increment, seeds)
                        selector_cpu = time.process_time() - selector_cpu_started
                        observations[profile_id] = torch.cat(
                            (observations[profile_id], batch.values)
                        )
                        selector_work += batch.work_units
                        selector_wall += batch.wall_seconds
                        selector_cpu_total += selector_cpu
                    selection_state = advance_sequential_crossover(
                        observations,
                        absolute_bounds={
                            profile_id: dcs_sampler.defensive_absolute_bound
                            for profile_id in profile_ids
                        },
                        costs_per_sample={
                            profile_id: dcs_sampler.cost_per_sample(profile_id)
                            for profile_id in profile_ids
                        },
                        candidate_profiles=candidate_profiles,
                        preprocessing_work={
                            candidate: work.total_work_units + selector_work
                            for candidate in candidate_profiles
                        },
                        sampling_variance_target=(cell.nominal_probability * relative_rmse) ** 2,
                        predeclared_looks=looks,
                        look_index=look_index,
                        familywise_alpha=float(config["selector"]["familywise_alpha"]),
                        simpler_candidate=f"start_{finest_level}",
                        previous_state=selection_state,
                        elimination_relative_tolerance=float(
                            config["selector"]["elimination_relative_tolerance"]
                        ),
                        practical_equivalence_relative_tolerance=float(
                            config["selector"]["practical_equivalence_relative_tolerance"]
                        ),
                        maximum_profile_work=route.effective_profile_work_cap,
                        maximum_profile_fraction_of_best_point=float(
                            config["router"]["maximum_profile_fraction"]
                        ),
                        minimum_final_samples_per_term=minimum_final_samples,
                    )
                    previous = cumulative
                    if selection_state.stopped:
                        break
                assert selection_state is not None and selection_state.frozen_decision is not None
                work = work.append(
                    _work_record(
                        "selector_profile",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=previous * len(profile_ids),
                        work_units=selector_work,
                        wall_seconds=selector_wall,
                        cpu_seconds=selector_cpu_total,
                    )
                )
                selected = selection_state.frozen_decision.selected_candidate
                prepared = prepare_v6_hybrid_policy(
                    HybridTarget(
                        f"{cell.cell_id}:v6-policy",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    selection_state.profiles,
                    policy_name="v6_policy",
                    cell_id=cell.cell_id,
                    route=route,
                    selection=selection_state.frozen_decision,
                    selected_profile_ids=candidate_profiles[selected],
                    protocol=f"{config['protocol_id']}:cluster-{cluster}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=minimum_final_samples,
                    allocation_safety_factor=float(sampling["allocation_safety_factor"]),
                    preparation_seed_ledger=ledger,
                )
                final_sampler = dcs_sampler
            elif route.action == "dcs_slis":
                assert dcs_profile is not None
                design = SingleTermDesign(
                    profile_id=dcs_profile.profile_id,
                    pilot_count=dcs_profile.moments.sample_count,
                    pilot_mean=dcs_profile.moments.sample_mean,
                    pilot_variance=dcs_profile.moments.sample_variance,
                    design_variance=(
                        max(
                            2.0 * dcs_profile.moments.sample_variance,
                            cell.nominal_probability**2,
                        )
                        if smoke
                        else float(sampling["allocation_safety_factor"])
                        * dcs_profile.moments.variance_interval[1]
                    ),
                    cost_per_sample=dcs_profile.cost_per_sample,
                    absolute_bound=dcs_sampler.defensive_absolute_bound,
                )
                prepared = prepare_v6_direct_policy(
                    HybridTarget(
                        f"{cell.cell_id}:v6-policy",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    design,
                    policy_name="v6_policy",
                    cell_id=cell.cell_id,
                    execution_method="dcs_slis",
                    protocol=f"{config['protocol_id']}:cluster-{cluster}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    route=route,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=minimum_final_samples,
                    preparation_seed_ledger=ledger,
                )
                final_sampler = dcs_sampler
            elif route.action == "crude":
                design = _design_from_pilot(
                    "crude",
                    screening,
                    cost_per_sample=crude_sampler.cost,
                    nominal_probability=cell.nominal_probability,
                    confidence_level=router_config.confidence_level,
                    pure_safety=1.0,
                    defensive_bound=1.0,
                    bounded_alpha=float(config["selector"]["familywise_alpha"]),
                )
                prepared = prepare_v6_direct_policy(
                    HybridTarget(
                        f"{cell.cell_id}:v6-policy",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    design,
                    policy_name="v6_policy",
                    cell_id=cell.cell_id,
                    execution_method="crude",
                    protocol=f"{config['protocol_id']}:cluster-{cluster}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    route=route,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=minimum_final_samples,
                    streams=("proposal",),
                    preparation_seed_ledger=ledger,
                )
                final_sampler = crude_sampler
            else:
                raise AssertionError("router remained unresolved")

            record_checkpoint = (
                None
                if checkpoint_directory is None
                else _record_checkpoint_path(
                    checkpoint_directory, cell_id=cell.cell_id, cluster=cluster
                )
            )
            if record_checkpoint is None:
                result = execute_v6_policy(
                    prepared,
                    final_sampler,
                    reference_probability=reference_probability,
                    reference_standard_error=reference_se,
                    final_peak_memory_bytes=0,
                )
            else:
                state_path = record_checkpoint.with_suffix(
                    record_checkpoint.suffix + ".v6.json"
                )
                result = execute_v6_policy_durable(
                    prepared,
                    final_sampler,
                    checkpoint_path=record_checkpoint,
                    resume=record_checkpoint.exists() or state_path.exists(),
                    chunks_per_checkpoint=1,
                    reference_probability=reference_probability,
                    reference_standard_error=reference_se,
                    final_peak_memory_bytes=0,
                )
            audit = audit_v6_policy(prepared, result)
            selection_fraction = (
                result.total_work.category_work("selector_profile")
                / result.total_work.total_work_units
                if result.total_work.total_work_units > 0.0
                else 0.0
            )
            records.append(
                {
                    "cell_id": cell.cell_id,
                    "cluster": cluster,
                    "nominal_probability": cell.nominal_probability,
                    "reference_probability": reference_probability,
                    "reference_standard_error": reference_se,
                    "route": asdict(route),
                    "router_inputs": {
                        "successes": int(torch.count_nonzero(screening)),
                        "trials": int(screening.numel()),
                        "screening_work": work.category_work("screening"),
                        "config": asdict(router_config),
                        "crude_work": (
                            None if crude_interval is None else asdict(crude_interval)
                        ),
                        "dcs_work": None if dcs_interval is None else asdict(dcs_interval),
                        "hybrid_opportunity": (
                            None
                            if hybrid_opportunity is None
                            else asdict(hybrid_opportunity)
                        ),
                    },
                    "selection": None if selection_state is None else asdict(selection_state),
                    "selection_work_fraction": selection_fraction,
                    "preparation": v6_policy_preparation_to_dict(prepared),
                    "result": asdict(result),
                    "audit": asdict(audit),
                }
            )
            final_ledger = SeedLedger.from_dict(result.core.seed_ledger_payload)
            for seed_record in final_ledger.records:
                master_ledger.allocate(seed_record.key)
            if progress_path is not None:
                save_v6_progress(
                    progress_path,
                    V6ProgressJournal(
                        "g11_v6_routed_policy", progress_identities, tuple(records)
                    ),
                )
                if record_checkpoint is not None:
                    _clear_record_checkpoint(record_checkpoint)
    selection_fractions = [float(record["selection_work_fraction"]) for record in records]
    median_fraction = _linear_quantile(selection_fractions, 0.5)
    p90_fraction = _linear_quantile(selection_fractions, 0.9)
    gates = {
        "complete_matrix": len(records) == len(cells) * clusters,
        "all_routes_resolved": all(
            record["route"]["action"] != "continue_screening" for record in records
        ),
        "all_runs_complete": all(record["result"]["core"]["complete"] for record in records),
        "all_design_targets_attained": all(
            record["result"]["core"]["design_target_attained"] for record in records
        ),
        "all_empirical_targets_attained": all(
            record["result"]["core"]["empirical_target_attained"] is True
            for record in records
        ),
        "no_resource_censoring": all(
            not record["result"]["core"]["resource_censored"] for record in records
        ),
        "all_independent_audits": all(record["audit"]["passed"] for record in records),
        "median_selection_fraction": median_fraction is not None
        and median_fraction <= float(config["gates"]["maximum_median_selection_fraction"]),
        "p90_selection_fraction": p90_fraction is not None
        and p90_fraction <= float(config["gates"]["maximum_p90_selection_fraction"]),
        "reference_not_used_by_router_schema": all(
            all("reference" not in key for key in record["route"])
            for record in records
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "frozen_manifest": manifest.frozen,
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
        "proposal_training_source_verified": (
            "task_controls" not in proposal
            or bool(proposal_training_audit and proposal_training_audit["verified"])
        ),
        "proposal_training_source_formal": (
            "task_controls" not in proposal
            or bool(
                proposal_training_audit
                and proposal_training_audit["formal_training_source_readiness"]
            )
        ),
    }
    return {
        "schema": "npi.g11.v6-routed-policy.v1",
        "protocol_id": config["protocol_id"],
        "config_schema": config["schema"],
        "work_certificate_scope": (
            "simultaneous bounded work intervals"
            if config["schema"] == _SCHEMA_V1
            else "replicated planning point-work only; no finite-sample work-regret certificate"
        ),
        "phase": config["phase"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_artifact_sha256": reference_hash,
        "proposal_training_audit": proposal_training_audit,
        "proposal_training_allocation": proposal_training_allocation_contract,
        "smoke": smoke,
        "records": records,
        "selection_fraction_summary": {"median": median_fraction, "p90": p90_fraction},
        "gates": gates,
        "formal_readiness": formal,
        "policy_qualified": all(gates.values()) and all(formal.values()),
        "seed_ledger": master_ledger.to_dict(),
        "seed_ledger_sha256": master_ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/routed_policy_development.yaml"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--proposal-training-source", type=Path)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.manifest,
        arguments.reference,
        smoke=arguments.smoke,
        checkpoint_directory=arguments.checkpoint_directory,
        resume=arguments.resume,
        proposal_training_source_path=arguments.proposal_training_source,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"qualified": result["policy_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
