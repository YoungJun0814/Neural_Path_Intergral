"""One achieved-RMSE contract for V6 baselines and the routed policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from src.path_integral.hybrid_allocation import (
    HybridCheckpoint,
    HybridPreparedRun,
    HybridResult,
    HybridTarget,
    HybridTermSampler,
    SingleTermDesign,
    execute_hybrid_run,
    prepare_hybrid_run,
    prepare_single_term_run,
)
from src.path_integral.mlmc import WorkLedgerEntry
from src.path_integral.rarity_router import FrozenRarityRoute
from src.path_integral.robust_crossover import FrozenCrossoverDecision, LevelProfileInterval
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.v6_work_ledger import V6WorkLedger, V6WorkRecord

V6ExecutionMethod = Literal["crude", "pure_cem", "defensive_cem", "dcs_slis", "hybrid"]


@dataclass(frozen=True)
class V6PolicyPreparedRun:
    schema: str
    policy_name: str
    cell_id: str
    execution_method: V6ExecutionMethod
    route: FrozenRarityRoute | None
    core: HybridPreparedRun
    preprocessing_work: V6WorkLedger
    policy_hash: str


@dataclass(frozen=True)
class V6PolicyResult:
    schema: str
    policy_name: str
    cell_id: str
    execution_method: V6ExecutionMethod
    policy_hash: str
    core: HybridResult
    total_work: V6WorkLedger
    result_hash: str


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_policy_identity(policy_name: str, cell_id: str) -> None:
    if not policy_name or policy_name.strip() != policy_name:
        raise ValueError("policy_name must be nonempty and stripped")
    if not cell_id or cell_id.strip() != cell_id:
        raise ValueError("cell_id must be nonempty and stripped")


def _validate_route(method: V6ExecutionMethod, route: FrozenRarityRoute | None) -> None:
    if route is None:
        if method in ("hybrid",):
            raise ValueError("Hybrid policy execution requires a frozen rarity route")
        return
    if route.action == "continue_screening":
        raise ValueError("an unresolved rarity route cannot prepare final sampling")
    expected = {
        "crude": "crude",
        "dcs_slis": "dcs_slis",
        "hybrid": "profile_hybrid",
    }
    if method not in expected or route.action != expected[method]:
        raise ValueError("execution method does not match the frozen rarity route")


def _preprocessing_entries(
    ledger: V6WorkLedger,
    *,
    policy_name: str,
    cell_id: str,
) -> tuple[WorkLedgerEntry, ...]:
    if any(record.method != policy_name or record.cell_id != cell_id for record in ledger.records):
        raise ValueError("preprocessing work identity does not match the policy and cell")
    if any(record.category == "final" for record in ledger.records):
        raise ValueError("preprocessing ledger cannot contain final work")
    return tuple(
        WorkLedgerEntry(
            role=record.category,
            level=None,
            samples=record.samples,
            work_units=record.work_units,
            wall_seconds=record.wall_seconds,
        )
        for record in ledger.records
    )


def _required_categories(
    method: V6ExecutionMethod, route: FrozenRarityRoute | None
) -> tuple[str, ...]:
    required: list[str] = []
    if route is not None:
        required.extend(("screening", "routing"))
    if method in ("pure_cem", "defensive_cem"):
        required.append("proposal_training")
    if method == "hybrid":
        required.append("selector_profile")
    required.append("allocation_pilot")
    return tuple(required)


def _validate_required_work(
    method: V6ExecutionMethod,
    route: FrozenRarityRoute | None,
    ledger: V6WorkLedger,
) -> None:
    present = {record.category for record in ledger.records}
    missing = set(_required_categories(method, route)) - present
    if missing:
        raise ValueError(f"policy preprocessing ledger is missing categories: {sorted(missing)}")


def _policy_hash(
    *,
    policy_name: str,
    cell_id: str,
    execution_method: V6ExecutionMethod,
    route: FrozenRarityRoute | None,
    core: HybridPreparedRun,
    work: V6WorkLedger,
) -> str:
    return _canonical_hash(
        {
            "schema": "npi.g11.v6-policy-preparation.v1",
            "policy_name": policy_name,
            "cell_id": cell_id,
            "execution_method": execution_method,
            "route": None if route is None else asdict(route),
            "core_preparation_hash": core.preparation_hash,
            "preprocessing_work_sha256": work.sha256,
        }
    )


def prepare_v6_direct_policy(
    target: HybridTarget,
    design: SingleTermDesign,
    *,
    policy_name: str,
    cell_id: str,
    execution_method: Literal["crude", "pure_cem", "defensive_cem", "dcs_slis"],
    protocol: str,
    regime: str,
    task: str,
    operation_work_cap: float,
    preprocessing_work: V6WorkLedger,
    route: FrozenRarityRoute | None = None,
    chunk_size: int = 4096,
    minimum_final_samples: int = 32,
    streams: tuple[str, ...] = ("proposal", "labels"),
    preparation_seed_ledger: SeedLedger | None = None,
) -> V6PolicyPreparedRun:
    """Prepare crude/CEM/DCS-SLIS under one target and work contract."""

    _validate_policy_identity(policy_name, cell_id)
    _validate_route(execution_method, route)
    _validate_required_work(execution_method, route, preprocessing_work)
    core = prepare_single_term_run(
        target,
        design,
        method=execution_method,
        protocol=protocol,
        regime=regime,
        task=task,
        operation_work_cap=operation_work_cap,
        chunk_size=chunk_size,
        minimum_final_samples=minimum_final_samples,
        streams=streams,
        preparation_ledger=preparation_seed_ledger,
        preprocessing_work_entries=_preprocessing_entries(
            preprocessing_work, policy_name=policy_name, cell_id=cell_id
        ),
    )
    policy_hash = _policy_hash(
        policy_name=policy_name,
        cell_id=cell_id,
        execution_method=execution_method,
        route=route,
        core=core,
        work=preprocessing_work,
    )
    return V6PolicyPreparedRun(
        schema="npi.g11.v6-policy-preparation.v1",
        policy_name=policy_name,
        cell_id=cell_id,
        execution_method=execution_method,
        route=route,
        core=core,
        preprocessing_work=preprocessing_work,
        policy_hash=policy_hash,
    )


def prepare_v6_hybrid_policy(
    target: HybridTarget,
    profiles: Sequence[LevelProfileInterval],
    *,
    policy_name: str,
    cell_id: str,
    route: FrozenRarityRoute,
    selection: FrozenCrossoverDecision,
    selected_profile_ids: Sequence[str],
    protocol: str,
    regime: str,
    task: str,
    operation_work_cap: float,
    preprocessing_work: V6WorkLedger,
    chunk_size: int = 4096,
    minimum_final_samples: int = 32,
    allocation_safety_factor: float = 1.0,
    streams: tuple[str, ...] = ("proposal", "labels"),
    preparation_seed_ledger: SeedLedger | None = None,
) -> V6PolicyPreparedRun:
    """Prepare Hybrid only after an economically admissible frozen route."""

    _validate_policy_identity(policy_name, cell_id)
    _validate_route("hybrid", route)
    _validate_required_work("hybrid", route, preprocessing_work)
    core = prepare_hybrid_run(
        target,
        profiles,
        selection=selection,
        selected_profile_ids=selected_profile_ids,
        protocol=protocol,
        regime=regime,
        task=task,
        operation_work_cap=operation_work_cap,
        chunk_size=chunk_size,
        minimum_final_samples=minimum_final_samples,
        allocation_safety_factor=allocation_safety_factor,
        streams=streams,
        preparation_ledger=preparation_seed_ledger,
        preprocessing_work_entries=_preprocessing_entries(
            preprocessing_work, policy_name=policy_name, cell_id=cell_id
        ),
    )
    policy_hash = _policy_hash(
        policy_name=policy_name,
        cell_id=cell_id,
        execution_method="hybrid",
        route=route,
        core=core,
        work=preprocessing_work,
    )
    return V6PolicyPreparedRun(
        schema="npi.g11.v6-policy-preparation.v1",
        policy_name=policy_name,
        cell_id=cell_id,
        execution_method="hybrid",
        route=route,
        core=core,
        preprocessing_work=preprocessing_work,
        policy_hash=policy_hash,
    )


def execute_v6_policy(
    prepared: V6PolicyPreparedRun,
    sampler: HybridTermSampler,
    *,
    checkpoint: HybridCheckpoint | None = None,
    maximum_chunks: int | None = None,
    reference_probability: float | None = None,
    reference_standard_error: float = 0.0,
    cumulative_final_cpu_seconds: float,
    final_peak_memory_bytes: int,
) -> V6PolicyResult:
    """Execute the frozen policy and attach a complete training-inclusive ledger.

    On resume, ``cumulative_final_cpu_seconds`` must cover the earlier checkpointed
    chunks as well as the current call; this avoids silently undercharging resumed
    execution.
    """

    expected_hash = _policy_hash(
        policy_name=prepared.policy_name,
        cell_id=prepared.cell_id,
        execution_method=prepared.execution_method,
        route=prepared.route,
        core=prepared.core,
        work=prepared.preprocessing_work,
    )
    if prepared.schema != "npi.g11.v6-policy-preparation.v1" or expected_hash != prepared.policy_hash:
        raise ValueError("V6 policy preparation hash is invalid")
    if not math.isfinite(cumulative_final_cpu_seconds) or cumulative_final_cpu_seconds < 0.0:
        raise ValueError("cumulative_final_cpu_seconds must be finite and nonnegative")
    if (
        isinstance(final_peak_memory_bytes, bool)
        or not isinstance(final_peak_memory_bytes, int)
        or final_peak_memory_bytes < 0
    ):
        raise ValueError("final_peak_memory_bytes must be a nonnegative integer")
    core_result = execute_hybrid_run(
        prepared.core,
        sampler,
        checkpoint=checkpoint,
        maximum_chunks=maximum_chunks,
        reference_probability=reference_probability,
        reference_standard_error=reference_standard_error,
    )
    final_entries = tuple(entry for entry in core_result.work.entries if entry.role == "final")
    total_work = prepared.preprocessing_work
    if final_entries:
        total_work = total_work.append(
            V6WorkRecord(
                category="final",
                method=prepared.policy_name,
                cell_id=prepared.cell_id,
                attempt=0,
                samples=sum(entry.samples for entry in final_entries),
                work_units=math.fsum(entry.work_units for entry in final_entries),
                wall_seconds=math.fsum(entry.wall_seconds for entry in final_entries),
                cpu_seconds=cumulative_final_cpu_seconds,
                peak_memory_bytes=final_peak_memory_bytes,
                successful=core_result.complete,
            )
        )
    result_payload: dict[str, object] = {
        "schema": "npi.g11.v6-policy-result.v1",
        "policy_hash": prepared.policy_hash,
        "core_preparation_hash": core_result.preparation_hash,
        "complete": core_result.complete,
        "resource_censored": core_result.resource_censored,
        "estimate": core_result.estimate,
        "empirical_sampling_variance": core_result.empirical_sampling_variance,
        "seed_ledger_hash": core_result.seed_ledger_hash,
        "total_work_sha256": total_work.sha256,
    }
    return V6PolicyResult(
        schema="npi.g11.v6-policy-result.v1",
        policy_name=prepared.policy_name,
        cell_id=prepared.cell_id,
        execution_method=prepared.execution_method,
        policy_hash=prepared.policy_hash,
        core=core_result,
        total_work=total_work,
        result_hash=_canonical_hash(result_payload),
    )
