"""One achieved-RMSE contract for V6 baselines and the routed policy."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.path_integral.hybrid_allocation import (
    HybridCheckpoint,
    HybridPreparedRun,
    HybridResult,
    HybridTarget,
    HybridTermSampler,
    SingleTermDesign,
    execute_hybrid_run,
    load_hybrid_checkpoint,
    prepare_hybrid_run,
    prepare_single_term_run,
    save_hybrid_checkpoint,
)
from src.path_integral.mlmc import WorkLedgerEntry
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.rarity_router import FrozenRarityRoute
from src.path_integral.robust_crossover import FrozenCrossoverDecision, LevelProfileInterval
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.v6_work_ledger import V6WorkLedger, V6WorkRecord

V6ExecutionMethod = Literal[
    "crude",
    "pure_cem",
    "defensive_cem",
    "dcs_slis",
    "raw_defensive",
    "hybrid",
]


@dataclass(frozen=True)
class V6PolicyPreparedRun:
    schema: str
    policy_name: str
    cell_id: str
    execution_method: V6ExecutionMethod
    route: FrozenRarityRoute | None
    audit_design: SingleTermDesign | None
    minimum_final_samples: int
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


def v6_policy_preparation_to_dict(prepared: V6PolicyPreparedRun) -> dict[str, object]:
    """Serialize every sufficient statistic required by the offline V6 auditor."""

    core = prepared.core
    return {
        "schema": prepared.schema,
        "policy_name": prepared.policy_name,
        "cell_id": prepared.cell_id,
        "execution_method": prepared.execution_method,
        "route": None if prepared.route is None else asdict(prepared.route),
        "audit_design": (
            None if prepared.audit_design is None else asdict(prepared.audit_design)
        ),
        "minimum_final_samples": prepared.minimum_final_samples,
        "core": {
            "protocol": core.protocol,
            "regime": core.regime,
            "task": core.task,
            "selected_candidate": core.selected_candidate,
            "target": asdict(core.target),
            "selection": asdict(core.selection),
            "allocations": [asdict(item) for item in core.allocations],
            "expected_final_work": core.expected_final_work,
            "operation_work_cap": core.operation_work_cap,
            "resource_censored": core.resource_censored,
            "censoring_reason": core.censoring_reason,
            "chunk_size": core.chunk_size,
            "streams": list(core.streams),
            "seed_ledger": core.ledger.to_dict(),
            "work_entries": [asdict(item) for item in core.work.entries],
            "preparation_hash": core.preparation_hash,
        },
        "preprocessing_work": prepared.preprocessing_work.to_dict(),
        "policy_hash": prepared.policy_hash,
    }


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
    elif not (method == "crude" and route is not None):
        # The natural-law screening sample is already a valid crude allocation
        # pilot. Requiring a second record would double-label the same work.
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
    audit_design: SingleTermDesign | None,
    minimum_final_samples: int,
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
            "audit_design": None if audit_design is None else asdict(audit_design),
            "minimum_final_samples": minimum_final_samples,
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
    execution_method: Literal[
        "crude", "pure_cem", "defensive_cem", "dcs_slis", "raw_defensive"
    ],
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
        audit_design=design,
        minimum_final_samples=minimum_final_samples,
        core=core,
        work=preprocessing_work,
    )
    return V6PolicyPreparedRun(
        schema="npi.g11.v6-policy-preparation.v1",
        policy_name=policy_name,
        cell_id=cell_id,
        execution_method=execution_method,
        route=route,
        audit_design=design,
        minimum_final_samples=minimum_final_samples,
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
    design_variance_overrides: Mapping[str, float] | None = None,
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
        design_variance_overrides=design_variance_overrides,
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
        audit_design=None,
        minimum_final_samples=minimum_final_samples,
        core=core,
        work=preprocessing_work,
    )
    return V6PolicyPreparedRun(
        schema="npi.g11.v6-policy-preparation.v1",
        policy_name=policy_name,
        cell_id=cell_id,
        execution_method="hybrid",
        route=route,
        audit_design=None,
        minimum_final_samples=minimum_final_samples,
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
    final_peak_memory_bytes: int,
    prior_final_cpu_seconds: float = 0.0,
) -> V6PolicyResult:
    """Execute the frozen policy and attach a complete training-inclusive ledger.

    On resume, ``prior_final_cpu_seconds`` must contain CPU time charged by earlier
    checkpointed chunks. The current call measures and adds its own process CPU time.
    """

    expected_hash = _policy_hash(
        policy_name=prepared.policy_name,
        cell_id=prepared.cell_id,
        execution_method=prepared.execution_method,
        route=prepared.route,
        audit_design=prepared.audit_design,
        minimum_final_samples=prepared.minimum_final_samples,
        core=prepared.core,
        work=prepared.preprocessing_work,
    )
    if prepared.schema != "npi.g11.v6-policy-preparation.v1" or expected_hash != prepared.policy_hash:
        raise ValueError("V6 policy preparation hash is invalid")
    if not math.isfinite(prior_final_cpu_seconds) or prior_final_cpu_seconds < 0.0:
        raise ValueError("prior_final_cpu_seconds must be finite and nonnegative")
    if (
        isinstance(final_peak_memory_bytes, bool)
        or not isinstance(final_peak_memory_bytes, int)
        or final_peak_memory_bytes < 0
    ):
        raise ValueError("final_peak_memory_bytes must be a nonnegative integer")
    cpu_started = time.process_time()
    core_result = execute_hybrid_run(
        prepared.core,
        sampler,
        checkpoint=checkpoint,
        maximum_chunks=maximum_chunks,
        reference_probability=reference_probability,
        reference_standard_error=reference_standard_error,
    )
    cumulative_final_cpu_seconds = prior_final_cpu_seconds + (
        time.process_time() - cpu_started
    )
    measured_peak_memory_bytes = max(
        final_peak_memory_bytes, process_peak_resident_memory_bytes()
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
                peak_memory_bytes=measured_peak_memory_bytes,
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


def _durable_state_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".v6.json")


def _write_durable_state(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.025 * 2**attempt)


def execute_v6_policy_durable(
    prepared: V6PolicyPreparedRun,
    sampler: HybridTermSampler,
    *,
    checkpoint_path: str | Path,
    resume: bool = False,
    chunks_per_checkpoint: int = 1,
    maximum_cycles: int | None = None,
    reference_probability: float | None = None,
    reference_standard_error: float = 0.0,
    final_peak_memory_bytes: int,
) -> V6PolicyResult:
    """Execute a V6 policy with a strict, policy-bound durable checkpoint.

    A call interrupted after a published cycle loses at most
    ``chunks_per_checkpoint`` chunks.  ``maximum_cycles`` is intended for schedulers
    and tests that deliberately yield; omit it to run through completion.
    """

    checkpoint = Path(checkpoint_path)
    state_path = _durable_state_path(checkpoint)
    if (
        isinstance(chunks_per_checkpoint, bool)
        or chunks_per_checkpoint < 1
        or isinstance(maximum_cycles, bool)
        or (maximum_cycles is not None and maximum_cycles < 1)
    ):
        raise ValueError("durable checkpoint cycle counts must be positive integers")
    prior_cpu = 0.0
    core_checkpoint = None
    if resume:
        if not checkpoint.exists() or not state_path.exists():
            raise FileNotFoundError("durable V6 resume requires checkpoint and state files")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or set(state) != {
            "schema",
            "status",
            "policy_hash",
            "core_preparation_hash",
            "prior_final_cpu_seconds",
        }:
            raise ValueError("malformed durable V6 state")
        if (
            state["schema"] != "npi.g11.v6-policy-checkpoint.v1"
            or state["status"] != "running"
            or state["policy_hash"] != prepared.policy_hash
            or state["core_preparation_hash"] != prepared.core.preparation_hash
        ):
            raise ValueError("durable V6 state does not match the prepared policy")
        prior_cpu = float(state["prior_final_cpu_seconds"])
        if not math.isfinite(prior_cpu) or prior_cpu < 0.0:
            raise ValueError("durable V6 state contains invalid CPU work")
        core_checkpoint = load_hybrid_checkpoint(checkpoint)
    elif checkpoint.exists() or state_path.exists():
        raise FileExistsError("fresh durable execution refuses an existing checkpoint")

    cycles = 0
    while True:
        result = execute_v6_policy(
            prepared,
            sampler,
            checkpoint=core_checkpoint,
            maximum_chunks=chunks_per_checkpoint,
            reference_probability=reference_probability,
            reference_standard_error=reference_standard_error,
            final_peak_memory_bytes=final_peak_memory_bytes,
            prior_final_cpu_seconds=prior_cpu,
        )
        cycles += 1
        if result.core.complete or result.core.resource_censored:
            # Keep the last published running checkpoint until the outer experiment
            # durably journals the completed record. If the process dies in that
            # narrow window, resume deterministically replays only the final suffix.
            return result
        if result.core.checkpoint is None:
            raise AssertionError("incomplete V6 execution did not return a core checkpoint")
        core_checkpoint = result.core.checkpoint
        final_records = tuple(
            record for record in result.total_work.records if record.category == "final"
        )
        prior_cpu = final_records[-1].cpu_seconds if final_records else prior_cpu
        save_hybrid_checkpoint(core_checkpoint, checkpoint)
        _write_durable_state(
            state_path,
            {
                "schema": "npi.g11.v6-policy-checkpoint.v1",
                "status": "running",
                "policy_hash": prepared.policy_hash,
                "core_preparation_hash": prepared.core.preparation_hash,
                "prior_final_cpu_seconds": prior_cpu,
            },
        )
        if maximum_cycles is not None and cycles >= maximum_cycles:
            return result
