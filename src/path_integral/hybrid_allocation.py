"""Pilot-frozen achieved-RMSE allocation and execution for hybrid estimators."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Protocol, cast

import torch

from src.path_integral.mlmc import LevelBatch, OnlineMoments, WorkLedger, WorkLedgerEntry
from src.path_integral.robust_crossover import (
    FrozenCrossoverDecision,
    LevelProfileInterval,
)
from src.path_integral.seed_ledger import SeedKey, SeedLedger


@dataclass(frozen=True)
class HybridTarget:
    """Absolute sampling error budget fixed by a nominal cell probability."""

    target_id: str
    nominal_probability: float
    relative_sampling_rmse: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not self.target_id or self.target_id.strip() != self.target_id:
            raise ValueError("target_id must be nonempty and stripped")
        if not math.isfinite(self.nominal_probability) or not (
            0.0 < self.nominal_probability <= 1.0
        ):
            raise ValueError("nominal_probability must lie in (0, 1]")
        if not math.isfinite(self.relative_sampling_rmse) or not (
            0.0 < self.relative_sampling_rmse < 1.0
        ):
            raise ValueError("relative_sampling_rmse must lie in (0, 1)")
        if not math.isfinite(self.confidence_level) or not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must lie in (0, 1)")

    @property
    def sampling_standard_error_target(self) -> float:
        return self.nominal_probability * self.relative_sampling_rmse

    @property
    def sampling_variance_target(self) -> float:
        return self.sampling_standard_error_target**2


@dataclass(frozen=True)
class HybridTermAllocation:
    """Frozen integer allocation for one independently sampled telescoping term."""

    profile_id: str
    absolute_bound: float
    design_variance: float
    cost_per_sample: float
    continuous_count: float
    final_count: int
    design_sampling_variance: float


@dataclass(frozen=True)
class HybridPreparedRun:
    """Pilot-measurable run contract; it contains no allocated final seeds."""

    protocol: str
    regime: str
    task: str
    selected_candidate: str
    target: HybridTarget
    selection: FrozenCrossoverDecision
    allocations: tuple[HybridTermAllocation, ...]
    expected_final_work: float
    operation_work_cap: float
    resource_censored: bool
    censoring_reason: str | None
    chunk_size: int
    streams: tuple[str, ...]
    ledger: SeedLedger
    work: WorkLedger
    preparation_hash: str


@dataclass(frozen=True)
class HybridTermEstimate:
    """Final estimate and empirical uncertainty of one independent term."""

    profile_id: str
    count: int
    mean: float
    variance: float
    sampling_variance: float


@dataclass(frozen=True)
class HybridCheckpoint:
    """Resume state tied to one immutable preparation hash."""

    schema: str
    preparation_hash: str
    selected_candidate: str
    allocations: tuple[int, ...]
    next_term: int
    next_offset: int
    moments: tuple[OnlineMoments, ...]
    ledger_payload: dict[str, object]
    work_entries: tuple[WorkLedgerEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "preparation_hash": self.preparation_hash,
            "selected_candidate": self.selected_candidate,
            "allocations": list(self.allocations),
            "next_term": self.next_term,
            "next_offset": self.next_offset,
            "moments": [asdict(item) for item in self.moments],
            "ledger": self.ledger_payload,
            "work_entries": [asdict(item) for item in self.work_entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HybridCheckpoint:
        if payload.get("schema") != "npi.g11.hybrid-checkpoint.v1":
            raise ValueError("unsupported hybrid checkpoint schema")
        raw_moments = payload.get("moments")
        raw_work = payload.get("work_entries")
        raw_allocations = payload.get("allocations")
        raw_ledger = payload.get("ledger")
        raw_next_term = payload.get("next_term")
        raw_next_offset = payload.get("next_offset")
        if (
            not isinstance(raw_moments, list)
            or not all(isinstance(item, dict) for item in raw_moments)
            or not isinstance(raw_work, list)
            or not all(isinstance(item, dict) for item in raw_work)
            or not isinstance(raw_allocations, list)
            or not all(isinstance(item, int) for item in raw_allocations)
            or not isinstance(raw_ledger, dict)
            or not isinstance(raw_next_term, int)
            or isinstance(raw_next_term, bool)
            or not isinstance(raw_next_offset, int)
            or isinstance(raw_next_offset, bool)
        ):
            raise ValueError("malformed hybrid checkpoint collections")
        try:
            moments = tuple(
                OnlineMoments(
                    count=int(item["count"]),
                    mean=float(item["mean"]),
                    m2=float(item["m2"]),
                )
                for item in raw_moments
            )
            work = tuple(
                WorkLedgerEntry(
                    role=str(item["role"]),
                    level=(None if item["level"] is None else int(item["level"])),
                    samples=int(item["samples"]),
                    work_units=float(item["work_units"]),
                    wall_seconds=float(item["wall_seconds"]),
                )
                for item in raw_work
            )
            checkpoint = cls(
                schema=str(payload["schema"]),
                preparation_hash=str(payload["preparation_hash"]),
                selected_candidate=str(payload["selected_candidate"]),
                allocations=tuple(raw_allocations),
                next_term=raw_next_term,
                next_offset=raw_next_offset,
                moments=moments,
                ledger_payload=cast(dict[str, object], raw_ledger),
                work_entries=work,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed hybrid checkpoint") from error
        if checkpoint.next_term < 0 or checkpoint.next_offset < 0:
            raise ValueError("malformed hybrid checkpoint position")
        return checkpoint


@dataclass(frozen=True)
class HybridResult:
    """Achieved-RMSE result with asymptotic and bounded-contribution intervals."""

    complete: bool
    resource_censored: bool
    censoring_reason: str | None
    selected_candidate: str
    estimate: float | None
    design_sampling_variance: float
    empirical_sampling_variance: float | None
    standard_error: float | None
    asymptotic_confidence_interval: tuple[float, float] | None
    bounded_confidence_interval: tuple[float, float] | None
    requested_relative_sampling_rmse: float
    achieved_relative_sampling_rmse_nominal: float | None
    achieved_relative_sampling_rmse_reference: float | None
    design_target_attained: bool
    empirical_target_attained: bool | None
    reference_z_score: float | None
    terms: tuple[HybridTermEstimate, ...]
    allocations: tuple[HybridTermAllocation, ...]
    work: WorkLedger
    seed_ledger_hash: str
    preparation_hash: str
    checkpoint: HybridCheckpoint | None


class HybridTermSampler(Protocol):
    def __call__(
        self,
        profile_id: str,
        role: str,
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch: ...


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _increase_to_design_target(allocations: list[dict[str, float | int]], target: float) -> None:
    def design_variance() -> float:
        return math.fsum(float(item["variance"]) / int(item["count"]) for item in allocations)

    iterations = 0
    while design_variance() > target * (1.0 + 1e-14):
        best = max(
            range(len(allocations)),
            key=lambda index: (
                (
                    float(allocations[index]["variance"]) / int(allocations[index]["count"])
                    - float(allocations[index]["variance"]) / (int(allocations[index]["count"]) + 1)
                )
                / float(allocations[index]["cost"]),
                -index,
            ),
        )
        allocations[best]["count"] = int(allocations[best]["count"]) + 1
        iterations += 1
        if iterations > 1_000_000:
            raise RuntimeError("integer allocation correction did not converge")


def prepare_hybrid_run(
    target: HybridTarget,
    profiles: Sequence[LevelProfileInterval],
    *,
    selection: FrozenCrossoverDecision,
    selected_profile_ids: Sequence[str],
    protocol: str,
    regime: str,
    task: str,
    operation_work_cap: float,
    chunk_size: int = 4096,
    minimum_final_samples: int = 32,
    allocation_safety_factor: float = 1.0,
    streams: tuple[str, ...] = ("proposal", "labels"),
    preparation_ledger: SeedLedger | None = None,
    preprocessing_work_entries: Sequence[WorkLedgerEntry] = (),
) -> HybridPreparedRun:
    """Freeze selection and integer allocation without allocating a final seed."""

    text_fields = (protocol, regime, task)
    if any(not value or value.strip() != value for value in text_fields):
        raise ValueError("protocol, regime, and task must be nonempty and stripped")
    if selection.selected_candidate == "":
        raise ValueError("selection must name a candidate")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if minimum_final_samples < 2:
        raise ValueError("minimum_final_samples must be at least two")
    if not math.isfinite(allocation_safety_factor) or allocation_safety_factor < 1.0:
        raise ValueError("allocation_safety_factor must be at least one")
    if not math.isfinite(operation_work_cap) or operation_work_cap <= 0.0:
        raise ValueError("operation_work_cap must be finite and positive")
    if not streams or len(set(streams)) != len(streams):
        raise ValueError("streams must be nonempty and unique")

    profile_map = {profile.profile_id: profile for profile in profiles}
    if len(profile_map) != len(tuple(profiles)):
        raise ValueError("profile identifiers must be unique")
    selected_ids = tuple(selected_profile_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected profiles must be nonempty and unique")
    if any(profile_id not in profile_map for profile_id in selected_ids):
        raise ValueError("selection references an unavailable profile")

    work = WorkLedger.empty()
    for entry in preprocessing_work_entries:
        if entry.role.startswith("final"):
            raise ValueError("preparation work cannot contain final-role entries")
        work.add(entry)
    ledger = SeedLedger(() if preparation_ledger is None else preparation_ledger.records)
    if any(record.key.role.startswith("final") for record in ledger.records):
        raise ValueError("preparation ledger already contains a final seed")

    terms = tuple(profile_map[profile_id] for profile_id in selected_ids)
    design_variances = tuple(
        allocation_safety_factor * profile.moments.variance_interval[1] for profile in terms
    )
    costs = tuple(profile.cost_per_sample for profile in terms)
    root_sum = math.fsum(
        math.sqrt(variance * cost) for variance, cost in zip(design_variances, costs, strict=True)
    )
    mutable: list[dict[str, float | int]] = []
    for _profile, variance, cost in zip(terms, design_variances, costs, strict=True):
        continuous = (
            root_sum * math.sqrt(variance / cost) / target.sampling_variance_target
            if variance > 0.0
            else 0.0
        )
        if continuous > (1 << 63) - 1:
            raise OverflowError("required allocation exceeds signed 64-bit range")
        mutable.append(
            {
                "variance": variance,
                "cost": cost,
                "continuous": continuous,
                "count": max(minimum_final_samples, math.ceil(continuous)),
            }
        )
    _increase_to_design_target(mutable, target.sampling_variance_target)
    allocations = tuple(
        HybridTermAllocation(
            profile_id=profile.profile_id,
            absolute_bound=profile.moments.absolute_bound,
            design_variance=float(item["variance"]),
            cost_per_sample=float(item["cost"]),
            continuous_count=float(item["continuous"]),
            final_count=int(item["count"]),
            design_sampling_variance=float(item["variance"]) / int(item["count"]),
        )
        for profile, item in zip(terms, mutable, strict=True)
    )
    expected_final_work = math.fsum(
        allocation.final_count * allocation.cost_per_sample for allocation in allocations
    )
    expected_total_work = work.total_work_units + expected_final_work
    censored = expected_total_work > operation_work_cap
    reason = "frozen achieved-RMSE allocation exceeds the operation-work cap" if censored else None
    preparation_payload: dict[str, object] = {
        "schema": "npi.g11.hybrid-preparation.v1",
        "protocol": protocol,
        "regime": regime,
        "task": task,
        "selected_candidate": selection.selected_candidate,
        "target": asdict(target),
        "selection": asdict(selection),
        "allocations": [asdict(item) for item in allocations],
        "expected_final_work": expected_final_work,
        "operation_work_cap": operation_work_cap,
        "resource_censored": censored,
        "chunk_size": chunk_size,
        "streams": list(streams),
        "preparation_seed_ledger_hash": ledger.sha256,
        "preprocessing_work": [asdict(item) for item in work.entries],
    }
    return HybridPreparedRun(
        protocol=protocol,
        regime=regime,
        task=task,
        selected_candidate=selection.selected_candidate,
        target=target,
        selection=selection,
        allocations=allocations,
        expected_final_work=expected_final_work,
        operation_work_cap=operation_work_cap,
        resource_censored=censored,
        censoring_reason=reason,
        chunk_size=chunk_size,
        streams=streams,
        ledger=ledger,
        work=work,
        preparation_hash=_canonical_hash(preparation_payload),
    )


def _execution_state(
    prepared: HybridPreparedRun, checkpoint: HybridCheckpoint | None
) -> tuple[int, int, list[OnlineMoments], SeedLedger, WorkLedger]:
    if checkpoint is None:
        return (
            0,
            0,
            [OnlineMoments() for _ in prepared.allocations],
            SeedLedger(prepared.ledger.records),
            WorkLedger(list(prepared.work.entries)),
        )
    expected = tuple(item.final_count for item in prepared.allocations)
    if (
        checkpoint.preparation_hash != prepared.preparation_hash
        or checkpoint.selected_candidate != prepared.selected_candidate
        or checkpoint.allocations != expected
        or len(checkpoint.moments) != len(expected)
    ):
        raise ValueError("checkpoint does not match the prepared hybrid run")
    if checkpoint.next_term > len(expected) or (
        checkpoint.next_term == len(expected) and checkpoint.next_offset != 0
    ):
        raise ValueError("checkpoint position is outside the prepared allocation")
    if (
        checkpoint.next_term < len(expected)
        and checkpoint.next_offset > expected[checkpoint.next_term]
    ):
        raise ValueError("checkpoint offset exceeds the prepared term allocation")
    expected_counts = [
        expected[index]
        if index < checkpoint.next_term
        else (checkpoint.next_offset if index == checkpoint.next_term else 0)
        for index in range(len(expected))
    ]
    if [item.count for item in checkpoint.moments] != expected_counts:
        raise ValueError("checkpoint moments do not match its execution position")
    if any(
        item.count < 0
        or not math.isfinite(item.mean)
        or not math.isfinite(item.m2)
        or item.m2 < -1e-12
        for item in checkpoint.moments
    ):
        raise ValueError("checkpoint moments are invalid")
    final_counts = [0 for _ in expected]
    for entry in checkpoint.work_entries:
        if (
            entry.samples < 0
            or not math.isfinite(entry.work_units)
            or entry.work_units < 0.0
            or not math.isfinite(entry.wall_seconds)
            or entry.wall_seconds < 0.0
        ):
            raise ValueError("checkpoint work ledger is invalid")
        if entry.role == "final":
            if entry.level is None or not 0 <= entry.level < len(expected):
                raise ValueError("checkpoint final work has an invalid term")
            final_counts[entry.level] += entry.samples
    if final_counts != expected_counts:
        raise ValueError("checkpoint final work counts do not match its moments")
    ledger = SeedLedger.from_dict(checkpoint.ledger_payload)
    prepared_keys = {record.key for record in prepared.ledger.records}
    if not prepared_keys.issubset({record.key for record in ledger.records}):
        raise ValueError("checkpoint is missing preparation seed records")
    return (
        checkpoint.next_term,
        checkpoint.next_offset,
        [OnlineMoments(item.count, item.mean, item.m2) for item in checkpoint.moments],
        ledger,
        WorkLedger(list(checkpoint.work_entries)),
    )


def _censored_result(prepared: HybridPreparedRun) -> HybridResult:
    design = math.fsum(item.design_sampling_variance for item in prepared.allocations)
    return HybridResult(
        complete=False,
        resource_censored=True,
        censoring_reason=prepared.censoring_reason,
        selected_candidate=prepared.selected_candidate,
        estimate=None,
        design_sampling_variance=design,
        empirical_sampling_variance=None,
        standard_error=None,
        asymptotic_confidence_interval=None,
        bounded_confidence_interval=None,
        requested_relative_sampling_rmse=prepared.target.relative_sampling_rmse,
        achieved_relative_sampling_rmse_nominal=None,
        achieved_relative_sampling_rmse_reference=None,
        design_target_attained=design <= prepared.target.sampling_variance_target,
        empirical_target_attained=None,
        reference_z_score=None,
        terms=(),
        allocations=prepared.allocations,
        work=WorkLedger(list(prepared.work.entries)),
        seed_ledger_hash=prepared.ledger.sha256,
        preparation_hash=prepared.preparation_hash,
        checkpoint=None,
    )


def execute_hybrid_run(
    prepared: HybridPreparedRun,
    sampler: HybridTermSampler,
    *,
    checkpoint: HybridCheckpoint | None = None,
    maximum_chunks: int | None = None,
    reference_probability: float | None = None,
    reference_standard_error: float = 0.0,
) -> HybridResult:
    """Execute only the frozen final allocation, optionally resuming by chunks."""

    if maximum_chunks is not None and maximum_chunks < 1:
        raise ValueError("maximum_chunks must be positive")
    if reference_probability is not None and (
        not math.isfinite(reference_probability) or not 0.0 < reference_probability <= 1.0
    ):
        raise ValueError("reference_probability must lie in (0, 1]")
    if not math.isfinite(reference_standard_error) or reference_standard_error < 0.0:
        raise ValueError("reference_standard_error must be finite and nonnegative")
    if reference_probability is None and reference_standard_error != 0.0:
        raise ValueError("reference uncertainty requires a reference probability")
    if prepared.resource_censored:
        if checkpoint is not None:
            raise ValueError("a resource-censored preparation cannot have a checkpoint")
        return _censored_result(prepared)

    term, offset, moments, ledger, work = _execution_state(prepared, checkpoint)
    chunks = 0
    while term < len(prepared.allocations):
        allocation = prepared.allocations[term]
        if offset >= allocation.final_count:
            term += 1
            offset = 0
            continue
        count = min(prepared.chunk_size, allocation.final_count - offset)
        replicate = offset // prepared.chunk_size
        seeds = {
            stream: ledger.allocate(
                SeedKey(
                    f"{prepared.protocol}/{prepared.selected_candidate}",
                    "final",
                    prepared.regime,
                    prepared.task,
                    term,
                    replicate,
                    stream,
                )
            )
            for stream in prepared.streams
        }
        batch = sampler(allocation.profile_id, "final", count, seeds)
        if batch.values.numel() != count:
            raise ValueError("sampler returned the wrong final count")
        expected_work = count * allocation.cost_per_sample
        if not math.isclose(batch.work_units, expected_work, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("sampler work differs from the frozen operation cost")
        if float(torch.amax(torch.abs(batch.values))) > allocation.absolute_bound * (1.0 + 1e-12):
            raise ValueError("final contribution exceeds its defensive bound")
        moments[term].update(batch.values)
        work.add(WorkLedgerEntry("final", term, count, batch.work_units, batch.wall_seconds))
        if work.total_work_units > prepared.operation_work_cap * (1.0 + 1e-12):
            raise AssertionError("executed work exceeded the frozen operation-work cap")
        offset += count
        chunks += 1
        if maximum_chunks is not None and chunks >= maximum_chunks:
            break

    complete = term >= len(prepared.allocations)
    if not complete and offset >= prepared.allocations[term].final_count:
        term += 1
        offset = 0
        complete = term >= len(prepared.allocations)
    design = math.fsum(item.design_sampling_variance for item in prepared.allocations)
    if not complete:
        state = HybridCheckpoint(
            schema="npi.g11.hybrid-checkpoint.v1",
            preparation_hash=prepared.preparation_hash,
            selected_candidate=prepared.selected_candidate,
            allocations=tuple(item.final_count for item in prepared.allocations),
            next_term=term,
            next_offset=offset,
            moments=tuple(moments),
            ledger_payload=ledger.to_dict(),
            work_entries=tuple(work.entries),
        )
        return HybridResult(
            complete=False,
            resource_censored=False,
            censoring_reason=None,
            selected_candidate=prepared.selected_candidate,
            estimate=None,
            design_sampling_variance=design,
            empirical_sampling_variance=None,
            standard_error=None,
            asymptotic_confidence_interval=None,
            bounded_confidence_interval=None,
            requested_relative_sampling_rmse=prepared.target.relative_sampling_rmse,
            achieved_relative_sampling_rmse_nominal=None,
            achieved_relative_sampling_rmse_reference=None,
            design_target_attained=design <= prepared.target.sampling_variance_target,
            empirical_target_attained=None,
            reference_z_score=None,
            terms=(),
            allocations=prepared.allocations,
            work=work,
            seed_ledger_hash=ledger.sha256,
            preparation_hash=prepared.preparation_hash,
            checkpoint=state,
        )

    estimates = tuple(
        HybridTermEstimate(
            profile_id=allocation.profile_id,
            count=moment.count,
            mean=moment.mean,
            variance=moment.variance,
            sampling_variance=moment.variance / moment.count,
        )
        for allocation, moment in zip(prepared.allocations, moments, strict=True)
    )
    estimate = math.fsum(item.mean for item in estimates)
    empirical = math.fsum(item.sampling_variance for item in estimates)
    standard_error = math.sqrt(max(0.0, empirical))
    critical = NormalDist().inv_cdf(0.5 + prepared.target.confidence_level / 2.0)
    asymptotic = (
        estimate - critical * standard_error,
        estimate + critical * standard_error,
    )
    family_alpha = 1.0 - prepared.target.confidence_level
    term_alpha = family_alpha / len(prepared.allocations)
    bounded_radius = math.fsum(
        allocation.absolute_bound
        * math.sqrt(2.0 * math.log(2.0 / term_alpha) / allocation.final_count)
        for allocation in prepared.allocations
    )
    bounded_raw = (
        estimate - bounded_radius,
        estimate + bounded_radius,
    )
    bounded_clipped = (
        max(0.0, estimate - bounded_radius),
        min(1.0, estimate + bounded_radius),
    )
    bounded = bounded_clipped if bounded_clipped[0] <= bounded_clipped[1] else bounded_raw
    achieved_nominal = standard_error / prepared.target.nominal_probability
    achieved_reference = (
        standard_error / reference_probability if reference_probability is not None else None
    )
    reference_z = None
    if reference_probability is not None:
        denominator = math.sqrt(empirical + reference_standard_error**2)
        reference_z = (
            (estimate - reference_probability) / denominator
            if denominator > 0.0
            else (
                0.0
                if estimate == reference_probability
                else math.copysign(math.inf, estimate - reference_probability)
            )
        )
    return HybridResult(
        complete=True,
        resource_censored=False,
        censoring_reason=None,
        selected_candidate=prepared.selected_candidate,
        estimate=estimate,
        design_sampling_variance=design,
        empirical_sampling_variance=empirical,
        standard_error=standard_error,
        asymptotic_confidence_interval=asymptotic,
        bounded_confidence_interval=bounded,
        requested_relative_sampling_rmse=prepared.target.relative_sampling_rmse,
        achieved_relative_sampling_rmse_nominal=achieved_nominal,
        achieved_relative_sampling_rmse_reference=achieved_reference,
        design_target_attained=design <= prepared.target.sampling_variance_target,
        empirical_target_attained=empirical <= prepared.target.sampling_variance_target,
        reference_z_score=reference_z,
        terms=estimates,
        allocations=prepared.allocations,
        work=work,
        seed_ledger_hash=ledger.sha256,
        preparation_hash=prepared.preparation_hash,
        checkpoint=None,
    )


def save_hybrid_checkpoint(
    checkpoint: HybridCheckpoint,
    path: str | Path,
    *,
    replace_attempts: int = 10,
    initial_retry_seconds: float = 0.025,
) -> None:
    """Durably publish JSON and retry transient Windows replacement failures."""

    if replace_attempts < 1 or initial_retry_seconds < 0.0:
        raise ValueError("invalid checkpoint retry policy")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(
        checkpoint.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(replace_attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt + 1 == replace_attempts:
                raise
            time.sleep(initial_retry_seconds * 2**attempt)


def load_hybrid_checkpoint(path: str | Path) -> HybridCheckpoint:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("hybrid checkpoint is unreadable or corrupt") from error
    if not isinstance(payload, dict):
        raise ValueError("hybrid checkpoint root must be an object")
    return HybridCheckpoint.from_dict(payload)
