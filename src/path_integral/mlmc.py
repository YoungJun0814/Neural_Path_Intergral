"""A target-explicit, seed-audited multilevel Monte Carlo engine."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch

from src.path_integral.seed_ledger import SeedKey, SeedLedger


@dataclass(frozen=True)
class FixedFinestGridTarget:
    """Estimate the expectation on one declared finest finite grid."""

    finest_level: int

    def __post_init__(self) -> None:
        if self.finest_level < 0:
            raise ValueError("finest_level must be nonnegative")


@dataclass(frozen=True)
class ContinuousTarget:
    """Continuous-target mode with a separately budgeted bias tolerance."""

    maximum_level: int
    bias_tolerance: float

    def __post_init__(self) -> None:
        if self.maximum_level < 1:
            raise ValueError("maximum_level must be positive")
        if not math.isfinite(self.bias_tolerance) or self.bias_tolerance <= 0.0:
            raise ValueError("bias_tolerance must be finite and positive")


MLMCTarget = FixedFinestGridTarget | ContinuousTarget


@dataclass(frozen=True)
class MLMCHierarchy:
    """A dyadic (or integer-refined) hierarchy with an explicit estimand type."""

    coarsest_steps: int
    refinement: int
    target: MLMCTarget

    def __post_init__(self) -> None:
        if self.coarsest_steps < 1:
            raise ValueError("coarsest_steps must be positive")
        if self.refinement < 2:
            raise ValueError("refinement must be at least two")

    @property
    def finest_level(self) -> int:
        if isinstance(self.target, FixedFinestGridTarget):
            return self.target.finest_level
        return self.target.maximum_level

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(range(self.finest_level + 1))

    def steps(self, level: int) -> int:
        if level not in self.levels:
            raise ValueError("level is outside the hierarchy")
        return self.coarsest_steps * self.refinement**level


@dataclass(frozen=True)
class LevelBatch:
    """One independently seeded batch of a level-zero term or correction."""

    values: torch.Tensor
    work_units: float
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.values.ndim != 1
            or not self.values.is_floating_point()
            or not torch.isfinite(self.values).all()
        ):
            raise ValueError("batch values must be a finite floating vector")
        if self.values.numel() < 1:
            raise ValueError("batch must contain at least one sample")
        if not math.isfinite(self.work_units) or self.work_units <= 0.0:
            raise ValueError("work_units must be finite and positive")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0.0:
            raise ValueError("wall_seconds must be finite and nonnegative")


class LevelSampler(Protocol):
    def __call__(
        self,
        level: int,
        role: Literal["pilot", "final"],
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch: ...


@dataclass
class OnlineMoments:
    """Mergeable Welford moments; sequential updates make resume deterministic."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        data = values.detach().to(device="cpu", dtype=torch.float64).tolist()
        for value in data:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    def merge(self, other: OnlineMoments) -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean, other.m2
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else math.nan


@dataclass(frozen=True)
class LevelPilotStatistics:
    level: int
    count: int
    mean: float
    variance: float
    cost_per_sample: float


@dataclass(frozen=True)
class LevelAllocation:
    level: int
    continuous_count: float
    final_count: int
    design_variance: float
    cost_per_sample: float


@dataclass(frozen=True)
class LevelEstimate:
    level: int
    count: int
    mean: float
    variance: float
    sampling_variance: float


@dataclass(frozen=True)
class WorkLedgerEntry:
    role: str
    level: int | None
    samples: int
    work_units: float
    wall_seconds: float


@dataclass
class WorkLedger:
    entries: list[WorkLedgerEntry]

    @classmethod
    def empty(cls) -> WorkLedger:
        return cls(entries=[])

    def add(self, entry: WorkLedgerEntry) -> None:
        if entry.samples < 0:
            raise ValueError("work-ledger sample count cannot be negative")
        if (
            not math.isfinite(entry.work_units)
            or entry.work_units < 0.0
            or not math.isfinite(entry.wall_seconds)
            or entry.wall_seconds < 0.0
        ):
            raise ValueError("work-ledger costs must be finite and nonnegative")
        self.entries.append(entry)

    @property
    def total_work_units(self) -> float:
        return math.fsum(entry.work_units for entry in self.entries)

    @property
    def total_wall_seconds(self) -> float:
        return math.fsum(entry.wall_seconds for entry in self.entries)


def _checkpoint_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"checkpoint {field} must be a nonempty stripped string")
    return value


def _checkpoint_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"checkpoint {field} must be an integer at least {minimum}")
    return value


def _checkpoint_real(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"checkpoint {field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"checkpoint {field} must be a finite real number")
    return result


def _checkpoint_moments(value: object) -> OnlineMoments:
    if not isinstance(value, dict) or set(value) != {"count", "mean", "m2"}:
        raise ValueError("invalid checkpoint moment record")
    record: dict[str, object] = dict(value)
    count = _checkpoint_integer(record["count"], "moment count")
    mean = _checkpoint_real(record["mean"], "moment mean")
    m2 = _checkpoint_real(record["m2"], "moment m2", minimum=0.0)
    if count < 2 and m2 != 0.0:
        raise ValueError("checkpoint moment m2 must be zero below two samples")
    if count == 0 and mean != 0.0:
        raise ValueError("checkpoint empty moment must have zero mean")
    return OnlineMoments(count=count, mean=mean, m2=m2)


def _checkpoint_work_entry(value: object) -> WorkLedgerEntry:
    fields = {"role", "level", "samples", "work_units", "wall_seconds"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid checkpoint work-ledger record")
    record: dict[str, object] = dict(value)
    raw_level = record["level"]
    level = None if raw_level is None else _checkpoint_integer(raw_level, "work-ledger level")
    entry = WorkLedgerEntry(
        role=_checkpoint_text(record["role"], "work-ledger role"),
        level=level,
        samples=_checkpoint_integer(record["samples"], "work-ledger samples"),
        work_units=_checkpoint_real(record["work_units"], "work-ledger work units", minimum=0.0),
        wall_seconds=_checkpoint_real(
            record["wall_seconds"], "work-ledger wall seconds", minimum=0.0
        ),
    )
    validator = WorkLedger.empty()
    validator.add(entry)
    return entry


@dataclass(frozen=True)
class MLMCPreparedRun:
    hierarchy: MLMCHierarchy
    protocol: str
    regime: str
    task: str
    streams: tuple[str, ...]
    pilot: tuple[LevelPilotStatistics, ...]
    allocations: tuple[LevelAllocation, ...]
    sampling_variance_target: float
    chunk_size: int
    ledger: SeedLedger
    work: WorkLedger


@dataclass(frozen=True)
class MLMCCheckpoint:
    schema: str
    protocol: str
    regime: str
    task: str
    allocations: tuple[int, ...]
    next_level: int
    next_offset: int
    moments: tuple[OnlineMoments, ...]
    ledger_payload: dict[str, object]
    work_entries: tuple[WorkLedgerEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol": self.protocol,
            "regime": self.regime,
            "task": self.task,
            "allocations": list(self.allocations),
            "next_level": self.next_level,
            "next_offset": self.next_offset,
            "moments": [asdict(item) for item in self.moments],
            "ledger": self.ledger_payload,
            "work_entries": [asdict(item) for item in self.work_entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MLMCCheckpoint:
        if payload.get("schema") != "npi.g11.mlmc-checkpoint.v1":
            raise ValueError("unsupported MLMC checkpoint schema")
        expected_fields = {
            "schema",
            "protocol",
            "regime",
            "task",
            "allocations",
            "next_level",
            "next_offset",
            "moments",
            "ledger",
            "work_entries",
        }
        if set(payload) != expected_fields:
            raise ValueError("malformed MLMC checkpoint fields")

        protocol = _checkpoint_text(payload["protocol"], "protocol")
        regime = _checkpoint_text(payload["regime"], "regime")
        task = _checkpoint_text(payload["task"], "task")
        next_level = _checkpoint_integer(payload["next_level"], "next_level")
        next_offset = _checkpoint_integer(payload["next_offset"], "next_offset")

        raw_allocations = payload["allocations"]
        if not isinstance(raw_allocations, list) or not raw_allocations:
            raise ValueError("checkpoint allocations must be a nonempty list")
        allocations = tuple(
            _checkpoint_integer(item, "allocation", minimum=1) for item in raw_allocations
        )

        raw_moments = payload["moments"]
        if not isinstance(raw_moments, list):
            raise ValueError("checkpoint moments must be a list")
        moments = tuple(_checkpoint_moments(item) for item in raw_moments)

        raw_ledger = payload["ledger"]
        if not isinstance(raw_ledger, dict):
            raise ValueError("checkpoint ledger must be an object")
        ledger_payload: dict[str, object] = dict(raw_ledger)
        try:
            SeedLedger.from_dict(ledger_payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint seed ledger") from error

        raw_work = payload["work_entries"]
        if not isinstance(raw_work, list):
            raise ValueError("checkpoint work entries must be a list")
        work = tuple(_checkpoint_work_entry(item) for item in raw_work)

        if len(moments) != len(allocations):
            raise ValueError("checkpoint moments and allocations must have equal lengths")
        if next_level > len(allocations):
            raise ValueError("checkpoint next level exceeds its allocation hierarchy")
        if next_level == len(allocations) and next_offset != 0:
            raise ValueError("completed checkpoint level must have zero offset")
        if next_level < len(allocations) and next_offset > allocations[next_level]:
            raise ValueError("checkpoint offset exceeds its level allocation")
        expected_moment_counts = tuple(
            allocation if level < next_level else next_offset if level == next_level else 0
            for level, allocation in enumerate(allocations)
        )
        if tuple(item.count for item in moments) != expected_moment_counts:
            raise ValueError("checkpoint moments do not match its execution position")
        if any(
            entry.role == "final" and (entry.level is None or entry.level >= len(allocations))
            for entry in work
        ):
            raise ValueError("checkpoint final-work level is outside its hierarchy")
        final_work_counts = tuple(
            sum(entry.samples for entry in work if entry.role == "final" and entry.level == level)
            for level in range(len(allocations))
        )
        if final_work_counts != expected_moment_counts:
            raise ValueError("checkpoint final-work ledger does not match its moments")

        return cls(
            schema="npi.g11.mlmc-checkpoint.v1",
            protocol=protocol,
            regime=regime,
            task=task,
            allocations=allocations,
            next_level=next_level,
            next_offset=next_offset,
            moments=moments,
            ledger_payload=ledger_payload,
            work_entries=work,
        )


@dataclass(frozen=True)
class MLMCResult:
    complete: bool
    estimate: float | None
    empirical_sampling_variance: float | None
    design_sampling_variance: float
    standard_error: float | None
    confidence_interval_95: tuple[float, float] | None
    levels: tuple[LevelEstimate, ...]
    pilot: tuple[LevelPilotStatistics, ...]
    allocations: tuple[LevelAllocation, ...]
    work: WorkLedger
    seed_ledger_hash: str
    checkpoint: MLMCCheckpoint | None


def _validate_run_inputs(
    sampling_variance_target: float,
    pilot_samples: int,
    chunk_size: int,
    streams: tuple[str, ...],
) -> None:
    if not math.isfinite(sampling_variance_target) or sampling_variance_target <= 0.0:
        raise ValueError("sampling_variance_target must be finite and positive")
    if pilot_samples < 2:
        raise ValueError("pilot_samples must be at least two")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not streams or len(set(streams)) != len(streams):
        raise ValueError("streams must be nonempty and unique")
    if any(not stream or stream.strip() != stream for stream in streams):
        raise ValueError("stream names must be nonempty and stripped")


def _batch_seeds(
    ledger: SeedLedger,
    *,
    protocol: str,
    role: str,
    regime: str,
    task: str,
    level: int,
    replicate: int,
    streams: tuple[str, ...],
) -> dict[str, int]:
    return {
        stream: ledger.allocate(SeedKey(protocol, role, regime, task, level, replicate, stream))
        for stream in streams
    }


def prepare_mlmc(
    hierarchy: MLMCHierarchy,
    sampler: LevelSampler,
    *,
    protocol: str,
    regime: str,
    task: str,
    sampling_variance_target: float,
    pilot_samples: int,
    chunk_size: int = 4096,
    minimum_final_samples: int = 2,
    allocation_safety_factor: float = 1.0,
    streams: tuple[str, ...] = ("proposal", "labels"),
    initial_work_entries: tuple[WorkLedgerEntry, ...] = (),
    minimum_pilot_nonzero: int = 0,
    maximum_pilot_samples: int | None = None,
) -> MLMCPreparedRun:
    """Run discarded pilots and freeze an integer allocation for final samples."""

    _validate_run_inputs(sampling_variance_target, pilot_samples, chunk_size, streams)
    if minimum_final_samples < 2:
        raise ValueError("minimum_final_samples must be at least two")
    if minimum_pilot_nonzero < 0:
        raise ValueError("minimum_pilot_nonzero must be nonnegative")
    maximum_pilot_samples = (
        pilot_samples if maximum_pilot_samples is None else maximum_pilot_samples
    )
    if maximum_pilot_samples < pilot_samples:
        raise ValueError("maximum_pilot_samples cannot be below pilot_samples")
    if not math.isfinite(allocation_safety_factor) or allocation_safety_factor < 1.0:
        raise ValueError("allocation_safety_factor must be at least one")
    ledger = SeedLedger()
    work = WorkLedger.empty()
    for entry in initial_work_entries:
        if entry.role in {"pilot", "final"}:
            raise ValueError("initial work cannot be relabeled as pilot or final")
        work.add(entry)
    pilots: list[LevelPilotStatistics] = []
    for level in hierarchy.levels:
        moments = OnlineMoments()
        nonzero = 0
        pilot_work_units = 0.0
        replicate = 0
        while moments.count < pilot_samples or nonzero < minimum_pilot_nonzero:
            remaining = maximum_pilot_samples - moments.count
            if remaining <= 0:
                break
            count = min(pilot_samples, remaining)
            seeds = _batch_seeds(
                ledger,
                protocol=protocol,
                role="pilot",
                regime=regime,
                task=task,
                level=level,
                replicate=replicate,
                streams=streams,
            )
            batch = sampler(level, "pilot", count, seeds)
            if batch.values.numel() != count:
                raise ValueError("sampler returned the wrong pilot count")
            moments.update(batch.values)
            nonzero += int(torch.count_nonzero(batch.values))
            pilot_work_units += batch.work_units
            work.add(WorkLedgerEntry("pilot", level, count, batch.work_units, batch.wall_seconds))
            replicate += 1
        if nonzero < minimum_pilot_nonzero:
            raise ValueError(
                f"level {level} pilot reached its cap with only {nonzero} nonzero terms"
            )
        variance = moments.variance
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("pilot variance must be finite and positive")
        cost = pilot_work_units / moments.count
        pilots.append(LevelPilotStatistics(level, moments.count, moments.mean, variance, cost))

    root_sum = math.fsum(math.sqrt(item.variance * item.cost_per_sample) for item in pilots)
    allocations: list[LevelAllocation] = []
    for item in pilots:
        continuous = (
            allocation_safety_factor
            * root_sum
            * math.sqrt(item.variance / item.cost_per_sample)
            / sampling_variance_target
        )
        final_count = max(minimum_final_samples, math.ceil(continuous))
        allocations.append(
            LevelAllocation(
                item.level,
                continuous,
                final_count,
                item.variance,
                item.cost_per_sample,
            )
        )
    design_variance = math.fsum(item.design_variance / item.final_count for item in allocations)
    if design_variance > sampling_variance_target / allocation_safety_factor + 1e-15:
        raise AssertionError("integer allocation failed its design variance target")
    return MLMCPreparedRun(
        hierarchy=hierarchy,
        protocol=protocol,
        regime=regime,
        task=task,
        streams=streams,
        pilot=tuple(pilots),
        allocations=tuple(allocations),
        sampling_variance_target=sampling_variance_target,
        chunk_size=chunk_size,
        ledger=ledger,
        work=work,
    )


def _initial_execution_state(
    prepared: MLMCPreparedRun,
    checkpoint: MLMCCheckpoint | None,
) -> tuple[int, int, list[OnlineMoments], SeedLedger, WorkLedger]:
    if checkpoint is None:
        return (
            0,
            0,
            [OnlineMoments() for _ in prepared.hierarchy.levels],
            SeedLedger(prepared.ledger.records),
            WorkLedger(list(prepared.work.entries)),
        )
    expected_allocations = tuple(item.final_count for item in prepared.allocations)
    if (
        checkpoint.protocol != prepared.protocol
        or checkpoint.regime != prepared.regime
        or checkpoint.task != prepared.task
        or checkpoint.allocations != expected_allocations
        or len(checkpoint.moments) != len(expected_allocations)
    ):
        raise ValueError("checkpoint does not match the prepared run")
    ledger = SeedLedger.from_dict(checkpoint.ledger_payload)
    pilot_keys = {record.key for record in prepared.ledger.records}
    if not pilot_keys.issubset({record.key for record in ledger.records}):
        raise ValueError("checkpoint is missing pilot seed records")
    return (
        checkpoint.next_level,
        checkpoint.next_offset,
        [OnlineMoments(item.count, item.mean, item.m2) for item in checkpoint.moments],
        ledger,
        WorkLedger(list(checkpoint.work_entries)),
    )


def execute_mlmc(
    prepared: MLMCPreparedRun,
    sampler: LevelSampler,
    *,
    checkpoint: MLMCCheckpoint | None = None,
    maximum_chunks: int | None = None,
) -> MLMCResult:
    """Execute a frozen allocation, optionally pausing only at batch boundaries."""

    if maximum_chunks is not None and maximum_chunks < 1:
        raise ValueError("maximum_chunks must be positive")
    level, offset, moments, ledger, work = _initial_execution_state(prepared, checkpoint)
    chunks = 0
    while level < len(prepared.allocations):
        target = prepared.allocations[level].final_count
        if offset >= target:
            level += 1
            offset = 0
            continue
        count = min(prepared.chunk_size, target - offset)
        replicate = offset // prepared.chunk_size
        seeds = _batch_seeds(
            ledger,
            protocol=prepared.protocol,
            role="final",
            regime=prepared.regime,
            task=prepared.task,
            level=level,
            replicate=replicate,
            streams=prepared.streams,
        )
        batch = sampler(level, "final", count, seeds)
        if batch.values.numel() != count:
            raise ValueError("sampler returned the wrong final count")
        moments[level].update(batch.values)
        work.add(WorkLedgerEntry("final", level, count, batch.work_units, batch.wall_seconds))
        offset += count
        chunks += 1
        if maximum_chunks is not None and chunks >= maximum_chunks:
            break

    complete = level >= len(prepared.allocations)
    if not complete and offset >= prepared.allocations[level].final_count:
        level += 1
        offset = 0
        complete = level >= len(prepared.allocations)
    design_variance = math.fsum(
        item.design_variance / item.final_count for item in prepared.allocations
    )
    if not complete:
        state = MLMCCheckpoint(
            schema="npi.g11.mlmc-checkpoint.v1",
            protocol=prepared.protocol,
            regime=prepared.regime,
            task=prepared.task,
            allocations=tuple(item.final_count for item in prepared.allocations),
            next_level=level,
            next_offset=offset,
            moments=tuple(moments),
            ledger_payload=ledger.to_dict(),
            work_entries=tuple(work.entries),
        )
        return MLMCResult(
            False,
            None,
            None,
            design_variance,
            None,
            None,
            (),
            prepared.pilot,
            prepared.allocations,
            work,
            ledger.sha256,
            state,
        )

    estimates = tuple(
        LevelEstimate(
            level=index,
            count=item.count,
            mean=item.mean,
            variance=item.variance,
            sampling_variance=item.variance / item.count,
        )
        for index, item in enumerate(moments)
    )
    empirical = math.fsum(item.sampling_variance for item in estimates)
    estimate = math.fsum(item.mean for item in estimates)
    standard_error = math.sqrt(empirical)
    interval = (
        estimate - 1.959963984540054 * standard_error,
        estimate + 1.959963984540054 * standard_error,
    )
    return MLMCResult(
        True,
        estimate,
        empirical,
        design_variance,
        standard_error,
        interval,
        estimates,
        prepared.pilot,
        prepared.allocations,
        work,
        ledger.sha256,
        None,
    )


def save_mlmc_checkpoint(checkpoint: MLMCCheckpoint, path: str | Path) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(
        checkpoint.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)


def load_mlmc_checkpoint(path: str | Path) -> MLMCCheckpoint:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MLMC checkpoint root must be an object")
    return MLMCCheckpoint.from_dict(payload)


def run_mlmc(
    hierarchy: MLMCHierarchy,
    sampler: LevelSampler,
    *,
    protocol: str,
    regime: str,
    task: str,
    sampling_variance_target: float,
    pilot_samples: int,
    chunk_size: int = 4096,
    minimum_final_samples: int = 2,
    allocation_safety_factor: float = 1.0,
    streams: tuple[str, ...] = ("proposal", "labels"),
    initial_work_entries: tuple[WorkLedgerEntry, ...] = (),
    minimum_pilot_nonzero: int = 0,
    maximum_pilot_samples: int | None = None,
) -> MLMCResult:
    """Convenience wrapper for an uninterrupted pilot plus final MLMC run."""

    prepared = prepare_mlmc(
        hierarchy,
        sampler,
        protocol=protocol,
        regime=regime,
        task=task,
        sampling_variance_target=sampling_variance_target,
        pilot_samples=pilot_samples,
        chunk_size=chunk_size,
        minimum_final_samples=minimum_final_samples,
        allocation_safety_factor=allocation_safety_factor,
        streams=streams,
        initial_work_entries=initial_work_entries,
        minimum_pilot_nonzero=minimum_pilot_nonzero,
        maximum_pilot_samples=maximum_pilot_samples,
    )
    return execute_mlmc(prepared, sampler)
