"""Strict training-inclusive work accounting for V6 method comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

V6WorkCategory = Literal[
    "environment_setup",
    "screening",
    "routing",
    "proposal_training",
    "selector_profile",
    "allocation_pilot",
    "final",
    "checkpoint",
    "failed_attempt",
    "retry",
    "audit",
]

_CATEGORIES = {
    "environment_setup",
    "screening",
    "routing",
    "proposal_training",
    "selector_profile",
    "allocation_pilot",
    "final",
    "checkpoint",
    "failed_attempt",
    "retry",
    "audit",
}


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a nonempty stripped string")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _real(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be a finite real number at least {minimum}")
    return result


@dataclass(frozen=True)
class V6WorkRecord:
    category: V6WorkCategory
    method: str
    cell_id: str
    attempt: int
    samples: int
    work_units: float
    wall_seconds: float
    cpu_seconds: float
    peak_memory_bytes: int
    successful: bool

    def __post_init__(self) -> None:
        if self.category not in _CATEGORIES:
            raise ValueError("unsupported V6 work category")
        _text(self.method, "method")
        _text(self.cell_id, "cell_id")
        _integer(self.attempt, "attempt")
        _integer(self.samples, "samples")
        _real(self.work_units, "work_units")
        _real(self.wall_seconds, "wall_seconds")
        _real(self.cpu_seconds, "cpu_seconds")
        _integer(self.peak_memory_bytes, "peak_memory_bytes")
        if not isinstance(self.successful, bool):
            raise ValueError("successful must be boolean")
        if self.category == "failed_attempt" and self.successful:
            raise ValueError("failed_attempt records cannot be successful")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> V6WorkRecord:
        fields = {
            "category",
            "method",
            "cell_id",
            "attempt",
            "samples",
            "work_units",
            "wall_seconds",
            "cpu_seconds",
            "peak_memory_bytes",
            "successful",
        }
        if set(payload) != fields:
            raise ValueError("malformed V6 work-record fields")
        category = payload["category"]
        if category not in _CATEGORIES:
            raise ValueError("unsupported V6 work category")
        successful = payload["successful"]
        if not isinstance(successful, bool):
            raise ValueError("successful must be boolean")
        return cls(
            category=cast(V6WorkCategory, category),
            method=_text(payload["method"], "method"),
            cell_id=_text(payload["cell_id"], "cell_id"),
            attempt=_integer(payload["attempt"], "attempt"),
            samples=_integer(payload["samples"], "samples"),
            work_units=_real(payload["work_units"], "work_units"),
            wall_seconds=_real(payload["wall_seconds"], "wall_seconds"),
            cpu_seconds=_real(payload["cpu_seconds"], "cpu_seconds"),
            peak_memory_bytes=_integer(payload["peak_memory_bytes"], "peak_memory_bytes"),
            successful=successful,
        )


@dataclass(frozen=True)
class V6WorkLedger:
    """Immutable ledger; failures and retries are deliberately never subtracted."""

    records: tuple[V6WorkRecord, ...] = ()

    def append(self, record: V6WorkRecord) -> V6WorkLedger:
        return V6WorkLedger(self.records + (record,))

    @property
    def total_work_units(self) -> float:
        return math.fsum(record.work_units for record in self.records)

    @property
    def total_wall_seconds(self) -> float:
        return math.fsum(record.wall_seconds for record in self.records)

    @property
    def total_cpu_seconds(self) -> float:
        return math.fsum(record.cpu_seconds for record in self.records)

    @property
    def peak_memory_bytes(self) -> int:
        return max((record.peak_memory_bytes for record in self.records), default=0)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def category_work(self, category: V6WorkCategory) -> float:
        if category not in _CATEGORIES:
            raise ValueError("unsupported V6 work category")
        return math.fsum(
            record.work_units for record in self.records if record.category == category
        )

    def require_categories(self, categories: tuple[V6WorkCategory, ...]) -> None:
        present = {record.category for record in self.records}
        missing = set(categories) - present
        if missing:
            raise ValueError(f"V6 work ledger is missing required categories: {sorted(missing)}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "npi.g11.v6-work-ledger.v1",
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> V6WorkLedger:
        if set(payload) != {"schema", "records"}:
            raise ValueError("malformed V6 work-ledger fields")
        if payload["schema"] != "npi.g11.v6-work-ledger.v1":
            raise ValueError("unsupported V6 work-ledger schema")
        raw_records = payload["records"]
        if not isinstance(raw_records, list):
            raise ValueError("V6 work-ledger records must be a list")
        records = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise ValueError("V6 work record must be an object")
            records.append(V6WorkRecord.from_dict(dict(raw)))
        return cls(tuple(records))
