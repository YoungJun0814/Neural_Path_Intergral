"""Deterministic, auditable seed allocation for research protocols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True)
class SeedKey:
    """One fully qualified random stream in a research protocol."""

    protocol: str
    role: str
    regime: str
    task: str
    level: int
    replicate: int
    stream: str

    def __post_init__(self) -> None:
        text_fields = (
            self.protocol,
            self.role,
            self.regime,
            self.task,
            self.stream,
        )
        if any(not value or value.strip() != value for value in text_fields):
            raise ValueError("seed text fields must be nonempty and already stripped")
        if self.level < 0 or self.replicate < 0:
            raise ValueError("seed level and replicate must be nonnegative")

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")


@dataclass(frozen=True)
class SeedRecord:
    key: SeedKey
    seed: int


def derive_seed(key: SeedKey) -> int:
    """Derive a positive 63-bit seed from every declared namespace field."""

    digest = hashlib.sha256(b"NPI-G11-SEED-V1\x00" + key.canonical_bytes()).digest()
    seed = int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)
    return seed or 1


class SeedLedger:
    """Allocate unique deterministic seeds and expose a canonical manifest hash."""

    def __init__(self, records: Iterable[SeedRecord] = ()) -> None:
        self._by_key: dict[SeedKey, SeedRecord] = {}
        self._by_seed: dict[int, SeedKey] = {}
        for record in records:
            self._insert(record)

    def _insert(self, record: SeedRecord) -> None:
        if record.key in self._by_key:
            raise ValueError(f"duplicate seed key: {record.key}")
        if record.seed <= 0 or record.seed >= 1 << 63:
            raise ValueError("seed must be in [1, 2**63)")
        if record.seed != derive_seed(record.key):
            raise ValueError("seed does not match its canonical key")
        previous = self._by_seed.get(record.seed)
        if previous is not None:
            raise ValueError(f"seed collision between {previous} and {record.key}")
        self._by_key[record.key] = record
        self._by_seed[record.seed] = record.key

    def allocate(self, key: SeedKey) -> int:
        """Allocate once; repeated allocation is rejected rather than hidden."""

        record = SeedRecord(key=key, seed=derive_seed(key))
        self._insert(record)
        return record.seed

    def lookup(self, key: SeedKey) -> int:
        try:
            return self._by_key[key].seed
        except KeyError as error:
            raise KeyError(f"unallocated seed key: {key}") from error

    @property
    def records(self) -> tuple[SeedRecord, ...]:
        return tuple(self._by_key[key] for key in sorted(self._by_key))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "npi.g11.seed-ledger.v1",
            "records": [
                {"key": asdict(record.key), "seed": record.seed}
                for record in self.records
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SeedLedger:
        if payload.get("schema") != "npi.g11.seed-ledger.v1":
            raise ValueError("unsupported seed-ledger schema")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("seed-ledger records must be a list")
        records: list[SeedRecord] = []
        for raw in raw_records:
            if not isinstance(raw, dict) or set(raw) != {"key", "seed"}:
                raise ValueError("invalid seed record")
            raw_key = raw["key"]
            if not isinstance(raw_key, dict) or not isinstance(raw["seed"], int):
                raise ValueError("invalid seed record fields")
            records.append(SeedRecord(SeedKey(**raw_key), raw["seed"]))
        return cls(records)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    def __len__(self) -> int:
        return len(self._by_key)
