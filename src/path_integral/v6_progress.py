"""Strict artifact-level progress journals for long V6 experiment matrices."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class V6ProgressJournal:
    experiment: str
    identities: dict[str, object]
    records: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.experiment or self.experiment.strip() != self.experiment:
            raise ValueError("progress experiment must be a nonempty stripped string")
        if not self.identities:
            raise ValueError("progress journal requires immutable identities")

    @property
    def records_sha256(self) -> str:
        return _hash(list(self.records))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "npi.g11.v6-progress.v1",
            "experiment": self.experiment,
            "identities": self.identities,
            "records": list(self.records),
            "records_sha256": self.records_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> V6ProgressJournal:
        if set(payload) != {
            "schema",
            "experiment",
            "identities",
            "records",
            "records_sha256",
        }:
            raise ValueError("malformed V6 progress journal")
        if payload["schema"] != "npi.g11.v6-progress.v1":
            raise ValueError("unsupported V6 progress schema")
        if not isinstance(payload["experiment"], str):
            raise ValueError("progress experiment must be a string")
        if not isinstance(payload["identities"], dict):
            raise ValueError("progress identities must be an object")
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or any(
            not isinstance(record, dict) for record in raw_records
        ):
            raise ValueError("progress records must be objects")
        journal = cls(
            experiment=payload["experiment"],
            identities=dict(payload["identities"]),
            records=tuple(dict(record) for record in raw_records),
        )
        if payload["records_sha256"] != journal.records_sha256:
            raise ValueError("V6 progress record hash mismatch")
        return journal


def load_v6_progress(
    path: str | Path,
    *,
    experiment: str,
    identities: dict[str, object],
) -> V6ProgressJournal:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V6 progress artifact must be an object")
    journal = V6ProgressJournal.from_dict(payload)
    if journal.experiment != experiment or journal.identities != identities:
        raise ValueError("V6 progress identity does not match the requested run")
    return journal


def save_v6_progress(path: str | Path, journal: V6ProgressJournal) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(journal.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    for attempt in range(10):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.025 * 2**attempt)


def v6_record_checkpoint_path(directory: str | Path, identity: str) -> Path:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return Path(directory) / "records" / f"{digest}.json"
