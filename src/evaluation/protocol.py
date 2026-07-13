"""Frozen experiment protocol loading and disjoint-seed validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FrozenSeedSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    evaluation: tuple[int, ...]

    def validate(self) -> None:
        groups = {
            "train": self.train,
            "validation": self.validation,
            "evaluation": self.evaluation,
        }
        for name, seeds in groups.items():
            if not seeds:
                raise ValueError(f"{name} seeds must not be empty")
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"{name} seeds contain duplicates")
            if any(seed < 0 for seed in seeds):
                raise ValueError(f"{name} seeds must be nonnegative")
        names = tuple(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = set(groups[left]) & set(groups[right])
                if overlap:
                    raise ValueError(f"{left}/{right} seed overlap: {sorted(overlap)}")


@dataclass(frozen=True)
class FrozenExperimentProtocol:
    protocol_id: str
    schema_version: int
    frozen: bool
    seeds: FrozenSeedSplit
    payload: dict[str, Any]
    sha256: str


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_frozen_protocol(path: str | Path) -> FrozenExperimentProtocol:
    """Load a YAML protocol, require it to be frozen, and fingerprint content."""
    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("protocol root must be a mapping")
    if raw.get("frozen") is not True:
        raise ValueError("benchmark protocol must explicitly set frozen: true")
    if not isinstance(raw.get("protocol_id"), str) or not raw["protocol_id"]:
        raise ValueError("protocol_id must be a nonempty string")
    if not isinstance(raw.get("schema_version"), int) or raw["schema_version"] <= 0:
        raise ValueError("schema_version must be a positive integer")

    raw_seeds = raw.get("seeds")
    if not isinstance(raw_seeds, dict):
        raise ValueError("seeds must be a mapping")
    try:
        seeds = FrozenSeedSplit(
            train=tuple(int(seed) for seed in raw_seeds["train"]),
            validation=tuple(int(seed) for seed in raw_seeds["validation"]),
            evaluation=tuple(int(seed) for seed in raw_seeds["evaluation"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("seeds must define train, validation, and evaluation lists") from error
    seeds.validate()
    return FrozenExperimentProtocol(
        protocol_id=raw["protocol_id"],
        schema_version=raw["schema_version"],
        frozen=True,
        seeds=seeds,
        payload=raw,
        sha256=_canonical_hash(raw),
    )
