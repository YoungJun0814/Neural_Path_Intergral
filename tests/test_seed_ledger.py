"""Seed namespace, collision, and serialization tests for G11."""

from __future__ import annotations

import copy

import pytest

from src.path_integral.seed_ledger import SeedKey, SeedLedger, derive_seed


def _key(replicate: int = 0, stream: str = "proposal") -> SeedKey:
    return SeedKey("g11-test", "pilot", "r0", "barrier", 2, replicate, stream)


def test_seed_depends_on_every_namespace_field() -> None:
    base = _key()
    fields = {
        "protocol": "g11-other",
        "role": "final",
        "regime": "r1",
        "task": "terminal",
        "level": 3,
        "replicate": 1,
        "stream": "labels",
    }
    seeds = {derive_seed(base)}
    for field, value in fields.items():
        payload = copy.copy(base.__dict__)
        payload[field] = value
        seeds.add(derive_seed(SeedKey(**payload)))
    assert len(seeds) == len(fields) + 1


def test_ledger_rejects_duplicate_allocation_and_round_trips() -> None:
    ledger = SeedLedger()
    for replicate in range(50):
        for stream in ("proposal", "labels", "bootstrap", "reference"):
            ledger.allocate(_key(replicate, stream))
    with pytest.raises(ValueError, match="duplicate seed key"):
        ledger.allocate(_key(0, "proposal"))
    restored = SeedLedger.from_dict(ledger.to_dict())
    assert restored.canonical_json() == ledger.canonical_json()
    assert restored.sha256 == ledger.sha256
    assert len({record.seed for record in restored.records}) == len(restored)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"protocol": ""},
        {"role": " pilot"},
        {"level": -1},
        {"replicate": -1},
        {"stream": ""},
    ],
)
def test_invalid_keys_are_rejected(kwargs: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "protocol": "g11-test",
        "role": "pilot",
        "regime": "r0",
        "task": "barrier",
        "level": 0,
        "replicate": 0,
        "stream": "proposal",
    }
    payload.update(kwargs)
    with pytest.raises(ValueError):
        SeedKey(**payload)


def test_tampered_seed_manifest_is_rejected() -> None:
    ledger = SeedLedger()
    ledger.allocate(_key())
    payload = ledger.to_dict()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict) and isinstance(record["seed"], int)
    record["seed"] += 1
    with pytest.raises(ValueError, match="does not match"):
        SeedLedger.from_dict(payload)
