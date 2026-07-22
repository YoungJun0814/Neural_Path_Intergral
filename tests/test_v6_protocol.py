"""Strict identity and evidence-boundary tests for the V6 protocol."""

from __future__ import annotations

import copy

import pytest

from src.path_integral.v6_protocol import (
    V6_CELL_MANIFEST_SCHEMA,
    V6CellManifest,
    V6RBergomiCell,
)


def _cell(cell_id: str = "h012-terminal-p1e3") -> V6RBergomiCell:
    return V6RBergomiCell(
        cell_id=cell_id,
        hurst=0.12,
        eta=1.1,
        xi=0.04,
        rho=-0.7,
        spot=100.0,
        maturity=0.25,
        finest_steps=128,
        task="terminal_left_tail",
        event_threshold=80.0,
        nominal_probability=1e-3,
        probability_band=(0.5e-3, 2e-3),
    )


def _manifest() -> V6CellManifest:
    return V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-cell-qualification-v1",
        phase="qualification",
        frozen=True,
        source_commit="a" * 40,
        dirty_tree=False,
        config_sha256="b" * 64,
        smoke=False,
        cells=(_cell(),),
    )


def test_v6_manifest_round_trip_and_hash_are_canonical() -> None:
    manifest = _manifest()
    restored = V6CellManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.sha256 == manifest.sha256
    assert len(manifest.sha256) == 64


@pytest.mark.parametrize("field", ["unexpected", "reference_probability"])
def test_v6_manifest_rejects_unknown_fields(field: str) -> None:
    payload = _manifest().to_dict()
    payload[field] = 1.0
    with pytest.raises(ValueError, match="fields"):
        V6CellManifest.from_dict(payload)


def test_v6_cell_rejects_unknown_field_and_nonrare_band() -> None:
    payload = _cell().to_dict()
    payload["oracle_threshold"] = 79.0
    with pytest.raises(ValueError, match="fields"):
        V6RBergomiCell.from_dict(payload)

    with pytest.raises(ValueError, match="nominal_probability"):
        V6RBergomiCell(**{**_cell().__dict__, "nominal_probability": 0.25})


def test_v6_evidence_rejects_dirty_smoke_unfrozen_or_uncommitted_inputs() -> None:
    base = _manifest().to_dict()
    invalid = (
        {**base, "dirty_tree": True},
        {**base, "smoke": True},
        {**base, "frozen": False},
        {**base, "source_commit": "uncommitted"},
    )
    for payload in invalid:
        with pytest.raises(ValueError):
            V6CellManifest.from_dict(payload)


def test_v6_manifest_rejects_duplicate_cell_identity_and_bool_numeric_fields() -> None:
    payload = _manifest().to_dict()
    payload["cells"] = [copy.deepcopy(payload["cells"][0]), copy.deepcopy(payload["cells"][0])]
    with pytest.raises(ValueError, match="unique"):
        V6CellManifest.from_dict(payload)

    cell = _cell().to_dict()
    cell["finest_steps"] = True
    with pytest.raises(ValueError, match="integer"):
        V6RBergomiCell.from_dict(cell)


def test_development_manifest_may_be_smoke_and_uncommitted_but_stays_identified() -> None:
    manifest = V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-development-smoke-v1",
        phase="development",
        frozen=False,
        source_commit="uncommitted",
        dirty_tree=True,
        config_sha256="c" * 64,
        smoke=True,
        cells=(_cell(),),
    )
    assert manifest.smoke and manifest.dirty_tree and not manifest.frozen
