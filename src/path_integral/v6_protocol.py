"""Strict identities for the G11 V6 dual-track research protocol.

The objects in this module describe scientific inputs, not runtime estimates.  They
therefore reject unknown fields and keep a canonical hash that can be recorded before
any qualification or confirmatory sample is drawn.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Literal, cast

V6_CELL_MANIFEST_SCHEMA = "npi.g11.v6-cell-manifest.v1"
V6Phase = Literal["development", "qualification", "confirmation"]
V6Task = Literal["terminal_left_tail", "discrete_lower_barrier"]

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _strict_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a nonempty stripped string")
    return value


def _strict_real(
    value: object,
    field: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite real number")
    if lower is not None and (result <= lower if lower_open else result < lower):
        comparator = ">" if lower_open else ">="
        raise ValueError(f"{field} must be {comparator} {lower}")
    if upper is not None and (result >= upper if upper_open else result > upper):
        comparator = "<" if upper_open else "<="
        raise ValueError(f"{field} must be {comparator} {upper}")
    return result


def _strict_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class V6RBergomiCell:
    """One fully declared finite-grid rare-event estimand."""

    cell_id: str
    hurst: float
    eta: float
    xi: float
    rho: float
    spot: float
    maturity: float
    finest_steps: int
    task: V6Task
    event_threshold: float
    nominal_probability: float
    probability_band: tuple[float, float]

    def __post_init__(self) -> None:
        _strict_text(self.cell_id, "cell_id")
        _strict_real(self.hurst, "hurst", lower=0.0, upper=0.5, lower_open=True, upper_open=True)
        _strict_real(self.eta, "eta", lower=0.0, lower_open=True)
        _strict_real(self.xi, "xi", lower=0.0, lower_open=True)
        _strict_real(self.rho, "rho", lower=-1.0, upper=1.0, lower_open=True, upper_open=True)
        _strict_real(self.spot, "spot", lower=0.0, lower_open=True)
        _strict_real(self.maturity, "maturity", lower=0.0, lower_open=True)
        steps = _strict_integer(self.finest_steps, "finest_steps", minimum=2)
        if not _is_power_of_two(steps):
            raise ValueError("finest_steps must be a power of two")
        if self.task not in ("terminal_left_tail", "discrete_lower_barrier"):
            raise ValueError("unsupported V6 task")
        threshold = _strict_real(
            self.event_threshold,
            "event_threshold",
            lower=0.0,
            upper=self.spot,
            lower_open=True,
            upper_open=True,
        )
        del threshold
        probability = _strict_real(
            self.nominal_probability,
            "nominal_probability",
            lower=0.0,
            upper=0.05,
            lower_open=True,
        )
        if not isinstance(self.probability_band, tuple) or len(self.probability_band) != 2:
            raise ValueError("probability_band must be a pair")
        lower = _strict_real(
            self.probability_band[0],
            "probability_band lower",
            lower=0.0,
            upper=0.05,
            lower_open=True,
        )
        upper = _strict_real(
            self.probability_band[1],
            "probability_band upper",
            lower=0.0,
            upper=0.05,
            lower_open=True,
        )
        if lower >= upper:
            raise ValueError("probability_band must be strictly increasing")
        if not lower <= probability <= upper:
            raise ValueError("nominal_probability must lie inside probability_band")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["probability_band"] = list(self.probability_band)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> V6RBergomiCell:
        fields = {
            "cell_id",
            "hurst",
            "eta",
            "xi",
            "rho",
            "spot",
            "maturity",
            "finest_steps",
            "task",
            "event_threshold",
            "nominal_probability",
            "probability_band",
        }
        if set(payload) != fields:
            raise ValueError("malformed V6 cell fields")
        raw_band = payload["probability_band"]
        if not isinstance(raw_band, list) or len(raw_band) != 2:
            raise ValueError("V6 cell probability_band must be a two-item list")
        task = payload["task"]
        if task not in ("terminal_left_tail", "discrete_lower_barrier"):
            raise ValueError("unsupported V6 task")
        return cls(
            cell_id=_strict_text(payload["cell_id"], "cell_id"),
            hurst=_strict_real(payload["hurst"], "hurst"),
            eta=_strict_real(payload["eta"], "eta"),
            xi=_strict_real(payload["xi"], "xi"),
            rho=_strict_real(payload["rho"], "rho"),
            spot=_strict_real(payload["spot"], "spot"),
            maturity=_strict_real(payload["maturity"], "maturity"),
            finest_steps=_strict_integer(payload["finest_steps"], "finest_steps", minimum=2),
            task=cast(V6Task, task),
            event_threshold=_strict_real(payload["event_threshold"], "event_threshold"),
            nominal_probability=_strict_real(
                payload["nominal_probability"], "nominal_probability"
            ),
            probability_band=(
                _strict_real(raw_band[0], "probability_band lower"),
                _strict_real(raw_band[1], "probability_band upper"),
            ),
        )


@dataclass(frozen=True)
class V6CellManifest:
    """Strict, hashable cell manifest separated by experimental phase."""

    schema: str
    protocol: str
    phase: V6Phase
    frozen: bool
    source_commit: str
    dirty_tree: bool
    config_sha256: str
    smoke: bool
    cells: tuple[V6RBergomiCell, ...]

    def __post_init__(self) -> None:
        if self.schema != V6_CELL_MANIFEST_SCHEMA:
            raise ValueError("unsupported V6 cell-manifest schema")
        protocol = _strict_text(self.protocol, "protocol")
        if not protocol.startswith("g11-v6-"):
            raise ValueError("V6 protocol must start with 'g11-v6-'")
        if self.phase not in ("development", "qualification", "confirmation"):
            raise ValueError("unsupported V6 phase")
        if not isinstance(self.frozen, bool):
            raise ValueError("frozen must be boolean")
        if not isinstance(self.dirty_tree, bool):
            raise ValueError("dirty_tree must be boolean")
        if not isinstance(self.smoke, bool):
            raise ValueError("smoke must be boolean")
        if self.source_commit != "uncommitted" and _HEX_40.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be lowercase 40-hex or 'uncommitted'")
        if _HEX_64.fullmatch(self.config_sha256) is None:
            raise ValueError("config_sha256 must be lowercase 64-hex")
        if not self.cells:
            raise ValueError("V6 cell manifest must contain at least one cell")
        identifiers = tuple(cell.cell_id for cell in self.cells)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("V6 cell identifiers must be unique")
        if self.phase != "development":
            if not self.frozen:
                raise ValueError("qualification and confirmation manifests must be frozen")
            if self.dirty_tree:
                raise ValueError("qualification and confirmation require a clean tree")
            if self.smoke:
                raise ValueError("smoke manifests cannot be scientific evidence")
            if self.source_commit == "uncommitted":
                raise ValueError("qualification and confirmation require a source commit")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol": self.protocol,
            "phase": self.phase,
            "frozen": self.frozen,
            "source_commit": self.source_commit,
            "dirty_tree": self.dirty_tree,
            "config_sha256": self.config_sha256,
            "smoke": self.smoke,
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> V6CellManifest:
        fields = {
            "schema",
            "protocol",
            "phase",
            "frozen",
            "source_commit",
            "dirty_tree",
            "config_sha256",
            "smoke",
            "cells",
        }
        if set(payload) != fields:
            raise ValueError("malformed V6 cell-manifest fields")
        raw_cells = payload["cells"]
        if not isinstance(raw_cells, list):
            raise ValueError("V6 manifest cells must be a list")
        cells: list[V6RBergomiCell] = []
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict):
                raise ValueError("V6 manifest cell must be an object")
            cells.append(V6RBergomiCell.from_dict(dict(raw_cell)))
        phase = payload["phase"]
        if phase not in ("development", "qualification", "confirmation"):
            raise ValueError("unsupported V6 phase")
        frozen = payload["frozen"]
        dirty_tree = payload["dirty_tree"]
        smoke = payload["smoke"]
        if not isinstance(frozen, bool):
            raise ValueError("frozen must be boolean")
        if not isinstance(dirty_tree, bool):
            raise ValueError("dirty_tree must be boolean")
        if not isinstance(smoke, bool):
            raise ValueError("smoke must be boolean")
        return cls(
            schema=_strict_text(payload["schema"], "schema"),
            protocol=_strict_text(payload["protocol"], "protocol"),
            phase=cast(V6Phase, phase),
            frozen=frozen,
            source_commit=_strict_text(payload["source_commit"], "source_commit"),
            dirty_tree=dirty_tree,
            config_sha256=_strict_text(payload["config_sha256"], "config_sha256"),
            smoke=smoke,
            cells=tuple(cells),
        )

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("ascii")).hexdigest()
