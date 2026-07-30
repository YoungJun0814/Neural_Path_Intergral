"""Immutable statistical contracts for sharded independent references."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

ReferenceStage = Literal["pilot", "final"]
ReferenceMethod = Literal["dcs_reference", "raw_crosscheck"]
REFERENCE_METHODS: tuple[ReferenceMethod, ...] = (
    "dcs_reference",
    "raw_crosscheck",
)
SHARD_SCHEMA = "npi.g11.v8-reference-shard.v1"
ALLOCATION_SCHEMA = "npi.g11.v8-reference-allocation-manifest.v1"


def canonical_json_bytes(payload: object) -> bytes:
    """Return the canonical UTF-8 representation used by all R1 hashes."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sha256_text(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: str, field: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be nonempty and already stripped")
    return value


def _git_commit(value: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("source_commit must be a lowercase full Git commit")
    return value


@dataclass(frozen=True)
class SufficientStatistics:
    """Mergeable ordinary-mean sufficient statistics."""

    count: int
    mean: float
    m2: float

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("sufficient-statistic count must be positive")
        if not math.isfinite(self.mean) or not math.isfinite(self.m2) or self.m2 < 0.0:
            raise ValueError("sufficient statistics must be finite with nonnegative M2")

    @classmethod
    def from_tensor(cls, values: torch.Tensor) -> SufficientStatistics:
        data = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if data.numel() < 1 or not bool(torch.isfinite(data).all()):
            raise FloatingPointError("reference statistics require a nonempty finite tensor")
        mean = float(torch.mean(data))
        centered = data - mean
        m2 = float(torch.sum(centered * centered))
        return cls(count=int(data.numel()), mean=mean, m2=m2)

    @classmethod
    def from_dict(cls, payload: Any) -> SufficientStatistics:
        if not isinstance(payload, dict) or set(payload) != {"count", "mean", "m2"}:
            raise ValueError("invalid sufficient-statistic payload")
        count, mean, m2 = payload["count"], payload["mean"], payload["m2"]
        if not isinstance(count, int) or isinstance(mean, bool) or isinstance(m2, bool):
            raise ValueError("invalid sufficient-statistic field types")
        if not isinstance(mean, (int, float)) or not isinstance(m2, (int, float)):
            raise ValueError("invalid sufficient-statistic numeric fields")
        return cls(count=count, mean=float(mean), m2=float(m2))

    def to_dict(self) -> dict[str, int | float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else math.nan

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.variance / self.count)

    def merge(self, other: SufficientStatistics) -> SufficientStatistics:
        total = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / total
        m2 = (
            self.m2
            + other.m2
            + delta * delta * self.count * other.count / total
        )
        return SufficientStatistics(count=total, mean=mean, m2=m2)


@dataclass(frozen=True, order=True)
class ReferenceShardIdentity:
    """A unique pilot or final shard in one reference namespace."""

    protocol_id: str
    namespace: str
    stage: ReferenceStage
    method: ReferenceMethod
    cell_id: str
    shard_index: int

    def __post_init__(self) -> None:
        _nonempty(self.protocol_id, "protocol_id")
        _nonempty(self.namespace, "namespace")
        _nonempty(self.cell_id, "cell_id")
        if self.stage not in ("pilot", "final"):
            raise ValueError("reference stage must be pilot or final")
        if self.method not in REFERENCE_METHODS:
            raise ValueError("unsupported reference method")
        if self.shard_index < 0:
            raise ValueError("reference shard index must be nonnegative")

    @classmethod
    def from_dict(cls, payload: Any) -> ReferenceShardIdentity:
        expected = {
            "protocol_id",
            "namespace",
            "stage",
            "method",
            "cell_id",
            "shard_index",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("invalid reference shard identity")
        return cls(**payload)

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @property
    def shard_id(self) -> str:
        digest = canonical_sha256(self.to_dict())[:20]
        return f"{self.stage}-{self.method}-{self.shard_index:06d}-{digest}"


def build_shard_artifact(
    *,
    identity: ReferenceShardIdentity,
    config_sha256: str,
    threshold_manifest_sha256: str,
    parent_sha256: str,
    source_commit: str,
    dirty_worktree: bool,
    environment_sha256: str,
    estimand: str,
    dtype: str,
    device: str,
    seed_key_sha256: str,
    requested_samples: int,
    contribution: SufficientStatistics,
    likelihood_normalization: SufficientStatistics,
    invalid_spot_count: int,
    invalid_variance_count: int,
    nonfinite_contribution_count: int,
    elapsed_wall_seconds: float,
    elapsed_cpu_seconds: float,
    peak_resident_memory_bytes: int,
) -> dict[str, Any]:
    """Build and validate one immutable shard payload."""

    artifact: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "identity": identity.to_dict(),
        "shard_id": identity.shard_id,
        "config_sha256": _sha256_text(config_sha256, "config_sha256"),
        "threshold_manifest_sha256": _sha256_text(
            threshold_manifest_sha256, "threshold_manifest_sha256"
        ),
        "parent_sha256": _sha256_text(parent_sha256, "parent_sha256"),
        "source_commit": _git_commit(source_commit),
        "dirty_worktree": dirty_worktree,
        "environment_sha256": _sha256_text(environment_sha256, "environment_sha256"),
        "estimand": _nonempty(estimand, "estimand"),
        "dtype": _nonempty(dtype, "dtype"),
        "device": _nonempty(device, "device"),
        "seed_key_sha256": _sha256_text(seed_key_sha256, "seed_key_sha256"),
        "requested_samples": requested_samples,
        "contribution": contribution.to_dict(),
        "likelihood_normalization": likelihood_normalization.to_dict(),
        "invalid_spot_count": invalid_spot_count,
        "invalid_variance_count": invalid_variance_count,
        "nonfinite_contribution_count": nonfinite_contribution_count,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "elapsed_cpu_seconds": elapsed_cpu_seconds,
        "peak_resident_memory_bytes": peak_resident_memory_bytes,
        "complete": True,
    }
    validate_shard_artifact(artifact)
    return artifact


def validate_shard_artifact(payload: Any) -> dict[str, Any]:
    expected = {
        "schema",
        "identity",
        "shard_id",
        "config_sha256",
        "threshold_manifest_sha256",
        "parent_sha256",
        "source_commit",
        "dirty_worktree",
        "environment_sha256",
        "estimand",
        "dtype",
        "device",
        "seed_key_sha256",
        "requested_samples",
        "contribution",
        "likelihood_normalization",
        "invalid_spot_count",
        "invalid_variance_count",
        "nonfinite_contribution_count",
        "elapsed_wall_seconds",
        "elapsed_cpu_seconds",
        "peak_resident_memory_bytes",
        "complete",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("malformed reference shard")
    if payload.get("schema") != SHARD_SCHEMA or payload.get("complete") is not True:
        raise ValueError("reference shard is not complete")
    identity = ReferenceShardIdentity.from_dict(payload["identity"])
    if payload.get("shard_id") != identity.shard_id:
        raise ValueError("reference shard identity hash mismatch")
    for field in (
        "config_sha256",
        "threshold_manifest_sha256",
        "parent_sha256",
        "environment_sha256",
        "seed_key_sha256",
    ):
        _sha256_text(payload[field], field)
    _git_commit(payload["source_commit"])
    for field in ("estimand", "dtype", "device"):
        _nonempty(payload[field], field)
    if not isinstance(payload["dirty_worktree"], bool):
        raise ValueError("dirty_worktree must be Boolean")
    requested = payload["requested_samples"]
    if not isinstance(requested, int) or requested < 2:
        raise ValueError("requested_samples must be an integer of at least two")
    contribution = SufficientStatistics.from_dict(payload["contribution"])
    normalization = SufficientStatistics.from_dict(payload["likelihood_normalization"])
    if contribution.count != requested or normalization.count != requested:
        raise ValueError("reference shard sufficient-statistic count mismatch")
    for field in (
        "invalid_spot_count",
        "invalid_variance_count",
        "nonfinite_contribution_count",
        "peak_resident_memory_bytes",
    ):
        value = payload[field]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
        if field != "peak_resident_memory_bytes" and value > requested:
            raise ValueError(f"{field} cannot exceed requested_samples")
    for field in ("elapsed_wall_seconds", "elapsed_cpu_seconds"):
        value = payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{field} must be finite and nonnegative")
    return payload


def seed_key_sha256(seed_key_payload: object) -> str:
    return canonical_sha256(seed_key_payload)
