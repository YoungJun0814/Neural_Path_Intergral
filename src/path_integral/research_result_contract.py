"""Versioned, inspectable measurement contract for post-audit experiments."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.path_integral.provenance import runtime_provenance, source_provenance

SCHEMA = "npi.post-audit.measurement.v1"


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_patch_digest(root: Path) -> str:
    """Hash tracked changes and untracked files without embedding their contents."""

    diff = subprocess.check_output(("git", "diff", "--binary", "HEAD"), cwd=root)
    paths = subprocess.check_output(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"), cwd=root
    ).split(b"\0")
    digest = hashlib.sha256(diff)
    for raw in sorted(path for path in paths if path):
        relative = raw.decode("utf-8", errors="surrogateescape")
        digest.update(raw + b"\0")
        digest.update(hashlib.sha256((root / relative).read_bytes()).digest())
    return digest.hexdigest()


def source_tree_digest(root: Path) -> str:
    """Hash runtime source/config contents independently of Git staging state."""

    paths = subprocess.check_output(
        (
            "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
            "--", "src", "experiments", "configs", "main.py", "train_driftnet.py",
        ),
        cwd=root,
    ).split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(set(path for path in paths if path)):
        relative = raw.decode("utf-8", errors="surrogateescape")
        path = root / relative
        if path.is_file():
            digest.update(raw + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def source_manifest(root: Path, *, config: dict[str, Any]) -> dict[str, Any]:
    provenance = source_provenance()
    return {
        "source_commit": provenance["source_commit"],
        "source_dirty": provenance["dirty_worktree"],
        "source_patch_digest": source_patch_digest(root),
        "source_tree_digest": source_tree_digest(root),
        "config_digest": canonical_digest(config),
        "runtime": runtime_provenance(dtype="float64"),
    }


@dataclass(frozen=True)
class Moments:
    """Central second-moment summary of independent inferential units."""

    count: int
    mean: float
    m2: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("moment count must be a positive integer")
        if (
            isinstance(self.mean, bool)
            or isinstance(self.m2, bool)
            or not math.isfinite(self.mean)
            or not math.isfinite(self.m2)
            or self.m2 < 0
        ):
            raise ValueError("moments must be finite with nonnegative m2")

    @classmethod
    def from_values(cls, values: torch.Tensor) -> Moments:
        if (
            values.ndim != 1
            or values.numel() == 0
            or values.dtype != torch.float64
            or values.device.type != "cpu"
            or not torch.isfinite(values).all()
        ):
            raise ValueError("inferential units must be finite CPU float64 vector")
        mean = float(torch.mean(values))
        m2 = float(torch.sum((values - mean).square()))
        return cls(int(values.numel()), mean, m2)

    def merge(self, other: Moments) -> Moments:
        n = self.count + other.count
        delta = other.mean - self.mean
        return Moments(
            count=n,
            mean=self.mean + delta * other.count / n,
            m2=self.m2 + other.m2 + delta * delta * self.count * other.count / n,
        )

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            raise ValueError("at least two independent units are needed")
        return self.m2 / (self.count - 1)

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.sample_variance / self.count)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class StageCost:
    stage: str
    wall_seconds: float
    cpu_seconds: float
    peak_memory_bytes: int
    proxy_work_units: float

    def __post_init__(self) -> None:
        if self.stage not in {"offline", "fit", "selection", "inference", "reference"}:
            raise ValueError("unsupported cost stage")
        if any(
            isinstance(x, bool) or not math.isfinite(x) or x < 0
            for x in (self.wall_seconds, self.cpu_seconds, self.proxy_work_units)
        ):
            raise ValueError("stage costs must be finite and nonnegative")
        if (
            isinstance(self.peak_memory_bytes, bool)
            or not isinstance(self.peak_memory_bytes, int)
            or self.peak_memory_bytes < 0
        ):
            raise ValueError("peak memory must be a nonnegative integer")

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
