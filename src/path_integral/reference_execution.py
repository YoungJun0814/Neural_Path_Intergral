"""One-shot execution of a frozen pilot or final reference shard."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .provenance import process_peak_resident_memory_bytes
from .reference_protocol import (
    ReferenceShardIdentity,
    SufficientStatistics,
    build_shard_artifact,
)


@dataclass(frozen=True)
class ReferenceBatch:
    """Complete, unfiltered output of one frozen reference shard draw."""

    contribution: torch.Tensor
    likelihood_normalization: torch.Tensor
    invalid_spot_count: int = 0
    invalid_variance_count: int = 0
    nonfinite_contribution_count: int = 0

    def __post_init__(self) -> None:
        contribution = self.contribution
        normalization = self.likelihood_normalization
        if contribution.device.type != "cpu" or normalization.device.type != "cpu":
            raise ValueError("R1 reference batches must remain on CPU")
        if contribution.dtype != torch.float64 or normalization.dtype != torch.float64:
            raise ValueError("R1 reference batches must use float64")
        if contribution.ndim != 1 or normalization.ndim != 1:
            raise ValueError("reference batch tensors must be one-dimensional")
        if contribution.numel() < 2 or normalization.numel() != contribution.numel():
            raise ValueError("reference batch tensors have invalid or unequal counts")
        for field in (
            "invalid_spot_count",
            "invalid_variance_count",
            "nonfinite_contribution_count",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or value < 0 or value > contribution.numel():
                raise ValueError(f"{field} is outside the reference batch")


def execute_reference_shard(
    *,
    identity: ReferenceShardIdentity,
    config_sha256: str,
    threshold_manifest_sha256: str,
    parent_sha256: str,
    source_commit: str,
    dirty_worktree: bool,
    environment_sha256: str,
    seed_key_sha256: str,
    requested_samples: int,
    draw: Callable[[], ReferenceBatch],
) -> dict[str, Any]:
    """Execute exactly one preallocated draw and return an immutable shard payload."""

    if requested_samples < 2:
        raise ValueError("reference shard requires at least two requested samples")
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    batch = draw()
    elapsed_wall = time.perf_counter() - wall_start
    elapsed_cpu = time.process_time() - cpu_start
    if batch.contribution.numel() != requested_samples:
        raise ValueError("draw count differs from the frozen shard allocation")
    nonfinite_count = int((~torch.isfinite(batch.contribution)).sum().item())
    nonfinite_normalization = int(
        (~torch.isfinite(batch.likelihood_normalization)).sum().item()
    )
    if (
        nonfinite_count != batch.nonfinite_contribution_count
        or nonfinite_count > 0
        or nonfinite_normalization > 0
    ):
        raise FloatingPointError(
            "reference draw produced nonfinite contribution or normalization values"
        )
    return build_shard_artifact(
        identity=identity,
        config_sha256=config_sha256,
        threshold_manifest_sha256=threshold_manifest_sha256,
        parent_sha256=parent_sha256,
        source_commit=source_commit,
        dirty_worktree=dirty_worktree,
        environment_sha256=environment_sha256,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        seed_key_sha256=seed_key_sha256,
        requested_samples=requested_samples,
        contribution=SufficientStatistics.from_tensor(batch.contribution),
        likelihood_normalization=SufficientStatistics.from_tensor(
            batch.likelihood_normalization
        ),
        invalid_spot_count=batch.invalid_spot_count,
        invalid_variance_count=batch.invalid_variance_count,
        nonfinite_contribution_count=batch.nonfinite_contribution_count,
        elapsed_wall_seconds=elapsed_wall,
        elapsed_cpu_seconds=elapsed_cpu,
        peak_resident_memory_bytes=process_peak_resident_memory_bytes(),
    )
