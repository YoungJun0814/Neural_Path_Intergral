"""Analytic telescoping, allocation, coverage, and resume tests for G11 MLMC."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from src.path_integral.mlmc import (
    ContinuousTarget,
    FixedFinestGridTarget,
    LevelBatch,
    MLMCHierarchy,
    execute_mlmc,
    load_mlmc_checkpoint,
    prepare_mlmc,
    save_mlmc_checkpoint,
)


class GaussianTelescopingSampler:
    """Cheap oracle with E[Y_0]+sum E[Y_l] = theta_L."""

    def __init__(self, theta: tuple[float, ...]) -> None:
        self.theta = theta

    def __call__(
        self,
        level: int,
        role: str,
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch:
        del role
        generator = torch.Generator().manual_seed(seeds["proposal"])
        mean = (
            self.theta[0]
            if level == 0
            else self.theta[level] - self.theta[level - 1]
        )
        scale = 0.8 * 2.0 ** (-0.6 * level)
        values = mean + scale * torch.randn(
            count, generator=generator, dtype=torch.float64
        )
        return LevelBatch(values, work_units=count * (2**level))


def _prepared(protocol: str = "g11-mlmc-test"):
    hierarchy = MLMCHierarchy(4, 2, FixedFinestGridTarget(3))
    sampler = GaussianTelescopingSampler((0.1, 0.16, 0.19, 0.205))
    prepared = prepare_mlmc(
        hierarchy,
        sampler,
        protocol=protocol,
        regime="gaussian",
        task="linear",
        sampling_variance_target=3e-4,
        pilot_samples=1024,
        chunk_size=127,
        allocation_safety_factor=1.1,
    )
    return sampler, prepared


def test_integer_allocation_meets_design_target_and_pilots_are_discarded() -> None:
    sampler, prepared = _prepared()
    result = execute_mlmc(prepared, sampler)
    assert result.complete
    assert result.design_sampling_variance <= prepared.sampling_variance_target / 1.1
    assert result.estimate == sum(item.mean for item in result.levels)
    assert all(
        item.count == allocation.final_count
        for item, allocation in zip(
            result.levels, prepared.allocations, strict=True
        )
    )
    assert {entry.role for entry in result.work.entries} == {"pilot", "final"}
    assert (
        sum(
            entry.samples for entry in result.work.entries if entry.role == "pilot"
        )
        == 4 * 1024
    )
    assert len({record.seed for record in prepared.ledger.records}) == len(
        prepared.ledger
    )


def test_checkpoint_resume_is_bitwise_identical_and_has_no_duplicate_samples(
    tmp_path: Path,
) -> None:
    sampler, prepared = _prepared("g11-resume-test")
    uninterrupted = execute_mlmc(prepared, sampler)
    partial = execute_mlmc(prepared, sampler, maximum_chunks=5)
    assert not partial.complete and partial.checkpoint is not None
    path = tmp_path / "checkpoint.json"
    save_mlmc_checkpoint(partial.checkpoint, path)
    restored = load_mlmc_checkpoint(path)
    resumed = execute_mlmc(prepared, sampler, checkpoint=restored)
    assert resumed.complete
    assert resumed.estimate == uninterrupted.estimate
    assert (
        resumed.empirical_sampling_variance
        == uninterrupted.empirical_sampling_variance
    )
    assert resumed.seed_ledger_hash == uninterrupted.seed_ledger_hash
    assert resumed.levels == uninterrupted.levels
    assert resumed.work.entries == uninterrupted.work.entries
    final_samples = sum(
        entry.samples for entry in resumed.work.entries if entry.role == "final"
    )
    assert final_samples == sum(item.final_count for item in prepared.allocations)


def test_checkpoint_tampering_is_rejected(tmp_path: Path) -> None:
    sampler, prepared = _prepared("g11-tamper-test")
    partial = execute_mlmc(prepared, sampler, maximum_chunks=1)
    assert partial.checkpoint is not None
    payload = partial.checkpoint.to_dict()
    allocations = payload["allocations"]
    assert isinstance(allocations, list)
    allocations[0] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = load_mlmc_checkpoint(path)
    with pytest.raises(ValueError, match="does not match"):
        execute_mlmc(prepared, sampler, checkpoint=checkpoint)


def test_fixed_and_continuous_targets_cannot_be_silently_relabelled() -> None:
    fixed = MLMCHierarchy(8, 2, FixedFinestGridTarget(4))
    continuous = MLMCHierarchy(8, 2, ContinuousTarget(4, 1e-3))
    assert fixed.finest_level == continuous.finest_level == 4
    assert isinstance(fixed.target, FixedFinestGridTarget)
    assert isinstance(continuous.target, ContinuousTarget)
    with pytest.raises(ValueError, match="positive"):
        ContinuousTarget(4, 0.0)


def test_gaussian_oracle_confidence_interval_has_nominal_small_run_coverage() -> None:
    truth = 0.205
    covered = 0
    repetitions = 1000
    for replicate in range(repetitions):
        hierarchy = MLMCHierarchy(2, 2, FixedFinestGridTarget(2))
        sampler = GaussianTelescopingSampler((0.1, 0.17, truth))
        prepared = prepare_mlmc(
            hierarchy,
            sampler,
            protocol=f"g11-coverage-{replicate}",
            regime="gaussian",
            task="linear",
            sampling_variance_target=2.5e-3,
            pilot_samples=128,
            chunk_size=512,
            allocation_safety_factor=1.0,
        )
        result = execute_mlmc(prepared, sampler)
        assert result.confidence_interval_95 is not None
        low, high = result.confidence_interval_95
        covered += int(low <= truth <= high)
    assert 0.93 <= covered / repetitions <= 0.97
