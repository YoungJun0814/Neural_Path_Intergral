"""Allocation, separation, censoring, and resume tests for V5 hybrid execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

import src.path_integral.hybrid_allocation as hybrid_module
from src.path_integral import (
    FrozenCrossoverDecision,
    HybridTarget,
    LevelBatch,
    SeedKey,
    SeedLedger,
    SingleTermDesign,
    WorkLedgerEntry,
    execute_hybrid_run,
    load_hybrid_checkpoint,
    prepare_hybrid_run,
    prepare_single_term_run,
    save_hybrid_checkpoint,
    update_profile_intervals,
)


class BoundedTelescopingSampler:
    def __init__(self, costs: Mapping[str, float]) -> None:
        self.costs = costs
        self.calls = 0

    def __call__(
        self,
        profile_id: str,
        role: str,
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch:
        assert role == "final"
        self.calls += 1
        generator = torch.Generator().manual_seed(seeds["proposal"])
        signs = (2 * torch.randint(0, 2, (count,), generator=generator) - 1).to(torch.float64)
        if profile_id == "base":
            values = 0.10 + 0.20 * signs
        elif profile_id == "correction":
            values = 0.05 + 0.10 * signs
        else:
            raise AssertionError("unexpected profile")
        return LevelBatch(values, count * self.costs[profile_id])


def _selection() -> FrozenCrossoverDecision:
    return FrozenCrossoverDecision(
        selected_candidate="hybrid",
        look_index=2,
        reason="pilot cap",
        surviving_candidates=("hybrid", "slis"),
        selected_work_interval=(100.0, 200.0),
        selected_point_work=140.0,
        worst_case_interval_regret_bound=1.25,
    )


def _profiles():
    generator = torch.Generator().manual_seed(15_100_101)
    signs = (2 * torch.randint(0, 2, (2048,), generator=generator) - 1).to(torch.float64)
    return update_profile_intervals(
        {"base": 0.10 + 0.20 * signs, "correction": 0.05 + 0.10 * signs},
        absolute_bounds={"base": 1.0, "correction": 1.0},
        costs_per_sample={"base": 1.0, "correction": 2.0},
        familywise_alpha=0.05,
        total_predeclared_looks=3,
    )


def _prepared(protocol: str = "g11-v5-hybrid-test", cap: float = 1e9):
    ledger = SeedLedger()
    ledger.allocate(SeedKey(protocol, "selection", "gaussian", "digital", 0, 0, "pilot"))
    prepared = prepare_hybrid_run(
        HybridTarget("finite-grid-digital", 0.15, 0.20),
        _profiles(),
        selection=_selection(),
        selected_profile_ids=("base", "correction"),
        protocol=protocol,
        regime="gaussian",
        task="digital",
        operation_work_cap=cap,
        chunk_size=37,
        minimum_final_samples=32,
        preparation_ledger=ledger,
        preprocessing_work_entries=(WorkLedgerEntry("selection", None, 4096, 100.0, 0.0),),
    )
    return prepared


def test_preparation_meets_integer_design_target_without_allocating_final_seeds() -> None:
    prepared = _prepared()
    assert not prepared.resource_censored
    assert sum(item.design_sampling_variance for item in prepared.allocations) <= (
        prepared.target.sampling_variance_target * (1.0 + 1e-14)
    )
    assert all(item.final_count >= 32 for item in prepared.allocations)
    assert all(record.key.role == "selection" for record in prepared.ledger.records)
    assert all(entry.role != "final" for entry in prepared.work.entries)


def test_independent_planning_variances_can_override_confidence_bounds() -> None:
    prepared = prepare_hybrid_run(
        HybridTarget("finite-grid-digital", 0.15, 0.20),
        _profiles(),
        selection=_selection(),
        selected_profile_ids=("base", "correction"),
        protocol="g11-v6-planning-override",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        minimum_final_samples=32,
        allocation_safety_factor=2.0,
        design_variance_overrides={"base": 0.04, "correction": 0.01},
    )
    assert [item.design_variance for item in prepared.allocations] == [0.08, 0.02]
    with pytest.raises(ValueError, match="match the selected profiles"):
        prepare_hybrid_run(
            HybridTarget("finite-grid-digital", 0.15, 0.20),
            _profiles(),
            selection=_selection(),
            selected_profile_ids=("base", "correction"),
            protocol="g11-v6-invalid-planning-override",
            regime="gaussian",
            task="digital",
            operation_work_cap=1e9,
            design_variance_overrides={"base": 0.04},
        )


def test_final_execution_is_independent_resumable_and_bitwise_identical(tmp_path: Path) -> None:
    prepared = _prepared("g11-v5-hybrid-resume")
    costs = {item.profile_id: item.cost_per_sample for item in prepared.allocations}
    uninterrupted = execute_hybrid_run(
        prepared,
        BoundedTelescopingSampler(costs),
        reference_probability=0.15,
        reference_standard_error=1e-4,
    )
    partial = execute_hybrid_run(prepared, BoundedTelescopingSampler(costs), maximum_chunks=3)
    assert not partial.complete and partial.checkpoint is not None
    checkpoint_path = tmp_path / "hybrid.json"
    save_hybrid_checkpoint(partial.checkpoint, checkpoint_path)
    restored = load_hybrid_checkpoint(checkpoint_path)
    resumed = execute_hybrid_run(
        prepared,
        BoundedTelescopingSampler(costs),
        checkpoint=restored,
        reference_probability=0.15,
        reference_standard_error=1e-4,
    )
    assert uninterrupted.complete and resumed.complete
    assert resumed.estimate == uninterrupted.estimate
    assert resumed.empirical_sampling_variance == uninterrupted.empirical_sampling_variance
    assert resumed.terms == uninterrupted.terms
    assert resumed.work.entries == uninterrupted.work.entries
    assert resumed.seed_ledger_hash == uninterrupted.seed_ledger_hash
    assert resumed.bounded_confidence_interval is not None
    assert resumed.asymptotic_confidence_interval is not None
    assert resumed.reference_z_score is not None


def test_infeasible_allocation_is_censored_before_any_final_seed_or_sample() -> None:
    prepared = _prepared("g11-v5-hybrid-censored", cap=101.0)
    assert prepared.resource_censored
    sampler = BoundedTelescopingSampler(
        {item.profile_id: item.cost_per_sample for item in prepared.allocations}
    )
    result = execute_hybrid_run(prepared, sampler)
    assert result.resource_censored and not result.complete
    assert sampler.calls == 0
    assert result.seed_ledger_hash == prepared.ledger.sha256
    assert all(record.key.role != "final" for record in prepared.ledger.records)


def test_execution_rejects_work_or_bound_changes_after_freeze() -> None:
    prepared = _prepared("g11-v5-hybrid-contract")

    class BadWorkSampler(BoundedTelescopingSampler):
        def __call__(self, profile_id, role, count, seeds):
            batch = super().__call__(profile_id, role, count, seeds)
            return LevelBatch(batch.values, batch.work_units + 1.0)

    costs = {item.profile_id: item.cost_per_sample for item in prepared.allocations}
    with pytest.raises(ValueError, match="frozen operation cost"):
        execute_hybrid_run(prepared, BadWorkSampler(costs), maximum_chunks=1)

    class BadBoundSampler(BoundedTelescopingSampler):
        def __call__(self, profile_id, role, count, seeds):
            batch = super().__call__(profile_id, role, count, seeds)
            values = batch.values.clone()
            values[0] = 1.1
            return LevelBatch(values, batch.work_units)

    with pytest.raises(ValueError, match="defensive bound"):
        execute_hybrid_run(prepared, BadBoundSampler(costs), maximum_chunks=1)


def test_checkpoint_hash_tampering_is_rejected() -> None:
    prepared = _prepared("g11-v5-hybrid-tamper")
    costs = {item.profile_id: item.cost_per_sample for item in prepared.allocations}
    partial = execute_hybrid_run(prepared, BoundedTelescopingSampler(costs), maximum_chunks=1)
    assert partial.checkpoint is not None
    payload = partial.checkpoint.to_dict()
    payload["preparation_hash"] = "0" * 64
    tampered = hybrid_module.HybridCheckpoint.from_dict(payload)
    with pytest.raises(ValueError, match="does not match"):
        execute_hybrid_run(prepared, BoundedTelescopingSampler(costs), checkpoint=tampered)


def test_checkpoint_parser_rejects_unknown_coerced_and_inconsistent_fields() -> None:
    prepared = _prepared("g11-v5-hybrid-strict-parser")
    costs = {item.profile_id: item.cost_per_sample for item in prepared.allocations}
    partial = execute_hybrid_run(prepared, BoundedTelescopingSampler(costs), maximum_chunks=1)
    assert partial.checkpoint is not None

    with_unknown = partial.checkpoint.to_dict()
    with_unknown["ignored"] = True
    with pytest.raises(ValueError, match="fields"):
        hybrid_module.HybridCheckpoint.from_dict(with_unknown)

    bool_allocation = partial.checkpoint.to_dict()
    bool_allocation["allocations"][0] = True
    with pytest.raises(ValueError, match="integer"):
        hybrid_module.HybridCheckpoint.from_dict(bool_allocation)

    coerced_moment = partial.checkpoint.to_dict()
    coerced_moment["moments"][0]["mean"] = "0.0"
    with pytest.raises(ValueError, match="finite real"):
        hybrid_module.HybridCheckpoint.from_dict(coerced_moment)

    inconsistent_work = partial.checkpoint.to_dict()
    inconsistent_work["work_entries"][-1]["samples"] -= 1
    with pytest.raises(ValueError, match="does not match"):
        hybrid_module.HybridCheckpoint.from_dict(inconsistent_work)


def test_checkpoint_writer_retries_transient_windows_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared("g11-v5-hybrid-replace")
    costs = {item.profile_id: item.cost_per_sample for item in prepared.allocations}
    partial = execute_hybrid_run(prepared, BoundedTelescopingSampler(costs), maximum_chunks=1)
    assert partial.checkpoint is not None
    real_replace = hybrid_module.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(hybrid_module.os, "replace", flaky_replace)
    path = tmp_path / "retry.json"
    save_hybrid_checkpoint(partial.checkpoint, path, replace_attempts=2, initial_retry_seconds=0.0)
    assert attempts == 2
    assert load_hybrid_checkpoint(path) == partial.checkpoint


def test_unbounded_single_term_uses_common_allocation_without_fake_hoeffding_bound() -> None:
    target = HybridTarget("pure-cem-tail", 0.15, 0.20)
    design = SingleTermDesign(
        profile_id="pure_cem",
        pilot_count=512,
        pilot_mean=0.15,
        pilot_variance=0.01,
        design_variance=0.0125,
        cost_per_sample=1.0,
        absolute_bound=None,
    )
    prepared = prepare_single_term_run(
        target,
        design,
        method="pure_cem",
        protocol="g11-v6-pure-cem-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        chunk_size=41,
    )
    sampler = BoundedTelescopingSampler({"base": 1.0})

    class PureCEMAdapter:
        def __call__(self, profile_id, role, count, seeds):
            batch = sampler("base", role, count, seeds)
            return LevelBatch(batch.values, batch.work_units)

    result = execute_hybrid_run(prepared, PureCEMAdapter())
    assert result.complete
    assert result.asymptotic_confidence_interval is not None
    assert result.bounded_confidence_interval is None
    assert result.allocations[0].absolute_bound is None


def test_bounded_single_term_retains_defensive_final_check() -> None:
    prepared = prepare_single_term_run(
        HybridTarget("defensive-tail", 0.15, 0.20),
        SingleTermDesign("defensive", 128, 0.15, 0.01, 0.01, 1.0, 0.25),
        method="defensive_cem",
        protocol="g11-v6-defensive-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
    )

    class ViolatingSampler:
        def __call__(self, profile_id, role, count, seeds):
            return LevelBatch(torch.full((count,), 0.30), float(count))

    with pytest.raises(ValueError, match="defensive bound"):
        execute_hybrid_run(prepared, ViolatingSampler())
