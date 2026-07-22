"""Unified V6 policy preparation, execution, and independent-audit tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from src.path_integral import (
    FrozenCrossoverDecision,
    HybridProfileOpportunity,
    HybridTarget,
    LevelBatch,
    RarityRouterConfig,
    RoutingWorkInterval,
    SingleTermDesign,
    V6WorkLedger,
    V6WorkRecord,
    audit_v6_policy,
    execute_v6_policy,
    execute_v6_policy_durable,
    freeze_rarity_route,
    prepare_v6_direct_policy,
    prepare_v6_hybrid_policy,
    update_profile_intervals,
)


class DirectSampler:
    def __call__(self, profile_id, role, count, seeds):
        generator = torch.Generator().manual_seed(seeds["proposal"])
        values = (torch.rand(count, generator=generator) < 0.1).to(torch.float64)
        return LevelBatch(values, float(count), wall_seconds=0.01)


def _record(category: str, policy: str = "v6_policy") -> V6WorkRecord:
    return V6WorkRecord(
        category=category,
        method=policy,
        cell_id="rare-cell",
        attempt=0,
        samples=64,
        work_units=64.0,
        wall_seconds=0.01,
        cpu_seconds=0.01,
        peak_memory_bytes=100,
        successful=True,
    )


def test_routed_direct_policy_executes_and_passes_independent_audit() -> None:
    route = freeze_rarity_route(
        successes=0,
        trials=256,
        screening_work=256.0,
        config=RarityRouterConfig(),
    )
    assert route.action == "dcs_slis"
    work = V6WorkLedger(
        (_record("screening"), _record("routing"), _record("allocation_pilot"))
    )
    prepared = prepare_v6_direct_policy(
        HybridTarget("rare-cell-target", 0.1, 0.2),
        SingleTermDesign("dcs", 64, 0.1, 0.09, 0.10, 1.0, 1.0),
        policy_name="v6_policy",
        cell_id="rare-cell",
        execution_method="dcs_slis",
        protocol="g11-v6-policy-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        preprocessing_work=work,
        route=route,
    )
    result = execute_v6_policy(
        prepared,
        DirectSampler(),
        final_peak_memory_bytes=1000,
    )
    assert result.core.complete
    assert result.total_work.category_work("screening") == 64.0
    assert result.total_work.category_work("final") > 0.0
    assert audit_v6_policy(prepared, result).passed


def test_pure_cem_requires_training_work_and_has_no_router() -> None:
    target = HybridTarget("cem-target", 0.1, 0.2)
    design = SingleTermDesign("cem", 64, 0.1, 0.09, 0.10, 1.0, None)
    with pytest.raises(ValueError, match="missing"):
        prepare_v6_direct_policy(
            target,
            design,
            policy_name="pure_cem",
            cell_id="rare-cell",
            execution_method="pure_cem",
            protocol="g11-v6-cem-test",
            regime="gaussian",
            task="digital",
            operation_work_cap=1e9,
            preprocessing_work=V6WorkLedger((_record("allocation_pilot", "pure_cem"),)),
        )


def test_policy_auditor_detects_result_hash_tampering() -> None:
    work = V6WorkLedger((_record("proposal_training", "pure_cem"), _record("allocation_pilot", "pure_cem")))
    prepared = prepare_v6_direct_policy(
        HybridTarget("cem-target", 0.1, 0.3),
        SingleTermDesign("cem", 64, 0.1, 0.02, 0.025, 1.0, None),
        policy_name="pure_cem",
        cell_id="rare-cell",
        execution_method="pure_cem",
        protocol="g11-v6-cem-audit-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        preprocessing_work=work,
    )
    result = execute_v6_policy(
        prepared,
        DirectSampler(),
        final_peak_memory_bytes=1000,
    )
    tampered = replace(result, result_hash="0" * 64)
    audit = audit_v6_policy(prepared, tampered)
    assert not audit.result_hash_valid
    assert not audit.passed


def test_economically_admitted_hybrid_policy_uses_registered_bounded_profiles() -> None:
    route = freeze_rarity_route(
        successes=0,
        trials=256,
        screening_work=256.0,
        config=RarityRouterConfig(
            maximum_hybrid_profile_work=1000.0,
            maximum_profile_fraction=0.25,
        ),
        crude_work=RoutingWorkInterval("crude", 9000.0, 10000.0, 11000.0),
        dcs_work=RoutingWorkInterval("dcs_slis", 7000.0, 8000.0, 9000.0),
        hybrid_opportunity=HybridProfileOpportunity(100.0, 5000.0, 1000.0),
    )
    assert route.action == "profile_hybrid"
    generator = torch.Generator().manual_seed(77)
    signs = (2 * torch.randint(0, 2, (256,), generator=generator) - 1).to(torch.float64)
    profiles = update_profile_intervals(
        {"base": 0.1 + 0.1 * signs, "correction": 0.02 * signs},
        absolute_bounds={"base": 1.0, "correction": 1.0},
        costs_per_sample={"base": 1.0, "correction": 2.0},
        familywise_alpha=0.05,
        total_predeclared_looks=1,
    )
    selection = FrozenCrossoverDecision(
        selected_candidate="hybrid",
        look_index=0,
        reason="synthetic registered look",
        surviving_candidates=("hybrid",),
        selected_work_interval=(100.0, 200.0),
        selected_point_work=150.0,
        worst_case_interval_regret_bound=1.0,
    )
    work = V6WorkLedger(
        (_record("screening"), _record("routing"), _record("selector_profile"))
    )
    prepared = prepare_v6_hybrid_policy(
        HybridTarget("hybrid-target", 0.1, 0.5),
        profiles,
        policy_name="v6_policy",
        cell_id="rare-cell",
        route=route,
        selection=selection,
        selected_profile_ids=("base", "correction"),
        protocol="g11-v6-hybrid-policy-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        preprocessing_work=work,
        minimum_final_samples=32,
    )

    class TwoTermSampler:
        def __call__(self, profile_id, role, count, seeds):
            del role, seeds
            values = (
                torch.full((count,), 0.1, dtype=torch.float64)
                if profile_id == "base"
                else torch.zeros(count, dtype=torch.float64)
            )
            cost = 1.0 if profile_id == "base" else 2.0
            return LevelBatch(values, count * cost)

    result = execute_v6_policy(
        prepared,
        TwoTermSampler(),
        final_peak_memory_bytes=0,
    )
    assert result.core.complete
    assert result.execution_method == "hybrid"
    assert audit_v6_policy(prepared, result).passed


def test_durable_policy_checkpoint_resumes_without_reusing_final_samples(tmp_path) -> None:
    work = V6WorkLedger(
        (_record("proposal_training", "pure_cem"), _record("allocation_pilot", "pure_cem"))
    )
    prepared = prepare_v6_direct_policy(
        HybridTarget("durable-target", 0.1, 0.5),
        SingleTermDesign("cem", 64, 0.1, 0.02, 0.025, 1.0, None),
        policy_name="pure_cem",
        cell_id="rare-cell",
        execution_method="pure_cem",
        protocol="g11-v6-durable-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        preprocessing_work=work,
        chunk_size=7,
        minimum_final_samples=32,
        streams=("proposal",),
    )
    checkpoint = tmp_path / "policy-checkpoint.json"
    partial = execute_v6_policy_durable(
        prepared,
        DirectSampler(),
        checkpoint_path=checkpoint,
        chunks_per_checkpoint=1,
        maximum_cycles=2,
        final_peak_memory_bytes=100,
    )
    assert not partial.core.complete
    assert checkpoint.exists()
    resumed = execute_v6_policy_durable(
        prepared,
        DirectSampler(),
        checkpoint_path=checkpoint,
        resume=True,
        chunks_per_checkpoint=1,
        final_peak_memory_bytes=100,
    )
    one_shot = execute_v6_policy(
        prepared,
        DirectSampler(),
        final_peak_memory_bytes=100,
    )
    assert resumed.core.complete
    assert resumed.core.terms == one_shot.core.terms
    assert resumed.core.seed_ledger_hash == one_shot.core.seed_ledger_hash
    assert resumed.core.work.total_work_units == one_shot.core.work.total_work_units

    tampered_checkpoint = tmp_path / "tampered.json"
    execute_v6_policy_durable(
        prepared,
        DirectSampler(),
        checkpoint_path=tampered_checkpoint,
        chunks_per_checkpoint=1,
        maximum_cycles=1,
        final_peak_memory_bytes=100,
    )
    state_path = tampered_checkpoint.with_suffix(".json.v6.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["policy_hash"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        execute_v6_policy_durable(
            prepared,
            DirectSampler(),
            checkpoint_path=tampered_checkpoint,
            resume=True,
            final_peak_memory_bytes=100,
        )
