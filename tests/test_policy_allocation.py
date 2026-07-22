"""Unified V6 policy preparation, execution, and independent-audit tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.path_integral import (
    HybridTarget,
    LevelBatch,
    RarityRouterConfig,
    SingleTermDesign,
    V6WorkLedger,
    V6WorkRecord,
    audit_v6_policy,
    execute_v6_policy,
    freeze_rarity_route,
    prepare_v6_direct_policy,
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
        cumulative_final_cpu_seconds=0.1,
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
        cumulative_final_cpu_seconds=0.1,
        final_peak_memory_bytes=1000,
    )
    tampered = replace(result, result_hash="0" * 64)
    audit = audit_v6_policy(prepared, tampered)
    assert not audit.result_hash_valid
    assert not audit.passed
