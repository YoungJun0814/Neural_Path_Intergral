"""Independent serialized-artifact audit tests."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from experiments.g11_v6_result_audit import _audit_record, _load_config, run
from src.path_integral import (
    HybridTarget,
    LevelBatch,
    SingleTermDesign,
    V6WorkLedger,
    V6WorkRecord,
    execute_v6_policy,
    prepare_v6_direct_policy,
    v6_policy_preparation_to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "result_audit_development.yaml"


class _Sampler:
    def __call__(self, profile_id, role, count, seeds):
        del profile_id, role
        generator = torch.Generator().manual_seed(seeds["proposal"])
        values = (torch.rand(count, generator=generator) < 0.1).to(torch.float64)
        return LevelBatch(values, float(count), wall_seconds=0.001)


def _work(category: str) -> V6WorkRecord:
    return V6WorkRecord(
        category=category,
        method="pure_cem",
        cell_id="audit-cell",
        attempt=0,
        samples=32,
        work_units=32.0,
        wall_seconds=0.001,
        cpu_seconds=0.001,
        peak_memory_bytes=0,
        successful=True,
    )


def _record() -> dict[str, object]:
    prepared = prepare_v6_direct_policy(
        HybridTarget("audit-target", 0.1, 0.5),
        SingleTermDesign("cem", 64, 0.1, 0.02, 0.025, 1.0, None),
        policy_name="pure_cem",
        cell_id="audit-cell",
        execution_method="pure_cem",
        protocol="g11-v6-independent-audit-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        preprocessing_work=V6WorkLedger(
            (_work("proposal_training"), _work("allocation_pilot"))
        ),
        minimum_final_samples=32,
        streams=("proposal",),
    )
    result = execute_v6_policy(prepared, _Sampler(), final_peak_memory_bytes=0)
    return {
        "cell_id": "audit-cell",
        "cluster": 0,
        "method": "pure_cem",
        "preparation": v6_policy_preparation_to_dict(prepared),
        "result": asdict(result),
    }


def test_v6_independent_audit_config_is_strict() -> None:
    config, digest = _load_config(CONFIG)
    assert "npi.g11.v6-routed-policy.v1" in config["accepted_source_schemas"]
    assert len(digest) == 64


def test_offline_auditor_recomputes_valid_record_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    record = _record()
    valid = _audit_record(record, relative=1e-13, absolute=1e-12)
    assert valid["passed"]

    allocation_tamper = copy.deepcopy(record)
    allocation_tamper["preparation"]["core"]["allocations"][0]["final_count"] += 1
    assert not _audit_record(
        allocation_tamper, relative=1e-13, absolute=1e-12
    )["passed"]

    work_tamper = copy.deepcopy(record)
    work_tamper["result"]["total_work"]["records"][-1]["work_units"] += 1.0
    assert not _audit_record(work_tamper, relative=1e-13, absolute=1e-12)["passed"]

    result_tamper = copy.deepcopy(record)
    result_tamper["result"]["core"]["estimate"] += 0.01
    assert not _audit_record(result_tamper, relative=1e-13, absolute=1e-12)["passed"]

    source = {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "smoke": True,
        "records": [record],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    artifact = run(CONFIG, source_path)
    assert artifact["gates"]["all_records_pass"]
    assert not artifact["gates"]["non_smoke_if_required"]
    assert not artifact["qualification_audit_passed"]

    seed_tamper = copy.deepcopy(record)
    seed_tamper["result"]["core"]["seed_ledger_payload"]["records"][-1]["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        _audit_record(seed_tamper, relative=1e-13, absolute=1e-12)
