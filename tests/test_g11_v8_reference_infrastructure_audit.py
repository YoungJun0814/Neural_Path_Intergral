from __future__ import annotations

import copy

from experiments.g11_v8_reference_infrastructure_audit import (
    ROOT,
    audit_reference_infrastructure,
    load_contract,
)


def test_r1_reference_infrastructure_contract_passes() -> None:
    contract, digest = load_contract(
        ROOT / "configs/g11_v8/reference_infrastructure_contract_v1.yaml"
    )
    report = audit_reference_infrastructure(contract, digest)

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["decision"]["r2_development_benchmark_authorized"] is True
    assert report["decision"]["formal_reference_complete"] is False
    assert report["decision"]["performance_claim_authorized"] is False


def test_r1_audit_fails_if_predecessor_hash_is_mutated() -> None:
    contract, digest = load_contract(
        ROOT / "configs/g11_v8/reference_infrastructure_contract_v1.yaml"
    )
    mutated = copy.deepcopy(contract)
    mutated["predecessor"]["sha256"] = "f" * 64
    report = audit_reference_infrastructure(mutated, digest)

    assert report["passed"] is False
    assert "predecessor_hash_and_status" in report["failures"]


def test_r1_audit_fails_if_performance_is_prematurely_authorized() -> None:
    contract, digest = load_contract(
        ROOT / "configs/g11_v8/reference_infrastructure_contract_v1.yaml"
    )
    mutated = copy.deepcopy(contract)
    mutated["performance_claim_authorized"] = True
    report = audit_reference_infrastructure(mutated, digest)

    assert report["passed"] is False
    assert "contract_scope_fail_closed" in report["failures"]
