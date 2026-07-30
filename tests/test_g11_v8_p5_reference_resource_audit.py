from __future__ import annotations

import copy

from experiments.g11_v8_p5_reference_pilot import _verify_authorization
from experiments.g11_v8_p5_reference_resource_audit import (
    ROOT,
    audit_resource_authorization,
    load_authorization,
)
from experiments.g11_v8_p5_sharded_reference_common import load_context

AUTHORIZATION = ROOT / "configs/g11_v8/p5_reference_pilot_authorization_v1.yaml"
CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"


def test_production_benchmark_authorizes_only_formal_pilot() -> None:
    authorization, digest = load_authorization(AUTHORIZATION)
    report = audit_resource_authorization(authorization, digest)
    verified = _verify_authorization(AUTHORIZATION, load_context(CONFIG))

    assert verified["decision"]["formal_pilot_execution_authorized"] is True
    assert report["passed"] is True
    assert report["decision"]["formal_pilot_execution_authorized"] is True
    assert report["decision"]["final_execution_authorized"] is False


def test_resource_audit_rejects_premature_final_authorization() -> None:
    authorization, digest = load_authorization(AUTHORIZATION)
    mutated = copy.deepcopy(authorization)
    mutated["final_execution"]["execution_authorized"] = True
    mutated["decision"]["final_execution_authorized"] = True
    report = audit_resource_authorization(mutated, digest)

    assert report["passed"] is False
    assert "final_remains_closed" in report["failures"]
    assert "decision_fail_closed" in report["failures"]
