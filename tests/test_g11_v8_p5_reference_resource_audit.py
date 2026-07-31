from __future__ import annotations

import copy

import pytest

from experiments.g11_v8_p5_reference_pilot import _verify_authorization
from experiments.g11_v8_p5_reference_resource_audit import (
    ROOT,
    audit_resource_authorization,
    load_authorization,
)
from experiments.g11_v8_p5_sharded_reference_common import load_context

AUTHORIZATION = ROOT / "configs/g11_v8/p5_reference_pilot_authorization_v2.yaml"
CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v4.yaml"
METHOD_ROLE_AUTHORIZATION = (
    ROOT / "configs/g11_v8/p5_reference_pilot_authorization_v3.yaml"
)
METHOD_ROLE_CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v5.yaml"


def test_archived_v2_authorization_fails_closed_after_source_change() -> None:
    authorization, digest = load_authorization(AUTHORIZATION)
    report = audit_resource_authorization(authorization, digest, AUTHORIZATION)

    assert report["passed"] is False
    assert "all_bound_inputs_and_implementation_pass" in report["failures"]
    assert report["decision"]["formal_pilot_execution_authorized"] is False
    assert report["decision"]["final_execution_authorized"] is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_authorization(AUTHORIZATION, load_context(CONFIG))


def test_resource_audit_rejects_premature_final_authorization() -> None:
    authorization, digest = load_authorization(AUTHORIZATION)
    mutated = copy.deepcopy(authorization)
    mutated["final_execution"]["execution_authorized"] = True
    mutated["decision"]["final_execution_authorized"] = True
    report = audit_resource_authorization(mutated, digest, AUTHORIZATION)

    assert report["passed"] is False
    assert "final_remains_closed" in report["failures"]
    assert "decision_fail_closed" in report["failures"]


def test_method_role_authorization_binds_current_runtime_and_only_pilot() -> None:
    authorization, digest = load_authorization(METHOD_ROLE_AUTHORIZATION)
    report = audit_resource_authorization(
        authorization,
        digest,
        METHOD_ROLE_AUTHORIZATION,
    )
    verified = _verify_authorization(
        METHOD_ROLE_AUTHORIZATION,
        load_context(METHOD_ROLE_CONFIG),
    )

    assert verified["decision"]["formal_pilot_execution_authorized"] is True
    assert report["passed"] is True
    assert report["decision"]["formal_pilot_execution_authorized"] is True
    assert report["decision"]["final_execution_authorized"] is False
