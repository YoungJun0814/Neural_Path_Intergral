from __future__ import annotations

import copy

from experiments.g11_v8_p5_threshold_binding_v2_audit import (
    ROOT,
    audit_binding_v2,
    load_binding_v2,
)


def test_fresh_r2_threshold_binding_passes() -> None:
    binding, digest = load_binding_v2(
        ROOT / "configs/g11_v8/p5_threshold_manifest_binding_v2.yaml"
    )
    report = audit_binding_v2(binding, digest)

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["decision"]["representative_benchmark_authorized"] is True
    assert report["decision"]["full_pilot_execution_authorized"] is False


def test_binding_v2_rejects_burned_or_opened_namespace() -> None:
    binding, digest = load_binding_v2(
        ROOT / "configs/g11_v8/p5_threshold_manifest_binding_v2.yaml"
    )
    burned = copy.deepcopy(binding)
    burned["reference_protocol"]["pilot_namespace"] = "p5-reference-v2"
    report = audit_binding_v2(burned, digest)
    assert report["passed"] is False
    assert "namespace_roster_exact" in report["failures"]
    assert "new_namespaces_disjoint" in report["failures"]

    opened = copy.deepcopy(binding)
    opened["current_namespace_outcomes_inspected_before_freeze"] = True
    report = audit_binding_v2(opened, digest)
    assert report["passed"] is False
    assert "current_namespaces_unopened" in report["failures"]
