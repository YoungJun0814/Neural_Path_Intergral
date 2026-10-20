from copy import deepcopy

import pytest

from experiments.post_audit_fine_reference_comparison import CELLS, DESIGNS, METHODS, compare


def inputs():
    components = [{"grid": g, "summary": {"log_mean": -18., "relative_se": .005}} for g in ("fine", "coarse")]
    resource = {"summaries": {"rows": [{"cell": c, "design": d, "components": deepcopy(components)} for c in CELLS for d in DESIGNS]}}
    guide = {"summaries": {"rows": [{"cell": c, "method": m, "components": deepcopy(components)} for c in CELLS for m in METHODS]}}
    return resource, guide


def test_cross_mechanism_complete_family_cannot_authorize_training():
    out = compare(*inputs())
    assert len(out["rows"]) == out["comparisons"] == 8
    assert out["all_development_agreements_pass"] and out["all_reported_precision_probes_pass"]
    assert out["next_priority"] == "preregister_new_confirmation_and_tail_coverage_gate"
    assert not any(out[k] for k in ("methods_certified_for_wall_comparison", "p1_qualified", "p2_qualified", "large_training_authorized"))
    assert all(r["interval_diagnosis"]["status"] == "inside_equivalence_margin" for r in out["rows"])


def test_cross_mechanism_rejects_prefix_and_detects_shifted_reference():
    resource, guide = inputs()
    resource["summaries"]["rows"].pop()
    with pytest.raises(ValueError):
        compare(resource, guide)
    resource, guide = inputs()
    guide["summaries"]["rows"][0]["components"][0]["summary"]["log_mean"] = -17.
    out = compare(resource, guide)
    assert out["next_priority"] == "investigate_reference_disagreement_or_uncertainty"
    assert out["rows"][0]["interval_diagnosis"]["status"] == "interval_entirely_outside_margin"


def test_equal_means_with_imprecise_smc_are_not_reported_as_detected_bias():
    resource, guide = inputs()
    for row in resource["summaries"]["rows"]:
        row["components"][0]["summary"]["relative_se"] = .1
    out = compare(resource, guide)
    assert not out["all_development_agreements_pass"]
    assert not out["all_reported_precision_probes_pass"]
    assert all(r["interval_diagnosis"]["status"] == "uncertainty_crosses_equivalence_margin" for r in out["rows"])
    assert all(r["interval_diagnosis"]["interval"][0] < 0 < r["interval_diagnosis"]["interval"][1] for r in out["rows"])
