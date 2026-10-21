import math

import pytest

from experiments.post_audit_model_restart_diagnosis import ARMS, CELLS, diagnose


def test_posthoc_risk_mismatch_keeps_uncertainty_and_global_flags_false():
    rows, risks, fits = [], [], []
    for c in CELLS:
        for rep in range(5):
            fits.append({"cell": c, "rep": rep,
                "fit_diagnostics": {a: {"bank_count": 8192, "ess": 4., "max_fraction": .5} for a in ("event", "risk")},
                "selections": {a: {"alpha": .25, "no_op": False} for a in ("event", "risk")}})
            for a in ARMS:
                rows.append({"cell": c, "rep": rep, "arm": a, "moments": {"relative_se": .01, "log_second_moment": math.log(.02)}})
            risks.append({"cell": c, "rep": rep, "risk_moments": {a: {"relative_se": .7,
                "maximum_fraction": .65, "log_mean": math.log(.2)} for a in ARMS}, "contrasts": []})
    summary = {"rows": rows, "fits": fits, "heldout_risk": risks, "development_candidate_arms": []}
    out = diagnose(summary)
    assert out["additional_physical_evaluations"] == 0
    assert out["p1_qualified"] is out["p2_qualified"] is out["large_training_authorized"] is False
    assert out["rows"][0]["heldout_over_direct_observed_m2"] == pytest.approx(10.)
    assert out["rows"][0]["heldout_m2_rse"] == .7
    assert out["rows"][0]["not_population_ratio_or_bias_proof"] is True
    with pytest.raises(ValueError, match="complete"):
        diagnose({**summary, "rows": rows[:-1]})
