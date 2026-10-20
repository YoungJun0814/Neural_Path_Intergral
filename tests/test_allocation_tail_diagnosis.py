import math

import pytest

from experiments.post_audit_allocation_tail_diagnosis import diagnose


def test_observed_tail_squares_are_not_treated_as_physical_cluster_probabilities():
    paths = [{"log_contribution": math.log(x), "log_integrated_variance": 0.,
              "log_maximum_left_variance": math.log(2.), "maximum_left_variance_time": .6,
              "sampler_label": 1} for x in range(16, 0, -1)]
    rows = [{"cell": c, "rep": rep, "arm": a, "top_sixteen_paths": paths,
             "summary": {"count": 32768, "log_sum": math.log(200.), "log_sum_squares": math.log(2000.),
                         "maximum_fraction": .08}}
            for c in ("a", "b") for rep in range(2) for a in ("uniform", "learned")]
    fits = [{"cell": c, "rep": rep, "weights": [.1, .4, .5] if rep == 0 else [.1, .5, .4]}
            for c in ("a", "b") for rep in range(2)]
    r = diagnose({"summary": {"rows": rows, "fits": fits}})
    first = r["rows"][0]
    assert first["maximum_sample_share_of_observed_squared_contributions"] == pytest.approx(256/2000)
    assert first["top_sixteen_observed_contribution_share"] == pytest.approx(136/200)
    assert first["top_sixteen_quarter_peak_time_observed_mass_only"] == pytest.approx([0., 0., .68, 0.])
    assert r["between_fit_weight_diagnostics"][0]["learned_weight_total_variation"] == pytest.approx(.1)
    assert r["new_physical_evaluations"] == 0 and r["post_hoc_exploratory"] is True
    assert r["p2_qualified"] is False


def test_tail_analysis_rejects_incomplete_family():
    with pytest.raises(ValueError):
        diagnose({"summary": {"rows": []}})
