import math

import pytest

from experiments.post_audit_window_priority_analysis import decode_label, observed_label_summary


def test_label_decode_exact_component_enumeration_not_clustering():
    centres = list(range(31))
    assert decode_label(0, centres)["kind"] == "natural"
    assert decode_label(1, centres) == {"kind": "shift", "template": 0, "centre_step": 0, "amplitude": 2., "price_amplitude": 0.}
    assert decode_label(372, centres) == {"kind": "shift", "template": 30, "centre_step": 30, "amplitude": 8., "price_amplitude": 4.}
    with pytest.raises(ValueError):
        decode_label(373, centres)


def test_observed_label_partition_and_top_mass_are_descriptive():
    sums = [None]*373
    sums[0], sums[1], sums[372] = math.log(1.), math.log(2.), math.log(3.)
    counts = [0]*373
    counts[0], counts[1], counts[372] = 1, 1, 1
    diagnostic = {"label_counts": counts, "label_log_sums": sums, "log_sum": math.log(6.), "count": 3,
                  "top": [{"log_contribution": math.log(3.), "features": [1.]*6}],
                  "feature_names": ["test"]*6, "observed_weighted_feature_means": [1.]*6}
    out = observed_label_summary(diagnostic, list(range(31)))
    assert out["natural_observed_fraction"] == pytest.approx(1/6)
    assert out["shift_price_amplitude_observed_fractions"]["4.0"] == pytest.approx(.5)
    assert out["top_sixteen_sample_contribution_fraction"] == pytest.approx(.5)
    assert out["top_ten_sampler_components"][0]["label"] == 372
