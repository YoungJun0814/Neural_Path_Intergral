import math
from copy import deepcopy

import pytest

from experiments.post_audit_bank_ensemble_analysis import analyze


def fixture():
    bins = [{"bin": i, "intensity_bin": i//16, "time_bin": (i%16)//4, "window_bin": i%4,
             "sample_count": 65536 if i == 60 else 0,
             **({"observed_contribution_fraction": 1., "observed_squared_contribution_fraction": 1.} if i == 60 else {})}
            for i in range(96)]
    rows = [{"cell": c, "arm": a, "summary": {"count": 65536, "log_mean": -math.log(65536),
                "log_sum": 0., "log_sum_squares": math.log((1+65535*.03**2)/65536),
                "relative_se": .03, "maximum_fraction": .001},
             "coverage": {"count": 65536, "top": [{"log_contribution": math.log(.001)}]*16},
             "bin_summaries": deepcopy(bins), "precision_1p5_pass": False, "score_wall": 10.,
             "block_sensitivity": {"between_unit_relative_se": .04}}
            for c in ("a", "b") for a in ("uniform", "fit0", "fit1", "ensemble")]
    return {"summary": {"rows": rows, "comparisons": []}}


def test_all_path_grouping_reconstructs_joint_partition_and_forecast_stays_unapproved():
    out = analyze(fixture())
    first = out["rows"][0]
    assert first["physical_bin_groups"]["time_bin"][3]["observed_contribution_fraction"] == 1.
    assert first["physical_bin_groups"]["window_bin"][0]["sample_count"] == 65536
    assert first["fresh_safety3_precision_forecast"]["draws"] == 786432
    assert first["fresh_safety3_precision_forecast"]["authorized"] is False
    assert out["new_physical_evaluations"] == 0 and out["p2_qualified"] is False


def test_synthesis_rejects_prefix_or_corrupt_joint_mass():
    data = fixture()
    data["summary"]["rows"].pop()
    with pytest.raises(ValueError):
        analyze(data)
    data = fixture()
    data["summary"]["rows"][0]["bin_summaries"][60]["observed_contribution_fraction"] = .9
    with pytest.raises(ValueError):
        analyze(data)
