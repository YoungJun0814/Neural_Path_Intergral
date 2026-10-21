import pytest

from experiments.post_audit_fine_smc_genealogy_analysis import analyze
from experiments.post_audit_fresh_ensemble_fine_smc import ARMS, CELLS, smc_jobs


def test_early_ancestry_diagnosis_is_descriptive_and_forecast_not_approval():
    records = [{**j, "stages": [{"beta": .001, "surviving_initial_ancestors": 128},
                {"beta": .01, "surviving_initial_ancestors": 64},
                {"beta": .02, "surviving_initial_ancestors": 32},
                {"beta": .1, "surviving_initial_ancestors": 16}]} for j in smc_jobs()]
    rows = [{"cell": c, "arm": a, "summary": {"relative_se": .1}, "score_wall": 80.,
             "minimum_stage_ess": [32.]*8, "initial_ancestors": [16]*8} for c in CELLS for a in ARMS]
    iid = [{"cell": c, "precision_and_sensitivity": {"pass": True}} for c in CELLS]
    result = analyze({"summary": {"smc_rows": rows, "iid_rows": iid}}, records)
    first = result["smc_rows"][0]
    assert first["ancestry_threshold_first_crossing"]["64"]["first_crossing_beta_median"] == .01
    assert first["fresh_safety3_whole_forecast"]["wholes"] == 1067
    assert first["fresh_safety3_whole_forecast"]["authorized"] is False
    assert result["new_physical_evaluations"] == 0 and result["p2_qualified"] is False


def test_genealogy_analysis_refuses_prefix_inference():
    with pytest.raises(ValueError, match="complete"):
        analyze({}, [])
