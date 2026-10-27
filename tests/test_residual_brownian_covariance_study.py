import copy
import math

import numpy as np
import pytest
import torch

from experiments import post_audit_residual_brownian_covariance as study
from src.path_integral.stratified_paired_risk import sufficient


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_exact_fixed_unit_count_and_unique_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    jobs = study.jobs(["c", "h"])
    assert len(jobs) == len(set(study.unit_id(j) for j in jobs)) == 96
    mapping, ledger = study.streams(["c", "h"])
    assert len(mapping) == len(set(mapping.values())) == 304
    assert ledger["collision_count"] == 0
    assert mapping == study.streams(["c", "h"])[0]


def test_fixed_caps_charge_before_running():
    budget = study.Budget(4, 10000, 30)
    budget.charge(4, 3, 2)
    with pytest.raises(TimeoutError):
        budget.charge(1, 3, 2)
    assert budget.physical == 4
    with pytest.raises(ValueError):
        budget.charge(True, 3, 2)


def test_known_stratum_mean_and_covariance_are_not_iid_flattened():
    rows = []
    for b in range(8):
        data = [torch.tensor([[1.], [3.]], dtype=torch.float64),
                torch.tensor([[10.], [14.]], dtype=torch.float64)]
        rows.append({"block": b, "strata": [sufficient(x.log()) for x in data]})
    result = study.family_summary(rows)
    assert math.exp(result["raw"]["log_mean"][0]) == pytest.approx(7.)
    # Within each stratum, eight duplicated fixed blocks -> 16 independent
    # observations in this synthetic sufficient-statistic calculation.
    all_data = [torch.cat([x]*8) for x in data]
    variance = sum(float(x.var(unbiased=True))/len(x)/4 for x in all_data)
    expected_rse = math.sqrt(variance)/7
    assert result["raw"]["relative_se"][0] == pytest.approx(expected_rse)
    assert result["blocks"]["relative_se"][0] == pytest.approx(0., abs=1e-7)


def test_unqualified_reference_does_not_get_agreement_label():
    result = study.agreement({}, {}, 0, qualified=False)
    assert not result["evaluated"] and not result["development_equivalence"]


def test_missing_duplicate_or_incomplete_blocks_rejected():
    with pytest.raises(ValueError):
        study.summarize([], ["c", "h"])
    with pytest.raises(ValueError):
        study.family_summary([])


def test_huge_common_contribution_cannot_qualify_a_small_ratio_se():
    logs = torch.tensor([[0., -1., -1., -1.], [20., 19., 19., 19.]], dtype=torch.float64)
    rows = [{"block": b, "strata": [sufficient(logs)]} for b in range(8)]
    summary = study.family_summary(rows)
    assert not study.risk_qualified(summary)


def test_complete_synthetic_family_keeps_all_global_flags_false():
    rng = np.random.default_rng(62)
    records = []
    for job in study.jobs(["c", "h"]):
        strata = 4 if job["phase"] == "heldout" else 1
        columns = 1 if job["phase"] == "direct" else 4
        values = [sufficient(torch.from_numpy(rng.normal(size=(32, columns)))) for _ in range(strata)]
        events = [sufficient(torch.from_numpy(rng.normal(size=(32, 1)))) for _ in range(strata)]
        records.append({**job, "strata": values, "event_strata": events})
    result = study.summarize(records, ["c", "h"])
    assert not result["P2_qualified"] and not result["production_adoption"]
    assert result["primary_arm"] == "coupled"
    bad = copy.deepcopy(records)
    bad[-1] = bad[-2]
    with pytest.raises(ValueError):
        study.summarize(bad, ["c", "h"])
