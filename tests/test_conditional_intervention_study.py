import copy

import numpy as np
import pytest
import torch

from experiments import post_audit_conditional_intervention as study
from src.path_integral.risk_path_attribution import moments


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_work_caps_charge_before_exceeding():
    budget = study.Work(3, 5, 30)
    budget.charge(2, 4)
    with pytest.raises(TimeoutError):
        budget.charge(2)
    with pytest.raises(TimeoutError):
        budget.charge(0, 2)
    assert (budget.full, budget.scalar) == (2, 4)
    with pytest.raises(ValueError):
        budget.charge(True)


def test_exact_zero_control_is_reported_without_inventing_positive_mass():
    logs = torch.tensor([[0., -torch.inf], [0., -torch.inf]], dtype=torch.float64)
    value = study.summary([moments(logs)])
    assert value["log_mean"] == [0., None]
    assert value["relative_se"] == [0., None]
    assert value["maximum_fraction"] == [.5, 0.]


def test_array_store_refuses_overwrite(tmp_path):
    path = tmp_path/"array.npz"
    study.store_array(path, {"a": np.array([1., 2.])})
    with pytest.raises(FileExistsError):
        study.store_array(path, {"a": np.array([3.])})
    with np.load(path, allow_pickle=False) as value:
        assert value["a"].tolist() == [1., 2.]


def test_stream_families_are_unique_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    a, ledger = study.streams(["c", "h"])
    b, _ = study.streams(["c", "h"])
    assert a == b and len(a) == len(set(a.values())) == 360
    assert ledger["collision_count"] == 0


def test_pair_and_summary_require_complete_fixed_family():
    with pytest.raises(ValueError):
        study.summarize([], ["c", "h"])


def test_complete_synthetic_family_preserves_p2_false_and_does_not_adopt():
    records = []
    rng = np.random.default_rng(31)
    for cell in ("c", "h"):
        for b in range(8):
            values = rng.normal(size=(32, 8))
            values[:, -1] = -np.inf
            records.append(study.make_record(cell, "assay", b, {"logs": values}))
        for design in ("q", "p"):
            for b in range(16):
                values = rng.normal(size=(32, 4))
                records.append(study.make_record(cell, design, b, {"risk": values, "event": values}))
    result = study.summarize(records, ["c", "h"])
    assert not result["P2_qualified"] and not result["production_adoption"]
    assert result["cells"][0]["removable_variance_over_M2_descriptive"][-1] == 0
    malformed = copy.deepcopy(records)
    malformed[-1] = malformed[-2]
    with pytest.raises(ValueError):
        study.summarize(malformed, ["c", "h"])


def test_q_and_p_risk_agreement_is_not_truth_certificate():
    summary = {"log_mean": [0.], "relative_se": [.01]}
    value = study.comparison(summary, summary, 0)
    assert value["development_equivalence"]
    assert value["not_finite_sample_coverage_or_truth_certificate"]
