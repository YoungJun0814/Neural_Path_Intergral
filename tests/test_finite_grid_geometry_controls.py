import copy

import pytest
import torch

from experiments.finite_grid_geometry_bank_family import JOBS, summary
from src.path_integral.finite_grid_geometry_controls import (
    ARMS,
    CONTRASTS,
    PER_WHOLE,
    diagnostic_contrasts,
    identity_mixture,
    stratified_moments,
)


def test_known_mass_variance_is_not_pooled_iid_variance():
    a = torch.tensor([[0., 0.], [2., 2.]], dtype=torch.float64)
    b = torch.tensor([[8., 8.], [10., 10.]], dtype=torch.float64)
    row = stratified_moments([a, b])
    assert row["mean"] == [5., 5.]
    assert row["covariance_of_mean"] == [[.5, .5], [.5, .5]]
    assert row["known_stratum_masses"] == [.5, .5]


def test_exact_shift_adapter_preserves_natural_floor_and_density():
    q = identity_mixture(torch.tensor([[0., 0.], [1., 2.]], dtype=torch.float64),
                         torch.tensor([.1, .9], dtype=torch.float64))
    draw = q.sample(64, path_seed=314, label_seed=315)
    assert q.defensive_mass == .1
    assert (draw.log_q_over_p >= torch.tensor(.1).double().log()-1e-7).all()
    torch.testing.assert_close(q.log_q_over_p(draw.samples), draw.log_q_over_p)
    assert PER_WHOLE == 342016


def test_two_reference_precision_and_agreement_cannot_be_bypassed_by_ratio():
    means = [1.]*7
    means[ARMS.index("ce-preserve")] = .5
    covariance = torch.eye(7, dtype=torch.float64)*1e-6
    h = {"mean": means, "covariance_of_mean": covariance.tolist(), "se": [.001]*7,
         "rse": [.002]*7}
    assert diagnostic_contrasts(h, h)[0]["signal"]
    bad = copy.deepcopy(h)
    bad["rse"][2] = .5
    assert not diagnostic_contrasts(h, bad)[0]["signal"]
    bad = copy.deepcopy(h)
    bad["mean"][2] = .95
    assert not diagnostic_contrasts(h, bad)[0]["signal"]


def test_missing_wholes_and_failures_cannot_be_successes():
    rows = [{"task_id": task, "whole": rep, "contrasts": [
        {"candidate": a, "baseline": b, "signal": True} for a, b in CONTRASTS]} for task, rep in JOBS]
    assert summary(rows)["decision"] == "GO_FRESH_FR3_HYPOTHESIS_TEST_ONLY"
    with pytest.raises(ValueError, match="matrix"):
        summary(rows[:-1])
    rows[0] = {**rows[0], "failed": True}
    assert summary(rows)["decision"] == "UNRESOLVED_MECHANISM_RUN_FAILURE"
    assert not summary(rows)["eligible_mechanism_hypotheses"]
