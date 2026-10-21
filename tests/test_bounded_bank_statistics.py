import copy
import math

import pytest
import torch

from src.path_integral.balanced_mixture_bank import balanced_counts, draw_bank, stream_layout
from src.path_integral.bounded_decision_protocol import allocate, d2_decision, repeated_decision
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.stratified_paired_risk import estimate, merge_sufficient, paired, sufficient


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def guide(k=372):
    weights = torch.full((k+1,), .9/k, dtype=torch.float64)
    weights[0] = .1
    return GaussianMixtureShiftSpec(torch.zeros((k+1, 2), dtype=torch.float64), weights)


@pytest.mark.parametrize("count", [1240, 2480, 19840, 29760, 39680, 158720])
def test_exact_tile_and_every_stratum(count):
    q = guide()
    counts = balanced_counts(q.weights, count)
    assert sum(counts) == count and min(counts) >= 3
    torch.testing.assert_close(torch.tensor(counts, dtype=torch.float64)/count, q.weights)


def test_bad_count_and_wrong_masses_rejected():
    with pytest.raises(ValueError):
        balanced_counts(guide().weights, 16384)
    w = guide().weights.clone()
    w[0], w[1] = .2, w[1]-.1
    with pytest.raises(ValueError):
        balanced_counts(w, 19840)


@pytest.mark.parametrize("design", ["iid", "stratified"])
def test_partial_chunks_replay_and_role_independence(design):
    q = guide(4)
    z, labels, digest = draw_bank(q, 160, design, {"phase": "test", "design": design}, chunk=37)
    z2, labels2, digest2 = draw_bank(q, 160, design, {"phase": "test", "design": design}, chunk=37)
    assert torch.equal(z, z2) and torch.equal(labels, labels2) and digest == digest2
    assert z.shape == (160, 2)
    if design == "stratified":
        assert labels.bincount().tolist() == list(balanced_counts(q.weights, 160))
    layout = stream_layout(q, 160, design, {"phase": "test", "design": design}, chunk=37)
    assert sum(x["count"] for x in layout) == 160
    assert all(("label_seed" in x) == (design == "iid") for x in layout)


def test_unequal_strata_mass_mean_covariance_and_signed_pair():
    values = [torch.tensor([[1., 4.], [2., 2.], [3., 3.]], dtype=torch.float64),
        torch.tensor([[4., 1.], [8., 6.], [5., 2.], [7., 4.]], dtype=torch.float64)]
    masses = torch.tensor([.25, .75], dtype=torch.float64)
    stats = estimate([sufficient(v.log()) for v in values], masses)
    mean = sum(w*v.mean(0) for w, v in zip(masses, values, strict=True))
    covariance = sum(w*w*torch.cov(v.T)/len(v) for w, v in zip(masses, values, strict=True))
    torch.testing.assert_close(torch.tensor(stats["log_mean"]).double().exp(), mean, rtol=1e-6, atol=1e-8)
    scale = torch.tensor(stats["log_scale"], dtype=torch.float64).exp()
    torch.testing.assert_close(torch.tensor(stats["scaled_covariance_of_mean"], dtype=torch.float64)*torch.outer(scale, scale), covariance)
    pair = paired(stats, 0, 1, comparisons=240)
    y = mean[0]-mean[1]
    r = float(y/mean[0])
    influence = torch.tensor([1-r, -1.], dtype=torch.float64)
    assert pair["relative_reduction"] == pytest.approx(r)
    assert pair["relative_se"] == pytest.approx(float((influence @ covariance @ influence).sqrt()/mean[0]))
    reverse = paired(stats, 1, 0, comparisons=240)
    assert reverse["relative_reduction"] < 0


def test_iid_single_stratum_and_exact_noop_merge():
    v = torch.arange(1, 21, dtype=torch.float64)
    logs = torch.stack((v.log(), v.log()), 1)
    merged = merge_sufficient([sufficient(logs[:10]), sufficient(logs[10:])])
    stats = estimate([merged], torch.ones(1, dtype=torch.float64))
    assert stats["relative_se"][0] == pytest.approx(float(v.std()/math.sqrt(20)/v.mean()))
    p = paired(stats, 0, 1, comparisons=240)
    assert abs(p["relative_reduction"]) < 1e-15 and not p["positive_signal"]
    assert p["relative_se"] == pytest.approx(0, abs=1e-14)
    changed = copy.deepcopy(merged)
    changed["log_cross_sum"][0][1] += .1
    with pytest.raises(ValueError):
        estimate([changed], torch.ones(1, dtype=torch.float64))


def test_constant_per_stratum_has_zero_se_without_erasing_normalizer():
    s = [sufficient(torch.full((10, 1), -5., dtype=torch.float64)), sufficient(torch.full((90, 1), -7., dtype=torch.float64))]
    stats = estimate(s, torch.tensor([.1, .9], dtype=torch.float64))
    assert math.exp(stats["log_mean"][0]) == pytest.approx(.1*math.exp(-5)+.9*math.exp(-7))
    assert stats["relative_se"][0] < 1e-8


def test_allocation_caps_no_clipping_or_zero_variance_success():
    assert allocate(.05, .06, kind="probability")["count"] == 65536
    assert allocate(.9, .9, kind="probability")["count"] is None
    assert allocate(0., 0., kind="risk")["status"] == "unallocated_unresolved_variance"
    assert allocate(.2, .2, kind="risk")["count"] == 39680


def test_d2_stratified_primary_does_not_require_iid_precision():
    cells = [{"cell": c, "methods": {"stratified": {"gates": [True]*12}, "iid": {"gates": [False]*12}},
        "contrasts": [{"contradiction": False}]*12} for c in ("a", "b")]
    assert d2_decision(cells)["d2_gate"] is True
    cells[1]["methods"]["stratified"]["gates"][0] = False
    assert d2_decision(cells)["decision"] == "STOP_UNRESOLVED_BUDGET"


def test_repeated_decision_preserves_failures_and_cost_gate():
    rows = [{"cell": c, "rep": r, "arm": a, "evaluable": True, "success": True,
        "cost_ratio": .7, "bank_effect": True} for c in ("a", "b") for r in range(5) for a in ("E-IID", "E-STRAT")]
    assert repeated_decision(rows, ("a", "b"))["decision"] == "CONTINUE_CANDIDATE"
    for r in rows:
        r["cost_ratio"] = 2.
    assert repeated_decision(rows, ("a", "b"))["decision"] == "PIVOT_PRACTICAL_CLAIM"
    for r in rows:
        r["success"] = False
    assert repeated_decision(rows, ("a", "b"))["decision"] == "PIVOT_WEIGHT_ONLY"
    for r in rows:
        r["evaluable"] = False
    assert repeated_decision(rows, ("a", "b"))["decision"] == "STOP_UNRESOLVED_BUDGET"
    with pytest.raises(ValueError):
        repeated_decision(rows[:-1], ("a", "b"))
