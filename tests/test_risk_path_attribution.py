import copy
import math

import pytest
import torch

import experiments.post_audit_frozen_q_path_attribution as runner
from experiments.post_audit_frozen_q_path_attribution import phase_key, physical_features
from src.path_integral.balanced_mixture_bank import stream_layout
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.risk_path_attribution import (
    FEATURES,
    candidate_rules,
    contrasts,
    decode,
    diagnostic_logs,
    estimate,
    merge,
    moments,
    rule_mask,
)
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_zero_aware_unequal_stratum_mean_and_covariance_match_hand_formula():
    values = [torch.tensor([[1., 0., 2.], [2., 0., 0.], [3., 0., 5.]], dtype=torch.float64),
        torch.tensor([[3., 1., 0.], [2., 0., 0.], [7., 0., 3.], [4., 0., 2.]], dtype=torch.float64)]
    masses = torch.tensor([.2, .8], dtype=torch.float64)
    result = estimate([moments(v.log()) for v in values], masses)
    mean = sum(p*v.mean(0) for p, v in zip(masses, values, strict=True))
    cov = sum(p*p*torch.cov(v.T)/len(v) for p, v in zip(masses, values, strict=True))
    scale = torch.tensor(result["scale"], dtype=torch.float64).exp()
    torch.testing.assert_close(torch.tensor(result["mean"], dtype=torch.float64)*scale, mean)
    torch.testing.assert_close(torch.tensor(result["covariance"], dtype=torch.float64)*torch.outer(scale, scale), cov)
    assert moments(values[0].log())["log_sum"][1] is None


def test_merge_preserves_exact_zero_and_constant_column():
    v = torch.tensor([[0., 2.], [0., 3.], [0., 4.], [0., 6.]], dtype=torch.float64)
    pooled = merge([moments(v[:2].log()), moments(v[2:].log())])
    assert pooled == moments(v.log())
    out = estimate([pooled], torch.ones(1, dtype=torch.float64))
    assert out["mean"][0] == out["covariance"][0][0] == 0
    torch.testing.assert_close(decode([None, 0.]), torch.tensor([-math.inf, 0.], dtype=torch.float64))


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_inputs_rejected(bad):
    with pytest.raises(ValueError):
        moments(torch.tensor([[0., bad], [0., 0.]], dtype=torch.float64))


def test_invalid_count_mass_or_asymmetry_rejected():
    part = moments(torch.zeros((4, 2), dtype=torch.float64))
    with pytest.raises(ValueError):
        estimate([part], torch.tensor([.5], dtype=torch.float64))
    bad = copy.deepcopy(part)
    bad["log_cross_sum"][0][1] += .1
    with pytest.raises(ValueError):
        estimate([bad], torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize("empty", [False, True])
def test_ratio_contrasts_match_direct_delta_covariance(empty):
    y = torch.arange(1, 21, dtype=torch.float64)
    logs = torch.stack((y.log(), (y*y).log(), (y*3).log()), 1)
    indicator = y > 15 if not empty else y < 0
    values = diagnostic_logs(logs, indicator).exp()
    result = estimate([moments(values.log())], torch.ones(1, dtype=torch.float64))
    contrast = contrasts(result, 0)
    mean = values.mean(0)
    cov = torch.cov(values.T)/len(values)
    tau, event, base = mean[5]/mean[1], mean[4]/mean[0], mean[3]
    assert contrast["risk_mass"] == pytest.approx(float(tau))
    assert contrast["event_mass"] == pytest.approx(float(event))
    assert contrast["G_mass"] == pytest.approx(float(base))
    gt, ge, gb = [torch.zeros(7, dtype=torch.float64) for _ in range(3)]
    gt[5], gt[1] = 1/mean[1], -tau/mean[1]
    ge[4], ge[0] = 1/mean[0], -event/mean[0]
    gb[3] = 1
    for row, grad in zip(contrast["intervals"], (gt, gt-2*gb, gt-ge), strict=True):
        assert row["se"] == pytest.approx(math.sqrt(max(0., float(grad @ cov @ grad))), abs=1e-12)
    if empty:
        assert contrast["enrichment"] is None and not contrast["all_lower_positive"]


def test_empty_tail_with_extreme_log_scale_does_not_multiply_zero_by_infinity():
    logs = torch.full((20, 3), -1000., dtype=torch.float64)
    result = estimate([moments(diagnostic_logs(logs, torch.zeros(20, dtype=torch.bool)))], torch.ones(1, dtype=torch.float64))
    out = contrasts(result, 0)
    assert out["risk_mass"] == 0 and all(math.isfinite(r["se"]) for r in out["intervals"])


def discovery_data():
    x = torch.arange(1000, dtype=torch.float64)/1000
    f = torch.stack([x]*6, 1)
    logs = torch.stack((torch.zeros_like(x), torch.where(x >= .9, 5., 0.), torch.where(x >= .9, 4., 0.)), 1).double()
    return {c: f.clone() for c in ("a", "b")}, {c: logs.clone() for c in ("a", "b")}


def test_one_shared_rule_weakest_cell_score_and_stable_tie_order():
    fs, logs = discovery_data()
    chosen = candidate_rules(fs, logs)
    assert len(chosen["candidates"]) == 12
    rule = chosen["selected"]
    assert rule["feature"] == FEATURES[0] and rule["direction"] == "high"
    for c in fs:
        mask = rule_mask(fs[c], rule, c)
        assert int(mask.sum()) == 100
        assert rule["cells"][c]["risk_mass"][0] > .9


def test_missing_or_constant_cell_does_not_become_eligible_by_quantile_ties():
    fs, logs = discovery_data()
    fs["b"][:] = 1
    chosen = candidate_rules(fs, logs)
    assert chosen["selected"] is None
    with pytest.raises(ValueError):
        candidate_rules({"a": fs["a"]}, {"a": logs["a"]})


def test_rule_identity_cannot_change_after_selection():
    fs, logs = discovery_data()
    rule = candidate_rules(fs, logs)["selected"]
    rule["feature"] = "arbitrary-new-feature"
    with pytest.raises(ValueError):
        rule_mask(fs["a"], rule, "a")


def test_single_quantities_features_match_cached_payoff_and_hand_variance():
    cache = CachedVolterraTerminal.create(problem(32))
    guide = PhysicalWindowGuide.create(cache.problem, kind="mixed-window", amplitudes=[2., 4., 6., 8.], price_amplitudes=[0., 2., 4.])
    z = torch.randn((40, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(98261))
    g, features, variance = physical_features(cache, guide, z)
    assert torch.equal(g, cache.log_probability(z))
    used = variance[:, :-1]
    torch.testing.assert_close(features[:, 0], (cache.problem.step_dt*used.sum(1)).log())
    torch.testing.assert_close(features[:, 2], used.argmax(1).double()/32)
    torch.testing.assert_close(features[:, 3], (used/used.sum(1, keepdim=True)).square().sum(1))
    assert (features[:, 5] >= 0).all() and (features[:, 5] <= 1).all()


def test_discovery_reuses_old_roles_but_confirmation_seeds_are_disjoint():
    cache = CachedVolterraTerminal.create(problem(32))
    guide = PhysicalWindowGuide.create(cache.problem, kind="mixed-window", amplitudes=[2.], price_amplitudes=[0.])
    streams = [stream_layout(guide.spec, 29760, "stratified", phase_key(p, "cell", b))
        for p in ("discovery", "confirmation") for b in range(8)]
    seeds = [r["path_seed"] for layout in streams for r in layout]
    assert len(set(seeds)) == len(seeds)
    assert phase_key("discovery", "cell", 3)["phase"] == "D2"


@pytest.mark.parametrize("outlier", [False, True])
def test_complete_confirmation_arrays_recompute_and_concentration_veto(tmp_path, monkeypatch, outlier):
    monkeypatch.setattr(runner, "COUNT", 80)
    q = GaussianMixtureShiftSpec(torch.zeros((5, 2), dtype=torch.float64), torch.tensor([.1, .225, .225, .225, .225], dtype=torch.float64))
    labels = torch.cat([torch.full((n,), j, dtype=torch.int64) for j, n in enumerate((8, 18, 18, 18, 18))])
    fs, ls = discovery_data()
    rule = candidate_rules(fs, ls)["selected"]
    rule["cells"] = {c: {"cut": .5} for c in ("a", "b")}
    records = []
    for cell in ("a", "b"):
        for block in range(8):
            f = torch.zeros((80, 6), dtype=torch.float64)
            f[::4, 0] = 1
            h = torch.where(f[:, 0] == 1, 100., 1.)
            logs = torch.stack((torch.zeros(80, dtype=torch.float64), h.log(), h.log()), 1)
            if outlier and block == 7:
                logs[0, 1:] = 25
            trace = {"features": f, "logs": logs, "labels": labels,
                "metadata": {"cell": cell, "block": block}}
            records.append(runner.save_trace(tmp_path, cell+str(block), trace))
    result = runner.confirmation_summary(tmp_path, records, {c: (None, q) for c in ("a", "b")}, rule)
    assert result["primary_confirmed"] is not outlier
    assert result["D3_authorized"] is False and result["model_changed"] is False
    if outlier:
        assert not any(c["outcomes"][0]["precision"] for c in result["cells"])
    bad = tmp_path/records[0]["array"]
    bad.write_bytes(b"bad compressed data")
    with pytest.raises(ValueError, match="hash"):
        runner.load_trace(tmp_path, records[0])


def test_trace_features_reuse_original_draw_and_full_moments(monkeypatch):
    import experiments.post_audit_bounded_bank_runtime as old
    monkeypatch.setattr(runner, "COUNT", 1240)
    cache = CachedVolterraTerminal.create(problem(32))
    guide = PhysicalWindowGuide.create(cache.problem, kind="mixed-window", amplitudes=[2., 4., 6., 8.], price_amplitudes=[0., 2., 4.])
    k = phase_key("discovery", "test", 0)
    weights = [guide.spec.weights.tolist()]*11
    before = old.risk_block((cache, guide.spec), weights, "stratified", 1240, k)
    trace = runner.draw_trace((cache, guide.spec), guide, weights, k)
    for a, b in (("old_strata", "strata"), ("old_estimator", "estimator"), ("seed_digest", "seed_digest"), ("sample_digest", "sample_digest")):
        assert trace["metadata"][a] == before[b]
    assert trace["features"].shape == (1240, 6)
    assert len(trace["metadata"]["tops"]) == 16
