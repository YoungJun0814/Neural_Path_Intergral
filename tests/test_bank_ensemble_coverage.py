import math

import pytest
import torch

from src.path_integral.bank_ensemble_coverage import (
    coverage_ledger,
    mean_defensive_weights,
    merge_coverage,
    path_bins,
    physical_path_features,
    summarize_bins,
)


def test_equal_bank_ensemble_preserves_defensive_floor_and_density_convexity():
    a = torch.tensor([.1, .1, .8], dtype=torch.float64)
    b = torch.tensor([.1, .8, .1], dtype=torch.float64)
    ensemble = mean_defensive_weights([a, b])
    torch.testing.assert_close(ensemble, torch.tensor([.1, .45, .45], dtype=torch.float64))
    z = torch.linspace(-6, 6, 401, dtype=torch.float64)
    m = torch.tensor([0., -2., 2.], dtype=torch.float64)
    ratios = (z[:, None]*m-.5*m.square()).exp()
    g2 = torch.exp(-z.square())
    assert float((g2/(ratios @ ensemble)).mean()) <= .5*float((g2/(ratios @ a)+g2/(ratios @ b)).mean())


def test_prefix_window_features_match_direct_sliding_window_oracle():
    v = torch.rand((4, 128), dtype=torch.float64, generator=torch.Generator().manual_seed(28001))+.01
    features = physical_path_features(v, maturity=2.)
    expected = v.unfold(1, 4, 1).sum(2).max(1).values/v.sum(1)
    torch.testing.assert_close(features[:, 2], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(features[:, 0].exp(), 2*v.mean(1))
    torch.testing.assert_close(features[:, 1].exp(), v.max(1).values/v.mean(1))


def test_constant_path_and_spike_have_distinct_energy_concentration():
    v = torch.ones((2, 128), dtype=torch.float64)
    v[1, 63] = 10000.
    f = physical_path_features(v, maturity=1.)
    assert f[0, 1] == 0 and f[0, 2] == 1/32
    assert f[1, 2] > .98 and f[1, 3] == 63/128


def test_fixed_bin_boundary_equalities_go_right():
    f = torch.tensor([[math.log(.25), 0., .25, .25], [math.log(4.), 0., .75, .75]], dtype=torch.float64)
    assert path_bins(f).tolist() == [(1*4+1)*4+1, (5*4+3)*4+3]


def test_all_samples_partition_reconstructs_ordinary_mean_and_zero_outside_bin_se():
    f = torch.tensor([[math.log(.1), 0., .1, .1], [math.log(.1), 0., .1, .1],
                      [math.log(2.), 0., .9, .9], [math.log(2.), 0., .9, .9]], dtype=torch.float64)
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    ledger = merge_coverage([coverage_ledger(x[:2].log(), f[:2]), coverage_ledger(x[2:].log(), f[2:])])
    assert math.exp(ledger["log_sum"])/ledger["count"] == pytest.approx(2.5)
    rows = summarize_bins(ledger)
    first = next(r for r in rows if r["sample_count"] and r["intensity_bin"] == 0)
    assert math.exp(first["log_ordinary_mean"]) == pytest.approx(.75)  # Not stratum mean 1.5.
    direct = torch.tensor([1., 2., 0., 0.], dtype=torch.float64)
    expected_rse = float(direct.std(unbiased=True)/2/direct.mean())
    assert first["relative_se"] == pytest.approx(expected_rse)
    empty = next(r for r in rows if not r["sample_count"])
    assert empty["relative_se"] is None and empty["status"] == "unobserved_not_zero_probability"


def test_merge_detects_corrupt_bin_partition_and_invalid_ensembles():
    with pytest.raises(ValueError):
        mean_defensive_weights([torch.tensor([.1, .45, .45], dtype=torch.float64)])
    with pytest.raises(ValueError):
        mean_defensive_weights([torch.tensor([.1, .89, .01], dtype=torch.float64)]*2)
    f = physical_path_features(torch.ones((2, 32), dtype=torch.float64), maturity=1.)
    ledger = coverage_ledger(torch.zeros(2, dtype=torch.float64), f)
    ledger["bin_counts"][0] += 1
    with pytest.raises(ValueError):
        merge_coverage([ledger])
