import math

import pytest
import torch

from src.path_integral.fixed_stratum_reference import proportional_guide_counts, stratified_summary
from src.path_integral.structural_v2_reference import log_moments


def test_declared_proportional_allocation_preserves_known_weights():
    weights = torch.tensor([.1, .3, .3, .3], dtype=torch.float64)
    counts = proportional_guide_counts(weights, multiplier=8)
    assert counts == (24, 72, 72, 72)
    torch.testing.assert_close(torch.tensor(counts, dtype=torch.float64)/sum(counts), weights)
    with pytest.raises(ValueError):
        proportional_guide_counts(torch.tensor([.2, .8], dtype=torch.float64), multiplier=8)


def test_stratum_estimator_and_se_match_direct_independent_sample_formula():
    values = [torch.tensor([1., 2., 3.], dtype=torch.float64), torch.tensor([4., 5., 6., 7.], dtype=torch.float64)]
    weights = torch.tensor([.25, .75], dtype=torch.float64)
    result = stratified_summary([log_moments(v.log()) for v in values], weights.log())
    mean = sum(float(w*v.mean()) for w, v in zip(weights, values, strict=True))
    variance = sum(float(w*w*v.var(unbiased=True)/len(v)) for w, v in zip(weights, values, strict=True))
    assert math.exp(result["log_mean"]) == pytest.approx(mean)
    assert result["relative_se"] == pytest.approx(math.sqrt(variance)/mean)
    maximum = max(float((w*v/len(v)).max()) for w, v in zip(weights, values, strict=True))
    assert result["maximum_fraction"] == pytest.approx(maximum/mean)


def test_removing_label_variance_does_not_remove_the_normalizer():
    # Different constants per stratum still have a non-unit weighted normalizer.
    rows = [log_moments(torch.full((9,), -5., dtype=torch.float64)),
            log_moments(torch.full((81,), -7., dtype=torch.float64))]
    result = stratified_summary(rows, torch.tensor([.1, .9], dtype=torch.float64).log())
    assert math.exp(result["log_mean"]) == pytest.approx(.1*math.exp(-5)+.9*math.exp(-7))
    assert result["relative_se"] < 1e-8


def test_fixed_mixture_density_stratification_known_integral():
    generator = torch.Generator().manual_seed(6201)
    weights = torch.tensor([.1, .9], dtype=torch.float64)
    means = torch.tensor([0., 3.], dtype=torch.float64)
    counts = proportional_guide_counts(weights, multiplier=2048)
    rows = []
    for mean, count in zip(means, counts, strict=True):
        z = mean+torch.randn(count, dtype=torch.float64, generator=generator)
        log_d = torch.logsumexp(z[:, None]*means-.5*means.square()+weights.log(), 1)
        rows.append(log_moments(-.5*(z-2).square()-log_d))
    result = stratified_summary(rows, weights.log())
    exact = 2**-.5*math.exp(-1.)
    assert abs(math.exp(result["log_mean"])/exact-1) < max(.02, 5*result["relative_se"])
