import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.residual_brownian_covariance import (
    ARMS,
    ResidualBrownianCovarianceFamily,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def family(dimension=6):
    means = torch.stack([torch.zeros(dimension, dtype=torch.float64),
                         torch.linspace(-1., 2., dimension, dtype=torch.float64),
                         torch.linspace(.5, -.8, dimension, dtype=torch.float64)])
    return ResidualBrownianCovarianceFamily(GaussianMixtureShiftSpec(
        means, torch.tensor([.1, .6, .3], dtype=torch.float64)), chunk=7)


def cov(f, j, arm):
    covariance = np.eye(f.dimension)
    if arm != "base":
        a, b = 2*j-1, 2*j
        covariance[a, a] = f.variance
        if arm in ("independent", "coupled"):
            covariance[b, b] = f.variance
        if arm == "coupled":
            covariance[a, b] = covariance[b, a] = f.price_sign*(f.variance-1)
    return covariance


def dense(f, z, arm):
    points = z.numpy()
    terms = []
    for k in range(f.base.components):
        w = float(f.base.weights[k])
        mean = f.base.means[k].numpy()
        if k == 0 or arm == "base":
            terms.append(math.log(w)+multivariate_normal.logpdf(points, mean=mean, cov=np.eye(f.dimension)))
        else:
            terms.append(math.log(w*(1-f.alpha))+multivariate_normal.logpdf(points, mean=mean, cov=np.eye(f.dimension)))
            for j in range(1, f.times+1):
                terms.append(math.log(w*f.alpha/f.times)+multivariate_normal.logpdf(points, mean=mean, cov=cov(f, j, arm)))
    return logsumexp(terms, axis=0)-multivariate_normal.logpdf(points, mean=np.zeros(f.dimension), cov=np.eye(f.dimension))


@pytest.mark.parametrize("sign", [-1, 1])
def test_all_shifted_component_time_terms_match_independent_dense_gaussians(sign):
    f = replace(family(), price_sign=sign)
    z = torch.randn((23, 6), dtype=torch.float64, generator=torch.Generator().manual_seed(63))
    actual = f.log_densities(z)
    for i, arm in enumerate(ARMS):
        np.testing.assert_allclose(actual[:, i].numpy(), dense(f, z, arm), rtol=1e-11, atol=1e-11)


def test_shifted_covariance_uses_z_minus_component_mean():
    f = family(4)
    z = f.base.means[1:2].clone()
    expected = dense(f, z, "coupled")
    assert float(f.log_densities(z)[0, 3]) == pytest.approx(float(expected), abs=1e-12)


def test_reference_matches_dense_zero_mean_mixture():
    f = family()
    z = torch.randn((31, 6), dtype=torch.float64, generator=torch.Generator().manual_seed(23))
    points = z.numpy()
    p = multivariate_normal.logpdf(points, mean=np.zeros(6), cov=np.eye(6))
    terms = [math.log(.1)+p]
    for j in range(1, f.times+1):
        terms.append(math.log(.9/f.times)+multivariate_normal.logpdf(points, mean=np.zeros(6), cov=cov(f, j, "coupled")))
    expected = logsumexp(terms, axis=0)-p
    np.testing.assert_allclose(f.reference_log_density(z).numpy(), expected, atol=1e-12)


@pytest.mark.parametrize("arm", ARMS)
def test_sampling_preserves_analytic_mixture_mean_and_matches_covariance(arm):
    f = family(4)
    sample = f.sample(70000, arm, path_seed=13, label_seed=14)
    mean = (f.base.weights[:, None]*f.base.means).sum(0)
    expected = sum(float(w)*torch.outer(m-mean, m-mean) for w, m in zip(f.base.weights, f.base.means, strict=True))
    expected += torch.eye(4, dtype=torch.float64)
    if arm != "base":
        expected += f.alpha*.9/f.times*sum(torch.from_numpy(cov(f, j, arm)-np.eye(4)) for j in range(1, f.times+1))
    torch.testing.assert_close(sample.mean(0), mean, rtol=0, atol=.03)
    torch.testing.assert_close(torch.cov(sample.T), expected, rtol=0, atol=.08)


def test_reference_sampling_matches_expected_covariance_and_unused_coordinate():
    f = family(6)
    sample = f.reference_sample(70000, path_seed=16, label_seed=17)
    expected = torch.eye(6, dtype=torch.float64)+.9/f.times*sum(
        torch.from_numpy(cov(f, j, "coupled")-np.eye(6)) for j in range(1, f.times+1))
    torch.testing.assert_close(sample.mean(0), torch.zeros(6, dtype=torch.float64), rtol=0, atol=.03)
    torch.testing.assert_close(torch.cov(sample.T), expected, rtol=0, atol=.08)
    assert float(expected[-1, -1]) == 1


def test_floor_baseline_retention_and_exact_alpha_zero():
    f = family()
    z = 5*torch.randn((99, 6), dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    density = f.log_densities(z)
    assert (density >= math.log(.1)-1e-12).all()
    assert (density[:, 1:] >= density[:, :1]+math.log(.5)-1e-12).all()
    zero = replace(f, alpha=0.).log_densities(z)
    assert torch.equal(zero, zero[:, :1].expand_as(zero))
    assert (f.reference_log_density(z) >= math.log(.1)-1e-12).all()


def test_guide_free_reference_does_not_depend_on_shift_means():
    f = family()
    means = f.base.means.clone()
    means[1:] *= 4
    altered = replace(f, base=GaussianMixtureShiftSpec(means, f.base.weights))
    z = torch.randn((9, 6), dtype=torch.float64, generator=torch.Generator().manual_seed(36))
    assert torch.equal(f.reference_log_density(z), altered.reference_log_density(z))


@pytest.mark.parametrize("kwargs", [{"variance": 1.}, {"variance": math.inf}, {"alpha": 1.},
                                    {"alpha": -.1}, {"price_sign": 0}, {"price_sign": True}, {"chunk": 0}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(family(), **kwargs)


def test_invalid_shape_nonfinite_and_duplicate_seeds_rejected():
    f = family()
    with pytest.raises(ValueError):
        f.log_densities(torch.ones((3, 5), dtype=torch.float64))
    with pytest.raises(ValueError):
        f.log_densities(torch.full((3, 6), math.nan, dtype=torch.float64))
    with pytest.raises(ValueError):
        f.sample(3, "coupled", path_seed=1, label_seed=1)
    with pytest.raises(ValueError):
        f.reference_sample(True, path_seed=1, label_seed=2)
