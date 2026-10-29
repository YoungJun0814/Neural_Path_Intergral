import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.grouped_covariance_density import GroupedCovarianceDensity, LocalMeanGroups
from src.path_integral.residual_brownian_covariance import ARMS, ResidualBrownianCovarianceFamily
from src.path_integral.window_coupled_covariance import WindowCoupledCovariance


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def family(dimension=12):
    means = torch.zeros((5, dimension), dtype=torch.float64)
    means[1] = torch.linspace(-1, 2, dimension, dtype=torch.float64)
    means[2] = means[1]
    means[2, 0] = 4
    means[3, 0] = 2
    means[4, -1] = -3
    return ResidualBrownianCovarianceFamily(GaussianMixtureShiftSpec(
        means, torch.tensor([.1, .2, .3, .25, .15], dtype=torch.float64)), chunk=7)


def points(dimension=12):
    return torch.randn((23, dimension), dtype=torch.float64,
                       generator=torch.Generator().manual_seed(5820))


@pytest.mark.parametrize("sign", [-1, 1])
def test_grouped_all_and_single_arms_match_original_for_duplicates_and_extremes(sign):
    f = replace(family(), price_sign=sign)
    z = torch.cat((points(), 30*points()))
    grouped = GroupedCovarianceDensity.create(f)
    expected = f.log_densities(z)
    torch.testing.assert_close(grouped.log_densities(z), expected, rtol=1e-12, atol=1e-10)
    assert grouped.group_counts() == [2]*f.times
    for i, arm in enumerate(ARMS):
        torch.testing.assert_close(grouped.log_density(z, arm), expected[:, i], rtol=1e-12, atol=1e-10)
    torch.testing.assert_close(grouped.log_densities(z, ("coupled", "base", "residual")),
                               expected[:, [3, 0, 1]], rtol=1e-12, atol=1e-10)


def test_exact_mean_grouping_never_rounds_adjacent_float_values():
    x = np.array([[1., 2.], [np.nextafter(1., 2.), 2.], [1., 2.]], dtype=np.float64)
    groups = LocalMeanGroups.create(x)
    assert len(groups.pair_means) == 2
    logs = np.array([[-10000., 0., -10001.]], dtype=np.float64)
    actual = groups.aggregate(logs)
    np.testing.assert_allclose(actual, [[np.logaddexp(-10000., -10001.), 0.]], atol=1e-12)


def test_group_small_global_weight_can_receive_large_later_correction():
    groups = LocalMeanGroups.create(np.array([[0., 0.], [1., 0.]], dtype=np.float64))
    logs = np.array([[0., -10000.]], dtype=np.float64)
    result = logsumexp(groups.aggregate(logs)+np.array([[0., 10001.]]), axis=1)
    np.testing.assert_allclose(result, [np.logaddexp(0., 1.)], atol=1e-12)


def test_grouped_alpha_zero_exact_and_original_mutation_does_not_stale_cache():
    f = replace(family(), alpha=0.)
    grouped = GroupedCovarianceDensity.create(f)
    z = points()
    before = grouped.log_densities(z)
    assert torch.equal(before, f.log_densities(z))
    f.base.means[1:, 0] += 20
    f.base.weights[1], f.base.weights[2] = .3, .2
    assert torch.equal(grouped.log_densities(z), before)


@pytest.mark.parametrize("arms", [(), ("x",), ("base", "base")])
def test_grouped_invalid_arms(arms):
    with pytest.raises(ValueError):
        GroupedCovarianceDensity.create(family()).log_densities(points(), arms)


@pytest.mark.parametrize("x", [np.ones((2, 3)), np.ones((0, 2)),
                              np.ones((2, 2), dtype=np.float32), np.full((2, 2), np.nan)])
def test_invalid_group_means(x):
    with pytest.raises(ValueError):
        LocalMeanGroups.create(x)


def dense_window(window, z, reference=False):
    f, terms = window.family, []
    p = multivariate_normal.logpdf(z.numpy(), mean=np.zeros(f.dimension), cov=np.eye(f.dimension))
    for k, (mean, weight) in enumerate(zip(f.base.means, f.base.weights, strict=True)):
        if reference:
            break
        terms.append(math.log(float(weight)*(1 if k == 0 else 1-f.alpha))
                     +multivariate_normal.logpdf(z.numpy(), mean=mean.numpy(), cov=np.eye(f.dimension)))
        if k:
            for w in window.directions.numpy():
                covariance = np.eye(f.dimension)+(window.eigenvalue-1)*np.outer(w, w)
                terms.append(math.log(float(weight)*f.alpha/len(window.windows))
                    +multivariate_normal.logpdf(z.numpy(), mean=mean.numpy(), cov=covariance))
    if reference:
        terms = [math.log(.1)+p]
        for w in window.directions.numpy():
            terms.append(math.log(.9/len(window.windows))+multivariate_normal.logpdf(
                z.numpy(), mean=np.zeros(f.dimension),
                cov=np.eye(f.dimension)+(window.eigenvalue-1)*np.outer(w, w)))
    return logsumexp(terms, axis=0)-p


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("width", [1, 2, 5])
def test_window_and_reference_match_dense_independent_gaussian_oracle(sign, width):
    window = WindowCoupledCovariance.create(replace(family(), price_sign=sign), width)
    z = points()
    np.testing.assert_allclose(window.log_density(z).numpy(), dense_window(window, z), atol=1e-11)
    np.testing.assert_allclose(window.reference_log_density(z).numpy(),
                               dense_window(window, z, True), atol=1e-11)


def test_width_one_recovers_old_local_distribution_and_equal_trace():
    f = family()
    window = WindowCoupledCovariance.create(f, 1)
    torch.testing.assert_close(window.log_density(points()), f.log_densities(points())[:, 3],
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(window.reference_log_density(points()), f.reference_log_density(points()),
                               rtol=1e-12, atol=1e-12)
    actual = WindowCoupledCovariance.create(family(64), 4)
    assert actual.windows[-1] == (29, 32)
    assert len(actual.windows) == 8
    torch.testing.assert_close(actual.directions @ actual.directions.T, torch.eye(8, dtype=torch.float64))
    assert (actual.directions[:, -1] == 0).all()
    for w in actual.directions:
        covariance = torch.eye(64, dtype=torch.float64)+8*torch.outer(w, w)
        assert float(covariance.trace()-64) == pytest.approx(8)
        assert float(torch.linalg.eigvalsh(covariance)[-1]) == pytest.approx(9)


@pytest.mark.parametrize("reference", [False, True])
def test_window_sampling_matches_mean_covariance_and_unused_coordinate(reference):
    w = WindowCoupledCovariance.create(family(), 2)
    f = w.family
    mean = torch.zeros(f.dimension, dtype=torch.float64) if reference else (f.base.weights[:, None]*f.base.means).sum(0)
    expected = torch.eye(f.dimension, dtype=torch.float64)
    mass = .9 if reference else .9*f.alpha
    expected += mass*(w.eigenvalue-1)/len(w.windows)*(w.directions.T @ w.directions)
    if not reference:
        expected += sum(float(weight)*torch.outer(m-mean, m-mean)
                        for weight, m in zip(f.base.weights, f.base.means, strict=True))
    fun = w.reference_sample if reference else w.sample
    z = fun(120000, path_seed=44, label_seed=45)
    torch.testing.assert_close(z.mean(0), mean, rtol=0, atol=.025)
    torch.testing.assert_close(torch.cov(z.T), expected, rtol=0, atol=.075)
    assert torch.equal(z, fun(120000, path_seed=44, label_seed=45))


def test_window_density_floor_retention_alpha_zero_and_snapshot():
    f = family()
    w = WindowCoupledCovariance.create(f, 2)
    z = 15*points()
    before = w.log_density(z)
    base = f.log_densities(z)[:, 0]
    assert (before >= math.log(.1)-1e-12).all()
    assert (before >= base+math.log1p(-f.alpha)-1e-12).all()
    f.base.means[1:, 0] += 10
    assert torch.equal(w.log_density(z), before)
    zero = WindowCoupledCovariance.create(replace(family(), alpha=0.), 2)
    torch.testing.assert_close(zero.log_density(z), zero.family.log_densities(z)[:, 0], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("width", [0, -1, 6, True, 1.5])
def test_invalid_window_width(width):
    with pytest.raises(ValueError):
        WindowCoupledCovariance.create(family(), width)


def test_bad_input_and_sampling_seeds_fail_closed():
    w = WindowCoupledCovariance.create(family(), 2)
    for fun in (w.log_density, w.reference_log_density,
                GroupedCovarianceDensity.create(family()).log_densities):
        with pytest.raises(ValueError):
            fun(torch.ones((3, 12), dtype=torch.float32))
    for fun in (w.sample, w.reference_sample):
        with pytest.raises(ValueError):
            fun(3, path_seed=5, label_seed=5)
        with pytest.raises(ValueError):
            fun(True, path_seed=5, label_seed=6)
