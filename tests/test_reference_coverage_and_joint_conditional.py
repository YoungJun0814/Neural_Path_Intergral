"""Independent density and conditional-law checks, not empirical tail certificates."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.joint_shift_conditional import JointShiftConditional
from src.path_integral.local_variance_reference import LocalVarianceReference


def mixture():
    means = torch.tensor([[0., 0.], [2., -1.], [-.5, 3.]], dtype=torch.float64)
    return DefensiveFiniteRankGaussianMixture(tuple(FiniteRankGaussianComponent(
        m, torch.empty((2, 0), dtype=torch.float64), torch.empty(0, dtype=torch.float64))
        for m in means), torch.tensor([.1, .3, .6], dtype=torch.float64))


def test_scale_density_independent_scipy():
    r = LocalVarianceReference(4, 2)
    z = torch.randn((31, 4), dtype=torch.float64, generator=torch.Generator().manual_seed(15))
    components = [math.log(.1)+multivariate_normal.logpdf(z.numpy(), cov=np.eye(4))]
    for k in range(2):
        covariance = np.eye(4)
        covariance[2*k:2*k+2, 2*k:2*k+2] *= 25
        components.append(math.log(.45)+multivariate_normal.logpdf(z.numpy(), cov=covariance))
    expected = logsumexp(components, axis=0)-multivariate_normal.logpdf(z.numpy(), cov=np.eye(4))
    assert np.max(np.abs(expected-r.log_q_over_p(z).numpy())) < 1e-12
    assert bool((r.log_q_over_p(z) >= math.log(.1)).all())


def test_scale_normalization_and_sampling():
    r = LocalVarianceReference(1, 1)
    def density(x):
        z = torch.tensor([[x]], dtype=torch.float64)
        return math.exp(float(r.log_q_over_p(z)[0]))*norm.pdf(x)
    actual, error = quad(density, -35, 35, epsabs=1e-10)
    assert abs(actual-1) < 1e-10 and error < 1e-8
    r4 = LocalVarianceReference(4, 2)
    z = r4.sample(120000, path_seed=37, label_seed=38)
    # Each coordinate is inflated with probability .9/2.
    assert torch.max(torch.abs(z.mean(0))) < .05
    assert torch.max(torch.abs(z.var(0)-11.8)) < .2
    assert torch.equal(z[:10], r4.sample(120000, path_seed=37, label_seed=38)[:10])


@pytest.mark.parametrize("kwargs", [{"dimension": True}, {"dimension": 3},
    {"dimension": 2, "variance": 1}, {"dimension": 2, "defensive_mass": float("nan")},
    {"dimension": 2, "block_size": 0}])
def test_scale_invalid(kwargs):
    with pytest.raises(ValueError):
        LocalVarianceReference(**kwargs)


def test_joint_rotation_factorization_and_posterior():
    angle = .37
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]], dtype=torch.float64)
    r = mixture()
    conditional = JointShiftConditional.create(r, rotation, 1)
    z = torch.randn((41, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(77))
    a, b = conditional.split(z)
    assert torch.max(torch.abs(conditional.join(a, b)-z)) < 1e-14
    factorized = conditional.log_marginal_over_prior(a)+conditional.log_conditional_over_prior(a, b)
    assert torch.max(torch.abs(factorized-r.log_q_over_p(z))) < 2e-14
    ma = conditional.means[:, 0].numpy()
    logs = np.log(r.weights.numpy())[None, :]+norm.logpdf(a.numpy(), loc=ma)
    posterior = logs-logsumexp(logs, axis=1, keepdims=True)
    assert np.max(np.abs(posterior-conditional.conditional_log_weights(a).numpy())) < 1e-14
    # Independently integrate conditional density and weighted inner expectation.
    fixed_a = torch.tensor([[.4]], dtype=torch.float64)
    def integrand(x, weighted=False):
        b = torch.tensor([[x]], dtype=torch.float64)
        density_ratio = float(conditional.log_conditional_over_prior(fixed_a, b)[0])
        density = math.exp(density_ratio)*norm.pdf(x)
        return density*(math.exp(-.5*x*x-density_ratio) if weighted else 1.)
    assert abs(quad(integrand, -20, 20)[0]-1) < 1e-9
    assert abs(quad(lambda x: integrand(x, True), -20, 20)[0]-1/math.sqrt(2)) < 1e-9


def test_conditional_sampling_moments_and_seed_validation():
    law = JointShiftConditional.create(mixture(), torch.eye(2, dtype=torch.float64), 1)
    a = torch.full((100000, 1), .4, dtype=torch.float64)
    b = law.sample_inner(a, path_seed=41, label_seed=42)
    weights = law.conditional_log_weights(a[:1]).exp()[0]
    means = law.means[:, 1]
    expected = float(weights @ means)
    variance = float(weights @ means.square())-expected**2+1
    assert abs(float(b.mean())-expected) < .025
    assert abs(float(b.var())-variance) < .06
    for path_seed, label_seed in ((1, 1), (True, 2), (1, 0)):
        with pytest.raises(ValueError):
            law.sample_inner(a[:1], path_seed=path_seed, label_seed=label_seed)
    with pytest.raises(ValueError):
        JointShiftConditional.create(mixture(), torch.ones((2, 2), dtype=torch.float64), 1)
    with pytest.raises(ValueError):
        JointShiftConditional.create(mixture(), torch.eye(3, dtype=torch.float64), 1)
    with pytest.raises(ValueError):
        law.join(a[:1], b[:2])


def test_joint_original_risk_integral_not_squared_conditional_mean():
    law = JointShiftConditional.create(mixture(), torch.eye(2, dtype=torch.float64), 1)
    fixed_a = torch.tensor([[.7]], dtype=torch.float64)
    original_q = mixture()
    def integrand(x, importance):
        b = torch.tensor([[x]], dtype=torch.float64)
        z = law.join(fixed_a, b)
        log_h = -float(z.square().sum())-float(original_q.log_q_over_p(z)[0])
        log_r = float(law.log_conditional_over_prior(fixed_a, b)[0])
        if importance:
            return math.exp(log_h-log_r)*math.exp(log_r)*norm.pdf(x)
        return math.exp(log_h)*norm.pdf(x)
    direct = quad(lambda x: integrand(x, False), -20, 20, epsabs=1e-10)[0]
    weighted = quad(lambda x: integrand(x, True), -20, 20, epsabs=1e-10)[0]
    assert abs(direct-weighted) < 1e-10


def test_joint_rejects_nonzero_covariance_rank():
    natural = FiniteRankGaussianComponent.natural(2)
    ranked = FiniteRankGaussianComponent(torch.zeros(2, dtype=torch.float64),
        torch.tensor([[1.], [0.]], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64))
    r = DefensiveFiniteRankGaussianMixture((natural, ranked), torch.tensor([.1, .9], dtype=torch.float64))
    with pytest.raises(ValueError):
        JointShiftConditional.create(r, torch.eye(2, dtype=torch.float64), 1)
