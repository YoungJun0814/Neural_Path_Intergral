import math

import pytest
import torch

from src.path_integral.continuum_probability_bound import (
    conditional_cdf_log_constant,
    continuum_bound_diagnostic,
)
from src.path_integral.cyclic_umbrella_smc import cyclic_pair_permutation, estimate_cyclic_umbrella
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_pair_permutation_is_exact_norm_preserving_with_uniform_inverse():
    z = torch.arange(4*12, dtype=torch.float64).reshape(4, 12)
    shifts = torch.tensor([0, 1, 2, 5])
    y = cyclic_pair_permutation(z, shifts)
    assert torch.equal(cyclic_pair_permutation(y, (-shifts) % 6), z)
    assert torch.equal(z.square().sum(1), y.square().sum(1))
    assert torch.equal(y[1, :2], z[1, -2:])


def test_cyclic_mh_satisfies_detailed_balance_not_unconditional_acceptance():
    z = torch.tensor([[3., 1., 0., -2., 1., 4.]], dtype=torch.float64)
    y = cyclic_pair_permutation(z, torch.tensor([1]))
    def log_h(x):
        return -.5*x[:, 0].square()
    beta = .63
    lp, lq = -.5*z.square().sum()+beta*log_h(z)[0], -.5*y.square().sum()+beta*log_h(y)[0]
    forward = lp+torch.minimum(beta*(log_h(y)-log_h(z))[0], torch.tensor(0.))
    reverse = lq+torch.minimum(beta*(log_h(z)-log_h(y))[0], torch.tensor(0.))
    torch.testing.assert_close(forward, reverse, atol=1e-12, rtol=0)
    assert float(log_h(z)[0]) != float(log_h(y)[0])


def test_permutation_preserves_prior_covariance():
    generator = torch.Generator().manual_seed(5823)
    z = torch.randn((70000, 8), dtype=torch.float64, generator=generator)
    y = cyclic_pair_permutation(z, torch.randint(4, (len(z),), generator=generator))
    torch.testing.assert_close(y.mean(0), torch.zeros(8, dtype=torch.float64), atol=.02, rtol=0)
    torch.testing.assert_close(torch.cov(y.T), torch.eye(8, dtype=torch.float64), atol=.025, rtol=0)


@pytest.mark.parametrize("cyclic", [False, True])
def test_exact_constant_family_components_and_joint_work(cyclic):
    cfg = WeightedSMCConfig(32, (0., .01, .1, .5, 1.), 2, .35, 1, 52,
                            resample_every=2, resampling_scheme="stratified")
    def components(z):
        return torch.tensor([math.log(.2), math.log(.7)], dtype=torch.float64).expand(len(z), -1)
    result = estimate_cyclic_umbrella(components, dimension=6, log_family_weights=torch.full(
        (2,), -math.log(2), dtype=torch.float64), config=cfg, use_cyclic=cyclic)
    assert result.log_normalizer == pytest.approx(math.log(.45), abs=1e-12)
    torch.testing.assert_close(result.log_component_normalizers,
        torch.tensor([math.log(.2), math.log(.7)], dtype=torch.float64), atol=1e-12, rtol=0)
    assert result.joint_evaluations == result.joint_gradient_requests == 32*(1+3*2)
    assert all(s["surviving_initial_ancestors"] == 32 for s in result.stages)


def test_analytic_gaussian_normalizer_and_family_identity_with_nonconstant_potential():
    means = torch.tensor([1.5, -1.], dtype=torch.float64)
    def components(z):
        return -.5*(z[:, :1]-means).square()
    logs = []
    for rep in range(24):
        cfg = WeightedSMCConfig(128, tuple((i/16)**2 for i in range(17)), 2, .35, 1, rep+100,
                                resample_every=2, resampling_scheme="stratified")
        r = estimate_cyclic_umbrella(components, dimension=6, log_family_weights=torch.full(
            (2,), -math.log(2), dtype=torch.float64), config=cfg, use_cyclic=True)
        assert r.log_normalizer == pytest.approx(float(torch.logsumexp(r.log_component_normalizers, 0))-math.log(2), abs=1e-12)
        logs.append(r.log_component_normalizers)
    estimates = torch.stack(logs).exp()
    expected = torch.exp(-means.square()/4)/math.sqrt(2)
    torch.testing.assert_close(estimates.mean(0), expected, atol=.025, rtol=0)


@pytest.mark.parametrize("eta,rho", [(1.5, -.7), (2., -.7), (.5, 0.), (1., .8)])
def test_log_absolute_constant_matches_independent_direct_formula(eta, rho):
    xi, t, h = .04, 1., .05
    m = t**(2*h)
    b1 = (xi*t)**-1*math.exp(eta**2*m)
    b2 = (xi*t)**-2*math.exp(3*eta**2*m)
    cv = xi**2*math.exp(eta**2*m)*(eta**2+eta**4*m*math.cosh(eta**2*m/2))
    expected = (math.sqrt(b1/(math.pi*(1-rho**2)))*(.5*math.sqrt(t*cv)+.5*abs(rho)*eta*math.sqrt(xi))
                +math.sqrt(b2)/(2*math.sqrt(math.pi*math.e))*math.sqrt(t*cv))
    actual = conditional_cdf_log_constant(eta=eta, xi=xi, rho=rho, maturity=t, hurst=h)
    assert math.exp(actual) == pytest.approx(expected, rel=1e-13)


def test_continuum_bound_scales_but_never_uses_observed_mu_as_true_lower_bound():
    kwargs = dict(eta=1.5, xi=.04, rho=-.7, maturity=1., hurst=.05, observed_log_mu=-18.1)
    a = continuum_bound_diagnostic(steps=4096, **kwargs)
    b = continuum_bound_diagnostic(steps=8192, **kwargs)
    assert b["log_unclipped_absolute_upper"]-a["log_unclipped_absolute_upper"] == pytest.approx(-.05*math.log(2))
    assert a["absolute_upper_clipped_at_probability_range"] == 1.
    assert not a["relative_bias_certificate_obtained"]
    assert a["observed_mu_is_not_certified_true_probability_lower_bound"]
    assert a["diagnostic_log10_required_N_using_observed_scale"] > 100


@pytest.mark.parametrize("kwargs", [{"eta": 0.}, {"xi": -1.}, {"rho": 1.}, {"hurst": .5}, {"eta": True}])
def test_invalid_continuum_domain_rejected(kwargs):
    base = dict(eta=1.5, xi=.04, rho=-.7, maturity=1., hurst=.05)
    with pytest.raises(ValueError):
        conditional_cdf_log_constant(**{**base, **kwargs})


def test_invalid_permutation_and_smc_config_rejected():
    with pytest.raises(ValueError):
        cyclic_pair_permutation(torch.zeros((2, 3), dtype=torch.float64), torch.tensor([0, 0]))
    with pytest.raises(ValueError):
        cyclic_pair_permutation(torch.zeros((2, 4), dtype=torch.float64), torch.tensor([0, 2]))
    cfg = WeightedSMCConfig(8, (0., 1.), 1, .35, 1, 1)
    with pytest.raises(ValueError):
        estimate_cyclic_umbrella(lambda z: z[:, :1], dimension=4, log_family_weights=torch.zeros(1),
                                 config=cfg, use_cyclic=True)
