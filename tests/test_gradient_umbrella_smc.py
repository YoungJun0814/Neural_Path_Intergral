import math

import pytest
import torch

from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.gradient_umbrella_smc import (
    capped_proposal_drift,
    estimate_gradient_umbrella_normalizers,
    evaluate_family_gradient,
    pcnl_log_acceptance,
)
from src.path_integral.structural_v2_reference import log_moments
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("cap", [.5, 20.])
def test_gradient_mh_expansion_matches_full_reverse_density_and_detailed_balance(cap):
    generator = torch.Generator().manual_seed(6101)
    z = torch.randn((97, 8), dtype=torch.float64, generator=generator)
    y = torch.randn(z.shape, dtype=torch.float64, generator=generator)
    h, new_h = -.7*(z-2).square().sum(1), -.7*(y-2).square().sum(1)
    v = capped_proposal_drift(-1.4*(z-2), maximum_norm=cap)
    new_v = capped_proposal_drift(-1.4*(y-2), maximum_norm=cap)
    beta, s = .73, .35
    a = math.sqrt(1-s*s)
    forward = -.5*(y-a*z-(1-a)*beta*v).square().sum(1)/(s*s)
    reverse = -.5*(z-a*y-(1-a)*beta*new_v).square().sum(1)/(s*s)
    current = -.5*z.square().sum(1)+beta*h
    target = -.5*y.square().sum(1)+beta*new_h
    exact = target-current+reverse-forward
    ratio = pcnl_log_acceptance(z, y, h, new_h, v, new_v, beta=beta, scale=s)
    torch.testing.assert_close(ratio, exact, rtol=1e-12, atol=1e-12)
    inverse = pcnl_log_acceptance(y, z, new_h, h, new_v, v, beta=beta, scale=s)
    torch.testing.assert_close(ratio, -inverse, rtol=0, atol=1e-12)
    torch.testing.assert_close(current+forward+ratio.clamp_max(0), target+reverse+inverse.clamp_max(0),
                               rtol=1e-12, atol=1e-12)
    assert torch.linalg.vector_norm(v, dim=1).max() <= cap*(1+1e-12)


def test_zero_drift_beta_zero_and_invalid_inputs():
    z = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    y = -z
    h, new_h = -z.square().sum(1), -.3*y.square().sum(1)
    zero = torch.zeros_like(z)
    torch.testing.assert_close(pcnl_log_acceptance(z, y, h, new_h, zero, zero, beta=.4, scale=.35), .4*(new_h-h))
    torch.testing.assert_close(pcnl_log_acceptance(z, y, h, new_h, z, y, beta=0., scale=.35), torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        capped_proposal_drift(z, maximum_norm=True)
    with pytest.raises(ValueError):
        pcnl_log_acceptance(z, y, h, new_h, z, y, beta=.4, scale=1.)
    with pytest.raises(ValueError):
        pcnl_log_acceptance(z, y, h, new_h, z.float(), y, beta=.4, scale=.35)
    with pytest.raises(ValueError):
        evaluate_family_gradient(lambda x: torch.full((len(x), 1), math.nan, dtype=torch.float64),
                                 z, torch.zeros(1, dtype=torch.float64), maximum_drift_norm=20.)


def test_cached_physical_family_gradient_finite_difference_and_row_separability():
    evaluator = CachedVolterraTerminal.create(problem(32))
    generator = torch.Generator().manual_seed(6102)
    z = .5*torch.randn((3, 64), dtype=torch.float64, generator=generator)
    means = torch.stack((torch.zeros(64, dtype=torch.float64), torch.full((64,), .8, dtype=torch.float64)))
    density = CachedShiftDensity(means, torch.tensor([.1, .9], dtype=torch.float64).log()-.5*means.square().sum(1))
    weights = torch.tensor([.3, .7], dtype=torch.float64).log()
    def family(x):
        g = evaluator.log_probability(x)
        return torch.stack((g, log_risk_potential(g, density.log_q_over_p(x), defensive_mass=.1)), 1)
    _, _, gradient = evaluate_family_gradient(family, z, weights, maximum_drift_norm=1e6)
    singleton = torch.cat([evaluate_family_gradient(family, row[None], weights, maximum_drift_norm=1e6)[2] for row in z])
    torch.testing.assert_close(gradient, singleton, rtol=1e-11, atol=1e-10)
    direction = torch.randn(z.shape, dtype=torch.float64, generator=generator)
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    epsilon = 1e-5
    hp = evaluate_family_gradient(family, z+epsilon*direction, weights, maximum_drift_norm=1e6)[1]
    hm = evaluate_family_gradient(family, z-epsilon*direction, weights, maximum_drift_norm=1e6)[1]
    torch.testing.assert_close((hp-hm)/(2*epsilon), (gradient*direction).sum(1), rtol=2e-6, atol=2e-6)


def test_gradient_mh_preserves_known_gaussian_stationary_law():
    generator = torch.Generator().manual_seed(6103)
    count, precision, beta, s = 30000, 2., .8, .35
    center = torch.tensor([3., -2.], dtype=torch.float64)
    mean = beta*precision/(1+beta*precision)*center
    std = (1+beta*precision)**-.5
    z = mean+std*torch.randn((count, 2), dtype=torch.float64, generator=generator)
    def potential(x):
        return -.5*precision*(x-center).square().sum(1)
    h = potential(z)
    v = capped_proposal_drift(-precision*(z-center), maximum_norm=20.)
    a = math.sqrt(1-s*s)
    y = a*z+(1-a)*beta*v+s*torch.randn(z.shape, dtype=torch.float64, generator=generator)
    new_h = potential(y)
    new_v = capped_proposal_drift(-precision*(y-center), maximum_norm=20.)
    take = torch.rand(count, dtype=torch.float64, generator=generator).log() < pcnl_log_acceptance(z, y, h, new_h, v, new_v, beta=beta, scale=s)
    moved = torch.where(take[:, None], y, z)
    assert (moved.mean(0)-mean).abs().max() < 5*std/math.sqrt(count)
    assert ((moved-mean).square().mean(0)-std**2).abs().max() < 5*std**2*math.sqrt(2/count)


def test_gradient_smc_known_multimodal_family_and_separate_gradient_cost():
    centers = torch.tensor([[3., 0.], [-3., 0.], [0., 2.]], dtype=torch.float64)
    variance = .8**2
    exact = variance/(1+variance)*torch.exp(-centers.square().sum(1)/(2*(1+variance)))
    mixture = torch.tensor([.2, .3, .5], dtype=torch.float64).log()
    def components(z):
        return -.5*(z[:, None, :]-centers[None, :, :]).square().sum(2)/variance
    values = []
    for rep in range(16):
        cfg = WeightedSMCConfig(512, tuple((k/31)**2 for k in range(32)), 4, .35, 1, 6110+rep*103,
                                resample_every=4, resampling_scheme="stratified")
        out = estimate_gradient_umbrella_normalizers(components, dimension=2, log_mixture_weights=mixture, config=cfg)
        assert out.potential_evaluations == out.gradient_evaluations == 512*(1+30*4)
        values.append(out.log_component_normalizers)
    for j in range(3):
        stats = log_moments(torch.stack(values)[:, j])
        ratio = math.exp(stats["log_mean"])/float(exact[j])
        assert abs(ratio-1) < max(.08, 4*ratio*stats["relative_se"])


def test_constant_family_frozen_weights_and_rejected_guidance():
    cfg = WeightedSMCConfig(32, (0., .1, .5, 1.), 4, .35, 1, 6190,
                            resample_every=2, resampling_scheme="stratified")
    constants = torch.tensor([-7., -9., -12.], dtype=torch.float64)
    weights = torch.tensor([.2, .3, .5], dtype=torch.float64).log()
    def components(z):
        weights.add_(.1)
        return constants.expand(len(z), -1)
    result = estimate_gradient_umbrella_normalizers(components, dimension=8, log_mixture_weights=weights, config=cfg)
    torch.testing.assert_close(result.log_component_normalizers, constants, rtol=0, atol=1e-12)
    assert result.gradient_evaluations == 0
    bad = WeightedSMCConfig(32, (0., 1.), 4, .35, 1, 6190, independence_every=2)
    with pytest.raises(ValueError):
        estimate_gradient_umbrella_normalizers(components, dimension=8, log_mixture_weights=weights, config=bad)
