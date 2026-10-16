import math

import pytest
import torch

from src.path_integral.guide_free_block_smc import estimate_guide_free_block_normalizer
from src.path_integral.local_variance_reference import LocalVarianceReference
from src.path_integral.proposal_bridge_smc import estimate_proposal_bridge
from src.path_integral.structural_v2_reference import log_moments
from src.path_integral.umbrella_normalizer_smc import (
    estimate_umbrella_normalizers,
    terminal_component_log_normalizers,
    umbrella_log_potential,
)
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_terminal_measure_identity_and_bounded_ratios():
    components = torch.tensor([[-1., -4., -2.], [-4., -1., -3.]], dtype=torch.float64)
    mixture = torch.tensor([.1, .3, .6], dtype=torch.float64).log()
    particle = torch.tensor([.4, .6], dtype=torch.float64).log()
    result = terminal_component_log_normalizers(-5., particle, components, mixture)
    assert abs(float(torch.logsumexp(result+mixture, 0))+5) < 1e-12
    h = umbrella_log_potential(components, mixture)
    assert torch.all(components-h[:, None] <= -mixture+1e-12)
    with pytest.raises(ValueError):
        umbrella_log_potential(components, mixture+1)
    with pytest.raises(ValueError):
        umbrella_log_potential(components, torch.tensor(0., dtype=torch.float64))


@pytest.mark.parametrize("blocks,guided", [(False, False), (True, False), (False, True)])
def test_known_distinct_gaussian_family_endpoints(blocks, guided):
    centers = torch.tensor([[3., 0.], [-3., 0.], [0., 2.]], dtype=torch.float64)
    variance = .8**2
    mixture = torch.tensor([.2, .3, .5], dtype=torch.float64).log()
    exact = variance/(1+variance)*torch.exp(-centers.square().sum(1)/(2*(1+variance)))
    def components(z):
        return -.5*(z[:, None, :]-centers[None, :, :]).square().sum(2)/variance
    logs = []
    for rep in range(16):
        cfg = WeightedSMCConfig(512, tuple((k/31)**2 for k in range(32)), 4, .35, 1, 5600+rep*103,
                                resample_every=4, resampling_scheme="stratified", independence_every=2 if guided else 0)
        out = estimate_umbrella_normalizers(components, dimension=2, log_mixture_weights=mixture,
                                            config=cfg, use_blocks=blocks,
                                            proposal=LocalVarianceReference(2, 1) if guided else None)
        assert out.potential_evaluations == 512*(1+30*4)
        logs.append(out.log_component_normalizers)
    for j in range(3):
        stats = log_moments(torch.stack(logs)[:, j])
        ratio = math.exp(stats["log_mean"])/float(exact[j])
        assert abs(ratio-1) < max(.08, 4*ratio*stats["relative_se"])


def test_single_component_reduces_to_guide_free_scalar_smc():
    cfg = WeightedSMCConfig(128, tuple((k/15)**2 for k in range(16)), 8, .35, 1, 5700,
                            resample_every=4, resampling_scheme="stratified")
    def potential(z):
        return -.2*(z-1).square().sum(1)
    vector = estimate_umbrella_normalizers(lambda z: potential(z)[:, None], dimension=8,
        log_mixture_weights=torch.zeros(1, dtype=torch.float64), config=cfg, use_blocks=True)
    scalar = estimate_guide_free_block_normalizer(potential, dimension=8, config=cfg, use_blocks=True)
    assert abs(vector.log_normalizer-scalar.log_normalizer) < 1e-12
    assert abs(float(vector.log_component_normalizers[0])-scalar.log_normalizer) < 1e-12
    assert vector.potential_evaluations == scalar.potential_evaluations


def test_constant_family_normalizers_do_not_become_self_normalized_ratios():
    cfg = WeightedSMCConfig(32, (0., .1, .5, 1.), 4, .35, 1, 5750,
                            resample_every=2, resampling_scheme="stratified")
    constants = torch.tensor([-7., -9., -12.], dtype=torch.float64)
    result = estimate_umbrella_normalizers(lambda z: constants.expand(len(z), -1), dimension=8,
        log_mixture_weights=torch.tensor([.2, .3, .5], dtype=torch.float64).log(), config=cfg, use_blocks=False)
    torch.testing.assert_close(result.log_component_normalizers, constants, rtol=0, atol=1e-12)


def test_single_guided_component_matches_proposal_bridge():
    cfg = WeightedSMCConfig(128, tuple((k/15)**2 for k in range(16)), 4, .35, 1, 5790,
                            resample_every=4, resampling_scheme="stratified", independence_every=2)
    r = LocalVarianceReference(2, 1)
    def potential(z):
        return -.2*(z-1).square().sum(1)
    vector = estimate_umbrella_normalizers(lambda z: potential(z)[:, None], dimension=2,
        log_mixture_weights=torch.zeros(1, dtype=torch.float64), config=cfg, use_blocks=False, proposal=r)
    scalar = estimate_proposal_bridge(potential, proposal=r, config=cfg, base="proposal")
    assert abs(vector.log_normalizer-scalar.log_normalizer) < 1e-12
    assert abs(float(vector.log_component_normalizers[0])-scalar.log_normalizer) < 1e-12
    assert vector.potential_evaluations == scalar.potential_evaluations


def test_mixture_weights_are_frozen_against_callback_side_effects():
    cfg = WeightedSMCConfig(16, (0., .5, 1.), 2, .35, 1, 5800)
    mixture = torch.tensor([.3, .7], dtype=torch.float64).log()
    constants = torch.tensor([-7., -9.], dtype=torch.float64)
    def components(z):
        mixture.add_(.1)
        return constants.expand(len(z), -1)
    out = estimate_umbrella_normalizers(components, dimension=2, log_mixture_weights=mixture,
                                       config=cfg, use_blocks=False)
    torch.testing.assert_close(out.log_component_normalizers, constants, rtol=0, atol=1e-12)
