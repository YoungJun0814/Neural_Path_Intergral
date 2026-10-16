import math

import pytest
import torch

from src.path_integral.guide_free_block_smc import (
    estimate_guide_free_block_normalizer,
    multiscale_block_pcn,
)
from src.path_integral.structural_v2_reference import log_moments
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_block_pcn_preserves_gaussian_prior_and_pair_symmetry():
    generator = torch.Generator().manual_seed(4120)
    z = torch.randn((80000, 8), dtype=torch.float64, generator=generator)
    c = multiscale_block_pcn(z, scale=.7, generator=generator)
    torch.testing.assert_close(c.mean(0), torch.zeros(8, dtype=torch.float64), rtol=0, atol=.016)
    torch.testing.assert_close(torch.cov(c.T), torch.eye(8, dtype=torch.float64), rtol=0, atol=.018)
    # A Gaussian-reversible, state-independent block mixture has symmetric
    # cross covariance even though the entire joint law is not Gaussian.
    cross = z.T @ c/len(z)
    torch.testing.assert_close(cross, cross.T, rtol=0, atol=.018)
    with pytest.raises(ValueError):
        multiscale_block_pcn(torch.zeros((2, 6), dtype=torch.float64), scale=.7, generator=generator)


@pytest.mark.parametrize("blocks", [False, True])
def test_bounded_gaussian_oracle_endpoint_and_exact_work(blocks):
    center = torch.tensor([3., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)
    variance = 1.2**2
    def potential(z):
        return -.5*(z-center).square().sum(1)/variance
    exact = (variance/(1+variance))**4*math.exp(-float(center.square().sum())/(2*(1+variance)))
    logs = []
    for rep in range(12):
        cfg = WeightedSMCConfig(512, tuple((k/31)**2 for k in range(32)), 4, .35, 1, 4300+rep*103,
                                resample_every=4, resampling_scheme="stratified")
        out = estimate_guide_free_block_normalizer(potential, dimension=8, config=cfg, use_blocks=blocks)
        assert out.potential_evaluations == 512*(1+30*4)
        logs.append(out.log_normalizer)
    stats = log_moments(torch.tensor(logs, dtype=torch.float64))
    ratio = math.exp(stats["log_mean"])/exact
    assert abs(ratio-1) < max(.08, 4*ratio*stats["relative_se"])


def test_constant_normalizer_and_invalid_guided_configuration():
    cfg = WeightedSMCConfig(32, (0., .1, .5, 1.), 4, .35, 1, 4420,
                            resample_every=2, resampling_scheme="stratified")
    out = estimate_guide_free_block_normalizer(lambda z: torch.full((len(z),), -7., dtype=torch.float64),
                                              dimension=8, config=cfg, use_blocks=True)
    assert abs(out.log_normalizer+7) < 1e-12
    with pytest.raises(ValueError):
        bad = WeightedSMCConfig(32, (0., 1.), 4, .35, 1, 4420, independence_every=2)
        estimate_guide_free_block_normalizer(lambda z: torch.zeros(len(z), dtype=torch.float64),
                                             dimension=8, config=bad, use_blocks=True)


def test_full_arm_matches_existing_weighted_smc():
    cfg = WeightedSMCConfig(128, tuple((k/15)**2 for k in range(16)), 8, .35, 1, 4450,
                            resample_every=4, resampling_scheme="stratified")
    def potential(z):
        return -.2*(z-1.).square().sum(1)
    new = estimate_guide_free_block_normalizer(potential, dimension=8, config=cfg, use_blocks=False)
    old = estimate_weighted_tempered_normalizer(potential, dimension=8, config=cfg)
    assert abs(new.log_normalizer-float(old.log_replicate_estimates[0])) < 1e-12
    assert new.potential_evaluations == old.potential_evaluations
