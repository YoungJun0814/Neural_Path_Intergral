import math

import pytest
import torch

from src.path_integral.collective_variance_reference import CollectiveVarianceReference
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.proposal_bridge_smc import directional_pcn, estimate_proposal_bridge
from src.path_integral.structural_v2_reference import log_moments
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def geometry():
    return CollectiveVarianceReference(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=torch.float64),
                                       torch.tensor([2, 3]))


def test_directional_prior_stationarity_and_covariance():
    r = geometry()
    generator = torch.Generator().manual_seed(851)
    z = torch.randn((131072, 4), dtype=torch.float64, generator=generator)
    new = directional_pcn(z, r.units, r.price_indices, scale=.35, generator=generator)
    assert float(new.mean(0).abs().max()) < .012
    torch.testing.assert_close(new.T @ new/len(z), torch.eye(4, dtype=torch.float64), rtol=0, atol=.015)
    retained = math.sqrt(1-.35**2)
    torch.testing.assert_close((new*z).mean(0), torch.full((4,), (1+retained)/2, dtype=torch.float64), rtol=0, atol=.015)


@pytest.mark.parametrize("base", ["natural", "proposal"])
def test_directional_normalizer_known_oracle(base):
    r = geometry()
    oracle = GaussianBumpOracle((.5, .5), ((2., 0., 2., 0.), (-2., 0., -2., 0.)), (1., 1.))
    logs = []
    for rep in range(12):
        cfg = WeightedSMCConfig(1024, tuple((k/31)**2 for k in range(32)), 4, .35, 1, 870+rep,
                                resample_every=4, resampling_scheme="stratified", independence_every=2)
        result = estimate_proposal_bridge(oracle.log_value, proposal=r, config=cfg, base=base,
                                          directional_units=r.units, directional_price_indices=r.price_indices)
        logs.append(result.log_normalizer)
        assert result.potential_evaluations == 1024*(1+30*4)
    summary = log_moments(torch.tensor(logs, dtype=torch.float64))
    assert abs(math.exp(summary["log_mean"])/math.exp(oracle.log_endpoint(1))-1) < max(.1, 4*summary["relative_se"])
