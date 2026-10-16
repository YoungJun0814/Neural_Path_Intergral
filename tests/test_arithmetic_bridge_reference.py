import math

import pytest
import torch

from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.proposal_bridge_smc import (
    arithmetic_bridge_log_density,
    estimate_proposal_bridge,
)
from src.path_integral.structural_v2_reference import log_moments
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig
from tests.test_directional_pcn_reference import geometry


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("base", ["natural", "proposal"])
def test_arithmetic_bridge_density_and_telescoping(base):
    h = torch.tensor([-100., -10., -1.], dtype=torch.float64)
    d = torch.tensor([-.8, 3., 5.], dtype=torch.float64)
    c = .0001
    temperatures = (0., .2, .5, .8, 1.)
    fs = [arithmetic_bridge_log_density(t, h, d, base=base, normalizer_scale=c) for t in temperatures]
    for t, f in zip(temperatures, fs, strict=True):
        direct = (1-t)*(d.exp() if base == "proposal" else torch.ones_like(d))+t*h.exp()/c
        torch.testing.assert_close(f, direct.log(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(sum(b-a for a, b in zip(fs[:-1], fs[1:], strict=True)),
                               h-math.log(c)-(d if base == "proposal" else 0))


@pytest.mark.parametrize("base", ["natural", "proposal"])
@pytest.mark.parametrize("scale_factor", [.1, 10.])
def test_arithmetic_bridge_normalizer_oracle_with_wrong_scale(base, scale_factor):
    r = geometry()
    oracle = GaussianBumpOracle((.5, .5), ((3., 0., 3., 0.), (-3., 0., -3., 0.)), (1., 1.))
    c = math.exp(oracle.log_endpoint(1))*scale_factor
    logs = []
    for rep in range(12):
        cfg = WeightedSMCConfig(1024, tuple(k/31 for k in range(32)), 4, .35, 1, 920+rep,
                               resample_every=4, resampling_scheme="stratified", independence_every=2)
        out = estimate_proposal_bridge(oracle.log_value, proposal=r, config=cfg, base=base,
                                       bridge="arithmetic", normalizer_scale=c,
                                       directional_units=r.units, directional_price_indices=r.price_indices)
        logs.append(out.log_normalizer)
        assert out.potential_evaluations == 1024*(1+30*4)
    summary = log_moments(torch.tensor(logs, dtype=torch.float64))
    assert abs(math.exp(summary["log_mean"]-oracle.log_endpoint(1))-1) < max(.1, 4*summary["relative_se"])
