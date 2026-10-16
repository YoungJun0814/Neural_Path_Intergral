import math
from dataclasses import replace

import pytest
import torch

from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.adjacent_latent_adapter import AdjacentLatentAdapter
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.collective_variance_reference import CollectiveVarianceReference
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.local_variance_reference import LocalVarianceReference
from src.path_integral.proposal_bridge_smc import (
    bridge_mutation_log_ratio,
    estimate_proposal_bridge,
)
from src.path_integral.rbergomi_fft import (
    AdjacentRBergomiFFTInnovations,
    simulate_coupled_rbergomi_adjacent_fft,
)
from src.path_integral.structural_v2_reference import log_moments
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig


def problem(steps=16):
    return _problem({"spot": 100., "maturity": 1., "hurst": .05, "eta": 1.5, "xi": .04, "rho": -.7},
                    {"id": "test", "threshold": 1.}, steps)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_collective_density_and_sampling_match_full_gaussian_formula():
    r = CollectiveVarianceReference.create(problem(8))
    q = r.as_gaussian_mixture()
    z = r.sample(30000, path_seed=123, label_seed=124)
    assert torch.max(torch.abs(r.log_q_over_p(z)-q.log_q_over_p(z))) < 3e-12
    expected = torch.eye(r.dimension, dtype=torch.float64)
    for u, j in zip(r.units, r.price_indices, strict=True):
        v = torch.zeros(r.dimension, dtype=torch.float64)
        v[j] = 1
        expected += .9/len(r.units)*24*(torch.outer(u, u)+torch.outer(v, v))
    assert torch.max(torch.abs(torch.cov(z.T)-expected)) < .25
    assert torch.max(torch.abs(z.mean(0))) < .06
    with pytest.raises(ValueError):
        replace(r, units=torch.ones_like(r.units))
    with pytest.raises(ValueError):
        r.sample(1, path_seed=1, label_seed=1)


@pytest.mark.parametrize("base", ["natural", "proposal"])
def test_proposal_bridge_known_shifted_bimodal_endpoint(base):
    oracle = GaussianBumpOracle((.000001, .999999), ((0., 0.), (6., 0.)), (.25, .5))
    r = LocalVarianceReference(2, 1)
    logs = []
    for rep in range(16):
        cfg = WeightedSMCConfig(1024, tuple((i/31)**2 for i in range(32)), 4, .35, 1,
                                8765+rep*103, resample_every=4, resampling_scheme="stratified", independence_every=2)
        output = estimate_proposal_bridge(oracle.log_value, proposal=r, config=cfg, base=base)
        assert output.potential_evaluations == 1024*(1+30*4)
        logs.append(output.log_normalizer)
    summary = log_moments(torch.tensor(logs, dtype=torch.float64))
    ratio = math.exp(summary["log_mean"]-oracle.log_endpoint(1))
    assert abs(ratio-1) < max(.1, 4*ratio*summary["relative_se"])


@pytest.mark.parametrize("base", ["natural", "proposal"])
@pytest.mark.parametrize("global_move", [False, True])
def test_bridge_mh_ratio_matches_direct_target_and_proposal_density(base, global_move):
    beta = .37
    h0, h1, d0, d1 = [torch.tensor([x], dtype=torch.float64) for x in (-3., -1., .8, .2)]
    target_difference = beta*(h1-h0)+((1-beta) if base == "proposal" else 0)*(d1-d0)
    expected = target_difference-(d1-d0) if global_move else target_difference
    actual = bridge_mutation_log_ratio(beta, h0, h1, d0, d1, base=base, global_move=global_move)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("steps", [16, 32, 64])
def test_adjacent_adapter_covariance_and_original_fft_path_identity(steps):
    p = problem(steps)
    adapter = AdjacentLatentAdapter.create(p)
    covariance = adapter.fine_matrix @ adapter.fine_matrix.T+adapter.augmentation_matrix @ adapter.augmentation_matrix.T
    torch.testing.assert_close(covariance, torch.eye(steps, dtype=torch.float64), rtol=0, atol=2e-12)
    generator = torch.Generator().manual_seed(415)
    fine = torch.randn((200, steps, 2), dtype=torch.float64, generator=generator)
    augmentation = torch.randn((200, steps//2, 1), dtype=torch.float64, generator=generator)
    coarse_z = adapter.coarse(fine.reshape(200, -1), augmentation.squeeze(2))
    paths = simulate_coupled_rbergomi_adjacent_fft(p.simulator(), S0=p.spot, T=p.maturity,
        fine_steps=steps, num_paths=200, innovations=AdjacentRBergomiFFTInnovations(
            torch.cat((fine[:, 0::2, :], augmentation), 2), fine[:, 1::2, :], torch.zeros((200, steps), dtype=torch.float64)))
    for level, prob, z in ((paths.fine, p, fine.reshape(200, -1)),
                          (paths.coarse, replace(p, steps=steps//2), coarse_z)):
        cache = CachedVolterraTerminal.create(prob)
        terminal, integrated, variance = cache.quantities(z)
        torch.testing.assert_close(terminal, level.log_spot[:, -1], rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(variance, level.variance, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(integrated, level.step_dt*level.variance[:, :-1].sum(1), rtol=2e-12, atol=2e-12)


def test_adjacent_augmented_coarse_natural_and_shift_marginal():
    p = problem(16)
    adapter = AdjacentLatentAdapter.create(p)
    generator = torch.Generator().manual_seed(487)
    z = torch.randn((60000, 32), dtype=torch.float64, generator=generator)
    eps = torch.randn((60000, 8), dtype=torch.float64, generator=generator)
    shift = torch.linspace(-.2, .2, 32, dtype=torch.float64)
    coarse = adapter.coarse(z+shift, eps)
    torch.testing.assert_close(coarse.mean(0), adapter.fine_matrix @ shift, rtol=0, atol=.014)
    torch.testing.assert_close(torch.cov(coarse.T), torch.eye(16, dtype=torch.float64), rtol=0, atol=.018)
    # Analytic exponential payoff E_p[exp(a.Z)] and its signed mesh difference.
    a = .2
    fine_f = torch.exp(a*(z[:, 0]+shift[0]))
    coarse_f = torch.exp(a*coarse[:, 0])
    likelihood = torch.exp(-z @ shift-.5*shift.square().sum())
    assert abs(float(((fine_f-coarse_f)*likelihood).mean())) < .005
    assert abs(float((coarse_f*likelihood).mean())-math.exp(.5*a*a)) < .005
