import math

import pytest
import torch

from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.conditional_intervention import (
    conditional_component_probabilities,
    conditional_replace,
    log_half_squared_difference,
    orbit_endpoints,
    require_blind_residuals,
    sign_orbit_log_probabilities,
)
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def mixture():
    return GaussianMixtureShiftSpec(torch.tensor([[0., 0.], [2., 3.]], dtype=torch.float64),
                                    torch.tensor([.3, .7], dtype=torch.float64))


def test_conditional_posterior_ignores_replaced_coordinates_and_uses_complement():
    z = torch.tensor([[0., -50.], [0., 50.], [3., 50.]], dtype=torch.float64)
    probabilities = conditional_component_probabilities(mixture(), z, [1])
    expected = torch.softmax(torch.tensor([math.log(.3), math.log(.7)-2.], dtype=torch.float64), 0)
    torch.testing.assert_close(probabilities[0], expected)
    torch.testing.assert_close(probabilities[0], probabilities[1])
    assert probabilities[2, 1] > probabilities[1, 1]


def test_full_group_draw_uses_prior_component_weights():
    z = torch.zeros((2, 2), dtype=torch.float64)
    torch.testing.assert_close(conditional_component_probabilities(mixture(), z, [0, 1]),
                               mixture().weights.expand(2, -1))


def test_conditional_resampling_preserves_complement_and_conditional_moments():
    z = torch.zeros((40000, 2), dtype=torch.float64)
    draw = conditional_replace(mixture(), z, [1], label_generator=torch.Generator().manual_seed(14),
                               gaussian_generator=torch.Generator().manual_seed(15))
    torch.testing.assert_close(draw[:, 0], z[:, 0], rtol=0, atol=0)
    p = float(conditional_component_probabilities(mixture(), z[:1], [1])[0, 1])
    assert abs(float(draw[:, 1].mean())-3*p) < .03
    assert abs(float(draw[:, 1].square().mean())-(1+9*p)) < .08


@pytest.mark.parametrize("group", [[], [2], [-1], [1, 1], [True]])
def test_invalid_coordinate_group_is_rejected(group):
    with pytest.raises(ValueError):
        conditional_component_probabilities(mixture(), torch.zeros((2, 2), dtype=torch.float64), group)


def test_half_squared_difference_handles_equal_and_tiny_logs_without_clipping():
    a = torch.tensor([-1000., -1000., 1.], dtype=torch.float64)
    b = torch.tensor([-1000., -1000.+1e-10, 0.], dtype=torch.float64)
    result = log_half_squared_difference(a, b)
    assert torch.isneginf(result[0]) and torch.isfinite(result[1:]).all()
    assert result[2].exp() == pytest.approx((math.e-1)**2/2)


def test_pick_freeze_known_answer_total_effect_identity():
    # Independent coordinates, f=x+2y, refresh y: E[(f-f')²/2]=4.
    generator = torch.Generator().manual_seed(32)
    z = torch.randn((100000, 3), dtype=torch.float64, generator=generator)
    difference = ((z[:, 0]+2*z[:, 1])-(z[:, 0]+2*z[:, 2])).square()/2
    assert abs(float(difference.mean())-4) < .05


@pytest.mark.parametrize("eta", [1.5, 2.])
def test_vectorized_orbit_matches_full_simulator_in_random_contexts(eta):
    from dataclasses import replace

    cache = CachedVolterraTerminal.create(replace(problem(32), eta=eta))
    spec = GaussianMixtureShiftSpec(torch.zeros((1, 64), dtype=torch.float64),
                                    torch.ones(1, dtype=torch.float64))
    z = torch.randn((19, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(83))
    orbit = sign_orbit_log_probabilities(cache, spec, z)
    for k, signs in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
        points = z.clone()
        points[:, [57, 61]] *= torch.tensor(signs, dtype=torch.float64)
        torch.testing.assert_close(orbit[:, k], cache.log_probability(points), rtol=2e-12, atol=2e-9)


def test_nonblind_mixture_cannot_use_unweighted_sign_average():
    means = torch.zeros((1, 64), dtype=torch.float64)
    means[0, 57] = .1
    with pytest.raises(ValueError):
        require_blind_residuals(GaussianMixtureShiftSpec(means, torch.ones(1, dtype=torch.float64)))


def test_orbit_jensen_order_and_p_q_denominators():
    g = torch.tensor([[.1, .2, .4, .8], [.3, .3, .3, .3]], dtype=torch.float64)
    d = torch.tensor([2., 3.], dtype=torch.float64)
    q, p = [orbit_endpoints(g.log(), d.log(), design=design) for design in ("q", "p")]
    risk = q["risk"].exp()
    assert (risk[:, 0:1] >= risk[:, 1:]-1e-15).all()
    assert (risk[:, 1:3] >= risk[:, 3:4]-1e-15).all()
    torch.testing.assert_close(p["risk"].exp(), risk*d[:, None])
    assert q["event"][0, 3].exp() == pytest.approx(float(g[0].mean()/d[0]))
    assert q["risk"][0, 0].exp() == pytest.approx(float(g[0].square().mean()/d[0]**2))
    assert q["risk"][0, 3].exp() == pytest.approx(float(g[0].mean()**2/d[0]**2))


def test_asymmetric_sign_payoff_projection_known_answer():
    # Four equiprobable signs: expectation preserved, second moment contracts.
    g = torch.tensor([[.02, .1, .3, .9]], dtype=torch.float64)
    endpoints = orbit_endpoints(g.log(), torch.zeros(1, dtype=torch.float64), design="q")
    assert endpoints["event"][0, 3].exp() == pytest.approx(float(g.mean()))
    assert endpoints["risk"][0, 0].exp() > endpoints["risk"][0, 3].exp()
