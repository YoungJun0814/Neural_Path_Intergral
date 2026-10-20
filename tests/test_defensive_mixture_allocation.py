import math

import numpy as np
import pytest
import torch

from src.path_integral.defensive_mixture_allocation import (
    allocation_weights,
    fit_mixture_allocation,
    log_second_moment_objective,
)


def bank():
    x = torch.linspace(-5, 5, 501, dtype=torch.float64)
    means = torch.tensor([0., -2., 2.], dtype=torch.float64)
    components = x[:, None]*means-.5*means.square()
    sampling = torch.logsumexp(components+torch.tensor([.1, .45, .45]).log(), 1)
    payoff = -.5*((x-2)/.6).square()
    return components, payoff, sampling


def test_convex_fit_preserves_floors_and_reduces_empirical_second_moment():
    c, g, d = bank()
    fitted = fit_mixture_allocation(c, g, d)
    assert fitted.weights[0] == .1
    assert (fitted.weights[1:] >= .05).all()
    assert float(fitted.weights.sum()) == pytest.approx(1)
    assert fitted.final_log_objective < fitted.initial_log_objective-.2
    assert all(b <= a+1e-11 for a, b in zip(fitted.history, fitted.history[1:], strict=False))
    assert fitted.weights[2] > fitted.weights[1]
    assert fitted.relative_empirical_fw_gap < 1e-6


def test_objective_is_ordinary_mean_convex_and_shift_stable():
    c, g, d = bank()
    a = allocation_weights(torch.tensor([.9, .1], dtype=torch.float64))
    b = allocation_weights(torch.tensor([.1, .9], dtype=torch.float64))
    fa, fb, fm = [math.exp(log_second_moment_objective(c, g, d, w)) for w in (a, b, .3*a+.7*b)]
    assert fm <= .3*fa+.7*fb
    direct = float((2*g-d-torch.logsumexp(c+a.log(), 1)).exp().mean())
    assert fa == pytest.approx(direct, rel=1e-12)
    assert log_second_moment_objective(c, g-700, d, a) == pytest.approx(math.log(fa)-1400)


def test_independent_gauss_hermite_probability_oracle_uses_full_mixture_density():
    means = torch.tensor([0., -2., 2.], dtype=torch.float64)
    weights = allocation_weights(torch.tensor([.1, .9], dtype=torch.float64))
    nodes, qw = np.polynomial.hermite.hermgauss(120)
    nodes = torch.tensor(nodes*math.sqrt(2), dtype=torch.float64)
    qw = torch.tensor(qw/math.sqrt(math.pi), dtype=torch.float64)
    estimate = 0.
    for j in range(3):
        z = nodes+means[j]
        log_g = -.5*((z-1)/.8).square()
        log_d = torch.logsumexp(z[:, None]*means-.5*means.square()+weights.log(), 1)
        estimate += float(weights[j]*(qw*(log_g-log_d).exp()).sum())
    exact = .8/math.sqrt(.8**2+1)*math.exp(-1/(2*(.8**2+1)))
    assert estimate == pytest.approx(exact, rel=1e-8)


@pytest.mark.parametrize("bad", [0, True, -1])
def test_invalid_fit_cap_rejected(bad):
    with pytest.raises(ValueError):
        fit_mixture_allocation(*bank(), maximum_iterations=bad)


def test_autograd_and_missing_natural_component_rejected():
    c, g, d = bank()
    with pytest.raises(ValueError):
        fit_mixture_allocation(c.requires_grad_(True), g, d)
    with pytest.raises(ValueError):
        fit_mixture_allocation(c.detach()+1, g, d)


def test_relative_frank_wolfe_gradient_matches_independent_finite_difference():
    c, g, d = bank()
    v = torch.tensor([.35, .65], dtype=torch.float64)
    w = allocation_weights(v)
    log_d = torch.logsumexp(c+w.log(), 1)
    terms = 2*g-d-log_d
    # Derivative of log F along the feasible direction (1,-1).
    theoretical = -.8*float((terms.softmax(0)*((c[:, 1]-log_d).exp()-(c[:, 2]-log_d).exp())).sum())
    direction = torch.tensor([1., -1.], dtype=torch.float64)
    eps = 1e-5
    values = [log_second_moment_objective(c, g, d, allocation_weights(v+s*eps*direction)) for s in (-1, 1)]
    assert (values[1]-values[0])/(2*eps) == pytest.approx(theoretical, rel=1e-8, abs=1e-10)


def test_bank_payoff_common_rescaling_does_not_change_learned_allocation():
    c, g, d = bank()
    a, b = fit_mixture_allocation(c, g, d), fit_mixture_allocation(c, g-800., d)
    torch.testing.assert_close(a.weights, b.weights, rtol=1e-10, atol=1e-12)
    assert b.final_log_objective == pytest.approx(a.final_log_objective-1600.)
