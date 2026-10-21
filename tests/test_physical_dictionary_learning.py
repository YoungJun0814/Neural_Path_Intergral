import math

import numpy as np
import pytest
import torch

from src.path_integral.defensive_mixture_allocation import (
    allocation_weights,
    fit_mixture_allocation,
    log_second_moment_objective,
)
from src.path_integral.physical_dictionary_learning import (
    event_kl_objective,
    event_training_diagnostics,
    fit_dictionary_weights,
)
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def bank():
    x = torch.linspace(-5, 5, 501, dtype=torch.float64)
    means = torch.tensor([0., -2., 2.], dtype=torch.float64)
    components = x[:, None]*means-.5*means.square()
    sampling = torch.logsumexp(components+allocation_weights(torch.tensor([.5, .5], dtype=torch.float64)).log(), 1)
    payoff = -.5*((x-2)/.6).square()
    return components, payoff, sampling


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
def test_fixed_dictionary_fit_preserves_domain_and_decreases_its_own_loss(objective):
    c, g, d = bank()
    before = tuple(t.clone() for t in (c, g, d))
    fit = fit_dictionary_weights(c, g, d, objective=objective)
    assert fit.objective == objective
    assert fit.weights.shape == (3,)
    assert fit.weights[0] == .1
    assert (fit.weights[1:] >= .05).all()
    assert float(fit.weights.sum()) == pytest.approx(1)
    assert fit.final_objective < fit.initial_objective-.2
    assert all(b <= a+1e-11 for a, b in zip(fit.history, fit.history[1:], strict=False))
    assert fit.weights[2] > fit.weights[1]
    assert fit.relative_empirical_gap <= 1e-6
    assert fit.iterations == len(fit.history)-1
    for actual, original in zip((c, g, d), before, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_m2_wrapper_is_exactly_the_existing_frozen_allocator():
    args = dict(maximum_iterations=23, relative_gap_tolerance=1e-10, line_search_iterations=17)
    expected = fit_mixture_allocation(*bank(), **args)
    actual = fit_dictionary_weights(*bank(), objective="risk-m2", **args)
    torch.testing.assert_close(actual.weights, expected.weights, rtol=0, atol=0)
    assert actual.initial_objective == expected.initial_log_objective
    assert actual.final_objective == expected.final_log_objective
    assert actual.relative_empirical_gap == expected.relative_empirical_fw_gap
    assert actual.iterations == expected.iterations
    assert actual.history == expected.history
    assert actual.objective_units == "log-ordinary-second-moment"
    assert actual.gap_units == "relative-ordinary-second-moment"


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
def test_frank_wolfe_direction_and_full_line_derivative_match_finite_differences(objective):
    c, g, d = bank()
    v = torch.tensor([.35, .65], dtype=torch.float64)
    w = allocation_weights(v)
    log_density = torch.logsumexp(c+w.log(), 1)
    if objective == "event-kl":
        target = (g-d).softmax(0)

        def loss(weights):
            return event_kl_objective(c, g, d, weights)
    else:
        target = (2*g-d-log_density).softmax(0)
        # Its log objective has the same normalized directional derivative at zero.

        def loss(weights):
            return log_second_moment_objective(c, g, d, weights)
    gradient = -.8*(((c[:, 1:]-log_density[:, None]).exp()).T @ target)
    vertex = torch.zeros_like(v)
    vertex[int(gradient.argmin())] = 1.
    direction = vertex-v
    gap = float(gradient @ v-gradient.min())
    eps = 1e-5
    finite_difference = (loss(allocation_weights(v+eps*direction))
                         -loss(allocation_weights(v-eps*direction)))/(2*eps)
    assert finite_difference == pytest.approx(-gap, rel=1e-8, abs=1e-10)

    vertex_density = torch.logsumexp(c+allocation_weights(vertex).log(), 1)
    delta = (vertex_density-log_density).exp()-1
    step = .37
    power = 1 if objective == "event-kl" else 2
    line_derivative = -float((target*delta/(1+step*delta).pow(power)).sum())
    if objective == "event-kl":
        def line_loss(t):
            return loss(allocation_weights(v+t*direction))
    else:
        # The frozen M2 line search differentiates F(t)/F(0), not log F(t).
        initial_log_loss = loss(w)

        def line_loss(t):
            return math.exp(loss(allocation_weights(v+t*direction))-initial_log_loss)
    numerical = (line_loss(step+eps)-line_loss(step-eps))/(2*eps)
    assert numerical == pytest.approx(line_derivative, rel=1e-8, abs=1e-10)


def test_event_loss_is_normalized_cross_entropy_and_gap_not_divided_by_negative_loss():
    c, g, d = bank()
    fit = fit_dictionary_weights(c, g, d, objective="event-kl")
    direct = -float((g-d).softmax(0) @ torch.logsumexp(c+fit.weights.log(), 1))
    assert fit.final_objective == pytest.approx(direct, abs=1e-13)
    assert fit.final_objective < 0
    assert fit.objective_units == "nats"
    assert fit.gap_units == "normalized-event-directional-gap"
    v = (fit.weights[1:]-.1/(c.shape[1]-1))/.8
    log_density = torch.logsumexp(c+fit.weights.log(), 1)
    gradient = -.8*((c[:, 1:]-log_density[:, None]).exp().T @ (g-d).softmax(0))
    gap = float(gradient @ v-gradient.min())
    assert fit.relative_empirical_gap == pytest.approx(max(0., gap), abs=1e-13)


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
def test_common_payoff_rescaling_changes_only_m2_loss_units(objective):
    c, g, d = bank()
    original = fit_dictionary_weights(c, g, d, objective=objective)
    scaled = fit_dictionary_weights(c, g-800, d, objective=objective)
    torch.testing.assert_close(original.weights, scaled.weights, rtol=1e-10, atol=1e-12)
    offset = -1600 if objective == "risk-m2" else 0
    assert scaled.final_objective == pytest.approx(original.final_objective+offset)


def test_event_diagnostics_keep_ordinary_mass_separate_from_fitting_normalization():
    _, g, d = bank()
    diagnostics = event_training_diagnostics(g, d)
    ordinary = float((g-d).exp().mean())
    target = (g-d).softmax(0)
    assert math.exp(diagnostics["ordinary_log_mean_event_mass"]) == pytest.approx(ordinary)
    assert diagnostics["effective_sample_size"] == pytest.approx(float(1/target.square().sum()))
    assert diagnostics["maximum_weight_fraction"] == pytest.approx(float(target.max()))
    assert diagnostics["samples"] == len(g)
    shifted = event_training_diagnostics(g-800, d)
    assert shifted["ordinary_log_mean_event_mass"] == pytest.approx(diagnostics["ordinary_log_mean_event_mass"]-800)
    assert shifted["effective_sample_size"] == pytest.approx(diagnostics["effective_sample_size"])


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
def test_373_component_physical_dictionary_geometry_and_uniform_floor_are_preserved(objective):
    guide = PhysicalWindowGuide.create(problem(32), kind="mixed-window",
                                       amplitudes=[2., 4., 6., 8.],
                                       price_amplitudes=[0., 2., 4.], defensive_mass=.1)
    means_before, weights_before = guide.spec.means.clone(), guide.spec.weights.clone()
    z = guide.sample(80, path_seed=7620, label_seed=7621)
    c = z @ guide.spec.means.T-.5*guide.spec.means.square().sum(1)
    d = guide.log_q_over_p(z)
    g = -.02*(z[:, 0]-2).square()
    fit = fit_dictionary_weights(c, g, d, objective=objective, maximum_iterations=8)
    assert fit.weights.shape == (373,)
    assert fit.weights[0] == .1
    assert (fit.weights[1:] >= .1/372).all()
    assert float(fit.weights.sum()) == pytest.approx(1)
    torch.testing.assert_close(guide.spec.means, means_before, rtol=0, atol=0)
    torch.testing.assert_close(guide.spec.weights, weights_before, rtol=0, atol=0)
    torch.testing.assert_close(c[:, 0], torch.zeros(len(z), dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
def test_independent_finite_gaussian_oracle_uses_full_learned_density_without_self_normalizing(objective):
    fit = fit_dictionary_weights(*bank(), objective=objective)
    means = torch.tensor([0., -2., 2.], dtype=torch.float64)
    nodes, quadrature_weights = np.polynomial.hermite.hermgauss(160)
    nodes = torch.tensor(nodes*math.sqrt(2), dtype=torch.float64)
    quadrature_weights = torch.tensor(quadrature_weights/math.sqrt(math.pi), dtype=torch.float64)
    probability = 0.
    for j in range(len(means)):
        x = nodes+means[j]
        log_g = -.5*((x-1)/.8).square()
        log_density = torch.logsumexp(x[:, None]*means-.5*means.square()+fit.weights.log(), 1)
        probability += float(fit.weights[j]*(quadrature_weights*(log_g-log_density).exp()).sum())
    analytic = .8/math.sqrt(1+.8**2)*math.exp(-1/(2*(1+.8**2)))
    assert probability == pytest.approx(analytic, rel=2e-9)


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
@pytest.mark.parametrize("bad", [0, True, -1])
def test_invalid_optimization_caps_are_rejected(objective, bad):
    with pytest.raises(ValueError):
        fit_dictionary_weights(*bank(), objective=objective, maximum_iterations=bad)
    with pytest.raises(ValueError):
        fit_dictionary_weights(*bank(), objective=objective, line_search_iterations=bad)


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
def test_only_finite_detached_cpu_float64_and_natural_first_banks_are_accepted(objective):
    c, g, d = bank()
    for args in ((c.float(), g, d), (c, g.requires_grad_(), d), (c+1, g.detach(), d),
                 (c[:1], g.detach()[:1], d[:1]), (c, g.detach(), d+math.inf)):
        with pytest.raises(ValueError):
            fit_dictionary_weights(*args, objective=objective)
    for tolerance in (0, -1, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            fit_dictionary_weights(c, g.detach(), d, objective=objective,
                                   relative_gap_tolerance=tolerance)


def test_invalid_objective_and_nonpositive_mixture_weights_are_rejected():
    with pytest.raises(ValueError):
        fit_dictionary_weights(*bank(), objective="unsupported")
    for weights in (torch.tensor([0., .5, .5], dtype=torch.float64),
                    torch.tensor([.1, .4, .4], dtype=torch.float64),
                    torch.tensor([.1, .45, .45], dtype=torch.float32)):
        with pytest.raises(ValueError):
            event_kl_objective(*bank(), weights)
