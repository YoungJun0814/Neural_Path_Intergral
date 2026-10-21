"""Finite-dimensional density, fitting-target, and replay oracles for q correction."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    sample_gaussian_mixture,
)
from src.path_integral.risk_aware_model_correction import (
    bank_weight_diagnostics,
    blend_model_correction,
    fit_model_correction,
    fit_model_corrections,
    full_mixture_log_q_over_p,
    log_second_moment_bank,
    shift_spec_from_parameters,
    shift_spec_parameters,
)


def _spec(means: list[list[float]], weights: list[float]) -> GaussianMixtureShiftSpec:
    return GaussianMixtureShiftSpec(
        torch.tensor(means, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
    )


def _parent() -> GaussianMixtureShiftSpec:
    return _spec([[0.0], [-1.0], [2.0]], [0.1, 0.7, 0.2])


def _bank() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, GaussianMixtureShiftSpec]:
    x = torch.linspace(-3.0, 3.0, 151, dtype=torch.float64)[:, None]
    guide = _spec([[0.0], [-2.0], [-1.0], [1.0], [3.0]], [.1, .2, .2, .2, .3])
    log_g = -0.5 * ((x[:, 0] - 0.8) / 0.7).square()
    return x, log_g, full_mixture_log_q_over_p(x, guide), _parent()


def test_full_density_matches_independent_normal_densities_for_many_components() -> None:
    means = [[0.0, 0.0], [-2.0, 1.0], [0.5, 3.0], [2.0, -1.0], [4.0, 2.0]]
    spec = _spec(means, [0.1, 0.2, 0.15, 0.3, 0.25])
    x = torch.tensor([[0.3, -0.8], [4.2, 1.6], [-3.0, 2.0]], dtype=torch.float64)
    reference = torch.distributions.Normal(torch.zeros(2, dtype=torch.float64), 1.0)
    log_p = reference.log_prob(x).sum(dim=1)
    actual_components = torch.stack([
        torch.distributions.Normal(mean, 1.0).log_prob(x).sum(dim=1)
        for mean in spec.means
    ], dim=1)
    expected = torch.logsumexp(actual_components + spec.weights.log(), dim=1) - log_p
    actual = full_mixture_log_q_over_p(x, spec)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    # A single selected component is not a valid mixture denominator.
    assert not torch.allclose(actual, actual_components[:, 2] - log_p)


def test_event_and_risk_targets_are_distinct_and_use_frozen_full_q0() -> None:
    x, log_g, log_guide, q0 = _bank()
    original_means, original_weights = q0.means.clone(), q0.weights.clone()
    fitted = fit_model_corrections(x, log_g, log_guide, q0, clusters=1)
    log_q0 = full_mixture_log_q_over_p(x, q0)
    event_terms = log_g - log_guide
    risk_terms = 2.0 * log_g - log_q0 - log_guide
    for fit, terms in ((fitted.event, event_terms), (fitted.risk, risk_terms)):
        expected_mean = terms.softmax(0) @ x
        torch.testing.assert_close(fit.proposal.means[1], expected_mean,
                                   rtol=1e-13, atol=1e-13)
        assert float(fit.proposal.weights[0]) == 0.1
        torch.testing.assert_close(fit.proposal.means[0], torch.zeros(1, dtype=torch.float64))
        assert fit.proposal.components == 2
        assert fit.diagnostics.log_mass == pytest.approx(
            float(torch.logsumexp(terms, 0)) - math.log(len(x)), abs=1e-13,
        )
        assert fit.diagnostics.ess == pytest.approx(1.0 / float(terms.softmax(0).square().sum()))
        assert fit.diagnostics.max_fraction == pytest.approx(float(terms.softmax(0).max()))
        assert fit.diagnostics.bank_count == len(x)
    assert abs(float(fitted.event.proposal.means[1] - fitted.risk.proposal.means[1])) > 0.05
    torch.testing.assert_close(q0.means, original_means, rtol=0, atol=0)
    torch.testing.assert_close(q0.weights, original_weights, rtol=0, atol=0)


def test_single_arm_entry_point_matches_joint_fit_and_common_rescaling() -> None:
    x, log_g, log_guide, q0 = _bank()
    joint = fit_model_corrections(x, log_g, log_guide, q0, clusters=1)
    for kind, expected, multiplier in (("event", joint.event, 1), ("risk", joint.risk, 2)):
        single = fit_model_correction(x, log_g, log_guide, q0, kind=kind, clusters=1)
        shifted = fit_model_correction(x, log_g - 900.0, log_guide, q0,
                                       kind=kind, clusters=1)
        torch.testing.assert_close(single.proposal.means, expected.proposal.means,
                                   rtol=0, atol=0)
        torch.testing.assert_close(single.proposal.weights, expected.proposal.weights,
                                   rtol=0, atol=0)
        torch.testing.assert_close(shifted.proposal.means, single.proposal.means,
                                   rtol=1e-11, atol=1e-12)
        assert shifted.diagnostics.log_mass == pytest.approx(
            single.diagnostics.log_mass - multiplier * 900.0, abs=1e-11,
        )
        assert shifted.diagnostics.ess == pytest.approx(single.diagnostics.ess, rel=1e-12)


def test_constant_target_has_analytic_mass_and_ordinary_not_normalized_mean() -> None:
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    natural = _spec([[0.0]], [1.0])
    log_g = torch.full((len(x),), math.log(0.3), dtype=torch.float64)
    log_guide = torch.zeros(len(x), dtype=torch.float64)
    fitted = fit_model_corrections(x, log_g, log_guide, natural, clusters=1)
    for fit, mass in ((fitted.event, 0.3), (fitted.risk, 0.09)):
        assert math.exp(fit.diagnostics.log_mass) == pytest.approx(mass)
        assert fit.diagnostics.ess == pytest.approx(len(x))
        assert fit.diagnostics.max_fraction == pytest.approx(1.0 / len(x))
        torch.testing.assert_close(fit.proposal.means, torch.zeros((2, 1), dtype=torch.float64),
                                   rtol=0, atol=1e-15)
    assert math.exp(log_second_moment_bank(x, log_g, log_guide, natural)) == pytest.approx(.09)
    # A self-normalized bank mean of its own weights would incorrectly equal 1/n.
    assert math.exp(fitted.risk.diagnostics.log_mass) != pytest.approx(1.0 / len(x))


def test_r_equals_q_bank_rows_are_exactly_squared_ordinary_is_contributions() -> None:
    x, log_g, _, q0 = _bank()
    log_r = full_mixture_log_q_over_p(x, q0)
    squared_contributions = (2.0 * (log_g - log_r)).exp()
    log_mass = log_second_moment_bank(x, log_g, log_r, q0)
    assert math.exp(log_mass) == pytest.approx(float(squared_contributions.mean()), rel=1e-13)
    fitted = fit_model_corrections(x, log_g, log_r, q0, clusters=1)
    assert fitted.risk.diagnostics.log_mass == pytest.approx(log_mass, abs=1e-13)
    # Check the same identity using normalized physical p,q,r densities.
    log_p = -0.5 * x[:, 0].square() - 0.5 * math.log(2.0 * math.pi)
    direct = (2.0 * log_g + 2.0 * log_p - (log_p + log_r) - (log_p + log_r)).exp()
    torch.testing.assert_close(direct, squared_contributions, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("alpha", [0.25, 0.5])
def test_blend_exact_full_density_and_two_alpha_population_safety_oracle(alpha: float) -> None:
    q0 = _parent()
    # Deliberately misdirected correction illustrates that the bound is not improvement.
    correction = _spec([[0.0], [7.0]], [0.1, 0.9])
    blend = blend_model_correction(q0, correction, alpha)
    nodes, weights = np.polynomial.hermite.hermgauss(160)
    x = torch.tensor(nodes * math.sqrt(2.0), dtype=torch.float64)[:, None]
    quadrature = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float64)
    d0 = full_mixture_log_q_over_p(x, q0)
    dr = full_mixture_log_q_over_p(x, correction)
    da = full_mixture_log_q_over_p(x, blend)
    expected = torch.logaddexp(d0 + math.log1p(-alpha), dr + math.log(alpha))
    torch.testing.assert_close(da, expected, rtol=1e-13, atol=1e-13)
    assert bool((da >= d0 + math.log1p(-alpha) - 2e-13).all())
    log_g = -0.5 * ((x[:, 0] + 2.0) / 0.6).square()
    m2_q0 = float((quadrature * (2.0 * log_g - d0).exp()).sum())
    m2_blend = float((quadrature * (2.0 * log_g - da).exp()).sum())
    assert m2_q0 < m2_blend <= m2_q0 / (1.0 - alpha)
    assert blend.natural_component_weight() == pytest.approx(0.1)
    assert bool((blend.weights > 0).all())


def test_alpha_zero_is_exact_baseline_without_zero_components() -> None:
    q0 = _parent()
    correction = _spec([[0.0], [1.0]], [.1, .9])
    assert blend_model_correction(q0, correction, 0.0) is q0
    assert blend_model_correction(q0, correction, 1.0) is correction
    assert q0.components == 3
    assert bool((q0.weights > 0).all())


def test_one_dimensional_full_density_probability_and_second_moment_oracles() -> None:
    spec = _spec([[0.0], [-1.2], [2.3]], [.1, .3, .6])
    nodes, weights = np.polynomial.hermite.hermgauss(120)
    normal_nodes = torch.tensor(nodes * math.sqrt(2.0), dtype=torch.float64)[:, None]
    quadrature = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float64)
    width, center = 0.8, 1.0
    probability, second_moment = 0.0, 0.0
    for mean, component_weight in zip(spec.means, spec.weights, strict=True):
        x = normal_nodes + mean
        log_g = -.5 * ((x[:, 0] - center) / width).square()
        d = full_mixture_log_q_over_p(x, spec)
        probability += float(component_weight * (quadrature * (log_g - d).exp()).sum())
        # Importance-weighted natural second moment, independent of proposal choice.
        second_moment += float(component_weight * (quadrature * (2 * log_g - d).exp()).sum())
    exact_mu = width / math.sqrt(width**2 + 1) * math.exp(-center**2 / (2 * (width**2 + 1)))
    exact_natural_m2 = width / math.sqrt(width**2 + 2) * math.exp(-center**2 / (width**2 + 2))
    assert probability == pytest.approx(exact_mu, rel=1e-9)
    assert second_moment == pytest.approx(exact_natural_m2, rel=1e-9)


def test_two_cluster_rank_zero_fit_and_json_replay_are_exactly_reproducible() -> None:
    generator = torch.Generator().manual_seed(48)
    x = torch.cat((
        -2.0 + .25 * torch.randn((80, 2), dtype=torch.float64, generator=generator),
        2.0 + .25 * torch.randn((80, 2), dtype=torch.float64, generator=generator),
    ))
    q0 = _spec([[0.0, 0.0], [-1.0, -1.0], [1.0, 1.0]], [.1, .45, .45])
    log_g = -.1 * x.square().sum(1)
    log_guide = full_mixture_log_q_over_p(x, q0)
    fitted = fit_model_corrections(x, log_g, log_guide, q0, clusters=2)
    for fit in (fitted.event, fitted.risk):
        assert fit.proposal.components == 3
        assert fit.proposal.means[1, 0] * fit.proposal.means[2, 0] < 0
        candidate = blend_model_correction(q0, fit.proposal, .5)
        saved = json.loads(json.dumps(shift_spec_parameters(candidate), allow_nan=False))
        replay = shift_spec_from_parameters(saved)
        torch.testing.assert_close(replay.means, candidate.means, rtol=0, atol=0)
        torch.testing.assert_close(replay.weights, candidate.weights, rtol=0, atol=0)
        draws = []
        for spec in (candidate, replay):
            draws.append(sample_gaussian_mixture(
                spec, 101, gaussian_generator=torch.Generator().manual_seed(741),
                label_generator=torch.Generator().manual_seed(742),
            ))
        torch.testing.assert_close(draws[0].target_coordinates, draws[1].target_coordinates,
                                   rtol=0, atol=0)
        torch.testing.assert_close(draws[0].labels, draws[1].labels, rtol=0, atol=0)
        torch.testing.assert_close(
            full_mixture_log_q_over_p(draws[0].target_coordinates, candidate),
            full_mixture_log_q_over_p(draws[1].target_coordinates, replay), rtol=0, atol=0,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["samples", "log_g", "log_guide"])
def test_nonfinite_bank_values_are_rejected_without_row_filtering(bad: float, field: str) -> None:
    x, log_g, log_guide, q0 = _bank()
    values = {"samples": x, "log_g": log_g, "log_guide": log_guide}
    values[field].view(-1)[0] = bad
    with pytest.raises(ValueError, match="finite"):
        fit_model_corrections(x, log_g, log_guide, q0, clusters=1)


def test_finite_input_overflow_is_rejected_and_log_mass_does_not_exponentiate() -> None:
    terms = torch.full((5,), 1000.0, dtype=torch.float64)
    diagnostics = bank_weight_diagnostics(terms)
    assert diagnostics.log_mass == pytest.approx(1000.0)
    assert diagnostics.ess == pytest.approx(5)
    x, log_g, log_guide, q0 = _bank()
    with pytest.raises(ValueError, match="finite"):
        fit_model_corrections(x, torch.full_like(log_g, 1e308), log_guide, q0)


@pytest.mark.parametrize("clusters", [0, 3, True, 1.0])
def test_invalid_cluster_capacity_rejected(clusters: int) -> None:
    with pytest.raises(ValueError):
        fit_model_corrections(*_bank(), clusters=clusters)


@pytest.mark.parametrize("alpha", [-.1, 1.1, math.nan, math.inf, True])
def test_invalid_blend_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        blend_model_correction(_parent(), _parent(), alpha)


def test_invalid_shapes_autograd_dtype_and_serialized_weights_are_rejected() -> None:
    x, log_g, log_guide, q0 = _bank()
    for bad_x in (x[:, 0], x.float(), x.clone().requires_grad_()):
        with pytest.raises(ValueError):
            fit_model_corrections(bad_x, log_g, log_guide, q0)
    with pytest.raises(ValueError):
        fit_model_corrections(x, log_g[:, None], log_guide, q0)
    with pytest.raises(ValueError):
        fit_model_correction(x, log_g, log_guide, q0, kind="unknown")
    with pytest.raises(ValueError):
        shift_spec_from_parameters({"means": [[0.0], [1.0]], "weights": [1.0, 0.0]})
    with pytest.raises(ValueError):
        full_mixture_log_q_over_p(x, _spec([[0.0, 0.0]], [1.0]))
