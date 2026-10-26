import math

import numpy as np
import pytest
import torch

from src.path_integral.conditional_marginal_g import (
    ACTIVE,
    aggregate_error,
    blend,
    consistent,
    fit_marginal,
    json_compatible,
    moment_error,
    precision,
    require_factorization,
    risk_training_logs,
    select_alpha,
    selection_logs,
    training_gate,
)
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_protocol import canonical_json_bytes


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_risk_target_and_selection_target_are_distinct_ordinary_means():
    m = torch.tensor([.1, .4], dtype=torch.float64)
    q = torch.tensor([.5, 1.5], dtype=torch.float64)
    s = torch.tensor([1.2, .8], dtype=torch.float64)
    g = torch.tensor([.8, 1.2], dtype=torch.float64)
    torch.testing.assert_close(risk_training_logs(m.log(), q.log(), s.log()).exp(), m/(q*s))
    torch.testing.assert_close(selection_logs(m.log(), q.log(), s.log(), g.log()).exp(), m.square()/(q.square()*s*g))


def test_exact_finite_reference_and_candidate_have_same_mean_not_same_variance():
    p = torch.tensor([.75, .25], dtype=torch.float64)
    q = torch.tensor([.8, 1.6], dtype=torch.float64)
    m2 = torch.tensor([.01, .8], dtype=torch.float64)
    target = p @ (m2/q)
    dg = m2/q/target
    y = m2/(q*dg)
    torch.testing.assert_close(y, target.expand_as(y))
    torch.testing.assert_close((p*dg) @ y, target)
    assert float((p*dg) @ selection_logs(m2.log(), q.log(), dg.log(), dg.log()).exp()) == pytest.approx(float(target.square()))


def test_numerical_error_propagates_all_rows_including_tail_dominated_tiny_row():
    m = torch.tensor([-10., -1000.], dtype=torch.float64)
    e = torch.tensor([-40., -130.], dtype=torch.float64)
    d = torch.zeros(2, dtype=torch.float64)
    assert aggregate_error(m, e, d) < 1e-6
    assert aggregate_error(m, e, d, squared=True) < 1e-6
    assert aggregate_error(m, e+100., d) > 1e-6


def test_moment_adapter_does_not_override_structural_failure():
    from src.path_integral.analytic_residual_batch import batched_moments
    from tests.test_analytic_residual_batch import profile
    row = batched_moments([profile()])["rows"][0]
    assert moment_error(row)["structural_numerics_pass"]
    row["critical"]["near_multiple_root"] = True
    assert not moment_error(row)["structural_numerics_pass"]


@pytest.mark.parametrize("mode", ["ess", "numeric", "structural", "block"])
def test_bank_gate_cannot_be_relaxed_by_lower_observed_variance(mode):
    w = torch.zeros(2048, dtype=torch.float64)
    numerical, structural = 1e-12, True
    if mode == "ess":
        w[0] = 100
    elif mode == "numeric":
        numerical = 1e-3
    elif mode == "structural":
        structural = False
    else:
        w[:256] = 100
    assert not training_gate(w, numerical, structural)["pass"]


def test_uniform_bank_and_block_precision_known_answer_pass():
    w = torch.zeros(2048, dtype=torch.float64)
    assert training_gate(w, 1e-12, True)["pass"]
    result = precision(w, numerical_proxy=1e-12)
    assert result["qualified"] and result["raw_rse"] <= 1e-8 and result["block_rse"] <= 1e-8
    assert consistent(result, result)
    assert not consistent(result, {**result, "log_mean": 1.})


def test_single_tail_path_veto_and_alpha_tie_rule():
    w = torch.zeros(128, dtype=torch.float64)
    w[0] = 30
    assert not precision(w)["qualified"]
    assert select_alpha([1., 1.-1e-13, 2.]) == 0
    assert select_alpha([2., 1., .5]) == .5
    with pytest.raises(ValueError):
        select_alpha([1., 2., math.nan])


def test_reduced_gaussian_embedding_density_and_sampler_are_exactly_blind():
    z = torch.randn((128, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(9301))
    fitted, info = fit_marginal(z, torch.zeros(128, dtype=torch.float64))
    assert info["active_coordinates"] == list(ACTIVE)
    require_factorization(fitted)
    changed = z.clone()
    changed[:, [61, 63]] += 4
    torch.testing.assert_close(fitted.log_q_over_p(z), fitted.log_q_over_p(changed), rtol=0, atol=1e-11)
    for c in fitted.components:
        reduced = FiniteRankGaussianComponent(c.mean[list(ACTIVE)], c.directions[list(ACTIVE)], c.variance_eigenvalues)
        torch.testing.assert_close(c.log_q_over_p(z), reduced.log_q_over_p(z[:, ACTIVE]), atol=1e-11, rtol=1e-12)
        transformed = c.transform_standard_normal(z)
        assert torch.equal(transformed[:, [61, 63]], z[:, [61, 63]])
    draw = fitted.sample(128, path_seed=9302, label_seed=9303)
    torch.testing.assert_close(draw.log_q_over_p, fitted.log_q_over_p(draw.samples))
    assert float(draw.log_q_over_p.min()) >= math.log(.1)-1e-12


def test_full_covariance_density_matches_multivariate_normal_known_answer():
    mean = torch.tensor([1., -.5], dtype=torch.float64)
    direction = torch.tensor([[.6], [.8]], dtype=torch.float64)
    c = FiniteRankGaussianComponent(mean, direction, torch.tensor([2.], dtype=torch.float64))
    x = torch.tensor([[0., 1.], [2., -1.]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)+direction @ direction.T
    normal = torch.distributions.MultivariateNormal(mean, covariance)
    base = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(c.log_q_over_p(x), normal.log_prob(x)-base.log_prob(x))


def test_blend_density_includes_every_mass_and_rejects_residual_changes():
    means = torch.zeros((2, 64), dtype=torch.float64)
    means[1, 2] = 1.
    q = GaussianMixtureShiftSpec(means, torch.tensor([.1, .9], dtype=torch.float64))
    fit = DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(64),), torch.ones(1, dtype=torch.float64))
    proposal = blend(q, fit, .5)
    x = torch.zeros((4, 64), dtype=torch.float64)
    expected = torch.logaddexp(math.log(.5)+blend(q, None, 0.).log_q_over_p(x), math.log(.5)+fit.log_q_over_p(x))
    torch.testing.assert_close(proposal.log_q_over_p(x), expected)
    assert proposal.defensive_mass == pytest.approx(.55)
    means[1, 61] = 1.
    with pytest.raises(ValueError, match="residual"):
        blend(q, None, 0.)


def test_all_gate_payloads_serialize_and_nonfinite_stays_rejected():
    w = torch.zeros(128, dtype=torch.float64)
    payload = {"training": training_gate(w, 1e-12, True), "precision": precision(w), "flag": np.bool_(True)}
    assert canonical_json_bytes(json_compatible(payload))
    with pytest.raises(ValueError):
        canonical_json_bytes(json_compatible({"bad": np.float64(np.nan)}))


@pytest.mark.parametrize("operation", [risk_training_logs, selection_logs])
def test_invalid_log_vectors_are_not_silently_used(operation):
    x = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(ValueError):
        operation(*([x]*(4 if operation is selection_logs else 3)))
