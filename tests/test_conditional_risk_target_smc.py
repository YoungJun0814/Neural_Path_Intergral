import math

import pytest
import torch

from src.path_integral.conditional_risk_target_smc import (
    STREAMS,
    RiskEvaluation,
    TargetSMCConfig,
    island_gate,
    lineage_summary,
    mode_diagnostics,
    mutation_ratio,
    normalizer_summary,
    run_target_smc,
)
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_protocol import canonical_json_bytes


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def qspec():
    return GaussianMixtureShiftSpec(torch.tensor([[0.], [2.]], dtype=torch.float64),
        torch.tensor([.2, .8], dtype=torch.float64))


def evaluation(q, x):
    d = torch.logsumexp(x @ q.means.T-.5*q.means.square().sum(1)+q.weights.log(), 1)
    return RiskEvaluation(-.4*(x[:, 0]-1.).square()-.3, d, x.clone(), 1e-12)


def seeds(offset=15000):
    return {name: offset+j for j, name in enumerate(STREAMS)}


@pytest.mark.parametrize("beta", [0., .3, 1.])
@pytest.mark.parametrize("kind", ["prior", "component", "global"])
def test_correct_density_ratio_and_wrong_single_Dq_target_are_distinguished(beta, kind):
    q = qspec()
    x, y = torch.tensor([[-1.]], dtype=torch.float64), torch.tensor([[1.5]], dtype=torch.float64)
    old, new = evaluation(q, x), evaluation(q, y)
    expected = beta*(new.log_m2-old.log_m2-2*(new.log_dq-old.log_dq))
    if kind == "prior":
        expected += new.log_dq-old.log_dq
    torch.testing.assert_close(mutation_ratio(kind, beta, old, new), expected)
    if beta > 0:
        wrong = beta*(new.log_m2-old.log_m2-(new.log_dq-old.log_dq))
        if kind == "prior":
            wrong += new.log_dq-old.log_dq
        assert not torch.allclose(expected, wrong)


@pytest.mark.parametrize("kind", ["component", "prior", "global"])
def test_nontrivial_mixture_transition_has_pairwise_detailed_balance(kind):
    q = qspec()
    x, y = torch.tensor([[-.5]], dtype=torch.float64), torch.tensor([[1.2]], dtype=torch.float64)
    old, new = evaluation(q, x), evaluation(q, y)
    beta, scale = .6, .35
    retained = math.sqrt(1-scale**2)
    normal = torch.distributions.Normal(torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
    log_px, log_py = normal.log_prob(x[0, 0]), normal.log_prob(y[0, 0])
    log_target_x = log_px+old.log_dq[0]+beta*old.log_f[0]
    log_target_y = log_py+new.log_dq[0]+beta*new.log_f[0]

    def kernel_log(a, b, value):
        if kind == "global":
            return normal.log_prob(b[0, 0])+evaluation(q, b).log_dq[0]
        if kind == "prior":
            return torch.distributions.Normal(retained*a[0, 0], scale).log_prob(b[0, 0])
        posterior = a @ q.means.T-.5*q.means.square().sum(1)+q.weights.log()-value.log_dq[:, None]
        means = q.means[:, 0]+retained*(a[0, 0]-q.means[:, 0])
        return torch.logsumexp(posterior[0]+torch.distributions.Normal(means, scale).log_prob(b[0, 0]), 0)

    ratio = mutation_ratio(kind, beta, old, new)[0]
    forward = log_target_x+kernel_log(x, y, old)+torch.minimum(ratio, torch.zeros_like(ratio))
    backward = log_target_y+kernel_log(y, x, new)+torch.minimum(-ratio, torch.zeros_like(ratio))
    torch.testing.assert_close(forward, backward, rtol=0, atol=1e-12)


@pytest.mark.parametrize("arm", ["prior", "component"])
def test_constant_potential_known_answer_includes_final_mutation_and_does_not_reset_lineage(arm):
    q = GaussianMixtureShiftSpec(torch.zeros((1, 2), dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    visits = []

    def constant(x):
        visits.append(len(x))
        return RiskEvaluation(torch.full((len(x),), -2., dtype=torch.float64), torch.zeros(len(x), dtype=torch.float64), x, 0.)

    config = TargetSMCConfig(128, (0., .25, .5, .75, 1.), arm, resample_every=1)
    result = run_target_smc(q, constant, config=config, seeds=seeds())
    assert result["log_normalizer"] == pytest.approx(-2., abs=1e-12)
    assert len(visits) == 1+4*3 and result["potential_contexts"] == 128*13
    assert result["stages"][-1]["proposed"]["global"] == 128
    assert not result["stages"][-1]["resampled"]
    ancestors = [s["post_mutation"]["initial_ancestors"] for s in result["stages"]]
    assert all(b <= a for a, b in zip([128]+ancestors[:-1], ancestors, strict=True))
    assert result["diagnostics"]["initial_ancestors"] == 128
    assert canonical_json_bytes(result)


@pytest.mark.parametrize("arm", ["prior", "component"])
def test_gaussian_bump_normalizer_and_target_moment_known_answers(arm):
    q = GaussianMixtureShiftSpec(torch.zeros((1, 1), dtype=torch.float64), torch.ones(1, dtype=torch.float64))

    def bump(x):
        return RiskEvaluation(-.5*(x[:, 0]-1.).square(), torch.zeros(len(x), dtype=torch.float64), x.clone(), 0.)

    config = TargetSMCConfig(512, tuple(j/8 for j in range(9)), arm, resample_every=2)
    values, means = [], []
    for rep in range(4):
        result = run_target_smc(q, bump, config=config, seeds=seeds(16000+rep*100))
        values.append(math.exp(result["log_normalizer"]))
        means.append(float(torch.tensor(result["final_weights"], dtype=torch.float64) @ torch.tensor(result["final_x"], dtype=torch.float64)[:, 0]))
    expected = math.exp(-.25)/math.sqrt(2)
    assert sum(values)/4 == pytest.approx(expected, rel=.04)
    assert sum(means)/4 == pytest.approx(.5, abs=.08)


def test_uniform_weights_and_unique_moved_particles_do_not_hide_single_ancestor():
    x = torch.arange(128, dtype=torch.float64)[:, None]
    weights = torch.full((128,), 1/128, dtype=torch.float64)
    d = lineage_summary(torch.zeros(128, dtype=torch.int64), weights, x)
    assert d["nominal_weight_ess"] == 128 and d["distinct_particles"] == 128
    assert d["weighted_ancestor_ess"] == 1 and d["maximum_ancestor_mass"] == 1
    s = [{"accepted": {"prior": 128}, "proposed": {"prior": 128}}]*32
    assert not island_gate(d, s, torch.ones(128, dtype=torch.float64))["pass"]


def test_normalizer_rse_requires_independent_islands_and_cannot_be_replaced_by_particle_ESS():
    assert normalizer_summary([-5.]*4)["qualified"]
    assert not normalizer_summary([-5., -5., -5., 5.])["qualified"]
    with pytest.raises(ValueError):
        normalizer_summary([-5.])


def test_descriptive_feature_clusters_report_missing_island_mode():
    islands = []
    for j in range(4):
        islands.append({"final_features": [[float(j), float(j*j)]]*128, "final_weights": [1/128]*128})
    result = mode_diagnostics(islands)
    assert not result["pass"] and result["descriptive_training_bank_clustering_not_heldout_or_mode_completeness_proof"]


@pytest.mark.parametrize("bad", ["positive_m2", "nan", "error", "overflow_F"])
def test_unresolved_potential_is_rejected_without_clipping(bad):
    m2, dq = torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
    error = 0.
    if bad == "positive_m2":
        m2[0] = .1
    elif bad == "nan":
        m2[0] = math.nan
    elif bad == "overflow_F":
        dq[0] = -1e308
    else:
        error = .01
    with pytest.raises(ValueError):
        RiskEvaluation(m2, dq, torch.zeros((4, 1), dtype=torch.float64), error).validate(4)


def test_invalid_stream_reuse_and_schedule_are_rejected():
    q = qspec()
    with pytest.raises(ValueError):
        TargetSMCConfig(128, (0., .5, .5, 1.), "prior")
    with pytest.raises(ValueError, match="streams"):
        run_target_smc(q, lambda x: evaluation(q, x), config=TargetSMCConfig(128, (0., .5, 1.), "prior"),
            seeds={k: 17 for k in STREAMS})


def test_positive_logF_is_valid_when_m2_is_bounded_and_Dq_is_small():
    v = RiskEvaluation(torch.full((4,), -.1, dtype=torch.float64),
        torch.full((4,), -2., dtype=torch.float64), torch.zeros((4, 1), dtype=torch.float64), 0.)
    v.validate(4)
    assert bool((v.log_f > 0).all())
