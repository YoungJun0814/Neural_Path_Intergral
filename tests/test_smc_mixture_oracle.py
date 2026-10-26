import math

import pytest
import torch

from src.path_integral.conditional_risk_target_smc import STREAMS, TargetSMCConfig, run_target_smc
from src.path_integral.reference_protocol import canonical_json_bytes
from src.path_integral.smc_mixture_oracle import (
    final_moments,
    mixture_evaluation,
    mixture_spec,
    oracle_summary,
    reference_suite,
    scalar_log_target,
)


@pytest.fixture(scope="module")
def refs():
    return reference_suite()


@pytest.mark.parametrize("beta", [0., .25, .5, .75, 1.])
def test_independent_scalar_reference_matches_actual_tensor_target(beta):
    x = torch.tensor([[-5.], [-1.], [0.], [1.], [2.], [5.]], dtype=torch.float64)
    v = mixture_evaluation(x)
    log_p = -.5*x[:, 0].square()-.5*math.log(2*math.pi)
    actual = log_p+v.log_dq+beta*v.log_f
    expected = torch.tensor([scalar_log_target(float(a), beta) for a in x[:, 0]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-13)
    assert bool((mixture_evaluation(x).log_f > 0).any())


def test_nontrivial_mixture_references_match_independent_quadrature_and_initial_analytic_law(refs):
    assert refs["pass"]
    assert refs["initial_analytic_error"] <= 1e-10
    assert max(r["maximum_independent_raw_disagreement"] for r in refs["rows"]) <= 1e-10
    assert refs["scalar_reference_evaluations"] > 0
    assert max(refs["endpoint_raw_absolute_tail_bounds_real_arithmetic"]) < 1e-28
    assert canonical_json_bytes(refs)


def test_wrong_single_Dq_endpoint_has_different_known_answer_and_fails_gate(refs):
    assert refs["wrong_closed_form_error"] <= 1e-10
    assert refs["wrong_endpoint_relative_normalizer_separation"] >= .1
    records = [{"replicate": i, "log_normalizer": math.log(refs["wrong_endpoint"]["normalizer"]),
        "moments": {k: refs["wrong_endpoint"][k] for k in ("mean", "second_moment", "right_mass")}} for i in range(16)]
    summary = oracle_summary(records, refs["rows"][-1])
    assert not summary["pass"] and not summary["checks"]["normalizer_relative_error"]


def test_correct_synthetic_known_answer_passes_without_treating_particles_as_independent(refs):
    truth = refs["rows"][-1]
    records = [{"replicate": i, "log_normalizer": math.log(truth["normalizer"]),
        "moments": {k: truth[k] for k in ("mean", "second_moment", "right_mass")}} for i in range(16)]
    assert oracle_summary(records, truth)["pass"]
    with pytest.raises(ValueError, match="sixteen"):
        oracle_summary(records[:-1], truth)
    records[-1]["replicate"] = 0
    with pytest.raises(ValueError):
        oracle_summary(records, truth)


def test_oracle_replay_binds_final_state_and_normalized_moments():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seeds = {s: 236100+j for j, s in enumerate(STREAMS)}
        config = TargetSMCConfig(32, (0., .25, .5, .75, 1.), "component", resample_every=2)
        first = run_target_smc(mixture_spec(), mixture_evaluation, config=config, seeds=seeds)
        second = run_target_smc(mixture_spec(), mixture_evaluation, config=config, seeds=seeds)
        assert first == second
        assert all(math.isfinite(v) for v in final_moments(first).values())
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("x", [torch.zeros((4, 2), dtype=torch.float64), torch.zeros((4, 1)),
    torch.tensor([[math.nan]], dtype=torch.float64)])
def test_invalid_oracle_coordinates_fail_without_clipping(x):
    with pytest.raises(ValueError):
        mixture_evaluation(x)


@pytest.mark.parametrize("beta", [-.1, 1.1, math.nan])
def test_invalid_reference_temperature_is_rejected(beta):
    with pytest.raises(ValueError):
        scalar_log_target(0., beta)
