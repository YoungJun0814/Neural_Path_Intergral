import math

import pytest
import torch

import experiments.post_audit_small_grid_model_restart as runner
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.risk_aware_model_correction import (
    full_mixture_log_q_over_p,
    shift_spec_parameters,
)
from src.path_integral.structural_v2_reference import log_moments
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def natural(d=64):
    return GaussianMixtureShiftSpec(torch.zeros((1, d), dtype=torch.float64), torch.ones(1, dtype=torch.float64))


def test_fixed_work_complete_family_and_seed_disjointness():
    assert len(runner.fit_jobs()) == 10
    assert len(runner.final_jobs()) == 320
    assert len({runner.final_id(j) for j in runner.final_jobs()}) == 320
    assert runner.PARENT_WORK == 3*256*(1+47*2) == 72960
    assert runner.FIT_WORK == runner.PARENT_WORK+8192+16384
    assert 10*(runner.FIT_WORK+4*8*32768+131072)+512 == runner.TOTAL_WORK
    streams = []
    for j in runner.fit_jobs():
        streams += [runner.seed(j["cell"], j["rep"], "parent-island", i) for i in range(3)]
        for name, count in (("train", 8192), ("selection", 16384), ("heldout-risk", 131072)):
            streams += [runner.seed(j["cell"], j["rep"], name, k, r) for k in range(count//2048) for r in ("path", "label")]
        for arm in runner.ARMS:
            streams += [runner.seed(j["cell"], j["rep"], "final-"+arm, b*100+k, r)
                        for b in range(8) for k in range(16) for r in ("path", "label")]
    assert len(streams) == len(set(streams))
    manifest = runner.manifest_for(runner.configuration(), {"snapshot_sha256": "test"})
    assert len(manifest["expected_units"]) == len(set(manifest["expected_units"])) == 343


def test_alpha_tie_selects_exact_baseline_not_new_model():
    q = natural(1)
    z = torch.linspace(-1, 1, 20, dtype=torch.float64)[:, None]
    g, d = -.5*z[:, 0].square(), torch.zeros(20, dtype=torch.float64)
    selected, details = runner.choose_alpha(z, g, d, q, q)
    assert details["alpha"] == 0. and details["no_op"] is True
    assert selected is q


def test_independent_iid_bank_replay_and_exact_final_moments(monkeypatch):
    cache, q = CachedVolterraTerminal.create(problem(32)), natural()
    j = {"cell": "oracle", "rep": 0, "arm": "parent", "block": 0}
    monkeypatch.setattr(runner, "FINAL_BLOCK", 2048)
    fit = {"parameters": {"parent": shift_spec_parameters(q)}, "proposal_digests": {"parent": "bound"}}
    first = runner.final_values(j, (cache, q), fit)
    second = runner.final_values(j, (cache, q), fit)
    assert first["moments"] == second["moments"]
    assert first["sample_digest"] == second["sample_digest"]
    z, g, d = runner.iid_bank(cache, q, j, "final-parent", 2048)
    assert first["moments"] == log_moments(g-d)
    torch.testing.assert_close(d, full_mixture_log_q_over_p(z, q))
    assert runner.stream_digest(j, "train", 8192) != runner.stream_digest(j, "selection", 8192)


def test_costs_do_not_charge_static_for_parent_or_hide_correction_stages():
    fit = {"cost": {"parent_smc": 2., "parent_fit": 3., "train_bank": 5.,
        "correction_fit": {"event": 7., "risk": 11.}, "selection_bank": 13.,
        "selection_density": {"event": 17., "risk": 19.}}}
    setup = {"simulator_setup": 23., "geometry_setup": 29.}
    parent = runner.workflow_cost("parent", fit, setup, 31.)
    static = runner.workflow_cost("static", fit, setup, 31.)
    event = runner.workflow_cost("event", fit, setup, 31.)
    risk = runner.workflow_cost("risk", fit, setup, 31.)
    assert parent["total_fixed_final_budget_wall"] == 23+2+3+31
    assert static["total_fixed_final_budget_wall"] == 23+29+31
    assert static["parent_training_and_fit"] == 0
    assert event["total_fixed_final_budget_wall"] == 23+29+2+3+5+7+13+17+31
    assert risk["total_fixed_final_budget_wall"] == 23+29+2+3+5+11+13+19+31


def test_complete_family_required_and_noop_never_success_even_with_precise_final():
    moment = log_moments(torch.full((32768,), math.log(.23), dtype=torch.float64))
    cost = {"parent_smc": 1., "parent_fit": 1., "train_bank": 1.,
        "correction_fit": {"event": 1., "risk": 1.}, "selection_bank": 1.,
        "selection_density": {"event": 1., "risk": 1.}}
    fits = [{**j, "proposal_digests": {a: "same" for a in runner.ARMS},
        "component_counts": {a: 3 for a in runner.ARMS}, "selections": {a: {"alpha": 0.} for a in ("event", "risk")},
        "cost": cost} for j in runner.fit_jobs()]
    finals = [{**j, "moments": moment, "wall": 1.} for j in runner.final_jobs()]
    risks = [{**j, "contrasts": [{"baseline": "parent", "candidate": a, "development_reduction_signal": False}
                for a in runner.ARMS[1:]]} for j in runner.fit_jobs()]
    setup = {c: {"simulator_setup": 1., "geometry_setup": 1.} for c in runner.CELLS}
    result = runner.summarize(fits, finals, risks, setup)
    assert result["development_candidate_arms"] == []
    assert result["p1_qualified"] is result["p2_qualified"] is result["large_training_authorized"] is False
    assert all(m["development_precision_passes"] == 5 for c in result["cells"] for m in c["methods"])
    with pytest.raises(ValueError, match="complete"):
        runner.summarize(fits, finals[:-1], risks, setup)
    with pytest.raises(ValueError, match="complete"):
        runner.summarize(fits, finals, risks[:-1], setup)


def test_heldout_is_paired_fresh_bank_and_exact_noop_does_not_win(monkeypatch):
    monkeypatch.setattr(runner, "HELDOUT", 2048)
    cache, q = CachedVolterraTerminal.create(problem(32)), natural()
    j = {"cell": "oracle", "rep": 0}
    parameters = shift_spec_parameters(q)
    fit = {"parameters": {a: parameters for a in runner.ARMS},
           "proposal_digests": {a: "same" for a in runner.ARMS}}
    result = runner.heldout_values(j, (cache, q), fit)
    assert len(result["contrasts"]) == 6
    assert all(c["same_proposal"] and not c["development_reduction_signal"] for c in result["contrasts"])
    assert all(c["raw"]["same_observed_values"] for c in result["contrasts"])
    assert result["risk_moments"]["parent"] == result["risk_moments"]["risk"]
    assert result["seed_digest"] != runner.stream_digest(j, "train", 2048)


def test_parent_training_uses_fresh_islands_and_expected_work():
    cache, q = CachedVolterraTerminal.create(problem(32)), natural()
    result = runner.fit_values({"cell": "oracle", "rep": 0}, (cache, q))
    assert result["actual_work"] == runner.FIT_WORK
    assert len(result["parent_smc_diagnostics"]) == 3
    assert set(result["parameters"]) == set(runner.ARMS)
    assert all(result["component_counts"][a] <= 6 for a in ("event", "risk"))
    assert result["train_seed_digest"] != result["selection_seed_digest"]
    assert all(result["selections"][a]["alpha"] in (0., .25, .5) for a in ("event", "risk"))
