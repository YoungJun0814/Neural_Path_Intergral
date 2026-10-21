import copy
import math

import pytest
import torch

import experiments.post_audit_dictionary_model_learning as runner
import experiments.post_audit_small_grid_model_restart as prior
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.physical_dictionary_learning import DictionaryWeightFit
from src.path_integral.risk_aware_model_correction import full_mixture_log_q_over_p
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


def test_fixed_complete_family_work_and_disjoint_random_roles():
    assert len(runner.jobs()) == 10
    assert len(runner.final_jobs()) == len({runner.final_id(j) for j in runner.final_jobs()}) == 240
    assert runner.FIT_WORK == runner.TRAIN+runner.SELECT == 32768
    assert 10*(runner.FIT_WORK+3*8*runner.FINAL+runner.HELDOUT)+512 == runner.TOTAL_WORK == 9503232
    streams = []
    for j in runner.jobs():
        for name, count in (("train", runner.TRAIN), ("selection", runner.SELECT), ("heldout-risk", runner.HELDOUT)):
            streams += [runner.seed(j, name, k, role) for k in range(count//runner.CHUNK) for role in ("path", "label")]
        for arm in runner.ARMS:
            streams += [runner.seed(j, "final-"+arm, 100*b+k, role)
                        for b in range(8) for k in range(runner.FINAL//runner.CHUNK) for role in ("path", "label")]
        assert runner.seed(j, "train") != prior.seed(j["cell"], j["rep"], "train")
    assert len(streams) == len(set(streams))
    manifest = runner.manifest_for({}, {"snapshot_sha256": "test"})
    assert len(manifest["expected_units"]) == len(set(manifest["expected_units"])) == 263
    assert manifest["limits"]["D"] == {"work": 9503232, "wall": 7200.}


def test_identical_weights_choose_exact_uniform_without_rounding_artifacts():
    components = torch.zeros((20, 373), dtype=torch.float64)
    w = torch.cat((torch.tensor([.1], dtype=torch.float64), torch.full((372,), .9/372, dtype=torch.float64)))
    g, d = torch.linspace(-9, -1, 20, dtype=torch.float64), torch.zeros(20, dtype=torch.float64)
    selected, details = runner.select_weights(components, g, d, w, w.clone())
    assert selected is w
    assert details["alpha"] == 0 and details["no_op"]
    assert len(set(details["selection_log_m2"])) == 1


def test_selection_uses_ordinary_paired_event_second_moment():
    components = torch.tensor([[0., 2.], [0., -1.], [0., .5]], dtype=torch.float64)
    w, fit = torch.tensor([.1, .9], dtype=torch.float64), torch.tensor([.7, .3], dtype=torch.float64)
    g, d = torch.tensor([-4., -1., -3.], dtype=torch.float64), torch.tensor([.2, .3, .4], dtype=torch.float64)
    selected, details = runner.select_weights(components, g, d, w, fit)
    expected = []
    for alpha in runner.ALPHAS:
        weights = (1-alpha)*w+alpha*fit
        density = (components.exp()*weights).sum(1)
        expected.append(float(((2*g-d).exp()/density).mean().log()))
    assert details["selection_log_m2"] == pytest.approx(expected)
    index = min(range(3), key=lambda i: (expected[i], runner.ALPHAS[i]))
    torch.testing.assert_close(selected, (1-runner.ALPHAS[index])*w+runner.ALPHAS[index]*fit)


def test_independent_bank_final_replay_full_density_and_immutable_means(monkeypatch):
    cache, q = CachedVolterraTerminal.create(problem(32)), natural()
    j = {"cell": "oracle", "rep": 0, "arm": "uniform", "block": 0}
    monkeypatch.setattr(runner, "FINAL", 2048)
    fit = {"weights": {"uniform": [1.]}, "weight_digests": {"uniform": "bound"}}
    proposal = runner.proposal((cache, q), fit, "uniform")
    assert proposal.means.data_ptr() == q.means.data_ptr()
    first = runner.final_values(j, (cache, q), fit)
    second = runner.final_values(j, (cache, q), fit)
    assert first["moments"] == second["moments"]
    assert first["sample_digest"] == second["sample_digest"]
    z, g, d = runner.bank((cache, q), q, j, "final-uniform", 2048)
    assert first["moments"] == log_moments(g-d)
    torch.testing.assert_close(d, full_mixture_log_q_over_p(z, q))
    torch.testing.assert_close(runner.component_logs(z, q), torch.zeros((2048, 1), dtype=torch.float64))
    assert runner.stream_digest(j, "train", 2048) != runner.stream_digest(j, "selection", 2048)
    with pytest.raises(ValueError, match="complete fixed chunks"):
        runner.bank((cache, q), q, j, "train", 2049)


def test_workflow_cost_does_not_hide_learning_or_charge_it_to_uniform():
    fit = {"cost": {"train_bank_and_components": 2., "fit": {"event-kl": 3., "risk-m2": 5.},
        "selection_bank_and_components": 7., "selection": {"event-kl": 11., "risk-m2": 13.}}}
    setup = {"simulator_setup": 17., "geometry_setup": 19.}
    assert runner.workflow_cost("uniform", fit, setup, 23.)["total_fixed_final_budget_wall"] == 17+19+23
    assert runner.workflow_cost("event-kl", fit, setup, 23.)["total_fixed_final_budget_wall"] == 17+19+2+3+7+11+23
    assert runner.workflow_cost("risk-m2", fit, setup, 23.)["total_fixed_final_budget_wall"] == 17+19+2+5+7+13+23


def test_complete_family_required_and_noop_cannot_become_candidate():
    moment = log_moments(torch.full((32768,), math.log(.23), dtype=torch.float64))
    cost = {"train_bank_and_components": 1., "fit": {a: 1. for a in runner.ARMS[1:]},
        "selection_bank_and_components": 1., "selection": {a: 1. for a in runner.ARMS[1:]}}
    fits = [{**j, "weight_digests": {a: "same" for a in runner.ARMS},
        "selections": {a: {"alpha": 0.} for a in runner.ARMS[1:]}, "cost": cost} for j in runner.jobs()]
    finals = [{**j, "moments": moment, "wall": 1.} for j in runner.final_jobs()]
    risks = [{**j, "comparisons": [{"baseline": "uniform", "candidate": a, "development_reduction_signal": False}
                for a in runner.ARMS[1:]]} for j in runner.jobs()]
    setup = {c: {"simulator_setup": 1., "geometry_setup": 1.} for c in runner.CELLS}
    result = runner.summarize(fits, finals, risks, setup)
    assert result["development_candidate_arms"] == []
    assert result["p1_qualified"] is result["p2_qualified"] is result["large_training_authorized"] is False
    assert all(m["precision_passes"] == 5 for c in result["cells"] for m in c["methods"])
    with pytest.raises(ValueError, match="complete"):
        runner.summarize(fits, finals[:-1], risks, setup)
    with pytest.raises(ValueError, match="complete"):
        runner.summarize(fits, finals, risks[:-1], setup)
    with pytest.raises(ValueError, match="complete"):
        runner.summarize(fits[:-1], finals, risks, setup)


def test_heldout_common_paths_fresh_role_and_exact_noop(monkeypatch):
    monkeypatch.setattr(runner, "HELDOUT", 2048)
    cache, q = CachedVolterraTerminal.create(problem(32)), natural()
    j = {"cell": "oracle", "rep": 0}
    fit = {"weights": {a: [1.] for a in runner.ARMS}, "weight_digests": {a: "same" for a in runner.ARMS}}
    result = runner.risk_values(j, (cache, q), fit)
    assert len(result["comparisons"]) == 3
    assert all(c["same_proposal"] and not c["development_reduction_signal"] for c in result["comparisons"])
    assert all(c["raw"]["same_observed_values"] for c in result["comparisons"])
    assert result["risk_moments"]["uniform"] == result["risk_moments"]["risk-m2"]
    assert result["seed_digest"] != runner.stream_digest(j, "train", 2048)
    runner.validate_risk_record(result, fit)
    changed = copy.deepcopy(result)
    changed["risk_moments"]["uniform"]["count"] += 1
    with pytest.raises(ValueError, match="moment count"):
        runner.validate_risk_record(changed, fit)
    changed = copy.deepcopy(result)
    changed["block_log_means"]["uniform"] = changed["block_log_means"]["uniform"][:-1]
    with pytest.raises(ValueError, match="risk block"):
        runner.validate_risk_record(changed, fit)
    changed = copy.deepcopy(result)
    changed["comparisons"][0]["raw"]["sample_count"] -= 1
    with pytest.raises(ValueError, match="paired sample"):
        runner.validate_risk_record(changed, fit)


def test_fit_shared_bank_but_separate_selection_and_parameter_floors(monkeypatch):
    monkeypatch.setattr(runner, "TRAIN", 2048)
    monkeypatch.setattr(runner, "SELECT", 2048)
    means = torch.zeros((373, 64), dtype=torch.float64)
    weights = torch.cat((torch.tensor([.1], dtype=torch.float64), torch.full((372,), .9/372, dtype=torch.float64)))
    q = GaussianMixtureShiftSpec(means, weights)
    cache = CachedVolterraTerminal.create(problem(32))
    seen = []

    def fake_fit(c, g, d, *, objective):
        seen.append((c, g, d, objective))
        return DictionaryWeightFit(weights.clone(), objective, 0., 0., 0., 0, (0.,))

    monkeypatch.setattr(runner, "fit_dictionary_weights", fake_fit)
    result = runner.fit_values({"cell": "oracle", "rep": 0}, (cache, q))
    assert len(seen) == 2
    assert all(seen[0][i] is seen[1][i] for i in range(3))
    assert result["train_seed_digest"] != result["selection_seed_digest"]
    assert set(result["weights"]) == set(runner.ARMS)
    assert all(result["selections"][a]["no_op"] for a in runner.ARMS[1:])
    assert all(len(w) == 373 and min(w) > 0 and w[0] == pytest.approx(.1) for w in result["weights"].values())
    assert torch.equal(q.means, means)
    runner.validate_fit_record(result, q)
    changed = copy.deepcopy(result)
    changed["selections"]["event-kl"]["alpha"] = .5
    with pytest.raises(ValueError, match="binding"):
        runner.validate_fit_record(changed, q)
    changed = copy.deepcopy(result)
    changed["weights"]["risk-m2"][0] = .2
    with pytest.raises(ValueError, match="weights/floors"):
        runner.validate_fit_record(changed, q)


def test_one_shot_run_never_overwrites_or_implicitly_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_parent", lambda: {})
    monkeypatch.setattr(runner, "configuration", lambda _: {})
    directory = tmp_path/runner.DIRECTORY
    directory.mkdir(parents=True)
    (directory/"existing.start.json").touch()
    with pytest.raises(RuntimeError, match="one-shot protocol"):
        runner.run()
    assert (directory/"existing.start.json").exists()
    assert len(list(directory.iterdir())) == 1
