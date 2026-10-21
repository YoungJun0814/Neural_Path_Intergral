import copy
import math

import pytest
import torch
import yaml

import experiments.post_audit_bounded_bank_learning as learning
import experiments.post_audit_bounded_bank_runtime as runtime
from experiments.post_audit_bounded_bank_learning import assess, comparison, risk_allocation
from experiments.post_audit_bounded_frozen_bank_contrast import (
    check_policy,
    families,
    identity,
    jobs,
    summarize,
)
from src.path_integral.balanced_mixture_bank import balanced_counts
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.stratified_paired_risk import estimate, sufficient
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def guide():
    w = torch.full((373,), .9/372, dtype=torch.float64)
    w[0] = .1
    return GaussianMixtureShiftSpec(torch.zeros((373, 64), dtype=torch.float64), w)


def test_real_conditional_bank_partial_chunks_replay_and_full_density():
    cache, q = CachedVolterraTerminal.create(problem(32)), guide()
    k = runtime.key("test", "oracle", "stratified")
    c, g, d, labels, sd, digest = runtime.evaluate(cache, q, 1240, "stratified", k)
    assert labels.bincount().tolist() == list(balanced_counts(q.weights, 1240))
    torch.testing.assert_close(d, torch.logsumexp(c+q.weights.log(), 1))
    assert torch.isfinite(g).all() and (g <= 0).all()
    second = runtime.evaluate(cache, q, 1240, "stratified", k)
    assert sd == second[4] and digest == second[5]
    assert sd != runtime.evaluate(cache, q, 1240, "iid", runtime.key("test", "oracle", "iid"))[4]
    record = runtime.risk_block((cache, q), [q.weights.tolist()]*3, "stratified", 1240, k)
    assert len(record["strata"]) == 373 and len(record["estimator"]["log_mean"]) == 4


def test_stream_collisions_and_unbalanced_blocks_are_rejected():
    q = guide()
    k = runtime.key("test", "oracle", "iid")
    with pytest.raises(ValueError, match="collision"):
        runtime.validate_streams({"oracle": (None, q)}, [(100, "iid", k), (100, "iid", k)])
    with pytest.raises(ValueError, match="eight complete"):
        runtime.summarize_blocks([], q, "iid", k, probability_first=True)


def test_frozen_family_all_ids_and_no_prefix_summary():
    q = guide()
    fits = [{"cell": c, "rep": r, "weights": {a: q.weights.tolist() for a in ("event-kl", "risk-m2")}}
        for c in ("a", "b") for r in range(5)]
    prepared = {c: (None, q) for c in ("a", "b")}
    family = families({"summary": {"fits": fits}}, prepared)
    assert all(len(f["weights"]) == 11 for f in family.values())
    assert len(jobs(("a", "b"))) == len({identity(j) for j in jobs(("a", "b"))}) == 32
    with pytest.raises(ValueError, match="complete D2"):
        summarize([], prepared, family)
    with pytest.raises(ValueError, match="all five"):
        families({"summary": {"fits": fits[:-1]}}, prepared)


def test_complete_D2_summary_balanced_sufficient_moments_and_gates():
    q = guide()
    prepared = {c: (None, q) for c in ("a", "b")}
    family = {c: {"ids": [str(i) for i in range(11)]} for c in prepared}

    def constant_stats(n):
        logs = -torch.arange(1, 13, dtype=torch.float64)
        return {"count": n, "log_sum": (logs+math.log(n)).tolist(),
            "log_cross_sum": (logs[:, None]+logs[None, :]+math.log(n)).tolist(), "log_maximum": logs.tolist()}

    counts = balanced_counts(q.weights, 29760)
    strata = [constant_stats(n) for n in counts]
    records = []
    for j in jobs(("a", "b")):
        parts = [constant_stats(29760)] if j["design"] == "iid" else strata
        record = {**j, "actual_work": 29760, "strata": parts,
            "estimator": estimate(parts, torch.ones(1, dtype=torch.float64) if j["design"] == "iid" else q.weights)}
        records.append(record)
    summary = summarize(records, prepared, family)
    assert summary["decision"]["d2_gate"]
    assert all(len(c["contrasts"]) == 12 for c in summary["cells"])


def test_all_fit_alpha_floor_and_seed_bindings():
    q = guide()
    fit = {"cell": "oracle", "rep": 0, "weights": {a: q.weights.tolist() for a in ("U", "E-IID", "E-STRAT")},
        "fits": {a: {"weights": q.weights.tolist()} for a in ("E-IID", "E-STRAT")},
        "selections": {a: {"selection_log_m2": [0., 0., 0.], "alpha": 0., "no_op": True} for a in ("E-IID", "E-STRAT")},
        "bank_diagnostics": {a: {"samples": 19840} for a in ("E-IID", "E-STRAT")},
        "train_seed_digests": {}}
    from src.path_integral.balanced_mixture_bank import stream_layout
    from src.path_integral.research_result_contract import canonical_digest
    for design, arm in (("iid", "E-IID"), ("stratified", "E-STRAT")):
        fit["train_seed_digests"][arm] = canonical_digest(stream_layout(q, 19840, design,
            runtime.key("D3", "oracle", design, role="train")))
    fit["selection_seed_digest"] = canonical_digest(stream_layout(q, 19840, "stratified",
        runtime.key("D3", "oracle", "stratified", role="selection")))
    runtime.validate_fit(fit, q)
    changed = copy.deepcopy(fit)
    changed["selections"]["E-STRAT"]["alpha"] = .5
    with pytest.raises(ValueError, match="binding"):
        runtime.validate_fit(changed, q)


def fake_risks():
    v = torch.linspace(1, 2, 200, dtype=torch.float64)
    stats = estimate([sufficient(torch.stack((v, .7*v, .4*v), 1).log())], torch.ones(1, dtype=torch.float64))
    return {d: {"summary": {"estimator": copy.deepcopy(stats), "block_estimator": copy.deepcopy(stats), "gates": [True]*3}}
        for d in ("iid", "stratified")}


def test_primary_pair_noop_and_iid_negative_veto():
    risk = fake_risks()
    assert comparison(risk, 0, 2, False)["primary_reduction"]
    assert not comparison(risk, 0, 2, True)["primary_reduction"]
    v = torch.linspace(1, 2, 200, dtype=torch.float64)
    opposite = estimate([sufficient(torch.stack((v, .7*v, 2*v), 1).log())], torch.ones(1, dtype=torch.float64))
    risk["iid"]["summary"]["block_estimator"] = opposite
    assert not comparison(risk, 0, 2, False)["primary_reduction"]


def test_shared_risk_infeasibility_never_drops_worst_q():
    pilot = {"summary": {"estimator": {"relative_se": [.1, .2, 2.]},
        "sensitivities": [{"between_unit_relative_se": .1}]*3}}
    assert risk_allocation(pilot)["status"] == "unallocated_shared_risk_group"
    pilot["summary"]["estimator"]["relative_se"] = [.1, .2, .1]
    assert risk_allocation(pilot)["count"] == 39680


def test_assess_costs_charge_learning_and_failed_probability_remains_in_denominator():
    fit = {"cell": "a", "rep": 0, "weights": {"U": [1.], "E-IID": [2.], "E-STRAT": [3.]},
        "costs": {a: {"train": 1., "fit": 2., "selection": 3.} for a in ("E-IID", "E-STRAT")}}
    pp = {"summary": {"estimator": {"relative_se": [.03]}, "sensitivities": [{"between_unit_relative_se": .03}]}, "wall": 5.}
    probabilities = {a: copy.deepcopy(pp) for a in ("U", "E-IID", "E-STRAT")}
    pilots = {"a-0"+a: {"wall": 7.} for a in probabilities}
    _, rows = assess(fit, probabilities, fake_risks(), pilots, {"a": {"simulator_setup": 11., "geometry_setup": 13.}})
    assert rows[0]["workflow_wall"] == {"U": 36., "E-IID": 42.}
    assert rows[1]["success"] and rows[1]["bank_effect"]
    del probabilities["U"]
    _, rows = assess(fit, probabilities, fake_risks(), pilots, {"a": {"simulator_setup": 11., "geometry_setup": 13.}})
    assert all(not r["success"] and not r["evaluable"] and math.isfinite(r["cost_ratio"]) for r in rows)


def test_conditional_learning_never_bypasses_failed_D2(tmp_path):
    with pytest.raises(ValueError, match="D2 gate failed"):
        learning.run(tmp_path, {}, {}, {}, {"decision": {"d2_gate": False}})
    assert not list(tmp_path.iterdir())


def test_config_and_runtime_cannot_silently_disagree():
    config = yaml.safe_load((runtime.ROOT/runtime.CONFIG).read_text())
    check_policy(config)
    config["probability_pilot"] = 16385
    with pytest.raises(ValueError, match="differs"):
        check_policy(config)


def test_conditional_learning_complete_synthetic_workflow_and_sealed_allocation(tmp_path, monkeypatch):
    # Orchestration test only: no Gaussian/Volterra payoff is generated.
    monkeypatch.setattr(learning, "validate_streams", lambda *_: 0)

    def fit(_prepared, cell, rep):
        return {"cell": cell, "rep": rep, "actual_work": 59520,
            "weights": {"U": [1.], "E-IID": [2.], "E-STRAT": [3.]},
            "costs": {a: {"train": 1., "fit": 1., "selection": 1.} for a in ("E-IID", "E-STRAT")}}

    def probability(_prepared, _weights, count, _key):
        return {"actual_work": count, "wall": 1., "summary": {"estimator": {"relative_se": [.02]},
            "sensitivities": [{"between_unit_relative_se": .02}]}}

    def risk(_prepared, _fit, count, _key):
        value = fake_risks()["stratified"]
        value.update(actual_work=count, wall=1.)
        value["summary"]["sensitivities"] = [{"between_unit_relative_se": .02}]*3
        return value

    monkeypatch.setattr(learning, "fit_pair", fit)
    monkeypatch.setattr(learning, "eight_probability", probability)
    monkeypatch.setattr(learning, "eight_risk", risk)
    prepared = {c: (None, None) for c in ("a", "b")}
    setup = {c: {"simulator_setup": 1., "geometry_setup": 1.} for c in prepared}
    output = learning.run(tmp_path, {"snapshot_sha256": "test"}, prepared, setup, {"decision": {"d2_gate": True}})
    assert len(output["fits"]) == len(output["confirmations"]) == 10
    assert len(output["rows"]) == 20
    assert all(a["risk"]["count"] == 39680 for a in output["allocations"])
    assert output["actual_work"] == 595200+491520+396800+30*65536+20*39680
    assert (tmp_path/"D3/allocation.json").exists()
    assert output["decision"]["p2_qualified"] is False
