import math

import pytest
import torch

import experiments.post_audit_fine_smc_early_bridge as runner
from experiments.post_audit_fresh_ensemble_fine_smc import seed as previous_seed
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.structural_v2_reference import log_moments
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_frozen_family_equal_work_crn_and_disjoint_previous_streams():
    jobs = runner.jobs()
    assert len(jobs) == len({runner.identity(j) for j in jobs}) == 48
    assert len({runner.smc_config(j).seed for j in jobs}) == 16
    final = {runner.seed(c, r) for c in runner.CELLS for r in range(8)}
    preflight = {runner.seed(c, 0, "preflight") for c in runner.CELLS}
    old = {previous_seed("smc-whole", c, a, r) for c in runner.CELLS
           for a in ("64points-2moves", "128points-1move") for r in range(8)}
    assert len(final) == 16 and len(preflight) == 2 and final.isdisjoint(preflight | old)
    assert preflight.isdisjoint(old)
    assert 48*16000+6*4000 == 792000
    for c in runner.CELLS:
        for rep in range(8):
            cfgs = [runner.smc_config({"cell": c, "rep": rep, "power": p}) for p in runner.POWERS]
            assert len({cfg.seed for cfg in cfgs}) == 1
            for cfg, power in zip(cfgs, runner.POWERS, strict=True):
                assert len(cfg.temperatures) == 64
                assert cfg.temperatures[0] == 0 and cfg.temperatures[-1] == 1
                assert all(a < b for a, b in zip(cfg.temperatures[:-1], cfg.temperatures[1:], strict=True))
                assert cfg.particles*(1+(len(cfg.temperatures)-2)*cfg.mutation_steps) == 16000
                assert sum((k % cfg.resample_every == 0) for k in range(1, 63)) == 15
                expected = {2: 12, 3: 21, 4: 28}[power]
                assert sum(0 < beta <= .04 for beta in cfg.temperatures[1:-1]) == expected
    with pytest.raises(ValueError):
        runner.seed("c", 0, "replacement")
    with pytest.raises(ValueError):
        runner.smc_config({"cell": "c", "rep": 0, "power": 5})


def test_constant_target_all_powers_preserves_normalizer_and_common_rng_ancestry(monkeypatch):
    cache = CachedVolterraTerminal.create(problem(4))

    def constant(evaluator, z):
        return torch.full((len(z), 1), math.log(.23), dtype=torch.float64)+0*z[:, :1]

    monkeypatch.setattr(runner, "fine_components", constant)
    records = [runner.values({"cell": "oracle", "rep": 0, "power": p}, cache) for p in runner.POWERS]
    for r in records:
        assert r["log_mu"] == pytest.approx(math.log(.23), abs=1e-12)
        assert r["actual_work"] == r["backward_evaluations"] == 16000
        assert all(s["surviving_initial_ancestors"] == 128 for s in r["stages"])
        assert all(s["mutation_acceptance"] == 1. for s in r["stages"][:-1])
    assert len({r["seed"] for r in records}) == 1


@pytest.mark.parametrize("power", [2, 3, 4])
def test_power_schedule_known_gaussian_normalizer_not_geometric_mean(power):
    # A bounded Gaussian potential matches the production callback contract.
    # E exp(-a (Z-b)^2/2) = exp(-a b^2/[2(1+a)])/sqrt(1+a).
    a, b = .6, .7

    def potential(z):
        return -.5*a*(z[:, :1]-b).square()

    estimates = []
    for rep in range(24):
        cfg = WeightedSMCConfig(64, tuple((k/15)**power for k in range(16)), 2, .35, 1,
            415000+rep, resample_every=4, resampling_scheme="stratified")
        result = estimate_gradient_umbrella_normalizers(potential, dimension=4,
            log_mixture_weights=torch.zeros(1, dtype=torch.float64), config=cfg)
        assert result.potential_evaluations == 64*29
        estimates.append(math.exp(result.log_normalizer))
    values = torch.tensor(estimates, dtype=torch.float64)
    se = float(values.std(unbiased=True))/math.sqrt(len(values))
    exact = math.exp(-a*b*b/(2*(1+a)))/math.sqrt(1+a)
    assert abs(float(values.mean())-exact) < 5*se


def test_milestone_is_left_recorded_stage_and_reports_actual_beta():
    r = {"stages": [{"beta": .02, "surviving_initial_ancestors": 50},
                    {"beta": .05, "surviving_initial_ancestors": 30}]}
    assert runner.ancestry_at(r, .01) == {"requested_beta": .01, "recorded_beta": 0., "ancestors": 128}
    assert runner.ancestry_at(r, .04) == {"requested_beta": .04, "recorded_beta": .02, "ancestors": 50}
    assert runner.ancestry_at(r, .05)["ancestors"] == 30


def test_complete_family_pairing_candidate_and_no_global_promotion():
    moment = log_moments(torch.full((8192,), math.log(.23), dtype=torch.float64))
    parent = {"summary": {"iid_rows": [{"cell": c, "summary": moment,
                "precision_and_sensitivity": {"pass": True}} for c in runner.CELLS]}}
    records = []
    for j in runner.jobs():
        spread = .02 if j["power"] == 2 else .001
        log_mu = math.log(.23)+spread*(j["rep"]-3.5)
        records.append({**j, "log_mu": log_mu, "wall": 1., "stages": [
            {"beta": b, "surviving_initial_ancestors": 20+j["power"],
             "ess_before_resampling": 80., "mutation_acceptance": .8} for b in (.001, .03, 1.)]})
    result = runner.summarize(records, parent)
    assert result["development_candidate_powers"] == [3, 4]
    assert len(result["paired_mean_diagnostics"]) == 4
    assert len(result["reference_comparisons"]) == 6
    assert result["p1_qualified"] is result["p2_qualified"] is result["large_training_authorized"] is False
    changed = [dict(r) for r in records]
    changed[1]["log_mu"] = math.log(.6)
    assert runner.summarize(changed, parent)["development_candidate_powers"] == [4]
    with pytest.raises(ValueError, match="complete"):
        runner.summarize(records[:-1], parent)
    with pytest.raises(ValueError, match="complete"):
        runner.summarize([records[0], *records[:-1]], parent)
