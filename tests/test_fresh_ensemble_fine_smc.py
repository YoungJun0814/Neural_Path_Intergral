import math
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

import experiments.post_audit_fresh_ensemble_fine_smc as runner
from experiments.post_audit_bank_ensemble_coverage import seed as previous_seed
from src.path_integral.bank_ensemble_coverage import coverage_ledger
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.structural_v2_reference import log_moments
from tests.test_p1_recovery_p2_parallel import problem
from tests.test_physical_window_guide import KWARGS


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_fixed_work_and_disjoint_streams_including_old_final():
    iid, smc = runner.iid_jobs(), runner.smc_jobs()
    assert len(iid) == 102 and len(smc) == 32
    assert len({runner.iid_id(j) for j in iid}) == 102
    assert len({runner.smc_id(j) for j in smc}) == 32
    seeds = [runner.seed("iid-final", j["cell"], j["block"], k, role)
             for j in iid for k in range(32) for role in ("path", "label")]
    seeds += [runner.seed("iid-preflight", c, 0, 0, role) for c in runner.CELLS for role in ("path", "label")]
    seeds += [runner.seed("smc-whole", j["cell"], j["arm"], j["rep"]) for j in smc]
    seeds += [runner.seed("smc-whole", c, "preflight", 0) for c in runner.CELLS]
    assert len(seeds) == len(set(seeds))
    old = {previous_seed(c, a, b, k, role) for c in runner.CELLS
           for a in ("uniform", "fit0", "fit1", "ensemble") for b in range(8)
           for k in range(32) for role in ("path", "label")}
    assert set(seeds).isdisjoint(old)
    assert runner.smc_work("64points-2moves") == 16000
    assert runner.smc_work("128points-1move") == 16256
    assert 102*8192+512 == 836096
    assert sum(runner.smc_work(j["arm"]) for j in smc)+960 == 517056
    assert 836096+517056 == 1353152


def test_precision_gate_uses_full_units_and_does_not_round_or_ignore_instability():
    stats = {"relative_se": .015}
    stable = {"maximum_unit_contribution_fraction": .15, "maximum_leave_one_out_relative_shift": .05,
              "between_unit_relative_se": .03}
    assert runner.precision_gate(stats, stable, iid=True)["pass"] is True
    assert runner.precision_gate({"relative_se": .01500001}, stable, iid=True)["pass"] is False
    unstable = {**stable, "between_unit_relative_se": .03000001}
    assert runner.precision_gate(stats, unstable, iid=True)["pass"] is False
    assert runner.precision_gate(stats, unstable, iid=False)["pass"] is True
    assert runner.precision_gate(stats, {**stable, "maximum_leave_one_out_relative_shift": .050001}, iid=False)["pass"] is False


def test_interval_status_distinguishes_uncertainty_and_detected_difference():
    base = {"symmetric_relative_difference": 0., "standard_error": .1, "confidence_z": 2.5, "pass": False}
    assert runner.interval_status(base) == "unresolved_equivalence_uncertainty"
    assert runner.interval_status({**base, "symmetric_relative_difference": .5}) == "detected_development_disagreement_not_bias_attribution"
    assert runner.interval_status({**base, "pass": True}) == "development_equivalence_pass"


def test_small_fine_target_gradient_and_iid_production_replay():
    p = replace(problem(32), rho=.3)
    cache = CachedVolterraTerminal.create(p)
    guide = PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS)
    z = .1*torch.randn((3, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(29001))
    leaf = z.clone().requires_grad_(True)
    values = runner.fine_components(cache, leaf)
    torch.testing.assert_close(values[:, 0], cache.log_probability(z))
    gradient = torch.autograd.grad(values.sum(), leaf)[0]
    direction = torch.randn(z.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(29002))
    direction /= direction.norm(dim=1, keepdim=True)
    eps = 1e-5
    numeric = (runner.fine_components(cache, z+eps*direction)-runner.fine_components(cache, z-eps*direction))[:, 0]/(2*eps)
    torch.testing.assert_close(numeric, (gradient*direction).sum(1), rtol=1e-5, atol=1e-6)
    j = {"cell": "small-oracle", "block": 0}
    prepared = cache, {"ensemble": guide.spec}
    a = runner.iid_values(j, prepared, count=runner.CHUNK, stream="iid-preflight")
    b = runner.iid_values(j, prepared, count=runner.CHUNK, stream="iid-preflight")
    assert a["moments"] == b["moments"] and a["coverage"] == b["coverage"]
    assert sum(a["coverage"]["bin_counts"]) == runner.CHUNK


def test_single_component_smc_constant_potential_preserves_product_and_work(monkeypatch):
    cache = CachedVolterraTerminal.create(problem(32))
    value = math.log(.23)
    monkeypatch.setattr(runner, "ARMS", {"small": (16, 8, 2, 2)})

    def constant(evaluator, z):
        return torch.full((len(z), 1), value, dtype=torch.float64)+0*z[:, :1]

    monkeypatch.setattr(runner, "fine_components", constant)
    result = runner.smc_values({"cell": "oracle", "arm": "small", "rep": 0}, cache)
    assert result["log_mu"] == pytest.approx(value, abs=1e-12)
    assert result["actual_work"] == 16*(1+6*2) == result["backward_evaluations"]
    assert result["guide_independent_kernel"] is True
    assert result["stages"][-1]["surviving_initial_ancestors"] <= 16


def test_allocation_uses_worse_raw_and_block_projection_and_rejects_prefix():
    rows = [{"cell": c, "arm": "ensemble", "summary": {"count": 65536, "relative_se": raw},
             "block_sensitivity": {"between_unit_relative_se": block}}
            for c, raw, block in zip(runner.CELLS, (.023118291269171816, .019300125890726813),
                                     (.023366503258361624, .020067971312115802), strict=True)]
    parent = {"summary": {"rows": rows}, "config": {}}
    assert runner.configuration(parent)["counts"] == runner.COUNTS
    changed = deepcopy(parent)
    changed["summary"]["rows"][0]["summary"]["count"] = 123
    with pytest.raises(ValueError):
        runner.configuration(changed)
    with pytest.raises(ValueError, match="complete"):
        runner.summarize([], [], parent)


def test_scoped_corroboration_never_promotes_global_flags_or_unqualified_pair():
    values = torch.full((8192,), math.log(.23), dtype=torch.float64)
    moment = log_moments(values)
    features = torch.tensor([[0., 0., .1, .5]], dtype=torch.float64).repeat(8192, 1)
    coverage = coverage_ledger(values, features)
    records = [{**j, "moments": moment, "coverage": coverage, "wall": 1.} for j in runner.iid_jobs()]
    stages = [{"surviving_initial_ancestors": 128, "ess_before_resampling": 128,
               "mutation_acceptance": acceptance} for acceptance in (.8, 0.)]
    smc = [{**j, "log_mu": math.log(.23), "stages": stages, "wall": 1.} for j in runner.smc_jobs()]
    parent = {"summary": {"rows": [{"cell": c, "arm": "ensemble", "summary": moment} for c in runner.CELLS]}}
    result = runner.summarize(records, smc, parent)
    assert all(r["scoped_development_corroboration_pass"] for r in result["comparisons"])
    assert result["p1_qualified"] is False and result["p2_qualified"] is False
    assert result["large_training_authorized"] is False
    smc[0] = {**smc[0], "log_mu": 0.}
    changed = runner.summarize(records, smc, parent)
    first = changed["comparisons"][0]
    assert first["both_accuracy_qualified"] is False
    assert first["scoped_development_corroboration_pass"] is False
    with pytest.raises(ValueError, match="complete"):
        runner.iid_stage_summary(records[:-1])
