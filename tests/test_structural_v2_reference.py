"""Independent formulas, inference units, fixed allocation, and negative audits."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr

from experiments import post_audit_v2_independent_reference as runner
from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.structural_v2_reference import (
    exact_smc_work,
    log_moments,
    merge_moments,
    precision_count,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)


def natural(d: int) -> DefensiveFiniteRankGaussianMixture:
    return DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(d),), torch.ones(1, dtype=torch.float64))


def test_exact_smc_work_matches_last_bridge_schedule() -> None:
    spec = {"particles": 16, "levels": 6, "mutation_steps": 3}
    result = estimate_weighted_tempered_normalizer(lambda x: torch.full((len(x),), math.log(.2), dtype=torch.float64),
        dimension=2, config=WeightedSMCConfig(16, tuple(np.linspace(0, 1, 6)), 3, .35, 1, 8))
    assert result.potential_evaluations == exact_smc_work(spec) == 208
    assert result.mean == pytest.approx(.2, abs=1e-14)
    assert exact_smc_work({"particles": 256, "levels": 96, "mutation_steps": 8}) == 192768


@pytest.mark.parametrize("key,value", [("particles", True), ("levels", 1), ("mutation_steps", 1.2)])
def test_invalid_work(key: str, value: object) -> None:
    spec = {"particles": 8, "levels": 4, "mutation_steps": 2}
    spec[key] = value  # type: ignore[assignment]
    with pytest.raises(ValueError):
        exact_smc_work(spec)


def test_log_moments_merge_matches_direct_and_not_block_se() -> None:
    x = torch.log(torch.arange(1, 65, dtype=torch.float64))
    merged = merge_moments([log_moments(part) for part in x.chunk(8)])
    direct = log_moments(x)
    for k in merged:
        assert merged[k] == pytest.approx(direct[k], rel=1e-12, abs=1e-12)
    s = sensitivity([log_moments(part)["log_mean"] for part in x.chunk(8)], bootstrap_seed=4)
    assert s["unit_count"] == 8 and merged["count"] == 64
    assert s["between_unit_relative_se"] > merged["relative_se"]


def test_equivalence_requires_upper_bound_not_zero_in_ci() -> None:
    a = {"log_mean": -40., "relative_se": .025}
    result = relative_equivalence(a, a, comparisons=30)
    assert not result["pass"] and result["upper_absolute_difference"] > .10
    b = {**a, "relative_se": .015}
    assert relative_equivalence(b, b, comparisons=30)["pass"]
    assert relative_equivalence({**b, "log_mean": -40.2}, b, comparisons=30)["pass"] is False


def test_allocation_fixed_caps_and_outlier_sensitivity() -> None:
    a = precision_count({"count": 8, "relative_se": .4}, target_rse=.015, safety_factor=3,
                        minimum=32, maximum=256)
    assert a["status"] == "unresolved_sample_budget"
    assert a["required_count"] > 17000
    s = sensitivity([0.] * 7 + [math.log(100)], bootstrap_seed=7)
    assert s["maximum_unit_contribution_fraction"] > .9
    assert s["maximum_leave_one_out_relative_shift"] > .8


def test_n1_digital_independent_closed_form_quadrature() -> None:
    model = {"spot": 100., "maturity": .7, "hurst": .12, "eta": 1.2, "xi": .04, "rho": -.7}
    cell = {"id": "toy", "threshold": 85.}
    problem = _problem(model, cell, 1)
    z, w = hermgauss(80)
    z, w = math.sqrt(2) * z, w / math.sqrt(math.pi)
    samples = torch.zeros((80, 2), dtype=torch.float64)
    samples[:, 0] = torch.from_numpy(z)
    actual = evaluate_rbergomi_conditional_terminal(problem, samples).payoffs.left_probability.numpy()
    variance = model["xi"] * model["maturity"]
    mean = math.log(model["spot"]) - .5 * variance
    independent = ndtr((math.log(85.) - mean - model["rho"] * math.sqrt(variance) * z) /
                       math.sqrt((1 - model["rho"]**2) * variance))
    assert np.max(np.abs(actual - independent)) < 2e-15
    assert float(w @ actual) == pytest.approx(float(ndtr((math.log(85.) - mean) / math.sqrt(variance))), abs=2e-14)


def test_n2_rough_payoff_independent_numpy_and_quadrature_refinement() -> None:
    model = {"spot": 100., "maturity": .5, "hurst": .15, "eta": .6, "xi": .04, "rho": 0.}
    cell = {"id": "toy", "threshold": 90.}
    problem = _problem(model, cell, 2)
    estimates = []
    for order in (24, 40):
        z, w = hermgauss(order)
        z, w = math.sqrt(2) * z, w / math.sqrt(math.pi)
        a, b = np.meshgrid(z, z, indexing="ij")
        samples = np.zeros((order**2, 4))
        samples[:, :2] = np.stack((a.ravel(), b.ravel()), axis=1)
        h, H, eta = model["maturity"] / 2, model["hurst"], model["eta"]
        alpha = H - .5
        cov = h**(alpha + 1) / (alpha + 1)
        var = h**(2 * alpha + 1) / (2 * alpha + 1)
        L = cov / math.sqrt(h) * samples[:, 0] + math.sqrt(var - cov**2 / h) * samples[:, 1]
        Y = math.sqrt(2 * H) * L
        v1 = model["xi"] * np.exp(eta * Y - .5 * eta**2 * 2 * H * var)
        integrated = h * (model["xi"] + v1)
        independent = ndtr((math.log(90. / 100.) + .5 * integrated) / np.sqrt(integrated))
        actual = evaluate_rbergomi_conditional_terminal(problem, torch.from_numpy(samples)).payoffs.left_probability.numpy()
        assert np.max(np.abs(actual - independent)) < 3e-14
        estimates.append((float(np.outer(w, w).ravel() @ independent),
                          float(np.outer(w, w).ravel() @ independent**2)))
    assert estimates[0] == pytest.approx(estimates[1], abs=2e-10)


@pytest.fixture
def tiny_run(monkeypatch: pytest.MonkeyPatch) -> dict:
    config = yaml.safe_load((Path(runner.ROOT) / "configs/post_audit/structural_v2_p1_pilot_v1.yaml").read_text())
    config["pilot"] = {"whole_smc_replicates": 4, "iid_count": 64, "iid_blocks": 4, "batch_size": 8}
    config["budget"]["minimum_free_disk_bytes"] = 0
    config["qualification"]["bootstrap_replicates"] = 20
    for schedule in config["schedules"]:
        schedule.update(particles=8, levels=4, mutation_steps=2)
    model = {"spot": 100., "maturity": 1., "hurst": .1, "eta": .4, "xi": .04, "rho": -.4}
    params = proposal_parameters(natural(4))
    rows = [{"cell": {"id": cell, "threshold": 80.}, "parent_training_rep": rep,
             "q_parameters": params, "q_digest": canonical_digest(params)} for cell in ("one", "two") for rep in range(5)]
    old = {"config": {"model": model, "smc": {"steps": 2}}}
    monkeypatch.setattr(runner, "inputs", lambda c: (old, rows))
    monkeypatch.setattr(runner, "file_sha256", lambda path: ("d" * 64, 100))
    monkeypatch.setattr(runner, "source_tree_digest", lambda root: "a" * 64)
    monkeypatch.setattr(runner, "audit_snapshot", lambda *args: {})
    monkeypatch.setattr(runner, "build_volterra_excursion_guide", lambda *args, **kwargs: natural(4))
    return runner.run(config, {"source_tree_digest": "a" * 64})


def test_pilot_never_authorizes_p2_and_full_audit(tiny_run: dict) -> None:
    assert len(tiny_run["records"]) == 22
    assert all(r["status"] == "completed" for r in tiny_run["records"])
    assert tiny_run["qualification"]["p2_authorized"] is False
    assert runner.audit(tiny_run)["records"] == 22
    assert all(u["role"] in {"allocation-pilot", "audit"} for u in tiny_run["sample_uses"])


@pytest.mark.parametrize("mutation", ["moment", "work", "gate", "grid", "role", "guide", "conversion", "seed_missing", "block_counts", "schedule"])
def test_negative_reference_audit(tiny_run: dict, mutation: str) -> None:
    result = copy.deepcopy(tiny_run)
    row = result["records"][0]
    if mutation == "moment":
        row["summary"]["log_mean"] += .1
    elif mutation == "work":
        result["potential_evaluations"] += 1
    elif mutation == "gate":
        result["qualification"]["p2_authorized"] = True
    elif mutation == "grid":
        result["records"].pop()
    elif mutation == "role":
        row["measurement_contract"]["se_unit"] = "whole_smc_run"
    elif mutation == "guide":
        row["guide_digest"] = "b" * 64
    elif mutation == "conversion":
        next(r for r in result["records"] if r["whole_runs"])["whole_runs"][0]["log_estimand_estimate"] += .1
    elif mutation == "seed_missing":
        result["sample_uses"].pop(0)
    elif mutation == "block_counts":
        row["blocks"][0]["count"] -= 1
        row["blocks"][1]["count"] += 1
    else:
        next(r for r in result["records"] if r["whole_runs"])["whole_runs"][0]["diagnostics"]["stages"].pop()
    with pytest.raises((ValueError, KeyError, StopIteration)):
        runner.audit(result)


def test_production_locked_for_unresolved_allocation(tiny_run: dict) -> None:
    tiny_run["allocation"]["status"] = "unresolved_production_budget"
    with pytest.raises(ValueError, match="production locked"):
        runner.production_config(tiny_run, Path(runner.ROOT) / "results/fake-pilot.json")


def test_production_equal_blocks_rounding_before_freeze(tiny_run: dict) -> None:
    original = copy.deepcopy(tiny_run["config"])
    tiny_run["allocation"] = {"status": "allocated", "production_jobs": [
        {"cell_id": "one", "parent_training_rep": 0, "estimand": "risk", "method": "static-iid",
         "count": 65, "forecast_potential_evaluations": 65, "forecast_wall_seconds": 1.}]}
    config = runner.production_config(tiny_run, Path(runner.ROOT) / "results/fake-pilot.json")
    assert config["production_jobs"][0]["count"] == 96
    assert config["production_jobs"][0]["production_rounding_from_pilot_forecast_count"] == 65
    assert tiny_run["config"] == original


def test_retained_terminal_modes_runner_and_tamper_audit(tiny_run: dict) -> None:
    config = copy.deepcopy(tiny_run["config"])
    for schedule in config["schedules"]:
        schedule["retain_final_particles"] = True
    payload = runner.run(config, tiny_run["source"])
    assert payload["diagnostic_path_evaluations"] == 320
    assert runner.audit(payload)["records"] == 22
    row = next(r for r in payload["records"] if r["whole_runs"])
    assert sum(row["mode_contributions"]["relative_mean_contribution"]) == pytest.approx(1.)
    for metric in ("partition", "terminal_weight_ess"):
        tampered = copy.deepcopy(payload)
        geometry = next(r for r in tampered["records"] if r["whole_runs"])["whole_runs"][0]["terminal_geometry"]
        geometry[metric] = {} if metric == "partition" else 1e9
        with pytest.raises(ValueError):
            runner.audit(tampered)
    row["mode_contributions"]["relative_mean_contribution"][0] += .1
    with pytest.raises(ValueError, match="mode contribution"):
        runner.audit(payload)


def test_probability_both_local_pilots_separate_selection(tiny_run: dict) -> None:
    config = copy.deepcopy(tiny_run["config"])
    config["probability_local_A_pilot"] = True
    payload = runner.run(config, tiny_run["source"])
    assert len(payload["records"]) == 24
    assert "selected_probability_local_schedule" in payload["allocation"]
    assert runner.audit(payload)["records"] == 24
