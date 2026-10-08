"""Independent oracles for the structural-plan S0/S1 implementation."""

import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from src.path_integral.conditional_second_moment import allocate_precision, log_risk_potential
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.proposal_family_diagnostics import refit_identity_mixture
from src.path_integral.r1_bottleneck_diagnostics import fit_projected_mean_shift
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)


def component(mean: float) -> FiniteRankGaussianComponent:
    return FiniteRankGaussianComponent(torch.tensor([mean], dtype=torch.float64),
                                       torch.empty((1, 0), dtype=torch.float64),
                                       torch.empty(0, dtype=torch.float64))


def test_constant_risk_identity_and_smc_normalizer() -> None:
    delta, c = 0.1, 0.3
    logs = torch.full((20,), math.log(c), dtype=torch.float64)
    h = log_risk_potential(logs, torch.zeros_like(logs), defensive_mass=delta)
    assert torch.exp(h).tolist() == pytest.approx([delta * c * c] * 20)
    result = estimate_weighted_tempered_normalizer(
        lambda x: log_risk_potential(torch.full((len(x),), math.log(c), dtype=torch.float64),
                                     torch.zeros(len(x), dtype=torch.float64), defensive_mass=delta),
        dimension=1, config=WeightedSMCConfig(particles=32, temperatures=(0., .2, .7, 1.),
                                              mutation_steps=1, pcn_scale=.35, replicates=4,
                                              seed=71, resampling_scheme="stratified"))
    assert result.mean / delta == pytest.approx(c * c, abs=2e-15)


def test_risk_identity_independent_gaussian_quadrature() -> None:
    delta, shift = 0.1, -2.0
    proposal = DefensiveFiniteRankGaussianMixture((component(0), component(shift)),
                                                  torch.tensor([delta, 1-delta], dtype=torch.float64))

    def transformed_integrand(x: float) -> float:
        logg = torch.special.log_ndtr(torch.tensor([(-1.0-x)/.8], dtype=torch.float64))
        ratio = proposal.log_q_over_p(torch.tensor([[x]], dtype=torch.float64))
        return norm.pdf(x) * float(log_risk_potential(logg, ratio, defensive_mass=delta).exp()[0]) / delta

    computed = quad(transformed_integrand, -12, 12, epsabs=1e-11)[0]
    oracle = quad(lambda x: ndtr((-1-x)/.8)**2 * norm.pdf(x)**2 /
                  (delta * norm.pdf(x) + (1-delta) * norm.pdf(x-shift)),
                  -12, 12, epsabs=1e-11)[0]
    assert computed == pytest.approx(oracle, rel=1e-11)


@pytest.mark.parametrize("logg,ratio,delta", [([0.1], [0.], .1),
                                             ([-math.inf], [0.], .1),
                                             ([-1.], [-4.], .1),
                                             ([-1.], [math.nan], .1),
                                             ([-1.], [0.], 0.)])
def test_risk_rejects_invalid_contract(logg: list[float], ratio: list[float], delta: float) -> None:
    with pytest.raises(ValueError):
        log_risk_potential(torch.tensor(logg, dtype=torch.float64),
                           torch.tensor(ratio, dtype=torch.float64), defensive_mass=delta)


def test_allocation_matches_sample_cv_and_never_truncates_to_cap() -> None:
    values = torch.tensor([1., 2., 4., 8.], dtype=torch.float64)
    allocation = allocate_precision(values.log(), target_relative_se=.1, safety_factor=2,
                                    batch_size=16, maximum_count=32)
    expected = math.ceil(2 * float(values.var() / values.mean().square()) / .01 / 16) * 16
    assert allocation.planned_count == expected
    assert allocation.status == "unresolved_sample_budget"
    assert allocation.planned_count > allocation.maximum_count
    assert allocate_precision(torch.zeros(10, dtype=torch.float64), target_relative_se=.1,
                              safety_factor=2, batch_size=16, maximum_count=32).planned_count == 16


def test_family_preserves_two_modes_and_exact_density() -> None:
    torch.set_num_threads(1)
    parent = DefensiveFiniteRankGaussianMixture((component(0), component(-3), component(3)),
                                              torch.tensor([.1, .45, .45], dtype=torch.float64))
    bank = parent.sample(6000, path_seed=63, label_seed=64)
    # Positive bounded symmetric target with separated high-payoff modes.
    logg = torch.logaddexp(-.5 * (bank.samples[:, 0]-3).square(),
                           -.5 * (bank.samples[:, 0]+3).square()) - math.log(2)
    preserved, diag = refit_identity_mixture(parent, bank.samples, logg, bank.log_p_over_q,
                                             steps=30, learning_rate=.02)
    single, _ = fit_projected_mean_shift(bank.samples, logg, bank.log_p_over_q,
                                         torch.eye(1, dtype=torch.float64), objective="kl", steps=30)
    assert len(preserved.components) == 3
    assert preserved.defensive_mass == pytest.approx(.1)
    assert diag["returned_empirical_kl_loss"] <= diag["initial_empirical_kl_loss"]
    x = torch.linspace(-8, 8, 200, dtype=torch.float64)[:, None]
    q_dense = sum(float(w) * torch.exp(-.5*(x[:, 0]-float(c.mean[0])).square()) /
                  math.sqrt(2*math.pi) for w, c in zip(preserved.weights, preserved.components, strict=True))
    p_dense = torch.exp(-.5*x[:, 0].square()) / math.sqrt(2*math.pi)
    assert torch.allclose(preserved.log_q_over_p(x).exp(), q_dense/p_dense, atol=1e-9, rtol=1e-11)
    target_weights = torch.softmax(logg + bank.log_p_over_q, dim=0)
    # The family contrast is not just parameter shape: separated mass is retained.
    preserved_kl = -(target_weights * preserved.log_q_over_p(bank.samples)).sum()
    single_kl = -(target_weights * single.log_q_over_p(bank.samples)).sum()
    assert preserved_kl < single_kl - .05


def test_runner_role_separation_and_budget_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments import post_audit_r2_family_diagnosis as runner

    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({"cells": [{"cell": {"id": "toy"},
                                         "new_reference": {"mean": .5, "standard_error": .001}}]}))
    config = {"torch_threads": 1, "reference_path": str(ref), "model": {},
              "cells": [{"id": "toy"}], "parent_training_replicates": 1,
              "smc": {"steps": 1, "islands": 1, "particles": 32, "levels": 3,
                      "bridge_power": 1, "mutation_steps": 1, "pcn_scale": .35,
                      "resample_every": 1, "resampling_scheme": "stratified"},
              "fit": {"bank_count": 64, "steps": 3, "learning_rate": .02, "maximum_norm": 20},
              "mixture": {"clusters": 2, "covariance_rank": 0, "feature_rank": 1},
              "allocation": {"pilot_count": 64, "batch_size": 32, "target_relative_se": .5,
                             "safety_factor": 2, "maximum_count_per_method": 128},
              "budget": {"max_potential_evaluations": 1000, "max_wall_seconds": 60},
              "qualification": {"reference_se_fraction_of_method_se": .2,
                                "relative_equivalence_margin": .25, "confidence_z": 1.96}}
    monkeypatch.setattr(runner, "_problem", lambda *args: SimpleNamespace(task_id="toy", local_dimension=1))
    monkeypatch.setattr(runner, "_log_potential", lambda problem, x: torch.special.log_ndtr(x[:, 0]))
    result = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    row = result["records"][0]
    assert row["allocation_frozen_before_final"]
    assert len(row["candidates"]) == 3
    assert all(x["final_completed"] for x in row["candidates"])
    records = result["seed_ledger"]["records"]
    roles = {x["key"]["role"] for x in records}
    assert roles == {"parent-training", "refit-bank", "allocation-pilot", "final"}
    assert len({x["seed"] for x in records}) == len(records)
    # Exercise the auditor against generated toy artifacts, not mutable repo results.
    from experiments.post_audit_r2_family_audit import audit

    archive_path = tmp_path / "source.zip"
    content = b"toy source snapshot"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("toy.txt", content)
        archive.writestr("RUN_CONFIG.json", json.dumps(config))
    result["source"].update({"config_digest": canonical_digest(config),
                             "snapshot_path": str(archive_path),
                             "snapshot_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                             "snapshot_file_hashes": {"toy.txt": hashlib.sha256(content).hexdigest()}})
    assert audit(result)["whole_parent_runs"] == 1
    corrupted = copy.deepcopy(result)
    corrupted["records"][0]["candidates"][0]["accuracy"]["mean"] *= 2
    with pytest.raises(ValueError, match="accuracy arithmetic"):
        audit(corrupted)
    corrupted = copy.deepcopy(result)
    corrupted["records"][0]["candidates"][0]["allocation"]["planned_count"] += 32
    with pytest.raises(ValueError, match="allocation arithmetic"):
        audit(corrupted)
    config["budget"]["max_potential_evaluations"] = 1
    failure = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert failure["records"][0]["status"] == "unresolved"
    assert "unresolved_potential_budget" in failure["records"][0]["failure_reasons"][0]
    assert not failure["records"][0]["precision_cost_comparison_authorized"]
    assert failure["records"][0]["physical_wall_seconds_including_failure"] >= 0
