"""Plan S2: matched work, independent-risk SE and failure-preserving runner."""

from __future__ import annotations

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

from src.path_integral.conditional_second_moment import (
    log_risk_potential,
    summarize_risk_replicates,
)
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)


def test_risk_se_is_between_whole_replicates() -> None:
    values = torch.tensor([.1, .2, .4, .8], dtype=torch.float64)
    actual = summarize_risk_replicates(values.log()+math.log(.1), defensive_mass=.1,
                                      expected_replicates=4, maximum_relative_se=.2)
    assert actual["mean"] == pytest.approx(float(values.mean()))
    assert actual["standard_error"] == pytest.approx(float(values.std()/2))
    assert actual["status"] == "unresolved_risk_precision"
    assert actual["se_unit"] == "independent_whole_smc_normalizer"
    assert summarize_risk_replicates(torch.tensor([-.1], dtype=torch.float64), defensive_mass=.1,
                                    expected_replicates=4, maximum_relative_se=.2)["mean"] is None
    assert summarize_risk_replicates(torch.tensor([-.1, -.2], dtype=torch.float64), defensive_mass=.1,
                                    expected_replicates=4, maximum_relative_se=.2)["status"] == "unresolved_incomplete_risk"


@pytest.mark.parametrize("logs", [torch.tensor([1., 2.], dtype=torch.float64),
                                 torch.tensor([math.nan, -1.], dtype=torch.float64),
                                 torch.tensor([[-1., -2.]], dtype=torch.float64),
                                 torch.tensor([-1., -2.], dtype=torch.float32)])
def test_invalid_normalizers_rejected(logs: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        summarize_risk_replicates(logs, defensive_mass=.1, expected_replicates=2, maximum_relative_se=.2)


@pytest.mark.parametrize("log_z,delta", [(-1000., .1), (0., 1e-320)])
def test_extreme_risk_summary_is_unresolved_not_silently_zero_or_inf(log_z: float, delta: float) -> None:
    result = summarize_risk_replicates(torch.full((8,), log_z, dtype=torch.float64),
        defensive_mass=delta, expected_replicates=8, maximum_relative_se=.2)
    assert result["status"] == "unresolved_risk_numerical_range"
    assert result["mean"] is None and result["standard_error"] is None
    assert math.isfinite(result["log_mean"])


def test_nonconstant_risk_normalizer_against_quadrature() -> None:
    torch.set_num_threads(1)
    delta = .1
    natural = FiniteRankGaussianComponent.natural(1)
    shifted = FiniteRankGaussianComponent(torch.tensor([-1.], dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.float64), torch.empty(0, dtype=torch.float64))
    q = DefensiveFiniteRankGaussianMixture((natural, shifted), torch.tensor([.1, .9], dtype=torch.float64))

    def risk(x: torch.Tensor) -> torch.Tensor:
        return log_risk_potential(torch.special.log_ndtr(-1-x[:, 0]), q.log_q_over_p(x), defensive_mass=delta)

    # No resampling/mutation: exact reduction to independent prior MC per whole run.
    result = estimate_weighted_tempered_normalizer(risk, dimension=1,
        config=WeightedSMCConfig(particles=256, temperatures=(0., .1, .4, 1.),
            mutation_steps=0, pcn_scale=.35, replicates=32, seed=9241, resample_every=100))
    oracle = quad(lambda x: ndtr(-1-x)**2 * norm.pdf(x)**2 /
                  (.1*norm.pdf(x)+.9*norm.pdf(x+1)), -12, 12, epsabs=1e-12)[0]
    actual = summarize_risk_replicates(result.log_replicate_estimates, defensive_mass=delta,
                                      expected_replicates=32, maximum_relative_se=.2)
    assert abs(actual["mean"]-oracle) < 6*actual["standard_error"]


def test_constant_potential_scale_does_not_change_smc_geometry() -> None:
    spec = WeightedSMCConfig(particles=64, temperatures=(0., .1, .4, 1.),
        mutation_steps=2, pcn_scale=.35, replicates=2, seed=234, retain_final_particles=True)
    first = estimate_weighted_tempered_normalizer(lambda x: torch.special.log_ndtr(-x[:, 0]),
                                                  dimension=1, config=spec)
    scaled = estimate_weighted_tempered_normalizer(lambda x: torch.special.log_ndtr(-x[:, 0])+math.log(.01),
                                                   dimension=1, config=spec)
    assert first.final_particles is not None and scaled.final_particles is not None
    assert torch.allclose(first.final_particles, scaled.final_particles, atol=1e-12, rtol=0)
    assert torch.allclose(scaled.log_replicate_estimates-first.log_replicate_estimates,
                          torch.full((2,), math.log(.01), dtype=torch.float64), atol=1e-12, rtol=0)


def test_bank_runner_and_tamper_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments import post_audit_r2_bank_risk_diagnostics as runner

    ref = tmp_path/"reference.json"
    ref.write_text(json.dumps({"cells": [{"cell": {"id": "toy"}, "new_reference": {
        "mean": .5, "standard_error": .001}}]}))
    config = {"mode": "bank_islands", "role": "toy", "torch_threads": 1,
        "reference_path": str(ref), "model": {}, "cells": [{"id": "toy"}],
        "parent_training_replicates": 1, "bank_candidates": [
            {"id": "one", "islands": 1, "particles": 48},
            {"id": "three", "islands": 3, "particles": 16},
            {"id": "six", "islands": 6, "particles": 8}],
        "smc": {"steps": 1, "levels": 3, "bridge_power": 1, "mutation_steps": 1,
                "pcn_scale": .35, "resample_every": 1, "resampling_scheme": "stratified"},
        "mixture": {"clusters": 2, "covariance_rank": 0, "feature_rank": 1},
        "evaluation": {"bank_diagnostic_count": 16, "direct_count": 64, "batch_size": 32},
        "risk": {"particles": 16, "replicates": 4, "levels": 3, "bridge_power": 1,
                 "mutation_steps": 1, "pcn_scale": .35, "resample_every": 1,
                 "resampling_scheme": "stratified", "maximum_relative_se": .2},
        "budget": {"max_potential_evaluations": 5000, "max_wall_seconds": 60},
        "qualification": {"target_relative_se": .1, "reference_se_fraction_of_method_se": .2,
                          "relative_equivalence_margin": .25, "confidence_z": 1.96}}
    monkeypatch.setattr(runner, "_problem", lambda *args: SimpleNamespace(task_id="toy", local_dimension=1))
    monkeypatch.setattr(runner, "_log_potential", lambda problem, x: torch.special.log_ndtr(x[:, 0]))
    result = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert len(result["records"]) == 3
    assert {r["bank_summary"]["training_potential_evaluations"] for r in result["records"]} == {144}
    assert all(len(r["risk_replicates"]) == 4 for r in result["records"])
    assert all(not r["iid_bank_diagnostic"]["used_for_fit"] for r in result["records"])
    seeds = result["seed_ledger"]["records"]
    assert len({s["seed"] for s in seeds}) == len(seeds)
    assert {s["key"]["role"] for s in seeds} == {
        "parent-training", "iid-bank-diagnostic", "direct-final", "independent-risk"}
    archive = tmp_path/"source.zip"
    content = b"toy source"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("toy.py", content)
        z.writestr("RUN_CONFIG.json", json.dumps(config))
    result["source"].update({"config_digest": canonical_digest(config), "snapshot_path": str(archive),
        "snapshot_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "snapshot_file_hashes": {"toy.py": hashlib.sha256(content).hexdigest()}})
    assert runner.audit(result)["jobs"] == 3
    tampered = copy.deepcopy(result)
    tampered["records"][0]["risk"]["mean"] *= 2
    with pytest.raises(ValueError, match="risk arithmetic"):
        runner.audit(tampered)
    config["budget"]["max_potential_evaluations"] = 1
    failed = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert len(failed["records"]) == 3
    assert all(r["status"] == "unresolved" and not r["qualified"] for r in failed["records"])
    assert all(r["physical_wall_seconds"] >= 0 and r["costs"]["offline"] >= 0 for r in failed["records"])
    config["budget"]["max_potential_evaluations"] = 5000

    def failing_potential(problem: object, x: torch.Tensor) -> torch.Tensor:
        raise FloatingPointError("injected diagnostic failure")

    monkeypatch.setattr(runner, "_log_potential", failing_potential)
    numerical_failure = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert numerical_failure["potential_evaluations"] == 48+16+8
    assert [r["potential_evaluations"] for r in numerical_failure["records"]] == [48, 16, 8]
    assert all(r["status"] == "unresolved" for r in numerical_failure["records"])
