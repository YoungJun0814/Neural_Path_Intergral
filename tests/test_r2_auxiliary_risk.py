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

from src.path_integral.conditional_second_moment import log_auxiliary_second_moment
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)


def mixture(shift: float) -> DefensiveFiniteRankGaussianMixture:
    return DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(1),
        FiniteRankGaussianComponent(torch.tensor([shift], dtype=torch.float64),
            torch.empty((1, 0), dtype=torch.float64), torch.empty(0, dtype=torch.float64))),
        torch.tensor([.1, .9], dtype=torch.float64))


def test_auxiliary_risk_independent_quadrature_and_direct_identity() -> None:
    q, r = mixture(-1), mixture(-2)

    def integrand(x: float) -> float:
        points = torch.tensor([[x]], dtype=torch.float64)
        logg = torch.special.log_ndtr(-1-points[:, 0])
        value = log_auxiliary_second_moment(logg, q.log_q_over_p(points), r.log_q_over_p(points))
        return float(value.exp()[0])*(.1*norm.pdf(x)+.9*norm.pdf(x+2))

    actual = quad(integrand, -12, 12, epsabs=1e-11)[0]
    oracle = quad(lambda x: ndtr(-1-x)**2*norm.pdf(x)**2/(.1*norm.pdf(x)+.9*norm.pdf(x+1)),
                  -12, 12, epsabs=1e-11)[0]
    assert actual == pytest.approx(oracle, rel=1e-11)
    x = torch.linspace(-4, 4, 100, dtype=torch.float64)[:, None]
    logg, logq = torch.special.log_ndtr(-1-x[:, 0]), q.log_q_over_p(x)
    assert torch.allclose(log_auxiliary_second_moment(logg, logq, logq), 2*(logg-logq), atol=1e-12)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_auxiliary_risk_rejects_nonfinite(bad: float) -> None:
    with pytest.raises(ValueError):
        log_auxiliary_second_moment(torch.tensor([bad], dtype=torch.float64),
                                    torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))


def test_auxiliary_runner_audit_and_failed_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments import post_audit_r2_auxiliary_risk as runner
    from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
    from src.path_integral.research_result_contract import canonical_digest, source_tree_digest

    params = proposal_parameters(mixture(-1))
    parent = tmp_path/"parent.json"
    parent.write_text(json.dumps({"config": {"model": {}, "smc": {"steps": 1}}, "records": [
        {"cell": {"id": "toy"}, "parent_training_rep": 0, "candidates": [
            {"method": "parent-as-is", "proposal_parameters": params, "proposal_digest": canonical_digest(params)}]}]}))
    config = {"torch_threads": 1, "proposal_artifact_path": str(parent), "method": "parent-as-is",
        "training": {"islands": 1, "particles": 16, "levels": 3, "bridge_power": 1,
                     "mutation_steps": 1, "pcn_scale": .35, "resample_every": 1, "resampling_scheme": "stratified"},
        "mixture": {"clusters": 2, "covariance_rank": 0, "feature_rank": 1},
        "evaluation": {"count": 128, "batch_size": 64, "maximum_relative_se": .2},
        "budget": {"max_potential_evaluations": 5000, "max_wall_seconds": 60}}
    monkeypatch.setattr(runner, "_problem", lambda *args: SimpleNamespace(task_id="toy", local_dimension=1))
    monkeypatch.setattr(runner, "_log_potential", lambda problem, x: torch.special.log_ndtr(-1-x[:, 0]))
    result = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert result["potential_evaluations"] == 2*48+3*128
    assert {t["potential_evaluations"] for t in result["records"][0]["training"]} == {48}
    assert not result["model_q_changed"]
    archive = tmp_path/"source.zip"
    content = b"toy auxiliary source"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("toy.py", content)
        z.writestr("RUN_CONFIG.json", json.dumps(config))
    result["source"].update({"config_digest": canonical_digest(config), "snapshot_path": str(archive),
        "snapshot_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "snapshot_file_hashes": {"toy.py": hashlib.sha256(content).hexdigest()}})
    assert runner.audit(result)["parents"] == 1
    config["training_repetitions"] = 3
    repeated = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert len(repeated["records"]) == 3
    assert {r["auxiliary_training_rep"] for r in repeated["records"]} == {0, 1, 2}
    assert len({r["estimators"][2]["r_digest"] for r in repeated["records"]}) == 3
    assert repeated["potential_evaluations"] == 3*result["potential_evaluations"]
    from experiments.post_audit_r2_auxiliary_stability import assess
    assert not assess(repeated)["oracle_authorized"]
    assert assess(repeated)["whole_fit_grid_complete"]
    duplicate = copy.deepcopy(repeated)
    duplicate["config"]["training_repetitions"] = 5
    duplicate["records"] = [copy.deepcopy(duplicate["records"][0]) for _ in range(5)]
    assert not assess(duplicate)["whole_fit_grid_complete"]
    assert not assess(duplicate)["all_risk_fits_stable"]
    ensemble_config = copy.deepcopy(config)
    ensemble_config["island_ensemble"] = True
    ensemble_config["evaluation"]["counts"] = {"risk-auxiliary": 256}
    ensemble_config["evaluation"]["fixed_r_block_count"] = 2
    ensemble = runner.run(ensemble_config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert ensemble["potential_evaluations"] == 3*(96+128+128+256)
    assert all(len(e["fixed_r_independent_blocks"]) == 2 for r in ensemble["records"] for e in r["estimators"])
    ensemble_archive = tmp_path/"ensemble.zip"
    with zipfile.ZipFile(ensemble_archive, "w") as z:
        z.writestr("RUN_CONFIG.json", json.dumps(ensemble_config))
    ensemble["source"].update({"config_digest": canonical_digest(ensemble_config),
        "snapshot_path": str(ensemble_archive), "snapshot_sha256": hashlib.sha256(ensemble_archive.read_bytes()).hexdigest(),
        "snapshot_file_hashes": {}})
    assert runner.audit(ensemble)["whole_training_jobs"] == 3
    tampered = copy.deepcopy(ensemble)
    tampered["records"][0]["estimators"][2]["fixed_r_independent_blocks"][0]["relative_se"] *= 2
    with pytest.raises(ValueError, match="block moments"):
        runner.audit(tampered)
    assert not assess(ensemble)["all_risk_fits_stable"]  # three fits cannot pass five-fit gate
    monkeypatch.setattr(runner, "build_volterra_excursion_guide", lambda *args, **kwargs: mixture(-3))
    guided_config = copy.deepcopy(ensemble_config)
    guided_config.update({"volterra_guide": {"amplitudes": [3.]}, "guide_output_mass": .5,
                          "static_control": True})
    guided_config["training"]["independence_every"] = 1
    guided = runner.run(guided_config, {"source_tree_digest": source_tree_digest(runner.ROOT)})

    def bind_and_audit(payload: dict, cfg: dict, name: str) -> dict:
        path = tmp_path/name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("RUN_CONFIG.json", json.dumps(cfg))
        payload["source"].update({"config_digest": canonical_digest(cfg), "snapshot_path": str(path),
            "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "snapshot_file_hashes": {}})
        return runner.audit(payload)

    assert bind_and_audit(guided, guided_config, "guided.zip")["whole_training_jobs"] == 3
    assert len(assess(guided)["records"][0]["estimators"]) == 4
    static_config = copy.deepcopy(guided_config)
    static_config["static_only"] = True
    static_config["training_repetitions"] = 5
    static_result = runner.run(static_config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert static_result["potential_evaluations"] == 5*2*128
    static_audit = bind_and_audit(static_result, static_config, "static.zip")
    assert static_audit["whole_training_jobs"] == 0
    assert static_audit["fixed_r_evaluation_repetitions"] == 5
    assert all(not r["training"] for r in static_result["records"])
    assert assess(static_result)["replication_unit"] == "fixed_r_iid_evaluation"
    assert not assess(static_result)["all_risk_fits_stable"]
    config.pop("training_repetitions")
    corrupted = copy.deepcopy(result)
    corrupted["records"][0]["estimators"][0]["summary"]["relative_se"] *= 2
    with pytest.raises(ValueError, match="IID moment"):
        runner.audit(corrupted)
    config["budget"]["max_potential_evaluations"] = 1
    failure = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert failure["records"][0]["status"] == "unresolved"
    assert failure["records"][0]["wall_seconds_including_failure"] >= 0
