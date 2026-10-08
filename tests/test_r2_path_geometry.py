"""Held-out geometry calibration, omitted-region toy and stable-risk protocol."""

import copy
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.path_integral.path_geometry_diagnostics import calibrate_path_geometry
from src.path_integral.research_result_contract import source_tree_digest


def test_independent_geometry_calibration_and_omitted_region() -> None:
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(215)
    scales = torch.tensor([4., .1, .1, .1, .1, .1, .1, .1], dtype=torch.float64)
    train = torch.randn((4096, 8), generator=gen, dtype=torch.float64)*scales
    cal = torch.randn((4096, 8), generator=gen, dtype=torch.float64)*scales
    geometry = calibrate_path_geometry(train, cal, torch.zeros((1, 8), dtype=torch.float64), rank=2)
    held = torch.randn((10000, 8), generator=gen, dtype=torch.float64)*scales
    fraction = float(geometry.outside(held).double().mean())
    assert .01 < fraction < .08  # sanity check, not a conditional coverage theorem
    shifted = held.clone()
    shifted[:, -1] += 8
    assert float(geometry.outside(shifted).double().mean()) > .99
    along = held.clone()
    along[:, 0] += 40
    assert float(geometry.outside(along).double().mean()) > .99
    assert torch.allclose(geometry.directions.T@geometry.directions,
                          torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_calibration_uses_upper_order_statistic_without_interpolation() -> None:
    train = torch.tensor([[-2., 0.], [2., 0.], [-1., .1], [1., -.1]], dtype=torch.float64)
    cal = torch.linspace(-3, 3, 99, dtype=torch.float64)[:, None].repeat(1, 2)
    geometry = calibrate_path_geometry(train, cal, torch.zeros((1, 2), dtype=torch.float64),
                                       rank=1, outside_probability=.1)
    distance, complement = geometry.scores(cal)
    # ceil((99+1)*(.95)) = 95; one-based upper order statistic.
    assert geometry.distance_threshold == float(distance.sort().values[94])
    assert geometry.complement_threshold == float(complement.sort().values[94])


@pytest.mark.parametrize("rank,alpha", [(2, .05), (0, .05), (1, .0001)])
def test_invalid_geometry_contract_rejected(rank: int, alpha: float) -> None:
    x = torch.zeros((100, 2), dtype=torch.float64)
    with pytest.raises(ValueError):
        calibrate_path_geometry(x, x.clone(), x[:1], rank=rank, outside_probability=alpha)


def test_restricted_smc_normalizer_is_raw_weighted_integral_not_particle_mean() -> None:
    from src.path_integral.weighted_tempered_smc import (
        WeightedSMCConfig,
        estimate_weighted_tempered_normalizer,
    )

    def logh(x: torch.Tensor) -> torch.Tensor:
        return torch.special.log_ndtr(x[:, 0])

    smc = estimate_weighted_tempered_normalizer(logh, dimension=1,
        config=WeightedSMCConfig(particles=128, temperatures=(0., .2, .6, 1.),
            mutation_steps=0, pcn_scale=.35, replicates=1, seed=815,
            resample_every=100, retain_final_particles=True))
    assert smc.final_particles is not None and smc.final_weights is not None
    selected = smc.final_particles[:, 0] > 0
    restricted = smc.mean*float(smc.final_weights[selected].sum())
    raw_prior_mc = float(torch.exp(logh(smc.final_particles))[selected].sum()/128)
    assert restricted == pytest.approx(raw_prior_mc, rel=1e-12)


def test_risk_geometry_runner_roles_and_equal_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments import post_audit_r2_risk_geometry as runner
    from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
    from src.path_integral.finite_rank_gaussian_transport import (
        DefensiveFiniteRankGaussianMixture,
        FiniteRankGaussianComponent,
    )
    from src.path_integral.research_result_contract import canonical_digest

    shifted = FiniteRankGaussianComponent(torch.tensor([1., 0.], dtype=torch.float64),
        torch.empty((2, 0), dtype=torch.float64), torch.empty(0, dtype=torch.float64))
    q = DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(2), shifted),
                                           torch.tensor([.1, .9], dtype=torch.float64))
    params = proposal_parameters(q)
    artifact = tmp_path/"parent.json"
    artifact.write_text(json.dumps({"config": {"model": {}, "smc": {"steps": 1}}, "records": [
        {"cell": {"id": "toy"}, "parent_training_rep": 0, "candidates": [
            {"method": "parent-as-is", "proposal_parameters": params, "proposal_digest": canonical_digest(params)}]}]}))
    config = {"torch_threads": 1, "proposal_artifact_path": str(artifact), "method": "parent-as-is",
        "geometry": {"training_count": 128, "calibration_count": 128, "rank": 1, "outside_probability": .1},
        "direct": {"count": 128, "batch_size": 64},
        "risk": {"particles": 16, "replicates": 4, "levels": 3, "mutation_steps": 1, "maximum_relative_se": .2},
        "candidates": [{"id": name, "bridge_power": power, "resample_every": every,
                        "resampling_scheme": "stratified", "pcn_scale": .35}
                       for name, power, every in [("a", 1, 1), ("b", 2, 1), ("c", 2, 2)]],
        "budget": {"max_potential_evaluations": 5000, "max_wall_seconds": 60},
        "decision": {"required_precision_passes_per_cell": 1, "minimum_precision_designs_for_crosscheck": 2,
                     "maximum_relative_normalizer_difference": .25}}
    monkeypatch.setattr(runner, "_problem", lambda *args: SimpleNamespace(task_id="toy", local_dimension=2))
    monkeypatch.setattr(runner, "_log_potential", lambda problem, x: torch.special.log_ndtr(x[:, 0]))
    result = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert {x["potential_evaluations"] for x in result["records"][0]["candidates"]} == {128+4*48}
    roles = {s["key"]["role"] for s in result["seed_ledger"]["records"]}
    assert roles == {"geometry-training", "geometry-calibration", "direct-final", "independent-risk"}
    assert all(0 <= c["outside_risk_m2_share"] <= 1+1e-12 for c in result["records"][0]["candidates"])
    archive = tmp_path/"source.zip"
    content = b"toy geometry source"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("toy.py", content)
        z.writestr("RUN_CONFIG.json", json.dumps(config))
    result["source"].update({"config_digest": canonical_digest(config), "snapshot_path": str(archive),
        "snapshot_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "snapshot_file_hashes": {"toy.py": hashlib.sha256(content).hexdigest()}})
    assert runner.audit(result)["parents"] == 1
    corrupted = copy.deepcopy(result)
    corrupted["records"][0]["geometry"]["distance_threshold"] *= 2
    with pytest.raises(ValueError, match="threshold replay"):
        runner.audit(corrupted)
    corrupted = copy.deepcopy(result)
    corrupted["decisions"] = []
    with pytest.raises(ValueError, match="cell decision"):
        runner.audit(corrupted)
    config["budget"]["max_potential_evaluations"] = 1
    failure = runner.run(config, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert all(c["status"] == "unresolved" for c in failure["records"][0]["candidates"])
    assert not failure["decisions"][0]["correction_experiment_ready"]
