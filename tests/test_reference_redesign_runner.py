"""Failure/role/streaming/allocation regression for the new reference runner."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from experiments import post_audit_v2_reference_redesign as runner
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.structural_v2_reference import log_moments
from tests.test_reference_redesign_core import mixture


def config() -> dict:
    return yaml.safe_load((runner.ROOT / "configs/post_audit/reference_redesign_micro_v1.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def toy(monkeypatch):
    cfg = config()
    cfg.update(outer_batches=2, outer_batch_size=32, inner_counts=[1, 4], whole_replicates=3)
    params = proposal_parameters(mixture(8))
    rows = [{"cell": {"id": name, "threshold": 80.}, "parent_training_rep": rep,
             "q_parameters": params, "q_digest": canonical_digest(params)}
            for name in ("canonical", "high") for rep in range(5)]
    metadata = {"config": {"model": {"spot": 100., "maturity": 1., "hurst": .1,
                                    "eta": 1.5, "xi": .04, "rho": -.7}, "smc": {"steps": 4}}}
    monkeypatch.setattr(runner, "validate_config", lambda cfg: None)
    monkeypatch.setattr(runner, "inputs", lambda cfg: (metadata, rows))
    monkeypatch.setattr(runner, "audit_snapshot", lambda *args: None)
    monkeypatch.setattr(runner, "file_sha256", lambda path: ("a" * 64, 0))
    payload = runner.run(cfg, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    return cfg, payload


def test_micro_replay_audit_and_original_q_binding(toy) -> None:
    _, payload = toy
    assert payload["status"] == "completed_design_pilot"
    assert not payload["model_q_changed"]
    assert not payload["qualification"]["p2_authorized"]
    assert runner.audit(payload)["records"] == 4
    risk = [r for r in payload["records"] if r["estimand"] == "risk"]
    assert all(r["measurement_contract"]["se_unit"] == "iid_outer_draw" for r in risk)
    torch.set_num_threads(2)
    assert runner.audit(payload)["records"] == 4
    assert torch.get_num_threads() == 1
    changed = copy.deepcopy(payload)
    changed["records"][0]["candidates"][0]["summary"]["relative_se"] = 0
    with pytest.raises(ValueError, match="summary"):
        runner.audit(changed)
    changed = copy.deepcopy(payload)
    changed["sample_uses"].pop()
    with pytest.raises(ValueError):
        runner.audit(changed)


def test_allocation_does_not_authorize_no_gain(toy) -> None:
    _, micro = toy
    kernel = copy.deepcopy(micro)
    kernel["records"] = []
    baseline = copy.deepcopy(micro)
    baseline["records"] = []
    allocated = runner.production_allocation(micro, kernel, baseline, bindings=[])
    assert allocated["production_config"] is None
    assert not allocated["p2_authorized"]


def test_budget_failure_aborts_bundle_not_selective_replacement(toy, monkeypatch) -> None:
    cfg, _ = toy
    cfg["budget"]["max_potential_evaluations"] = 10
    failed = runner.run(cfg, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    assert failed["status"] == "protocol_failure"
    assert failed["records"][0]["status"] == "protocol_failure"
    assert all(r["status"] == "not_run_after_protocol_failure" for r in failed["records"][1:])
    assert not failed["qualification"]["p2_authorized"]


def test_fixed_contract_rejects_unsealed_production_and_tuning() -> None:
    cfg = config()
    runner.validate_config(cfg)
    for field, value in (("inner_counts", [1, 2]), ("pilot_parent_rep", 1), ("mode", "production")):
        changed = copy.deepcopy(cfg)
        changed[field] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed)


def test_production_streams_batch_moments_not_all_final_samples(toy, monkeypatch) -> None:
    cfg, payload = toy
    monkeypatch.setattr(runner, "verify_production", lambda cfg: None)
    cfg.update(mode="production", production_batch_size=32,
               production_jobs=[{"cell_id": "canonical", "parent_training_rep": 0,
                    "estimand": "risk", "method": "nested", "count": 64, "inner": 4, "block": None}])
    produced = runner.run(cfg, {"source_tree_digest": source_tree_digest(runner.ROOT)})
    record = produced["records"][0]
    assert record["status"] == "completed"
    assert len(record["batch_moments"]) == 2
    assert "log_unit_contributions" not in record
    assert record["summary"]["count"] == 64
    assert not produced["qualification"]["p2_authorized"]
    assert runner.audit(produced)["records"] == 1
    assert payload["status"] == "completed_design_pilot"


def test_allocation_builds_full_grid_and_blocks_over_budget(toy) -> None:
    _, micro = toy
    micro["selection"] = {"selected": {"block": None, "inner": 4, "worst_cv2_wall_per_outer": .1},
                          "candidates": [{"worst_cv2_wall_per_outer": .1}]}
    for r in micro["records"]:
        for c in r["candidates"]:
            c["single_pass_wall_seconds"] = .001
    kernel = copy.deepcopy(micro)
    summary = log_moments(torch.tensor([0., .01, -.01, .01, 0., 0., -.01, 0.], dtype=torch.float64))
    kernel["records"] = [{"cell_id": cell, "parent_training_rep": 0, "estimand": kind,
        "method": "elliptical_slice", "status": "completed", "summary": summary,
        "wall_seconds": .01, "counters": {"potential_equivalent_evaluations": 100}}
        for cell in ("canonical", "high") for kind in ("risk", "probability")]
    baseline = copy.deepcopy(micro)
    baseline["records"] = [{"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
        "estimand": "risk", "method": "static-iid", "status": "completed", "summary": summary,
        "wall_seconds": .001} for row in micro["frozen_qs"]]
    allocated = runner.production_allocation(micro, kernel, baseline, bindings=[])
    assert len(allocated["production_jobs"]) == 34
    assert allocated["status"] == "allocated"
    assert allocated["production_config"] is not None
    for r in kernel["records"]:
        r["wall_seconds"] = 1e6
    failed = runner.production_allocation(micro, kernel, baseline, bindings=[])
    assert "production_wall_cap" in failed["failure_reasons"]
    assert failed["production_config"] is None


def test_sealed_allocation_cannot_be_edited(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "file_sha256", lambda path: ("a" * 64, 0))
    cfg = {"mode": "production", "pilot_bindings": [], "production_jobs": [1],
           "production_jobs_digest": canonical_digest([2]), "allocation_path": "missing"}
    with pytest.raises(ValueError, match="allocation changed"):
        runner.verify_production(cfg)
