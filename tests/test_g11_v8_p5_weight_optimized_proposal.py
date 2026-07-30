from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.g11_v8_p5_weight_optimized_proposal import (
    RETAINED_REQUIREMENTS,
    UNRESOLVED_REQUIREMENTS,
    _floored_weights,
    _raw_second_moment_objective,
    load_weight_optimized_config,
    run_weight_optimized_proposal,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_weight_optimized_proposal_v1.yaml"


def test_weight_optimized_config_binds_complete_requirement_partition() -> None:
    config, digest = load_weight_optimized_config(CONFIG)
    retained = {
        (item["cell_id"], item["method"])
        for item in config["retained_requirements"]
    }
    unresolved = {
        (item["cell_id"], "raw_crosscheck")
        for item in config["raw_weight_optimization"]
    } | {
        (item["cell_id"], "dcs_reference")
        for item in config["dcs_candidate_grids"]
    }
    assert retained == RETAINED_REQUIREMENTS
    assert unresolved == UNRESOLVED_REQUIREMENTS
    assert retained.isdisjoint(unresolved)
    assert len(retained | unresolved) == 8
    assert len(digest) == 64


def test_floored_weights_preserve_defensive_natural_component() -> None:
    logits = torch.tensor([-2.0, 0.0, 1.0], dtype=torch.float64)
    weights = _floored_weights(
        logits,
        natural_weight=0.08,
        minimum_nonnatural_weight=0.005,
    )
    assert float(weights[0]) == pytest.approx(0.08)
    assert float(torch.sum(weights)) == pytest.approx(1.0)
    assert bool((weights[1:] >= 0.005).all())
    assert 1.0 / float(weights[0]) == pytest.approx(12.5)


def test_off_policy_objective_matches_direct_constant_density_case() -> None:
    logits = torch.zeros(2, dtype=torch.float64)
    component_log = torch.zeros((4, 3), dtype=torch.float64)
    base_log = torch.zeros(4, dtype=torch.float64)
    event = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    objective = _raw_second_moment_objective(
        logits,
        component_log_q_over_p=component_log,
        base_log_q_over_p=base_log,
        hard_event=event,
        natural_weight=0.08,
        minimum_nonnatural_weight=0.005,
    )
    assert float(objective) == pytest.approx(0.5)


def test_weight_optimized_smoke_is_exact_and_fail_closed() -> None:
    result = run_weight_optimized_proposal(CONFIG, smoke=True)
    seeds = [record["seed"] for record in result["seed_records"]]
    assert result["schema"].endswith("result.v1")
    assert result["raw_weight_theory"]["self_normalized"] is False
    assert result["dcs_weight_theory"]["raw_off_policy_optimizer_applied"] is False
    assert result["weight_fits"]
    assert all(
        fit["optimized_empirical_second_moment"]
        <= fit["base_empirical_second_moment"] * (1.0 + 1e-12)
        for fit in result["weight_fits"]
    )
    assert result["validation_design"]["block_partition"].startswith(
        "independent_seeded"
    )
    assert len(seeds) == len(set(seeds))
    assert result["decision"]["proposal_manifest_build_authorized"] is False
    assert result["decision"]["new_formal_pilot_authorized"] is False
    assert result["decision"]["final_execution_authorized"] is False
    json.dumps(result, allow_nan=False)
