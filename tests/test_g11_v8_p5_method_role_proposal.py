from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v8_p5_method_role_proposal import (
    RETAINED_REQUIREMENTS,
    UNRESOLVED_REQUIREMENTS,
    _defensive_weights,
    load_method_role_config,
    run_method_role_proposal,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_method_role_proposal_v1.yaml"


def test_method_role_config_partitions_all_eight_requirements() -> None:
    config, digest = load_method_role_config(CONFIG)
    retained = {
        (item["cell_id"], item["method"])
        for item in config["retained_requirements"]
    }
    unresolved = {
        (item["cell_id"], "raw_crosscheck")
        for item in config["raw_candidates"]
    } | {
        (item["cell_id"], "dcs_reference")
        for item in config["dcs_candidates"]
    }
    assert retained == RETAINED_REQUIREMENTS
    assert unresolved == UNRESOLVED_REQUIREMENTS
    assert retained.isdisjoint(unresolved)
    assert len(retained | unresolved) == 8
    assert len(digest) == 64


def test_defensive_reweighting_preserves_positive_simplex() -> None:
    weights = _defensive_weights([0.08, 0.2, 0.3, 0.42], 0.20)
    assert weights[0] == pytest.approx(0.20)
    assert sum(weights) == pytest.approx(1.0)
    assert all(weight > 0.0 for weight in weights)
    assert 1.0 / weights[0] == pytest.approx(5.0)


def test_method_role_smoke_uses_distinct_precision_targets() -> None:
    result = run_method_role_proposal(CONFIG, smoke=True)
    seeds = [record["seed"] for record in result["seed_records"]]
    by_method = {candidate["method"]: candidate for candidate in result["candidates"]}
    raw = by_method["raw_crosscheck"]
    dcs = by_method["dcs_reference"]
    assert raw["target_standard_error"] == pytest.approx(5e-6)
    assert dcs["target_standard_error"] == pytest.approx(2e-7)
    assert result["method_role_precision"][
        "raw_crosscheck_relative_standard_error_target"
    ] == 0.05
    assert result["method_role_precision"][
        "dcs_relative_standard_error_target"
    ] == 0.02
    assert len(seeds) == 12
    assert len(seeds) == len(set(seeds))
    assert result["decision"]["proposal_manifest_build_authorized"] is False
    assert result["decision"]["new_formal_pilot_authorized"] is False
    assert result["decision"]["final_execution_authorized"] is False
    json.dumps(result, allow_nan=False)
