from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_p5_production_scale_proposal import (
    DCS_METHOD,
    EXPECTED_REQUIREMENTS,
    RAW_METHOD,
    _dcs_rank_one_mixture,
    _finite_or_none,
    _rank_one_diagnostic,
    _raw_full_rank_mixture,
    load_production_scale_config,
    run_production_scale_proposal,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_production_scale_proposal_v1.yaml"
RECOVERY_CONFIG = (
    ROOT / "configs/g11_v8/p5_production_scale_proposal_v2.yaml"
)


def test_production_scale_config_binds_exact_failed_requirement_roster() -> None:
    config, digest = load_production_scale_config(CONFIG)
    recovery, recovery_digest = load_production_scale_config(RECOVERY_CONFIG)
    actual = {
        (cell["cell_id"], method)
        for cell in config["cells"]
        for method in cell["target_methods"]
    }
    assert actual == EXPECTED_REQUIREMENTS
    assert len(digest) == 64
    assert len(recovery_digest) == 64
    assert recovery["training_namespace"].endswith("-v2")
    assert recovery["validation_namespace"].endswith("-v2")
    assert recovery["prior_execution_failure"]["sha256"]
    assert config["decision"]["new_formal_pilot_authorized"] is False
    assert config["decision"]["final_execution_authorized"] is False


def test_production_scale_config_rejects_namespace_reuse(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["validation_namespace"] = config["training_namespace"]
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance or target roster"):
        load_production_scale_config(changed)


def test_raw_profile_mixture_allows_full_rank_but_retains_natural_bound() -> None:
    profiles: list[tuple[tuple[float, float], ...]] = [
        ((1.0, -1.0), (2.0, -2.0)),
        ((-1.0, -2.0), (1.0, -1.0)),
        ((0.5, -1.5), (-0.5, -0.5)),
    ]
    schedules, weights = _raw_full_rank_mixture(
        profiles, scales=[0.5, 1.0], natural_weight=0.08
    )
    diagnostic = _rank_one_diagnostic(schedules, maturity=1.0, steps=8)
    assert len(schedules) == 7
    assert all(value == 0.0 for pair in schedules[0] for value in pair)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.08)
    assert diagnostic["rank_one_price_span"] is False
    assert 1.0 / weights[0] == pytest.approx(12.5)


def test_dcs_profile_mixture_is_exactly_rank_one() -> None:
    profile = ((1.0, -1.0), (2.0, -2.0))
    schedules, weights = _dcs_rank_one_mixture(
        profile,
        scales=[0.0, 0.5, 1.0],
        weights=[0.2, 0.3, 0.5],
    )
    diagnostic = _rank_one_diagnostic(schedules, maturity=1.0, steps=8)
    assert diagnostic["rank_one_price_span"] is True
    assert sum(weights) == pytest.approx(1.0)


def test_production_scale_smoke_is_seed_disjoint_and_fail_closed() -> None:
    result = run_production_scale_proposal(RECOVERY_CONFIG, smoke=True)
    seeds = [record["seed"] for record in result["seed_records"]]
    assert result["schema"].endswith("result.v2")
    assert result["required_selection_count"] == 1
    assert result["candidates"]
    assert {candidate["method"] for candidate in result["candidates"]} == {
        RAW_METHOD
    }
    assert DCS_METHOD not in {
        candidate["method"] for candidate in result["candidates"]
    }
    assert len(seeds) == len(set(seeds))
    assert result["decision"]["selected_proposals_frozen"] is False
    assert result["decision"]["new_formal_pilot_authorized"] is False
    assert result["decision"]["final_execution_authorized"] is False
    assert result["decision"]["performance_claim_authorized"] is False
    json.dumps(result, allow_nan=False)


def test_undefined_diagnostic_is_json_null_not_nonfinite() -> None:
    assert _finite_or_none(math.inf) is None
    assert _finite_or_none(-math.inf) is None
    assert _finite_or_none(math.nan) is None
    assert _finite_or_none(1.25) == 1.25
