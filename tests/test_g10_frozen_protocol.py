"""Integrity tests for G10 calibration, validation, and stop-gate artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _canonical(payload: dict) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_g10_results_match_protocol_bytes_and_preserve_decisions() -> None:
    pairs = (
        (
            "configs/g10_control_span_development.yaml",
            "results/g10_control_span_development_v1_2026-07-19.json",
            True,
        ),
        (
            "configs/g10_control_span_correction_development.yaml",
            "results/g10_control_span_correction_development_v1_2026-07-19.json",
            True,
        ),
        (
            "configs/g10_control_span_calibration.yaml",
            "results/g10_control_span_calibration_v1_2026-07-19.json",
            True,
        ),
        (
            "configs/g10_control_span_frozen.yaml",
            "results/g10_control_span_frozen_v1_2026-07-19.json",
            False,
        ),
        (
            "configs/g10_rank_two_development.yaml",
            "results/g10_rank_two_development_v1_2026-07-19.json",
            False,
        ),
    )
    for config_path, result_path, expected_passed in pairs:
        expected_hash = hashlib.sha256((ROOT / config_path).read_bytes()).hexdigest()
        result = _json(result_path)
        assert result["protocol_sha256"] == expected_hash
        assert result["smoke"] is False
        assert result["passed"] is expected_passed


def test_g10_frozen_result_retains_failed_headline_and_passed_structure() -> None:
    result = _json("results/g10_control_span_frozen_v1_2026-07-19.json")
    assert result["aggregate"]["core_regimes"] == 12
    assert result["aggregate"]["stress_regimes"] == 6
    assert result["gates"]["geometric_total_work"] is False
    assert result["gates"]["total_work_lower_95"] is True
    assert result["gates"]["improved_core_regime_fraction"] is True
    assert result["gates"]["core_likelihood_pass_fraction"] is True
    assert result["gates"]["geometric_correction_work"] is True
    assert result["gates"]["core_exactness"] is True
    assert result["gates"]["stress_exactness"] is True
    assert result["aggregate"]["geometric_raw_over_marginalized_single_work_ratio"] == (
        pytest.approx(1.3346085352080408, rel=1e-12)
    )
    assert result["aggregate"]["geometric_raw_over_marginalized_correction_work_ratio"] == (
        pytest.approx(2.3953396730864966, rel=1e-12)
    )
    assert result["theory_contract"]["self_normalized"] is False


def test_every_g10_source_hash_is_canonical_and_current() -> None:
    for config_path in (
        "configs/g10_control_span_development.yaml",
        "configs/g10_control_span_correction_development.yaml",
        "configs/g10_control_span_calibration.yaml",
        "configs/g10_control_span_frozen.yaml",
        "configs/g10_rank_two_development.yaml",
    ):
        config = _yaml(config_path)
        sources = []
        for key in ("calibration_source", "selection_source", "reference_source"):
            if key in config:
                sources.append(config[key])
        sources.extend(config.get("calibration_sources", []))
        for source in sources:
            assert _canonical(_json(source["path"])) == source["canonical_json_sha256"]


def test_g10_calibration_and_validation_random_stream_seeds_are_disjoint() -> None:
    calibration = _yaml("configs/g10_control_span_calibration.yaml")
    frozen = _yaml("configs/g10_control_span_frozen.yaml")
    regime_count = 18
    alpha_count = len(calibration["mixture"]["natural_weight_candidates"])
    calibration_seed_count = len(calibration["evaluation"]["calibration_seeds"])
    level_count = len(frozen["hierarchy"]["correction_fine_steps"])
    validation_seed_count = len(frozen["validation"]["seeds"])

    calibration_simulation = {
        int(seed) + 100_000 * regime
        for regime in range(regime_count)
        for seed in calibration["evaluation"]["calibration_seeds"]
    }
    calibration_labels = {
        int(calibration["evaluation"]["label_seed_base"])
        + 100_000 * regime
        + 10_000 * alpha
        + seed_index
        for regime in range(regime_count)
        for alpha in range(alpha_count)
        for seed_index in range(calibration_seed_count)
    }
    validation_simulation = {
        int(seed) + 100_000 * regime
        for regime in range(regime_count)
        for seed in frozen["validation"]["seeds"]
    }
    single_labels = {
        int(frozen["validation"]["single_label_seed_base"])
        + 100_000 * regime
        + seed_index
        for regime in range(regime_count)
        for seed_index in range(validation_seed_count)
    }
    correction_labels = {
        int(frozen["validation"]["correction_label_seed_base"])
        + 100_000 * regime
        + 10_000 * seed_index
        + level
        for regime in range(regime_count)
        for seed_index in range(validation_seed_count)
        for level in range(level_count)
    }
    streams = (
        calibration_simulation,
        calibration_labels,
        validation_simulation,
        single_labels,
        correction_labels,
    )
    for left_index, left in enumerate(streams):
        assert len(left) > 0
        for right in streams[left_index + 1 :]:
            assert left.isdisjoint(right)


def test_frozen_natural_weights_came_only_from_declared_candidates() -> None:
    config = _yaml("configs/g10_control_span_calibration.yaml")
    result = _json("results/g10_control_span_calibration_v1_2026-07-19.json")
    candidates = {float(value) for value in config["mixture"]["natural_weight_candidates"]}
    assert result["passed_regimes"] == 18
    for regime in result["regimes"]:
        assert float(regime["selected_natural_weight"]) in candidates
        assert regime["passed"] is True
