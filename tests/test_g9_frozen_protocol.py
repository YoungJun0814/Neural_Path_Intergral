"""Integrity checks for the frozen G9 calibration and falsification artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest
import yaml

from experiments.g9_mgvs_frozen import _canonical_json_sha256

ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_frozen_result_matches_protocol_hash_and_preserves_failed_decision() -> None:
    config_path = ROOT / "configs/g9_mgvs_frozen.yaml"
    result = _json("results/g9_mgvs_frozen_v3_2026-07-18.json")
    assert result["protocol_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert result["smoke"] is False
    assert result["passed"] is False
    assert result["gates"]["geometric_correction_work"] is True
    assert result["gates"]["geometric_total_work"] is False
    assert result["gates"]["core_regime_fraction"] is False
    assert result["gates"]["core_exactness"] is True
    assert result["gates"]["stress_exactness"] is True
    assert result["aggregate"]["core_regimes"] == 12
    assert result["aggregate"]["stress_regimes"] == 6


def test_aggregate_interval_clusters_the_fixed_suite_by_validation_seed() -> None:
    result = _json("results/g9_mgvs_frozen_v3_2026-07-18.json")
    core = [regime for regime in result["regimes"] if regime["group"] == "core"]
    clustered = []
    for seed_index in range(len(result["validation_seeds"])):
        log_ratios = [
            math.log(float(regime["runs"][seed_index]["raw_over_smoothed_single_work_ratio"]))
            for regime in core
        ]
        clustered.append(statistics.mean(log_ratios))
    lower = math.exp(
        statistics.mean(clustered)
        - 1.833 * statistics.stdev(clustered) / math.sqrt(len(clustered))
    )
    assert result["aggregate"]["total_work_ratio_lower_95_one_sided"] == pytest.approx(
        lower, rel=1e-12
    )
    assert result["aggregate"]["regime_heterogeneity_sensitivity_lower_95_one_sided"] < 1.0


def test_every_frozen_source_is_verified_by_canonical_json_hash() -> None:
    config = _yaml("configs/g9_mgvs_frozen.yaml")
    sources = [*config["calibration_sources"], config["direction_source"]]
    sources.extend(config["legacy_baselines"].values())
    for source in sources:
        payload = _json(source["path"])
        assert _canonical_json_sha256(payload) == source["canonical_json_sha256"]


def test_all_protocol_results_match_their_config_bytes() -> None:
    pairs = (
        ("configs/g9_mgvs_calibration.yaml", "results/g9_mgvs_calibration_2026-07-18.json"),
        (
            "configs/g9_mgvs_stress_calibration.yaml",
            "results/g9_mgvs_stress_calibration_2026-07-18.json",
        ),
        (
            "configs/g9_direction_calibration.yaml",
            "results/g9_direction_calibration_2026-07-18.json",
        ),
    )
    for config_path, result_path in pairs:
        expected = hashlib.sha256((ROOT / config_path).read_bytes()).hexdigest()
        assert _json(result_path)["protocol_sha256"] == expected


def test_calibration_reference_and_validation_seeds_are_disjoint() -> None:
    frozen = _yaml("configs/g9_mgvs_frozen.yaml")
    direction = _yaml("configs/g9_direction_calibration.yaml")
    calibration_seeds: set[int] = set()
    regime_count = 0
    for source in frozen["calibration_sources"]:
        payload = _json(source["path"])
        assert payload["passed"] is True
        for regime in payload["regimes"]:
            calibration_seeds.add(int(regime["training_seed"]))
            calibration_seeds.add(int(regime["validation_seed"]))
            regime_count += 1
    direction_seeds = {
        int(direction["evaluation"]["seed_base"]) + 1_000 * regime + offset
        for regime in range(regime_count)
        for offset in (*range(len(frozen["hierarchy"]["fine_steps"])), 900)
    }
    reference_seeds = {
        int(frozen["reference"]["seed_base"]) + regime for regime in range(regime_count)
    }
    validation_seeds = {int(value) for value in frozen["seeds"]["validation"]}
    assert calibration_seeds.isdisjoint(direction_seeds)
    assert calibration_seeds.isdisjoint(reference_seeds)
    assert calibration_seeds.isdisjoint(validation_seeds)
    assert direction_seeds.isdisjoint(reference_seeds)
    assert direction_seeds.isdisjoint(validation_seeds)
    assert reference_seeds.isdisjoint(validation_seeds)


def test_direction_selection_is_frozen_and_strictly_from_declared_candidates() -> None:
    config = _yaml("configs/g9_direction_calibration.yaml")
    result = _json("results/g9_direction_calibration_2026-07-18.json")
    candidates = {float(value) for value in config["direction_family"]["decay_candidates"]}
    assert result["passed"] is True
    assert result["total_regimes"] == 18
    for regime in result["regimes"]:
        for selection in regime["selections"].values():
            assert float(selection["selected_decay"]) in candidates
            assert float(selection["selected_over_flat_variance"]) <= 1.0 + 1e-12
