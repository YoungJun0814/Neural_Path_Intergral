"""Fixed-grid runner smoke and statistical-unit semantics."""

import math

import pytest
import torch
import yaml

from experiments.post_audit_v2_overnight_reference_recovery import (
    ROOT,
    TOYS,
    seed,
    smc_unit,
    validate_config,
)


def test_fixed_config_and_namespaces():
    config = yaml.safe_load((ROOT/"configs/post_audit/overnight_reference_recovery_v1.yaml").read_text())
    validate_config(config)
    config["unit_count"] += 1
    with pytest.raises(ValueError):
        validate_config(config)
    values = [seed(f"method-{i}", stream) for i in range(100) for stream in ("path", "label", "pair")]
    assert len(values) == len(set(values))


def test_island_uses_arithmetic_normalizer_mean():
    config = yaml.safe_load((ROOT/"configs/post_audit/overnight_reference_recovery_v1.yaml").read_text())
    config["smc"]["levels"] = 4
    # smc_unit fixed work validator assumes the preregistered 64 points.
    config["smc"]["levels"] = 64
    torch.set_num_threads(1)
    result = smc_unit(TOYS["centered"].log_value, dimension=2, identity="unit-test-only",
                      islands=4, config=config, toy=True)
    expected = sum(math.exp(i["log_normalizer"]) for i in result["islands"])/4
    assert math.isclose(math.exp(result["log_estimate"]), expected, rel_tol=1e-13)
    assert result["potential_calls"] == 512*63
    assert result["inference_unit"] == "aggregate_whole_run"
    assert len(result["seeds"]) == len(set(result["seeds"])) == 4
    assert len(result["islands"][0]["snapshots"]) == 190
