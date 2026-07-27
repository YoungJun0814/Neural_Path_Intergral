from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from experiments.g11_v8_p5_reference import _update_moments_from_batch, load_reference_config
from src.path_integral import OnlineMoments

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v8" / "p5_independent_reference_execution_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "g11_v8" / "p5_independent_reference_execution_v2.yaml"


def test_p5_reference_config_is_strict_and_hash_bound() -> None:
    config, digest = load_reference_config(CONFIG)
    assert config["reference_contract"]["methods"] == ["dcs_reference", "raw_crosscheck"]
    assert config["reference_seed_namespace"] != config["final_method_seed_namespace"]
    assert len(digest) == 64


def test_p5_reference_v2_uses_pre_final_conservative_allocation() -> None:
    config, digest = load_reference_config(CONFIG_V2)
    sampling = config["sampling"]
    assert config["reference_seed_namespace"] != config["final_method_seed_namespace"]
    assert sampling["allocation_variance_statistic"] == "maximum_replicate_variance"
    assert sampling["allocation_safety_factor"] >= 1.0
    assert len(digest) == 64


def test_p5_reference_v2_rejects_undisclosed_outcome_use(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_V2.read_text(encoding="utf-8"))
    payload["outcome_data_used"] = False
    path = tmp_path / "bad-v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="disclose"):
        load_reference_config(path)


def test_p5_reference_batch_moments_preserve_mean_and_unbiased_variance() -> None:
    values = torch.tensor([0.25, -0.75, 1.5, 2.0, -1.0], dtype=torch.float64)
    moments = OnlineMoments()
    _update_moments_from_batch(moments, values[:2])
    _update_moments_from_batch(moments, values[2:])
    assert moments.count == values.numel()
    assert moments.mean == pytest.approx(float(values.mean()))
    assert moments.variance == pytest.approx(float(torch.var(values, unbiased=True)))


def test_p5_reference_config_rejects_changed_method_roster(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["reference_contract"]["methods"] = ["dcs_reference"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="methods"):
        load_reference_config(path)
