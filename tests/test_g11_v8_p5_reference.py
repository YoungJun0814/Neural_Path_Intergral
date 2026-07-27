from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_p5_reference import load_reference_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v8" / "p5_independent_reference_execution_v1.yaml"


def test_p5_reference_config_is_strict_and_hash_bound() -> None:
    config, digest = load_reference_config(CONFIG)
    assert config["reference_contract"]["methods"] == ["dcs_reference", "raw_crosscheck"]
    assert config["reference_seed_namespace"] != config["final_method_seed_namespace"]
    assert len(digest) == 64


def test_p5_reference_config_rejects_changed_method_roster(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["reference_contract"]["methods"] = ["dcs_reference"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="methods"):
        load_reference_config(path)
