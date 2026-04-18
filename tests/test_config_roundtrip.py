"""configs/default.yaml must parse and contain all keys required by the CLI."""
from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_yaml_parses():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)


def test_required_top_level_sections():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("seed", "heston", "rbergomi", "girsanov", "simulate", "calibrate", "train_ipm"):
        assert key in cfg, f"missing top-level key: {key}"


def test_heston_has_required_params():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    heston = cfg["heston"]
    for key in ("mu", "kappa", "theta", "xi", "rho", "jump_lambda", "jump_mean", "jump_std"):
        assert key in heston, f"heston missing: {key}"


def test_rbergomi_has_required_params():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rb = cfg["rbergomi"]
    for key in ("H", "eta", "xi", "rho"):
        assert key in rb, f"rbergomi missing: {key}"
    assert 0.0 < rb["H"] < 0.5


def test_girsanov_flag_present():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    g = cfg["girsanov"]
    assert "u_bound" in g
    assert "apply_v_drift_correction" in g


def test_train_ipm_has_data_driven_kurtosis_key():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    t = cfg["train_ipm"]
    # null means compute from data; we explicitly allow it
    assert "target_kurtosis" in t
