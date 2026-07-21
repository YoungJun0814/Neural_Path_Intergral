"""V4 parameter-separated crossover qualification tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.g11_v4_crossover_qualification import (
    _cell_result,
    _load,
    _load_cells,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_v4_matrix_uses_parameter_separated_valid_calibration_cells() -> None:
    config, config_hash = _load(ROOT / "configs" / "g11_v4_crossover_qualification.yaml")
    cells, inputs = _load_cells(config)
    assert len(config_hash) == 64
    assert len(cells) == 27
    assert len(inputs) == 14
    assert {cell["changed_parameter"] for cell in cells} == {"base", "H", "eta", "rho"}
    assert sum(cell["cem_enabled"] for cell in cells) == 4
    assert not any(
        cell["regime"] == "oat_rho_m085" and cell["task_id"] == "barrier_1e-06" for cell in cells
    )


def test_small_cell_profile_compares_all_dcs_start_levels() -> None:
    config, _config_hash = _load(ROOT / "configs" / "g11_v4_crossover_qualification.yaml")
    cells, _inputs = _load_cells(config)
    small = deepcopy(config)
    small["sampling"] = {"paths_per_profile": 32, "repetitions": 1, "engine": "fft"}
    cell = deepcopy(cells[0])
    cell["cem_enabled"] = False
    result = _cell_result(small, cell)
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert len(run["dcs_single_levels"]) == 5
    assert len(run["dcs_corrections"]) == 4
    assert set(run["decisions"]) == {"0.10", "0.20", "0.30"}
    assert result["seed_count"] == 20
