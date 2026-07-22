"""Training-inclusive matched-work audit tests for V5 G6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v5_g6_efficiency_audit import audit


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    baseline_method = {"samples": 100, "variance": 0.04, "work_units": 1000.0}
    baseline = {
        "schema": "npi.g11.v5-baseline-qualification.v1",
        "baseline_qualified": True,
        "records": [
            {
                "model_id": "model",
                "task": "terminal",
                "cluster": 0,
                "cem": {"training_work_units": 100.0},
                "crude_mc": baseline_method,
                "pure_cem_slis": {**baseline_method, "work_units": 1100.0},
                "defensive_cem_mixture": {**baseline_method, "work_units": 1100.0},
            }
        ],
    }
    hybrid = {
        "schema": "npi.g11.v5-confirmatory.v1",
        "qualification_passed": True,
        "records": [
            {
                "cell_id": "model:terminal:cluster-0",
                "model_id": "model",
                "task_name": "terminal",
                "reference": {"probability": 0.01, "standard_error": 0.0001},
                "result": {
                    "complete": True,
                    "resource_censored": False,
                    "requested_relative_sampling_rmse": 0.1,
                    "selected_candidate": "start_4",
                    "work": {
                        "entries": [
                            {"role": "selection", "work_units": 100.0},
                            {"role": "final", "work_units": 100.0},
                        ]
                    },
                },
            }
        ],
    }
    return _write(tmp_path / "hybrid.json", hybrid), _write(
        tmp_path / "baseline.json", baseline
    )


def test_efficiency_audit_includes_training_and_matches_cells(tmp_path: Path) -> None:
    hybrid, baseline = _inputs(tmp_path)
    result = audit(hybrid, baseline, minimum_final_samples=2)
    record = result["records"][0]
    assert record["baseline_projections"]["pure_cem_slis"]["training_work"] == 100.0
    assert result["gates"]["complete_matched_matrix"]
    assert result["gates"]["all_cells_rare_at_most_5_percent"]
    assert result["g6_efficiency_passed"]


def test_efficiency_audit_rejects_missing_matched_baseline(tmp_path: Path) -> None:
    hybrid, baseline = _inputs(tmp_path)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["records"] = []
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing matched baseline"):
        audit(hybrid, baseline)
