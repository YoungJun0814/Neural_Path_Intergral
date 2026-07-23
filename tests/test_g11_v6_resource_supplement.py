"""V6 resource-supplement tests."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.g11_v6_resource_supplement import summarize_resources


def test_resource_supplement_aggregates_wall_cpu_and_peak(tmp_path: Path) -> None:
    source = {
        "schema": "npi.g11.v6-secondary-baselines.v1",
        "environment": {"torch_threads": 4},
        "records": [
            {
                "cell_id": "cell",
                "cluster": 0,
                "method": "fixed_dcs_slis",
                "result": {
                    "total_work": {
                        "records": [
                            {
                                "wall_seconds": 1.0,
                                "cpu_seconds": 2.0,
                                "peak_memory_bytes": 1024,
                                "work_units": 10.0,
                            },
                            {
                                "wall_seconds": 3.0,
                                "cpu_seconds": 4.0,
                                "peak_memory_bytes": 4096,
                                "work_units": 20.0,
                            },
                        ]
                    }
                },
            }
        ],
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    result = summarize_resources(path)
    summary = result["method_summaries"]["fixed_dcs_slis"]
    assert summary["total_wall_seconds"] == 4.0
    assert summary["total_cpu_seconds"] == 6.0
    assert summary["maximum_peak_resident_memory_bytes"] == 4096
    assert result["passed"]
