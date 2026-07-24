"""Prespecified secondary accuracy co-gate tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.g11_v6_accuracy_analysis import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "g11_v6"
    / "secondary_accuracy_qualification_v1.yaml"
)


def test_secondary_accuracy_uses_method_cell_aggregate_gates(
    tmp_path: Path,
) -> None:
    records = []
    for method in ("fixed_dcs_slis", "fixed_raw_defensive"):
        for cluster in range(24):
            records.append(
                {
                    "cell_id": "cell",
                    "cluster": cluster,
                    "method": method,
                    "nominal_probability": 0.01,
                    "reference_probability": 0.0101,
                    "reference_standard_error": 1e-5,
                    "result": {
                        "core": {
                            "requested_relative_sampling_rmse": 0.2,
                            "estimate": 0.0101,
                            "empirical_target_attained": True,
                        }
                    },
                }
            )
    source = {
        "schema": "npi.g11.v6-secondary-baselines.v1",
        "smoke": False,
        "secondary_baselines_qualified": True,
        "records": records,
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "source_artifact_sha256": source_hash,
                "qualification_audit_passed": True,
            }
        ),
        encoding="utf-8",
    )
    result = run(CONFIG, source_path, audit_path)
    assert len(result["accuracy"]) == 2
    assert result["gates"]["all_accuracy_co_gates"]
    assert result["gates"]["source_operationally_qualified"]
    assert result["gates"]["independent_audit_matches_source"]
