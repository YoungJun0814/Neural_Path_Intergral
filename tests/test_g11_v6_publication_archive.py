"""Deterministic V6 publication-archive tests."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import yaml

from experiments.g11_v6_publication_archive import build_publication_archive


def _source(path: Path) -> dict:
    controls = {
        "terminal_left_tail": [[1.0, -2.0], [3.0, -4.0]],
        "discrete_lower_barrier": [[2.0, -1.0], [4.0, -3.0]],
    }
    records = []
    for index, (task, control) in enumerate(controls.items()):
        records.append(
            {
                "method": "pure_cem",
                "cem_fit": {"control": control},
                "preparation": {"core": {"task": task}},
                "result": {
                    "core": {"complete": True},
                    "total_work": {
                        "records": [
                            {
                                "category": "proposal_training",
                                "samples": 10,
                                "work_units": 20.0 + index,
                                "wall_seconds": 1.0 + index,
                                "cpu_seconds": 2.0 + index,
                            }
                        ]
                    },
                },
            }
        )
    artifact = {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "records": records,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return controls


def test_publication_archive_contains_hash_verified_v3_bank_and_source(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "training.json"
    controls = _source(source_path)
    proposal = {
        "weights": [0.2, 0.3, 0.5],
        "task_controls": {
            task: [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5 * value for value in segment] for segment in control],
                control,
            ]
            for task, control in controls.items()
        },
        "training_source_artifact_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "training_derivation": "componentwise_median_pure_cem_then_zero_half_full_bank",
        "training_source_record_count": 2,
        "training_total_samples": 20,
        "training_total_work_units": 41.0,
        "training_total_wall_seconds": 3.0,
        "training_total_cpu_seconds": 5.0,
        "training_amortization_record_count": 2,
    }
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "npi.g11.v6-routed-policy.config.v3",
                "protocol_id": "g11-v6-archive-test",
                "proposal": proposal,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    note = tmp_path / "note.txt"
    note.write_text("evidence", encoding="utf-8")
    output = tmp_path / "archive.zip"
    receipt = build_publication_archive(
        output,
        policy_config_path=config_path,
        proposal_training_source_path=source_path,
        additional_files={"note": note},
    )
    assert receipt["proposal_training_source_verified"]
    assert receipt["entry_count"] == 3
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert {entry["role"] for entry in manifest["entries"]} == {
            "policy_config",
            "proposal_training_source",
            "note",
        }
