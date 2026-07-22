"""Strict V6 progress-journal tests."""

from __future__ import annotations

import json

import pytest

from src.path_integral import V6ProgressJournal, load_v6_progress, save_v6_progress


def test_v6_progress_round_trip_and_identity_rejection(tmp_path) -> None:
    path = tmp_path / "progress.json"
    identities = {"config_sha256": "1" * 64, "smoke": True}
    journal = V6ProgressJournal(
        "experiment", identities, ({"cell_id": "a", "cluster": 0},)
    )
    save_v6_progress(path, journal)
    restored = load_v6_progress(
        path, experiment="experiment", identities=identities
    )
    assert restored.records == journal.records
    with pytest.raises(ValueError, match="identity"):
        load_v6_progress(
            path,
            experiment="experiment",
            identities={"config_sha256": "2" * 64, "smoke": True},
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["cluster"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_v6_progress(path, experiment="experiment", identities=identities)
