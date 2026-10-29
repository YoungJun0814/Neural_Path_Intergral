import json

import pytest

from experiments import post_audit_p2_hamiltonian_replay_recovery as recovery
from experiments import post_audit_p2_hamiltonian_tail as original


def test_numeric_replay_excludes_only_wall_not_seed_or_normalizer():
    left = {"wall_seconds": 10., "seed": 7, "log_mu": -18., "stages": [{"beta": .5}]}
    assert recovery.numeric_equal(left, {**left, "wall_seconds": 100.})
    assert not recovery.numeric_equal(left, {**left, "seed": 8})
    assert not recovery.numeric_equal(left, {**left, "log_mu": -18.01})
    assert not recovery.numeric_equal(left, {**left, "stages": []})


def test_original_operational_failure_is_preserved_not_a_completed_study():
    base = recovery.ROOT/original.DIRECTORY
    failure = json.loads((base/"failure.json").read_text())
    assert failure["completed_pilot"] == 20
    assert failure["completed_precision"] == 0
    assert failure["error"].startswith("TimeoutError")
    assert not (base/"result.json").exists()
    rows = [p for p in base.glob("pilot-*.json") if p.name != "pilot-result.json"]
    assert len(rows) == 20


def test_missing_original_family_is_rejected_before_new_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        recovery.load_original()
