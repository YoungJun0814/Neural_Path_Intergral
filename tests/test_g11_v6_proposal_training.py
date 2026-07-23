"""Reference-free V6 proposal-training contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.g11_v6_proposal_training import _load_config, run
from src.path_integral import (
    V6_CELL_MANIFEST_SCHEMA,
    SeedLedger,
    V6CellManifest,
    V6RBergomiCell,
    V6WorkLedger,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "proposal_training_development_v1.yaml"


def _cell(cell_id: str, task: str, threshold: float) -> V6RBergomiCell:
    return V6RBergomiCell(
        cell_id=cell_id,
        hurst=0.12,
        eta=1.1,
        xi=0.04,
        rho=-0.6,
        spot=100.0,
        maturity=0.25,
        finest_steps=128,
        task=task,
        event_threshold=threshold,
        nominal_probability=1.0e-3,
        probability_band=(5.0e-4, 2.0e-3),
    )


def _manifest(path: Path, *, dirty: bool = False) -> None:
    manifest = V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-test-development-manifest",
        phase="development",
        frozen=False,
        source_commit="a" * 40,
        dirty_tree=dirty,
        config_sha256="b" * 64,
        smoke=False,
        cells=(
            _cell(
                "h012-terminal-p1e-03",
                "terminal_left_tail",
                86.0,
            ),
            _cell(
                "h012-barrier-p1e-03",
                "discrete_lower_barrier",
                82.0,
            ),
        ),
    )
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")


def test_proposal_training_config_prespecifies_upper_tail_and_two_tasks() -> None:
    config, digest = _load_config(CONFIG)
    assert config["training"]["elite_quantile"] == 0.90
    assert config["selected_cell_ids"] == [
        "h012-terminal-p1e-03",
        "h012-barrier-p1e-03",
    ]
    assert config["clusters"] == 3
    assert len(digest) == 64


def test_reference_free_training_smoke_has_strict_ledgers(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    result = run(CONFIG, manifest_path, smoke=True)
    assert result["schema"] == "npi.g11.v6-proposal-training.v1"
    assert len(result["records"]) == 2
    assert "reference_artifact_sha256" not in result
    assert {record["task"] for record in result["records"]} == {
        "terminal_left_tail",
        "discrete_lower_barrier",
    }
    seed_ledger = SeedLedger.from_dict(result["seed_ledger"])
    work_ledger = V6WorkLedger.from_dict(result["work_ledger"])
    assert seed_ledger.sha256 == result["seed_ledger_sha256"]
    assert work_ledger.sha256 == result["work_ledger_sha256"]
    assert len(seed_ledger) == len(work_ledger.records) == 2
    assert not result["formal_readiness"]["non_smoke"]
    assert not result["proposal_training_qualified"]


def test_formal_training_rejects_a_dirty_development_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "dirty-manifest.json"
    _manifest(manifest_path, dirty=True)
    try:
        run(CONFIG, manifest_path, smoke=False)
    except ValueError as error:
        assert "clean committed development manifest" in str(error)
    else:
        raise AssertionError("dirty proposal-training manifest was accepted")
