from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_p5_barrier_proposal_falsification import (
    ROOT,
    _seeds,
    load_falsification_config,
)
from src.path_integral import derive_seed

CONFIG = (
    ROOT
    / "configs/g11_v8/p5_barrier_reference_proposal_falsification_v1.yaml"
)


def test_barrier_proposal_falsification_contract_is_frozen() -> None:
    config, digest = load_falsification_config(CONFIG)

    assert len(digest) == 64
    assert len(config["cells"]) == 6
    assert len(config["candidates"]) == 4
    assert config["decision"]["new_full_pilot_authorized"] is False


def test_candidate_cell_replicate_seed_families_are_disjoint() -> None:
    keys = [
        key
        for candidate in ("a", "b")
        for cell in ("cell-a", "cell-b")
        for replicate in range(2)
        for key in _seeds("protocol", "namespace", candidate, cell, replicate)
    ]
    seeds = [derive_seed(key) for key in keys]

    assert len(keys) == len(set(keys))
    assert len(seeds) == len(set(seeds))


def test_falsification_loader_rejects_post_freeze_candidate_change(
    tmp_path: Path,
) -> None:
    config, _ = load_falsification_config(CONFIG)
    mutated = copy.deepcopy(config)
    mutated["candidates"][0]["weights"][0] += 0.01
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="mixture"):
        load_falsification_config(path)
