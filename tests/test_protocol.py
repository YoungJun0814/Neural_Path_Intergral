from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.protocol import FrozenSeedSplit, load_frozen_protocol


def test_repository_g2_protocol_is_frozen_and_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = load_frozen_protocol(root / "configs" / "g2_heston_benchmark.yaml")
    assert protocol.frozen
    assert protocol.protocol_id == "g2-heston-terminal-left-tail-v1"
    assert len(protocol.seeds.evaluation) == 20
    assert len(protocol.sha256) == 64


def test_overlapping_seed_groups_are_rejected() -> None:
    split = FrozenSeedSplit(train=(1, 2), validation=(3, 4), evaluation=(2, 5))
    with pytest.raises(ValueError, match="overlap"):
        split.validate()
