from __future__ import annotations

from experiments.g11_v8_p5_dense_amplitude_proposal import (
    ROOT,
    load_dense_config,
)

CONFIG = ROOT / "configs/g11_v8/p5_dense_amplitude_proposal_v1.yaml"


def test_dense_amplitude_contract_is_frozen_and_fail_closed() -> None:
    config, digest = load_dense_config(CONFIG)

    assert len(digest) == 64
    assert len(config["cells"]) == 2
    assert len(config["proposal_families"]) == 4
    assert all(
        len(family["weights"]) == 7 for family in config["proposal_families"]
    )
    assert config["decision"]["proposal_manifest_freeze_authorized"] is False
    assert config["decision"]["new_full_pilot_authorized"] is False
