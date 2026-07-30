from __future__ import annotations

from experiments.g11_v8_p5_dense_amplitude_proposal import (
    ROOT,
    load_dense_config,
)

CONFIG = ROOT / "configs/g11_v8/p5_dense_amplitude_proposal_v1.yaml"
CONFIG_V2 = ROOT / "configs/g11_v8/p5_dense_amplitude_proposal_v2.yaml"


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

    config_v2, digest_v2 = load_dense_config(CONFIG_V2)
    assert len(digest_v2) == 64
    assert len(config_v2["cells"]) == 1
    assert len(config_v2["proposal_families"]) == 5
    assert all(
        len(family["weights"]) == 8
        for family in config_v2["proposal_families"]
    )
