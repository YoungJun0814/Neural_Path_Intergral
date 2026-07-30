from __future__ import annotations

from experiments.g11_v8_p5_reference_proposal_manifest import (
    ROOT,
    load_manifest_config,
)

CONFIG = ROOT / "configs/g11_v8/p5_reference_proposal_manifest_v1.yaml"


def test_reference_proposal_manifest_build_contract_is_frozen() -> None:
    config, digest = load_manifest_config(CONFIG)

    assert len(digest) == 64
    assert config["override_sources"]["total_override_count"] == 11
    assert config["reference_protocol"]["methods"] == [
        "dcs_reference",
        "raw_crosscheck",
    ]
    assert config["decision"]["new_full_pilot_authorized"] is False
