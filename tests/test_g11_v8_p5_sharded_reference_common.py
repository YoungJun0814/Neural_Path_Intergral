from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_p5_sharded_reference_common import (
    ROOT,
    load_context,
    load_sharded_reference_config,
    reference_seed_material,
)
from src.path_integral.reference_protocol import ReferenceShardIdentity

CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"


def test_sharded_reference_v3_context_is_hash_bound() -> None:
    context = load_context(CONFIG)

    assert len(context.cells_by_id) == 24
    assert context.config["protocol_id"] == context.binding["reference_protocol"]["id"]
    assert (
        context.config["sampling"]["pilot_namespace"]
        != context.config["sampling"]["final_namespace"]
    )
    assert len(context.environment_sha256) == 64


def test_seed_material_is_disjoint_by_namespace_method_stage_and_index() -> None:
    base = dict(
        protocol_id="g11-v8-p5-sharded-reference-development-v1",
        namespace="namespace-a",
        stage="pilot",
        method="dcs_reference",
        cell_id="cell",
        shard_index=0,
    )
    identities = [
        ReferenceShardIdentity(**base),
        ReferenceShardIdentity(**{**base, "namespace": "namespace-b"}),
        ReferenceShardIdentity(**{**base, "stage": "final"}),
        ReferenceShardIdentity(**{**base, "method": "raw_crosscheck"}),
        ReferenceShardIdentity(**{**base, "shard_index": 1}),
    ]
    materials = [reference_seed_material(identity) for identity in identities]

    assert len({item[0] for item in materials}) == len(materials)
    assert len({seed for item in materials for seed in item[1:]}) == 2 * len(materials)


def test_sharded_reference_loader_rejects_post_freeze_sampling_change(
    tmp_path: Path,
) -> None:
    config, _ = load_sharded_reference_config(CONFIG)
    mutated = copy.deepcopy(config)
    mutated["sampling"]["maximum_final_samples"] -= 1
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling contract"):
        load_sharded_reference_config(path)
