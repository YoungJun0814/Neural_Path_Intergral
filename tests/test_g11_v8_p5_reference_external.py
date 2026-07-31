from __future__ import annotations

from pathlib import Path

from experiments.g11_v8_p5_reference_allocation_amendment import (
    reconstruct_amended_allocation,
)
from experiments.g11_v8_p5_reference_external import (
    PARTITION_RULE,
    allocation_chunks,
    partition_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v6.yaml"
PACKAGE = ROOT / "results/g11_v8_p5_reference_pilot_package_v6_2026-07-31.json"
FAILURE = (
    ROOT / "results/g11_v8_p5_reference_allocation_failure_v6_2026-07-31.json"
)
AMENDMENT = (
    ROOT
    / "results/g11_v8_p5_reference_allocation_amendment_receipt_v1_2026-07-31.json"
)


def test_external_partitions_are_complete_disjoint_and_balanced() -> None:
    manifest, _ = reconstruct_amended_allocation(
        CONFIG,
        PACKAGE,
        FAILURE,
        AMENDMENT,
    )
    full = allocation_chunks(manifest)
    partitions = [
        partition_chunks(manifest, partition_index=index, partition_count=4)
        for index in range(4)
    ]
    partition_ids = [
        {identity.shard_id for identity, _ in partition}
        for partition in partitions
    ]

    assert PARTITION_RULE == (
        "sorted_shard_id_global_index_modulo_partition_count"
    )
    assert len(full) == 41_487
    assert set().union(*partition_ids) == {
        identity.shard_id for identity, _ in full
    }
    assert sum(len(ids) for ids in partition_ids) == len(full)
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1
