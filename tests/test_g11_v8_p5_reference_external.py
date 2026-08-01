from __future__ import annotations

from pathlib import Path

import torch

from experiments.g11_v8_p5_reference_allocation_amendment import (
    reconstruct_amended_allocation,
)
from experiments.g11_v8_p5_reference_external import (
    PARTITION_RULE,
    allocation_chunks,
    partition_chunks,
)
from experiments.g11_v8_p5_sharded_reference_common import (
    draw_actual_reference_batch,
    load_context,
)
from src.path_integral.reference_protocol import ReferenceShardIdentity

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v6.yaml"
PACKAGE = ROOT / "results/g11_v8_p5_reference_pilot_package_v6_2026-07-31.json"
FAILURE = ROOT / "results/g11_v8_p5_reference_allocation_failure_v6_2026-07-31.json"
AMENDMENT = ROOT / "results/g11_v8_p5_reference_allocation_amendment_receipt_v1_2026-07-31.json"


def test_external_partitions_are_complete_disjoint_and_balanced() -> None:
    manifest, _ = reconstruct_amended_allocation(
        CONFIG,
        PACKAGE,
        FAILURE,
        AMENDMENT,
    )
    full = allocation_chunks(manifest)
    partitions = [
        partition_chunks(manifest, partition_index=index, partition_count=4) for index in range(4)
    ]
    partition_ids = [{identity.shard_id for identity, _ in partition} for partition in partitions]

    assert PARTITION_RULE == ("sorted_shard_id_global_index_modulo_partition_count")
    assert len(full) == 41_487
    assert set().union(*partition_ids) == {identity.shard_id for identity, _ in full}
    assert sum(len(ids) for ids in partition_ids) == len(full)
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1


def test_final_shard_403_is_finite_in_canonical_log_spot_domain() -> None:
    """Regress the deterministic underflow found by the first local final run."""

    context = load_context(CONFIG)
    identity = ReferenceShardIdentity(
        protocol_id="g11-v8-p5-sharded-reference-resource-cap-v1",
        namespace="v8-r2-reference-resource-cap-final-v1",
        stage="final",
        cell_id="h0.20-discrete_lower_barrier-p1e-05",
        method="dcs_reference",
        shard_index=403,
    )

    batch = draw_actual_reference_batch(context, identity, 4096)

    assert batch.contribution.shape == (4096,)
    assert torch.isfinite(batch.contribution).all()
    assert torch.isfinite(batch.likelihood_normalization).all()
