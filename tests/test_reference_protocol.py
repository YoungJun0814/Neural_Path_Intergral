from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from src.path_integral.reference_aggregation import (
    _chunk_counts,
    aggregate_final_shards,
    build_allocation_manifest,
    validate_allocation_manifest,
)
from src.path_integral.reference_execution import (
    ReferenceBatch,
    execute_reference_shard,
)
from src.path_integral.reference_protocol import (
    REFERENCE_METHODS,
    ReferenceShardIdentity,
    SufficientStatistics,
    build_shard_artifact,
    canonical_sha256,
    seed_key_sha256,
    validate_shard_artifact,
)
from src.path_integral.reference_shards import (
    find_completed_shards,
    shard_path,
    write_shard_atomic,
)
from src.path_integral.resource_planner import (
    ReferenceBenchmarkObservation,
    forecast_reference_resources,
)

CONFIG_SHA = "a" * 64
THRESHOLD_SHA = "b" * 64
PILOT_PARENT_SHA = "c" * 64
ENVIRONMENT_SHA = "d" * 64
SOURCE_COMMIT = "1" * 40
PROTOCOL = "g11-v8-reference-test-v1"
PILOT_NAMESPACE = "test-pilot"
FINAL_NAMESPACE = "test-final"
CELLS = ("cell-a", "cell-b")


def _artifact(
    identity: ReferenceShardIdentity,
    values: torch.Tensor,
    *,
    parent_sha256: str,
    dirty_worktree: bool = False,
) -> dict[str, Any]:
    count = int(values.numel())
    return build_shard_artifact(
        identity=identity,
        config_sha256=CONFIG_SHA,
        threshold_manifest_sha256=THRESHOLD_SHA,
        parent_sha256=parent_sha256,
        source_commit=SOURCE_COMMIT,
        dirty_worktree=dirty_worktree,
        environment_sha256=ENVIRONMENT_SHA,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        seed_key_sha256=seed_key_sha256(identity.to_dict()),
        requested_samples=count,
        contribution=SufficientStatistics.from_tensor(values),
        likelihood_normalization=SufficientStatistics.from_tensor(
            torch.ones(count, dtype=torch.float64)
        ),
        invalid_spot_count=0,
        invalid_variance_count=0,
        nonfinite_contribution_count=0,
        elapsed_wall_seconds=0.1,
        elapsed_cpu_seconds=0.2,
        peak_resident_memory_bytes=1024,
    )


def _pilot_shards(
    *,
    target_scale: float = 1.0,
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, float]]:
    shards: list[tuple[dict[str, Any], str]] = []
    values = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64) * target_scale
    for cell in CELLS:
        for method in REFERENCE_METHODS:
            for replicate in range(3):
                identity = ReferenceShardIdentity(
                    protocol_id=PROTOCOL,
                    namespace=PILOT_NAMESPACE,
                    stage="pilot",
                    method=method,
                    cell_id=cell,
                    shard_index=replicate,
                )
                artifact = _artifact(
                    identity,
                    values + 0.1 * replicate,
                    parent_sha256=PILOT_PARENT_SHA,
                )
                shards.append((artifact, canonical_sha256(artifact)))
    return shards, {cell: 0.5 for cell in CELLS}


def _manifest(
    *,
    maximum_final_samples: int = 1000,
    target_scale: float = 1.0,
) -> dict[str, Any]:
    shards, targets = _pilot_shards(target_scale=target_scale)
    return build_allocation_manifest(
        protocol_id=PROTOCOL,
        config_sha256=CONFIG_SHA,
        threshold_manifest_sha256=THRESHOLD_SHA,
        pilot_parent_sha256=PILOT_PARENT_SHA,
        pilot_namespace=PILOT_NAMESPACE,
        final_namespace=FINAL_NAMESPACE,
        expected_cells=CELLS,
        expected_methods=REFERENCE_METHODS,
        pilot_replicates=3,
        pilot_shards=shards,
        target_standard_errors=targets,
        allocation_safety_factor=1.0,
        minimum_final_samples=4,
        maximum_final_samples=maximum_final_samples,
        final_chunk_size=4,
        source_commit=SOURCE_COMMIT,
        environment_sha256=ENVIRONMENT_SHA,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        design_informed_by_prior_development_outcomes=True,
        current_namespace_outcomes_inspected_before_freeze=False,
    )


def _final_shards(
    manifest: dict[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    digest = canonical_sha256(manifest)
    artifacts: list[tuple[dict[str, Any], str]] = []
    entries = manifest["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        chunks = entry["chunks"]
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, dict)
            identity = ReferenceShardIdentity.from_dict(chunk["identity"])
            count = chunk["requested_samples"]
            assert isinstance(count, int)
            values = torch.tensor(
                [float(index % 2) for index in range(count)], dtype=torch.float64
            )
            artifact = _artifact(identity, values, parent_sha256=digest)
            artifacts.append((artifact, canonical_sha256(artifact)))
    return artifacts


def test_sufficient_statistics_merge_matches_direct_unbiased_variance() -> None:
    left = torch.tensor([1.0, 4.0, -2.0], dtype=torch.float64)
    right = torch.tensor([7.0, 3.0, 2.0, -1.0], dtype=torch.float64)
    merged = SufficientStatistics.from_tensor(left).merge(
        SufficientStatistics.from_tensor(right)
    )
    direct = torch.cat((left, right))

    assert merged.count == direct.numel()
    assert merged.mean == pytest.approx(float(direct.mean()), abs=1e-15)
    assert merged.variance == pytest.approx(
        float(torch.var(direct, unbiased=True)), abs=1e-14
    )


def test_canonical_hash_is_order_stable_and_artifact_is_cpu_reproducible() -> None:
    assert canonical_sha256({"b": 2, "a": [1, 3]}) == canonical_sha256(
        {"a": [1, 3], "b": 2}
    )
    identity = ReferenceShardIdentity(
        PROTOCOL, PILOT_NAMESPACE, "pilot", "dcs_reference", "cell-a", 0
    )
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    first = _artifact(identity, values, parent_sha256=PILOT_PARENT_SHA)
    second = _artifact(identity, values, parent_sha256=PILOT_PARENT_SHA)

    assert first == second
    assert canonical_sha256(first) == canonical_sha256(second)


@pytest.mark.parametrize(
    ("total", "chunk_size", "expected"),
    [(2, 4, [2]), (3, 4, [3]), (4, 4, [4]), (5, 4, [3, 2]), (10, 4, [4, 4, 2])],
)
def test_chunk_partition_has_no_singleton(
    total: int, chunk_size: int, expected: list[int]
) -> None:
    assert _chunk_counts(total, chunk_size) == expected


def test_shard_validation_rejects_count_or_identity_corruption() -> None:
    identity = ReferenceShardIdentity(
        PROTOCOL, PILOT_NAMESPACE, "pilot", "dcs_reference", "cell-a", 0
    )
    artifact = _artifact(
        identity,
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        parent_sha256=PILOT_PARENT_SHA,
    )
    corrupted = copy.deepcopy(artifact)
    corrupted["requested_samples"] = 3
    with pytest.raises(ValueError, match="count mismatch"):
        validate_shard_artifact(corrupted)

    corrupted = copy.deepcopy(artifact)
    corrupted["shard_id"] = "post-hoc-id"
    with pytest.raises(ValueError, match="identity hash"):
        validate_shard_artifact(corrupted)


def test_atomic_shard_write_refuses_overwrite_and_discovers_completed(
    tmp_path: Path,
) -> None:
    identity = ReferenceShardIdentity(
        PROTOCOL, PILOT_NAMESPACE, "pilot", "raw_crosscheck", "cell-a", 0
    )
    artifact = _artifact(
        identity,
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        parent_sha256=PILOT_PARENT_SHA,
    )
    path, digest = write_shard_atomic(tmp_path, artifact)
    assert path == shard_path(tmp_path, identity)
    assert path.is_file()
    assert digest == canonical_sha256(artifact)
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_shard_atomic(tmp_path, artifact)
    completed = find_completed_shards(tmp_path)
    assert set(completed) == {identity.shard_id}
    assert completed[identity.shard_id][2] == digest


def test_allocation_is_fixed_from_complete_clean_pilots() -> None:
    manifest = _manifest()
    assert manifest["all_resources_feasible"] is True
    assert manifest["final_execution_authorized"] is True
    assert manifest["pilot_namespace"] != manifest["final_namespace"]
    entries = manifest["entries"]
    assert isinstance(entries, list)
    assert len(entries) == len(CELLS) * len(REFERENCE_METHODS)
    for entry in entries:
        assert entry["allocation_variance_statistic"] == "maximum_replicate_variance"
        assert entry["authorized_final_samples"] == entry["requested_final_samples"]
        assert sum(chunk["requested_samples"] for chunk in entry["chunks"]) == entry[
            "authorized_final_samples"
        ]


def test_allocation_fails_before_final_when_resource_cap_is_insufficient() -> None:
    manifest = _manifest(maximum_final_samples=4, target_scale=100.0)
    assert manifest["all_resources_feasible"] is False
    assert manifest["final_execution_authorized"] is False
    assert any(not entry["resource_feasible"] for entry in manifest["entries"])
    assert all(
        entry["authorized_final_samples"] is None
        for entry in manifest["entries"]
        if not entry["resource_feasible"]
    )


def test_allocation_rejects_dirty_duplicate_or_incomplete_pilots() -> None:
    shards, targets = _pilot_shards()
    dirty_payload = copy.deepcopy(shards[0][0])
    dirty_payload["dirty_worktree"] = True
    dirty = [(dirty_payload, canonical_sha256(dirty_payload)), *shards[1:]]
    kwargs = dict(
        protocol_id=PROTOCOL,
        config_sha256=CONFIG_SHA,
        threshold_manifest_sha256=THRESHOLD_SHA,
        pilot_parent_sha256=PILOT_PARENT_SHA,
        pilot_namespace=PILOT_NAMESPACE,
        final_namespace=FINAL_NAMESPACE,
        expected_cells=CELLS,
        expected_methods=REFERENCE_METHODS,
        pilot_replicates=3,
        target_standard_errors=targets,
        allocation_safety_factor=1.0,
        minimum_final_samples=4,
        maximum_final_samples=1000,
        final_chunk_size=4,
        source_commit=SOURCE_COMMIT,
        environment_sha256=ENVIRONMENT_SHA,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        design_informed_by_prior_development_outcomes=True,
        current_namespace_outcomes_inspected_before_freeze=False,
    )
    with pytest.raises(ValueError, match="clean-source"):
        build_allocation_manifest(pilot_shards=dirty, **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        build_allocation_manifest(pilot_shards=[*shards, shards[0]], **kwargs)
    with pytest.raises(ValueError, match="set mismatch"):
        build_allocation_manifest(pilot_shards=shards[:-1], **kwargs)


def test_allocation_rejects_forged_digest_or_reused_seed_stream() -> None:
    shards, targets = _pilot_shards()
    kwargs = dict(
        protocol_id=PROTOCOL,
        config_sha256=CONFIG_SHA,
        threshold_manifest_sha256=THRESHOLD_SHA,
        pilot_parent_sha256=PILOT_PARENT_SHA,
        pilot_namespace=PILOT_NAMESPACE,
        final_namespace=FINAL_NAMESPACE,
        expected_cells=CELLS,
        expected_methods=REFERENCE_METHODS,
        pilot_replicates=3,
        target_standard_errors=targets,
        allocation_safety_factor=1.0,
        minimum_final_samples=4,
        maximum_final_samples=1000,
        final_chunk_size=4,
        source_commit=SOURCE_COMMIT,
        environment_sha256=ENVIRONMENT_SHA,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        design_informed_by_prior_development_outcomes=True,
        current_namespace_outcomes_inspected_before_freeze=False,
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        build_allocation_manifest(
            pilot_shards=[(shards[0][0], "f" * 64), *shards[1:]],
            **kwargs,
        )

    reused_seed = copy.deepcopy(shards[1][0])
    reused_seed["seed_key_sha256"] = shards[0][0]["seed_key_sha256"]
    with pytest.raises(ValueError, match="duplicate pilot seed"):
        build_allocation_manifest(
            pilot_shards=[
                shards[0],
                (reused_seed, canonical_sha256(reused_seed)),
                *shards[2:],
            ],
            **kwargs,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["entries"].pop(),
        lambda manifest: manifest["entries"][0].__setitem__(
            "requested_final_samples",
            manifest["entries"][0]["requested_final_samples"] + 1,
        ),
        lambda manifest: manifest["entries"][0]["chunks"][0].__setitem__(
            "requested_samples",
            manifest["entries"][0]["chunks"][0]["requested_samples"] + 1,
        ),
    ],
)
def test_allocation_manifest_validation_rejects_post_freeze_mutation(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    manifest = copy.deepcopy(_manifest())
    mutation(manifest)
    with pytest.raises(ValueError):
        validate_allocation_manifest(manifest)


def test_interrupted_resume_aggregate_matches_uninterrupted(tmp_path: Path) -> None:
    manifest = _manifest()
    digest = canonical_sha256(manifest)
    shards = _final_shards(manifest)
    interrupted = tmp_path / "interrupted"
    uninterrupted = tmp_path / "uninterrupted"

    halfway = len(shards) // 2
    for artifact, _ in shards[:halfway]:
        write_shard_atomic(interrupted, artifact)
    completed = find_completed_shards(interrupted)
    for artifact, _ in shards:
        if artifact["shard_id"] not in completed:
            write_shard_atomic(interrupted, artifact)
        write_shard_atomic(uninterrupted, artifact)

    resumed_records = [
        (payload, shard_digest)
        for _, payload, shard_digest in find_completed_shards(interrupted).values()
    ]
    uninterrupted_records = [
        (payload, shard_digest)
        for _, payload, shard_digest in find_completed_shards(uninterrupted).values()
    ]
    resumed = aggregate_final_shards(manifest, digest, resumed_records)
    direct = aggregate_final_shards(manifest, digest, uninterrupted_records)

    assert resumed == direct
    assert resumed["complete_reference_matrix"] is True
    assert resumed["all_target_standard_errors"] is True
    assert resumed["all_likelihood_normalizations"] is True
    assert resumed["all_independent_methods_agree"] is True
    assert resumed["reference_acceptance_pass"] is True


def test_aggregate_rejects_missing_duplicate_or_foreign_parent() -> None:
    manifest = _manifest()
    digest = canonical_sha256(manifest)
    shards = _final_shards(manifest)
    with pytest.raises(ValueError, match="set mismatch"):
        aggregate_final_shards(manifest, digest, shards[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_final_shards(manifest, digest, [*shards, shards[0]])

    foreign = copy.deepcopy(shards[0][0])
    foreign["parent_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parent-manifest"):
        aggregate_final_shards(
            manifest,
            digest,
            [(foreign, canonical_sha256(foreign)), *shards[1:]],
        )


def test_aggregate_rejects_forged_digest_or_reused_seed_stream() -> None:
    manifest = _manifest()
    digest = canonical_sha256(manifest)
    shards = _final_shards(manifest)
    with pytest.raises(ValueError, match="digest mismatch"):
        aggregate_final_shards(
            manifest,
            digest,
            [(shards[0][0], "f" * 64), *shards[1:]],
        )

    reused_seed = copy.deepcopy(shards[1][0])
    reused_seed["seed_key_sha256"] = shards[0][0]["seed_key_sha256"]
    with pytest.raises(ValueError, match="final seed"):
        aggregate_final_shards(
            manifest,
            digest,
            [
                shards[0],
                (reused_seed, canonical_sha256(reused_seed)),
                *shards[2:],
            ],
        )

    pilot_overlap = copy.deepcopy(shards[0][0])
    pilot_overlap["seed_key_sha256"] = manifest["entries"][0][
        "pilot_seed_key_sha256s"
    ][0]
    with pytest.raises(ValueError, match="pilot-overlapping"):
        aggregate_final_shards(
            manifest,
            digest,
            [
                (pilot_overlap, canonical_sha256(pilot_overlap)),
                *shards[1:],
            ],
        )


def test_aggregate_rejects_invalid_paths_or_backend_mismatch() -> None:
    manifest = _manifest()
    digest = canonical_sha256(manifest)
    shards = _final_shards(manifest)

    invalid = copy.deepcopy(shards[0][0])
    invalid["invalid_spot_count"] = 1
    with pytest.raises(ValueError, match="invalid paths"):
        aggregate_final_shards(
            manifest,
            digest,
            [(invalid, canonical_sha256(invalid)), *shards[1:]],
        )

    wrong_dtype = copy.deepcopy(shards[0][0])
    wrong_dtype["dtype"] = "float32"
    with pytest.raises(ValueError, match="dtype or device"):
        aggregate_final_shards(
            manifest,
            digest,
            [(wrong_dtype, canonical_sha256(wrong_dtype)), *shards[1:]],
        )


def test_aggregate_fails_independent_method_disagreement() -> None:
    manifest = _manifest()
    digest = canonical_sha256(manifest)
    shards = _final_shards(manifest)
    first_raw_index = next(
        index
        for index, (payload, _) in enumerate(shards)
        if payload["identity"]["method"] == "raw_crosscheck"
    )
    changed_payload = copy.deepcopy(shards[first_raw_index][0])
    count = changed_payload["requested_samples"]
    changed_payload["contribution"] = SufficientStatistics.from_tensor(
        torch.full((count,), 100.0, dtype=torch.float64)
    ).to_dict()
    changed = list(shards)
    changed[first_raw_index] = (
        changed_payload,
        canonical_sha256(changed_payload),
    )
    result = aggregate_final_shards(manifest, digest, changed)

    assert result["all_independent_methods_agree"] is False
    assert result["reference_acceptance_pass"] is False


def test_resource_forecast_is_conservative_and_fail_closed() -> None:
    observations = [
        ReferenceBenchmarkObservation(100, 128, 2.0, 4.0, 1000, 50),
        ReferenceBenchmarkObservation(100, 128, 4.0, 6.0, 1200, 60),
    ]
    feasible = forecast_reference_resources(
        observations,
        total_paths=1000,
        steps=128,
        workers=4,
        parallel_efficiency=0.5,
        safety_factor=2.0,
        available_memory_bytes=10000,
        maximum_wall_seconds=1000.0,
    )
    assert feasible["conservative_single_worker_path_steps_per_second"] == 3200.0
    assert feasible["launch_authorized"] is True

    blocked = forecast_reference_resources(
        observations,
        total_paths=1000,
        steps=128,
        workers=4,
        parallel_efficiency=0.5,
        safety_factor=2.0,
        available_memory_bytes=1000,
        maximum_wall_seconds=1.0,
    )
    assert blocked["memory_feasible"] is False
    assert blocked["wall_feasible"] is False
    assert blocked["launch_authorized"] is False


def test_reference_identity_and_statistics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="namespace"):
        ReferenceShardIdentity(
            PROTOCOL, "", "pilot", "dcs_reference", "cell-a", 0
        )
    with pytest.raises(ValueError, match="nonnegative"):
        ReferenceShardIdentity(
            PROTOCOL, PILOT_NAMESPACE, "pilot", "dcs_reference", "cell-a", -1
        )
    with pytest.raises(FloatingPointError, match="finite"):
        SufficientStatistics.from_tensor(torch.tensor([math.nan], dtype=torch.float64))


def test_reference_shard_executor_enforces_frozen_count_and_backend() -> None:
    identity = ReferenceShardIdentity(
        PROTOCOL, FINAL_NAMESPACE, "final", "dcs_reference", "cell-a", 0
    )

    def valid_draw() -> ReferenceBatch:
        return ReferenceBatch(
            contribution=torch.tensor([0.0, 1.0], dtype=torch.float64),
            likelihood_normalization=torch.ones(2, dtype=torch.float64),
        )

    artifact = execute_reference_shard(
        identity=identity,
        config_sha256=CONFIG_SHA,
        threshold_manifest_sha256=THRESHOLD_SHA,
        parent_sha256=PILOT_PARENT_SHA,
        source_commit=SOURCE_COMMIT,
        dirty_worktree=False,
        environment_sha256=ENVIRONMENT_SHA,
        seed_key_sha256=seed_key_sha256(identity.to_dict()),
        requested_samples=2,
        draw=valid_draw,
    )
    assert artifact["requested_samples"] == 2
    assert artifact["dtype"] == "float64"
    assert artifact["device"] == "cpu"

    with pytest.raises(ValueError, match="draw count"):
        execute_reference_shard(
            identity=identity,
            config_sha256=CONFIG_SHA,
            threshold_manifest_sha256=THRESHOLD_SHA,
            parent_sha256=PILOT_PARENT_SHA,
            source_commit=SOURCE_COMMIT,
            dirty_worktree=False,
            environment_sha256=ENVIRONMENT_SHA,
            seed_key_sha256=seed_key_sha256(identity.to_dict()),
            requested_samples=3,
            draw=valid_draw,
        )
    with pytest.raises(ValueError, match="float64"):
        ReferenceBatch(
            contribution=torch.tensor([0.0, 1.0], dtype=torch.float32),
            likelihood_normalization=torch.ones(2, dtype=torch.float32),
        )


def test_reference_shard_executor_rejects_nonfinite_without_artifact() -> None:
    identity = ReferenceShardIdentity(
        PROTOCOL, FINAL_NAMESPACE, "final", "raw_crosscheck", "cell-a", 0
    )

    def nonfinite_draw() -> ReferenceBatch:
        return ReferenceBatch(
            contribution=torch.tensor([0.0, math.nan], dtype=torch.float64),
            likelihood_normalization=torch.ones(2, dtype=torch.float64),
            nonfinite_contribution_count=1,
        )

    with pytest.raises(FloatingPointError, match="nonfinite"):
        execute_reference_shard(
            identity=identity,
            config_sha256=CONFIG_SHA,
            threshold_manifest_sha256=THRESHOLD_SHA,
            parent_sha256=PILOT_PARENT_SHA,
            source_commit=SOURCE_COMMIT,
            dirty_worktree=False,
            environment_sha256=ENVIRONMENT_SHA,
            seed_key_sha256=seed_key_sha256(identity.to_dict()),
            requested_samples=2,
            draw=nonfinite_draw,
        )
