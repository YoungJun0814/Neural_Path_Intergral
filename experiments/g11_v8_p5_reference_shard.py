"""Execute or resume final R2 shards from an immutable allocation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_sharded_reference_common import (
    execute_actual_reference_shard,
    load_context,
)
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_aggregation import validate_allocation_manifest
from src.path_integral.reference_protocol import (
    REFERENCE_METHODS,
    ReferenceShardIdentity,
    canonical_sha256,
)
from src.path_integral.reference_shards import (
    find_completed_shards,
    write_shard_atomic,
)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="ascii"))
    validate_allocation_manifest(payload)
    return payload, canonical_sha256(payload)


def _validate_existing(
    artifact: dict[str, Any],
    *,
    context: Any,
    manifest: dict[str, Any],
    manifest_sha256: str,
    requested_samples: int,
) -> None:
    if (
        artifact["config_sha256"] != context.config_sha256
        or artifact["threshold_manifest_sha256"]
        != context.binding["threshold_manifest_sha256"]
        or artifact["parent_sha256"] != manifest_sha256
        or artifact["source_commit"] != manifest["source_commit"]
        or artifact["dirty_worktree"] is not False
        or artifact["environment_sha256"] != manifest["environment_sha256"]
        or artifact["requested_samples"] != requested_samples
    ):
        raise ValueError("completed final shard conflicts with its frozen allocation")


def run_final_shards(
    config_path: Path,
    manifest_path: Path,
    output_directory: Path,
    *,
    cell_id: str | None = None,
    method: str | None = None,
    shard_index: int | None = None,
) -> dict[str, Any]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("formal final execution requires a clean Git worktree")
    context = load_context(config_path)
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    if (
        manifest["config_sha256"] != context.config_sha256
        or manifest["threshold_manifest_sha256"]
        != context.binding["threshold_manifest_sha256"]
        or manifest["source_commit"] != provenance["source_commit"]
        or manifest["environment_sha256"] != context.environment_sha256
        or manifest["final_namespace"]
        != context.config["sampling"]["final_namespace"]
        or manifest["final_execution_authorized"] is not True
    ):
        raise ValueError("allocation manifest is not executable in this source state")
    if cell_id is not None and cell_id not in context.cells_by_id:
        raise ValueError("unknown final cell ID")
    if method is not None and method not in REFERENCE_METHODS:
        raise ValueError("unknown final reference method")
    expected_chunks: list[tuple[ReferenceShardIdentity, int]] = []
    for entry in manifest["entries"]:
        if cell_id is not None and entry["cell_id"] != cell_id:
            continue
        if method is not None and entry["method"] != method:
            continue
        for chunk in entry["chunks"]:
            identity = ReferenceShardIdentity.from_dict(chunk["identity"])
            if shard_index is not None and identity.shard_index != shard_index:
                continue
            expected_chunks.append((identity, int(chunk["requested_samples"])))
    if not expected_chunks:
        raise ValueError("final shard selection is empty")
    completed = find_completed_shards(output_directory)
    executed = 0
    skipped = 0
    for identity, requested in expected_chunks:
        existing = completed.get(identity.shard_id)
        if existing is not None:
            _validate_existing(
                existing[1],
                context=context,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                requested_samples=requested,
            )
            skipped += 1
            print(
                json.dumps(
                    {"shard_id": identity.shard_id, "status": "skipped"},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        artifact = execute_actual_reference_shard(
            context,
            identity,
            requested_samples=requested,
            parent_sha256=manifest_sha256,
            source_commit=manifest["source_commit"],
            dirty_worktree=False,
        )
        path, digest = write_shard_atomic(output_directory, artifact)
        executed += 1
        print(
            json.dumps(
                {
                    "shard_id": identity.shard_id,
                    "status": "completed",
                    "path": str(path),
                    "sha256": digest,
                    "wall_seconds": artifact["elapsed_wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "schema": "npi.g11.v8-p5-reference-final-run-receipt.v1",
        "protocol_id": context.config["protocol_id"],
        "allocation_manifest_sha256": manifest_sha256,
        "source_commit": manifest["source_commit"],
        "environment_sha256": manifest["environment_sha256"],
        "expected_in_selection": len(expected_chunks),
        "executed": executed,
        "skipped": skipped,
        "selection_complete": executed + skipped == len(expected_chunks),
        "formal_reference_complete": False,
        "performance_claim_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cell-id")
    parser.add_argument("--method", choices=REFERENCE_METHODS)
    parser.add_argument("--shard-index", type=int)
    arguments = parser.parse_args()
    receipt = run_final_shards(
        arguments.config,
        arguments.allocation_manifest,
        arguments.output_directory,
        cell_id=arguments.cell_id,
        method=arguments.method,
        shard_index=arguments.shard_index,
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
