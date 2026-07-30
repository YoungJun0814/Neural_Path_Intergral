"""Execute or resume the exact frozen R2 pilot-shard roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml

from experiments.g11_v8_p5_sharded_reference_common import (
    ROOT,
    execute_actual_reference_shard,
    load_context,
)
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_protocol import (
    REFERENCE_METHODS,
    ReferenceMethod,
    ReferenceShardIdentity,
)
from src.path_integral.reference_shards import (
    find_completed_shards,
    write_shard_atomic,
)

AUTHORIZATION_SCHEMA = "npi.g11.v8-p5-reference-pilot-authorization.v1"
IMPLEMENTATION_SCHEMA = "npi.g11.v8-p5-reference-implementation-manifest.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("pilot authorization contains a malformed binding")
    relative = record.get("path")
    if not isinstance(relative, str):
        raise ValueError("pilot authorization path must be a string")
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("pilot authorization path is absent or outside the repository")
    if record.get("sha256") != _sha256(path):
        raise ValueError("pilot authorization SHA-256 mismatch")
    return path


def _verify_authorization(path: Path, context: Any) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("unexpected formal-pilot authorization schema")
    config_path = _bound_path(value.get("execution_config"))
    benchmark_path = _bound_path(value.get("representative_benchmark"))
    implementation_path = _bound_path(value.get("implementation_manifest"))
    if _sha256(config_path) != context.config_sha256:
        raise ValueError("pilot authorization binds a different execution config")
    implementation_value = yaml.safe_load(
        implementation_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(implementation_value, dict)
        or implementation_value.get("schema") != IMPLEMENTATION_SCHEMA
    ):
        raise ValueError("unexpected reference implementation manifest")
    records = implementation_value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("reference implementation manifest is empty")
    for record in records:
        _bound_path(record)
    if (
        implementation_value.get("decision", {}).get(
            "formal_pilot_code_state_frozen"
        )
        is not True
    ):
        raise ValueError("reference implementation manifest is not frozen")
    benchmark_value = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (
        not isinstance(benchmark_value, dict)
        or benchmark_value.get("dirty_worktree") is not False
        or benchmark_value.get("config_sha256") != context.config_sha256
        or benchmark_value.get("threshold_binding_sha256")
        != context.binding_sha256
        or benchmark_value.get("decision", {}).get("status")
        != "representative_benchmark_pass"
    ):
        raise ValueError("pilot authorization benchmark is not clean and compatible")
    formal = value.get("formal_pilot")
    decision = value.get("decision")
    if (
        value.get("formal_pilot_outcomes_inspected_before_authorization") is not False
        or not isinstance(formal, dict)
        or formal.get("namespace")
        != context.config["sampling"]["pilot_namespace"]
        or formal.get("total_paths") != 12582912
        or formal.get("laptop_execution_authorized") is not True
        or not isinstance(decision, dict)
        or decision.get("formal_pilot_execution_authorized") is not True
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
    ):
        raise ValueError("formal-pilot authorization is not fail-closed")
    return value


def _validate_existing(
    artifact: dict[str, Any],
    *,
    context: Any,
    requested_samples: int,
    source_commit: str,
) -> None:
    if (
        artifact["config_sha256"] != context.config_sha256
        or artifact["threshold_manifest_sha256"]
        != context.binding["threshold_manifest_sha256"]
        or artifact["parent_sha256"] != context.reference_parent_sha256
        or artifact["source_commit"] != source_commit
        or artifact["dirty_worktree"] is not False
        or artifact["environment_sha256"] != context.environment_sha256
        or artifact["requested_samples"] != requested_samples
    ):
        raise ValueError("completed pilot shard conflicts with the frozen execution")


def run_pilots(
    config_path: Path,
    authorization_path: Path,
    output_directory: Path,
    *,
    cell_id: str | None = None,
    method: str | None = None,
    replicate: int | None = None,
) -> dict[str, Any]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("formal pilot execution requires a clean Git worktree")
    context = load_context(config_path)
    _verify_authorization(authorization_path, context)
    sampling = context.config["sampling"]
    pilot_replicates = int(sampling["pilot_replicates"])
    requested = int(sampling["pilot_samples_per_replicate"])
    if cell_id is not None and cell_id not in context.cells_by_id:
        raise ValueError("unknown pilot cell ID")
    if method is not None and method not in REFERENCE_METHODS:
        raise ValueError("unknown pilot reference method")
    if replicate is not None and not 0 <= replicate < pilot_replicates:
        raise ValueError("pilot replicate is outside the frozen roster")
    cells = [cell_id] if cell_id is not None else list(context.cells_by_id)
    methods = [method] if method is not None else list(REFERENCE_METHODS)
    replicates = [replicate] if replicate is not None else list(range(pilot_replicates))
    completed = find_completed_shards(output_directory)
    executed = 0
    skipped = 0
    expected = 0
    for selected_cell in cells:
        for selected_method in methods:
            for selected_replicate in replicates:
                assert selected_cell is not None
                assert selected_method is not None
                assert selected_replicate is not None
                expected += 1
                identity = ReferenceShardIdentity(
                    protocol_id=context.config["protocol_id"],
                    namespace=sampling["pilot_namespace"],
                    stage="pilot",
                    method=cast(ReferenceMethod, selected_method),
                    cell_id=selected_cell,
                    shard_index=selected_replicate,
                )
                existing = completed.get(identity.shard_id)
                if existing is not None:
                    _validate_existing(
                        existing[1],
                        context=context,
                        requested_samples=requested,
                        source_commit=provenance["source_commit"],
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
                    parent_sha256=context.reference_parent_sha256,
                    source_commit=provenance["source_commit"],
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
        "schema": "npi.g11.v8-p5-reference-pilot-run-receipt.v1",
        "protocol_id": context.config["protocol_id"],
        "config_sha256": context.config_sha256,
        "threshold_binding_sha256": context.binding_sha256,
        "pilot_namespace": sampling["pilot_namespace"],
        "source_commit": provenance["source_commit"],
        "environment_sha256": context.environment_sha256,
        "expected_in_selection": expected,
        "executed": executed,
        "skipped": skipped,
        "selection_complete": executed + skipped == expected,
        "formal_reference_complete": False,
        "performance_claim_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cell-id")
    parser.add_argument("--method", choices=REFERENCE_METHODS)
    parser.add_argument("--replicate", type=int)
    arguments = parser.parse_args()
    receipt = run_pilots(
        arguments.config,
        arguments.authorization,
        arguments.output_directory,
        cell_id=arguments.cell_id,
        method=arguments.method,
        replicate=arguments.replicate,
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
