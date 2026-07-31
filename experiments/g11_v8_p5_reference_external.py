"""Environment-authorized, disjoint external execution for the V6 final reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import psutil

from experiments.g11_v8_p5_reference_allocation_amendment import (
    reconstruct_amended_allocation,
)
from experiments.g11_v8_p5_sharded_reference_common import (
    ROOT,
    execute_actual_reference_shard,
    load_context,
)
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_aggregation import aggregate_final_shards
from src.path_integral.reference_protocol import (
    ReferenceShardIdentity,
    canonical_json_bytes,
    canonical_sha256,
)
from src.path_integral.reference_shards import (
    find_completed_shards,
    write_json_atomic_nonoverwriting,
    write_shard_atomic,
)
from src.path_integral.resource_planner import (
    ReferenceBenchmarkObservation,
    forecast_reference_resources,
)

BENCHMARK_SCHEMA = "npi.g11.v8-p5-reference-external-benchmark.v1"
AUTHORIZATION_SCHEMA = "npi.g11.v8-p5-reference-final-authorization.v1"
RUN_RECEIPT_SCHEMA = "npi.g11.v8-p5-reference-external-worker-receipt.v1"
AUDIT_SCHEMA = "npi.g11.v8-p5-reference-external-final-audit.v1"
PARTITION_RULE = "sorted_shard_id_global_index_modulo_partition_count"
IMPLEMENTATION_PATHS = (
    "src/path_integral/reference_protocol.py",
    "src/path_integral/reference_execution.py",
    "src/path_integral/reference_shards.py",
    "src/path_integral/reference_aggregation.py",
    "src/path_integral/resource_planner.py",
    "src/path_integral/controllers/markov.py",
    "src/path_integral/mixture.py",
    "src/path_integral/rbergomi_fft.py",
    "src/path_integral/rbergomi_mixture.py",
    "src/path_integral/control_span_smoothing.py",
    "src/path_integral/gaussian_smoothing.py",
    "src/physics_engine.py",
    "experiments/g11_v8_p5_reference.py",
    "experiments/g11_v8_p5_sharded_reference_common.py",
    "experiments/g11_v8_p5_reference_external.py",
    "configs/g11_v8/p5_sharded_reference_execution_v6.yaml",
    "configs/g11_v8/p5_threshold_manifest_binding_v2.yaml",
    "results/g11_v8_p5_reference_proposal_manifest_v3_2026-07-31.json",
    "results/g11_v8_p5_reference_proposal_manifest_audit_v3_2026-07-31.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("external-reference input must be a JSON mapping")
    return value


def _implementation_records() -> list[dict[str, str]]:
    records = []
    for relative in IMPLEMENTATION_PATHS:
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise ValueError("external implementation path is absent")
        records.append({"path": relative, "sha256": _sha256(path)})
    return records


def _validate_implementation(records: Any) -> None:
    if not isinstance(records, list) or records != _implementation_records():
        raise ValueError("external implementation hashes do not match authorization")


def allocation_chunks(
    manifest: dict[str, Any],
) -> list[tuple[ReferenceShardIdentity, int]]:
    chunks = [
        (
            ReferenceShardIdentity.from_dict(chunk["identity"]),
            int(chunk["requested_samples"]),
        )
        for entry in manifest["entries"]
        for chunk in entry["chunks"]
    ]
    chunks.sort(key=lambda item: item[0].shard_id)
    if len({identity.shard_id for identity, _ in chunks}) != len(chunks):
        raise ValueError("external allocation contains duplicate shard identities")
    return chunks


def partition_chunks(
    manifest: dict[str, Any],
    *,
    partition_index: int,
    partition_count: int,
) -> list[tuple[ReferenceShardIdentity, int]]:
    if partition_count < 1 or not 0 <= partition_index < partition_count:
        raise ValueError("external partition index/count is invalid")
    return [
        item
        for index, item in enumerate(allocation_chunks(manifest))
        if index % partition_count == partition_index
    ]


def _reconstruct(
    config_path: Path,
    package_path: Path,
    failure_receipt_path: Path,
    amendment_receipt_path: Path,
) -> dict[str, Any]:
    manifest, _ = reconstruct_amended_allocation(
        config_path,
        package_path,
        failure_receipt_path,
        amendment_receipt_path,
    )
    return manifest


def run_external_benchmark(
    config_path: Path,
    package_path: Path,
    failure_receipt_path: Path,
    amendment_receipt_path: Path,
    *,
    partition_count: int,
    topology_logical_cpus: int,
    topology_node_count: int,
    parallel_efficiency: float,
    available_memory_bytes: int,
    maximum_wall_seconds: float,
) -> dict[str, Any]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("external benchmark requires a clean Git worktree")
    if (
        partition_count < 1
        or topology_logical_cpus < 1
        or topology_node_count < 1
        or not math.isfinite(parallel_efficiency)
        or not 0.0 < parallel_efficiency <= 1.0
        or available_memory_bytes < 1
        or not math.isfinite(maximum_wall_seconds)
        or maximum_wall_seconds <= 0.0
    ):
        raise ValueError("external benchmark resource contract is invalid")
    context = load_context(config_path)
    manifest = _reconstruct(
        config_path,
        package_path,
        failure_receipt_path,
        amendment_receipt_path,
    )
    sample_count = int(context.config["sampling"]["final_chunk_size"])
    observations: list[dict[str, Any]] = []
    forecast_inputs: list[ReferenceBenchmarkObservation] = []
    for cell_index, cell_id in enumerate(
        context.config["benchmark"]["representative_cells"]
    ):
        for method_index, method in enumerate(
            context.config["benchmark"]["methods"]
        ):
            identity = ReferenceShardIdentity(
                context.config["protocol_id"],
                context.config["benchmark"]["namespace"],
                "pilot",
                method,
                cell_id,
                10_000 + 2 * cell_index + method_index,
            )
            artifact = execute_actual_reference_shard(
                context,
                identity,
                requested_samples=sample_count,
                parent_sha256=context.reference_parent_sha256,
                source_commit=provenance["source_commit"],
                dirty_worktree=False,
                benchmark=True,
            )
            cell = context.cells_by_id[cell_id]
            observation = ReferenceBenchmarkObservation(
                sample_count,
                int(cell["finest_steps"]),
                float(artifact["elapsed_wall_seconds"]),
                float(artifact["elapsed_cpu_seconds"]),
                int(artifact["peak_resident_memory_bytes"]),
                len(canonical_json_bytes(artifact)) + 1,
            )
            forecast_inputs.append(observation)
            observations.append(
                {
                    "cell_id": cell_id,
                    "method": method,
                    "paths": sample_count,
                    "steps": observation.steps,
                    "wall_seconds": observation.wall_seconds,
                    "cpu_seconds": observation.cpu_seconds,
                    "path_steps_per_wall_second": (
                        observation.path_steps_per_wall_second
                    ),
                    "peak_resident_memory_bytes": (
                        observation.peak_resident_memory_bytes
                    ),
                    "artifact_bytes": observation.artifact_bytes,
                    "seed_key_sha256": artifact["seed_key_sha256"],
                }
            )
    total_paths = sum(
        int(entry["requested_final_samples"]) for entry in manifest["entries"]
    )
    forecast = forecast_reference_resources(
        forecast_inputs,
        total_paths=total_paths,
        steps=int(context.config["benchmark"]["forecast_steps"]),
        workers=partition_count,
        parallel_efficiency=parallel_efficiency,
        safety_factor=float(context.config["benchmark"]["forecast_safety_factor"]),
        available_memory_bytes=available_memory_bytes,
        maximum_wall_seconds=maximum_wall_seconds,
    )
    seed_keys = [item["seed_key_sha256"] for item in observations]
    gates = {
        "complete_observation_roster": len(observations) == 6,
        "unique_benchmark_seed_keys": len(seed_keys) == len(set(seed_keys)),
        "allocation_statistically_authorized": manifest[
            "final_execution_authorized"
        ]
        is True,
        "cpu_topology_not_oversubscribed": topology_logical_cpus
        >= partition_count * int(context.environment["torch_threads"]),
        "memory_feasible": forecast["memory_feasible"] is True,
        "wall_feasible": forecast["wall_feasible"] is True,
    }
    return {
        "schema": BENCHMARK_SCHEMA,
        "protocol_id": context.config["protocol_id"],
        "config_sha256": context.config_sha256,
        "allocation_manifest_sha256": canonical_sha256(manifest),
        "benchmark_namespace": context.config["benchmark"]["namespace"],
        "partition_rule": PARTITION_RULE,
        "partition_count": partition_count,
        "topology_logical_cpus": topology_logical_cpus,
        "topology_node_count": topology_node_count,
        "torch_threads_per_worker": context.environment["torch_threads"],
        "parallel_efficiency": parallel_efficiency,
        "available_memory_bytes": available_memory_bytes,
        "maximum_wall_seconds": maximum_wall_seconds,
        "total_requested_final_samples": total_paths,
        "total_final_chunks": len(allocation_chunks(manifest)),
        "observations": observations,
        "forecast": forecast,
        "environment": context.environment,
        "environment_sha256": context.environment_sha256,
        "gates": gates,
        "decision": {
            "status": (
                "external_final_benchmark_pass"
                if all(gates.values())
                else "external_final_benchmark_fail"
            ),
            "final_execution_authorized": False,
            "authorization_build_allowed": all(gates.values()),
            "reference_complete": False,
            "performance_claim_authorized": False,
        },
        **provenance,
    }


def build_final_authorization(
    config_path: Path,
    benchmark_path: Path,
) -> dict[str, Any]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("final authorization requires a clean Git worktree")
    context = load_context(config_path)
    benchmark = _load_json(benchmark_path)
    gates = benchmark.get("gates")
    if (
        benchmark.get("schema") != BENCHMARK_SCHEMA
        or benchmark.get("protocol_id") != context.config["protocol_id"]
        or benchmark.get("config_sha256") != context.config_sha256
        or benchmark.get("source_commit") != provenance["source_commit"]
        or benchmark.get("dirty_worktree") is not False
        or benchmark.get("environment_sha256") != context.environment_sha256
        or not isinstance(gates, dict)
        or not all(gates.values())
        or benchmark.get("decision", {}).get("authorization_build_allowed")
        is not True
    ):
        raise ValueError("external benchmark does not authorize final execution")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "protocol_id": context.config["protocol_id"],
        "config_sha256": context.config_sha256,
        "allocation_manifest_sha256": benchmark["allocation_manifest_sha256"],
        "benchmark_file_sha256": _sha256(benchmark_path),
        "benchmark_canonical_sha256": canonical_sha256(benchmark),
        "execution_source_commit": provenance["source_commit"],
        "dirty_worktree": False,
        "implementation_records": _implementation_records(),
        "final_environment": context.environment,
        "final_environment_sha256": context.environment_sha256,
        "partition_rule": PARTITION_RULE,
        "partition_count": benchmark["partition_count"],
        "topology_logical_cpus": benchmark["topology_logical_cpus"],
        "topology_node_count": benchmark["topology_node_count"],
        "torch_threads_per_worker": benchmark["torch_threads_per_worker"],
        "total_requested_final_samples": benchmark[
            "total_requested_final_samples"
        ],
        "total_final_chunks": benchmark["total_final_chunks"],
        "forecast": benchmark["forecast"],
        "decision": {
            "status": "external_final_execution_authorized",
            "final_execution_authorized": True,
            "reference_complete": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def _validate_authorization(
    config_path: Path,
    manifest: dict[str, Any],
    authorization_path: Path,
) -> tuple[Any, dict[str, Any]]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("external final execution requires a clean Git worktree")
    context = load_context(config_path)
    authorization = _load_json(authorization_path)
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("protocol_id") != context.config["protocol_id"]
        or authorization.get("config_sha256") != context.config_sha256
        or authorization.get("allocation_manifest_sha256")
        != canonical_sha256(manifest)
        or authorization.get("execution_source_commit")
        != provenance["source_commit"]
        or authorization.get("dirty_worktree") is not False
        or authorization.get("final_environment_sha256")
        != context.environment_sha256
        or authorization.get("final_environment") != context.environment
        or authorization.get("partition_rule") != PARTITION_RULE
        or authorization.get("torch_threads_per_worker")
        != context.environment["torch_threads"]
        or int(authorization.get("topology_logical_cpus", 0))
        < int(authorization.get("partition_count", 0))
        * int(context.environment["torch_threads"])
        or int(authorization.get("topology_node_count", 0)) < 1
        or authorization.get("total_requested_final_samples")
        != sum(
            int(entry["requested_final_samples"])
            for entry in manifest["entries"]
        )
        or authorization.get("total_final_chunks")
        != len(allocation_chunks(manifest))
        or authorization.get("decision", {}).get("final_execution_authorized")
        is not True
    ):
        raise ValueError("external final authorization is incompatible")
    _validate_implementation(authorization.get("implementation_records"))
    return context, authorization


def run_external_partition(
    config_path: Path,
    manifest_path: Path,
    authorization_path: Path,
    output_directory: Path,
    *,
    partition_index: int,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    context, authorization = _validate_authorization(
        config_path,
        manifest,
        authorization_path,
    )
    selected = partition_chunks(
        manifest,
        partition_index=partition_index,
        partition_count=int(authorization["partition_count"]),
    )
    completed = find_completed_shards(output_directory)
    executed = 0
    skipped = 0
    manifest_sha256 = canonical_sha256(manifest)
    for identity, requested in selected:
        existing = completed.get(identity.shard_id)
        if existing is not None:
            payload = existing[1]
            if (
                payload["parent_sha256"] != manifest_sha256
                or payload["source_commit"]
                != authorization["execution_source_commit"]
                or payload["environment_sha256"]
                != authorization["final_environment_sha256"]
                or payload["requested_samples"] != requested
            ):
                raise ValueError("external completed shard conflicts with authorization")
            skipped += 1
            continue
        artifact = execute_actual_reference_shard(
            context,
            identity,
            requested_samples=requested,
            parent_sha256=manifest_sha256,
            source_commit=authorization["execution_source_commit"],
            dirty_worktree=False,
        )
        write_shard_atomic(output_directory, artifact)
        executed += 1
    return {
        "schema": RUN_RECEIPT_SCHEMA,
        "protocol_id": context.config["protocol_id"],
        "allocation_manifest_sha256": manifest_sha256,
        "authorization_sha256": canonical_sha256(authorization),
        "partition_rule": PARTITION_RULE,
        "partition_index": partition_index,
        "partition_count": authorization["partition_count"],
        "expected_in_partition": len(selected),
        "executed": executed,
        "skipped": skipped,
        "partition_complete": executed + skipped == len(selected),
        "reference_complete": False,
        "performance_claim_authorized": False,
    }


def aggregate_external_reference(
    config_path: Path,
    manifest_path: Path,
    authorization_path: Path,
    final_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    _, authorization = _validate_authorization(
        config_path,
        manifest,
        authorization_path,
    )
    completed = find_completed_shards(final_directory)
    aggregate = aggregate_final_shards(
        manifest,
        canonical_sha256(manifest),
        [(payload, digest) for _, payload, digest in completed.values()],
        final_source_commit=authorization["execution_source_commit"],
        final_environment_sha256=authorization["final_environment_sha256"],
    )
    return aggregate, authorization


def audit_external_reference(
    config_path: Path,
    manifest_path: Path,
    authorization_path: Path,
    final_directory: Path,
    aggregate_path: Path,
) -> dict[str, Any]:
    recorded = _load_json(aggregate_path)
    recomputed, authorization = aggregate_external_reference(
        config_path,
        manifest_path,
        authorization_path,
        final_directory,
    )
    checks = {
        "aggregate_exactly_recomputed": canonical_sha256(recorded)
        == canonical_sha256(recomputed),
        "final_source_and_environment_authorized": recorded.get("source_commit")
        == authorization["execution_source_commit"]
        and recorded.get("environment_sha256")
        == authorization["final_environment_sha256"],
        "complete_reference_matrix": recorded.get("complete_reference_matrix")
        is True
        and len(recorded.get("cells", [])) == 48
        and len(recorded.get("method_agreements", [])) == 24,
        "all_precision_normalization_agreement_gates": recorded.get(
            "all_target_standard_errors"
        )
        is True
        and recorded.get("all_likelihood_normalizations") is True
        and recorded.get("all_independent_methods_agree") is True
        and recorded.get("reference_acceptance_pass") is True,
        "performance_claim_refused": recorded.get(
            "performance_claim_authorized"
        )
        is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "allocation_manifest_sha256": authorization[
            "allocation_manifest_sha256"
        ],
        "authorization_sha256": canonical_sha256(authorization),
        "aggregate_sha256": canonical_sha256(recorded),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "external_final_reference_pass"
                if not failures
                else "external_final_reference_fail"
            ),
            "reference_complete": not failures,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def _add_allocation_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--amendment-receipt", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark")
    _add_allocation_inputs(benchmark)
    benchmark.add_argument("--partition-count", type=int, required=True)
    benchmark.add_argument("--topology-logical-cpus", type=int, required=True)
    benchmark.add_argument("--topology-node-count", type=int, required=True)
    benchmark.add_argument("--parallel-efficiency", type=float, default=0.70)
    benchmark.add_argument("--available-memory-bytes", type=int)
    benchmark.add_argument("--maximum-wall-seconds", type=float, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--config", type=Path, required=True)
    authorize.add_argument("--benchmark", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--allocation-manifest", type=Path, required=True)
    worker.add_argument("--authorization", type=Path, required=True)
    worker.add_argument("--output-directory", type=Path, required=True)
    worker.add_argument("--partition-index", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--allocation-manifest", type=Path, required=True)
    aggregate.add_argument("--authorization", type=Path, required=True)
    aggregate.add_argument("--final-directory", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--allocation-manifest", type=Path, required=True)
    audit.add_argument("--authorization", type=Path, required=True)
    audit.add_argument("--final-directory", type=Path, required=True)
    audit.add_argument("--aggregate", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "benchmark":
        result = run_external_benchmark(
            arguments.config,
            arguments.package,
            arguments.failure_receipt,
            arguments.amendment_receipt,
            partition_count=arguments.partition_count,
            topology_logical_cpus=arguments.topology_logical_cpus,
            topology_node_count=arguments.topology_node_count,
            parallel_efficiency=arguments.parallel_efficiency,
            available_memory_bytes=(
                arguments.available_memory_bytes
                if arguments.available_memory_bytes is not None
                else int(psutil.virtual_memory().total)
            ),
            maximum_wall_seconds=arguments.maximum_wall_seconds,
        )
        write_json_atomic_nonoverwriting(arguments.output, result)
        print(json.dumps(result["decision"], sort_keys=True))
        if result["decision"]["status"] != "external_final_benchmark_pass":
            raise SystemExit(1)
        return
    if arguments.command == "authorize":
        result = build_final_authorization(arguments.config, arguments.benchmark)
        write_json_atomic_nonoverwriting(arguments.output, result)
        print(json.dumps(result["decision"], sort_keys=True))
        return
    if arguments.command == "worker":
        result = run_external_partition(
            arguments.config,
            arguments.allocation_manifest,
            arguments.authorization,
            arguments.output_directory,
            partition_index=arguments.partition_index,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if arguments.command == "aggregate":
        result, _ = aggregate_external_reference(
            arguments.config,
            arguments.allocation_manifest,
            arguments.authorization,
            arguments.final_directory,
        )
        write_json_atomic_nonoverwriting(arguments.output, result)
        print(
            json.dumps(
                {
                    "reference_acceptance_pass": result[
                        "reference_acceptance_pass"
                    ],
                    "performance_claim_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    result = audit_external_reference(
        arguments.config,
        arguments.allocation_manifest,
        arguments.authorization,
        arguments.final_directory,
        arguments.aggregate,
    )
    write_json_atomic_nonoverwriting(arguments.output, result)
    print(json.dumps(result["decision"], sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
