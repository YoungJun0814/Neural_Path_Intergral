"""Fail-closed R1 audit for immutable sharded reference infrastructure."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

import src.path_integral as path_integral
from src.path_integral.reference_aggregation import (
    aggregate_final_shards,
    build_allocation_manifest,
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
    write_shard_atomic,
)
from src.path_integral.resource_planner import (
    ReferenceBenchmarkObservation,
    forecast_reference_resources,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-reference-infrastructure-contract.v1"
REPORT_SCHEMA = "npi.g11.v8-reference-infrastructure-audit.v1"
EXPECTED_MODULES = {
    "src/path_integral/reference_execution.py",
    "src/path_integral/reference_protocol.py",
    "src/path_integral/reference_shards.py",
    "src/path_integral/reference_aggregation.py",
    "src/path_integral/resource_planner.py",
}
EXPECTED_EXPORTS = {
    "ReferenceBatch",
    "ReferenceShardIdentity",
    "SufficientStatistics",
    "aggregate_final_shards",
    "build_allocation_manifest",
    "build_shard_artifact",
    "canonical_sha256",
    "find_completed_shards",
    "forecast_reference_resources",
    "validate_allocation_manifest",
    "validate_shard_artifact",
    "write_shard_atomic",
    "execute_reference_shard",
}
EXPECTED_GATE_CHECKS = {
    "predecessor_hash_and_status",
    "implementation_files_present",
    "public_exports_present",
    "canonical_hash_stability",
    "sufficient_statistics_exactness",
    "immutable_nonoverwriting_storage",
    "corrupted_shard_rejected",
    "duplicate_or_missing_shard_rejected",
    "seed_collision_rejected",
    "allocation_cap_fails_closed",
    "interrupted_resume_exact",
    "reference_acceptance_gates_propagate",
    "resource_forecast_fails_closed",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unexpected R1 reference-infrastructure contract")
    return payload, hashlib.sha256(raw).hexdigest()


def _statistics(values: list[float]) -> SufficientStatistics:
    mean = sum(values) / len(values)
    return SufficientStatistics(
        count=len(values),
        mean=mean,
        m2=sum((value - mean) ** 2 for value in values),
    )


def _artifact(
    identity: ReferenceShardIdentity,
    values: list[float],
    *,
    config_sha256: str,
    parent_sha256: str,
    seed_role: str = "independent",
) -> dict[str, Any]:
    return build_shard_artifact(
        identity=identity,
        config_sha256=config_sha256,
        threshold_manifest_sha256="a" * 64,
        parent_sha256=parent_sha256,
        source_commit="0" * 40,
        dirty_worktree=False,
        environment_sha256="e" * 64,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        seed_key_sha256=seed_key_sha256(
            {"identity": identity.to_dict(), "role": seed_role}
        ),
        requested_samples=len(values),
        contribution=_statistics(values),
        likelihood_normalization=_statistics([1.0] * len(values)),
        invalid_spot_count=0,
        invalid_variance_count=0,
        nonfinite_contribution_count=0,
        elapsed_wall_seconds=0.01,
        elapsed_cpu_seconds=0.01,
        peak_resident_memory_bytes=1024,
    )


def _mechanism_checks(contract_sha256: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["canonical_hash_stability"] = canonical_sha256(
        {"b": [2, 3], "a": 1}
    ) == canonical_sha256({"a": 1, "b": [2, 3]})

    left = _statistics([1.0, 2.0])
    right = _statistics([3.0, 4.0, 5.0])
    merged = left.merge(right)
    checks["sufficient_statistics_exactness"] = (
        merged.count == 5 and merged.mean == 3.0 and merged.variance == 2.5
    )

    protocol = "g11-v8-r1-synthetic-mechanism-v1"
    pilot_namespace = "v8-r1-reference-smoke-pilot"
    final_namespace = "v8-r1-reference-smoke-final"
    cells = ("smoke-terminal", "smoke-barrier")
    pilot_parent = "b" * 64
    pilot_shards: list[tuple[dict[str, Any], str]] = []
    for cell in cells:
        for method in REFERENCE_METHODS:
            for replicate in range(3):
                identity = ReferenceShardIdentity(
                    protocol,
                    pilot_namespace,
                    "pilot",
                    method,
                    cell,
                    replicate,
                )
                artifact = _artifact(
                    identity,
                    [0.0, 1.0, 0.0, 1.0],
                    config_sha256=contract_sha256,
                    parent_sha256=pilot_parent,
                )
                pilot_shards.append((artifact, canonical_sha256(artifact)))

    manifest = build_allocation_manifest(
        protocol_id=protocol,
        config_sha256=contract_sha256,
        threshold_manifest_sha256="a" * 64,
        pilot_parent_sha256=pilot_parent,
        pilot_namespace=pilot_namespace,
        final_namespace=final_namespace,
        expected_cells=cells,
        expected_methods=REFERENCE_METHODS,
        pilot_replicates=3,
        pilot_shards=pilot_shards,
        target_standard_errors={cell: 0.5 for cell in cells},
        allocation_safety_factor=1.0,
        minimum_final_samples=5,
        maximum_final_samples=100,
        final_chunk_size=4,
        source_commit="0" * 40,
        environment_sha256="e" * 64,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        design_informed_by_prior_development_outcomes=True,
        current_namespace_outcomes_inspected_before_freeze=False,
    )
    manifest_sha256 = canonical_sha256(manifest)

    final_shards: list[tuple[dict[str, Any], str]] = []
    for entry in manifest["entries"]:
        for chunk in entry["chunks"]:
            identity = ReferenceShardIdentity.from_dict(chunk["identity"])
            count = int(chunk["requested_samples"])
            values = [float(index % 2) for index in range(count)]
            artifact = _artifact(
                identity,
                values,
                config_sha256=contract_sha256,
                parent_sha256=manifest_sha256,
            )
            final_shards.append((artifact, canonical_sha256(artifact)))

    corrupted = copy.deepcopy(final_shards[0][0])
    corrupted["requested_samples"] += 1
    try:
        validate_shard_artifact(corrupted)
    except ValueError:
        checks["corrupted_shard_rejected"] = True
    else:
        checks["corrupted_shard_rejected"] = False

    try:
        aggregate_final_shards(manifest, manifest_sha256, final_shards[:-1])
    except ValueError:
        missing_rejected = True
    else:
        missing_rejected = False
    try:
        aggregate_final_shards(
            manifest,
            manifest_sha256,
            [*final_shards, final_shards[0]],
        )
    except ValueError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    checks["duplicate_or_missing_shard_rejected"] = (
        missing_rejected and duplicate_rejected
    )

    collision_payload = copy.deepcopy(final_shards[1][0])
    collision_payload["seed_key_sha256"] = final_shards[0][0]["seed_key_sha256"]
    collision_shards = list(final_shards)
    collision_shards[1] = (
        collision_payload,
        canonical_sha256(collision_payload),
    )
    try:
        aggregate_final_shards(manifest, manifest_sha256, collision_shards)
    except ValueError:
        checks["seed_collision_rejected"] = True
    else:
        checks["seed_collision_rejected"] = False

    high_variance_pilots: list[tuple[dict[str, Any], str]] = []
    for artifact, _ in pilot_shards:
        payload = copy.deepcopy(artifact)
        payload["contribution"] = _statistics([0.0, 100.0, 0.0, 100.0]).to_dict()
        high_variance_pilots.append((payload, canonical_sha256(payload)))
    capped = build_allocation_manifest(
        protocol_id=protocol,
        config_sha256=contract_sha256,
        threshold_manifest_sha256="a" * 64,
        pilot_parent_sha256=pilot_parent,
        pilot_namespace=pilot_namespace,
        final_namespace=f"{final_namespace}-capped",
        expected_cells=cells,
        expected_methods=REFERENCE_METHODS,
        pilot_replicates=3,
        pilot_shards=high_variance_pilots,
        target_standard_errors={cell: 0.01 for cell in cells},
        allocation_safety_factor=1.0,
        minimum_final_samples=5,
        maximum_final_samples=5,
        final_chunk_size=4,
        source_commit="0" * 40,
        environment_sha256="e" * 64,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        design_informed_by_prior_development_outcomes=True,
        current_namespace_outcomes_inspected_before_freeze=False,
    )
    checks["allocation_cap_fails_closed"] = (
        capped["final_execution_authorized"] is False
        and all(not entry["chunks"] for entry in capped["entries"])
    )

    with tempfile.TemporaryDirectory(prefix="npi-r1-audit-") as temporary:
        root = Path(temporary)
        interrupted = root / "interrupted"
        uninterrupted = root / "uninterrupted"
        halfway = len(final_shards) // 2
        for artifact, _ in final_shards[:halfway]:
            write_shard_atomic(interrupted, artifact)
        completed = find_completed_shards(interrupted)
        for artifact, _ in final_shards:
            if artifact["shard_id"] not in completed:
                write_shard_atomic(interrupted, artifact)
            write_shard_atomic(uninterrupted, artifact)
        first_artifact = final_shards[0][0]
        try:
            write_shard_atomic(uninterrupted, first_artifact)
        except FileExistsError:
            checks["immutable_nonoverwriting_storage"] = True
        else:
            checks["immutable_nonoverwriting_storage"] = False
        resumed_records = [
            (payload, digest)
            for _, payload, digest in find_completed_shards(interrupted).values()
        ]
        direct_records = [
            (payload, digest)
            for _, payload, digest in find_completed_shards(uninterrupted).values()
        ]
        resumed = aggregate_final_shards(
            manifest, manifest_sha256, resumed_records
        )
        direct = aggregate_final_shards(manifest, manifest_sha256, direct_records)
    checks["interrupted_resume_exact"] = (
        canonical_sha256(resumed) == canonical_sha256(direct)
    )
    checks["reference_acceptance_gates_propagate"] = (
        resumed["reference_acceptance_pass"] is True
        and resumed["performance_claim_authorized"] is False
    )

    observations = [
        ReferenceBenchmarkObservation(100, 128, 2.0, 3.0, 2048, 64),
        ReferenceBenchmarkObservation(100, 128, 4.0, 5.0, 4096, 80),
    ]
    resource = forecast_reference_resources(
        observations,
        total_paths=10_000,
        steps=128,
        workers=4,
        parallel_efficiency=0.5,
        safety_factor=2.0,
        available_memory_bytes=1024,
        maximum_wall_seconds=1.0,
    )
    checks["resource_forecast_fails_closed"] = (
        resource["launch_authorized"] is False
        and resource["memory_feasible"] is False
        and resource["wall_feasible"] is False
    )
    return checks


def audit_reference_infrastructure(
    contract: dict[str, Any],
    contract_sha256: str,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Audit the frozen contract plus executable synthetic falsification checks."""

    checks: dict[str, bool] = {}
    predecessor = contract.get("predecessor")
    if isinstance(predecessor, dict):
        predecessor_path = root / str(predecessor.get("path", ""))
        try:
            predecessor_payload = yaml.safe_load(
                predecessor_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, yaml.YAMLError):
            predecessor_payload = None
        checks["predecessor_hash_and_status"] = (
            predecessor_path.is_file()
            and _sha256(predecessor_path) == predecessor.get("sha256")
            and isinstance(predecessor_payload, dict)
            and predecessor_payload.get("decision", {}).get("status")
            == predecessor.get("required_status")
        )
    else:
        checks["predecessor_hash_and_status"] = False

    implementation = contract.get("implementation")
    modules = (
        implementation.get("modules") if isinstance(implementation, dict) else None
    )
    checks["implementation_files_present"] = (
        isinstance(modules, list)
        and set(modules) == EXPECTED_MODULES
        and all((root / path).is_file() for path in modules)
    )
    checks["public_exports_present"] = (
        isinstance(implementation, dict)
        and implementation.get("public_package_exports_required") is True
        and all(hasattr(path_integral, name) for name in EXPECTED_EXPORTS)
    )

    gate = contract.get("gate")
    declared_gate_checks = (
        gate.get("required_checks") if isinstance(gate, dict) else None
    )
    checks["gate_roster_exact"] = (
        isinstance(declared_gate_checks, list)
        and set(declared_gate_checks) == EXPECTED_GATE_CHECKS
        and len(declared_gate_checks) == len(EXPECTED_GATE_CHECKS)
    )
    checks["contract_scope_fail_closed"] = (
        contract.get("schema") == SCHEMA
        and contract.get("phase") == "r1_reference_infrastructure"
        and contract.get("evidence_class") == "synthetic_mechanism_verification"
        and contract.get("performance_claim_authorized") is False
        and isinstance(implementation, dict)
        and implementation.get("cpu_float64_only") is True
        and implementation.get("gpu_reference_authorized") is False
        and contract.get("decision", {}).get("r2_reference_execution_authorized")
        is False
        and contract.get("decision", {}).get("submission_authorized") is False
    )

    try:
        checks.update(_mechanism_checks(contract_sha256))
    except Exception:
        for name in EXPECTED_GATE_CHECKS - set(checks):
            checks[name] = False
        checks["mechanism_exception_free"] = False
    else:
        checks["mechanism_exception_free"] = True

    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "contract_sha256": contract_sha256,
        "evidence_class": "synthetic_mechanism_verification",
        "checks": checks,
        "failures": sorted(name for name, value in checks.items() if not value),
        "passed": passed,
        "decision": {
            "status": (
                "r1_reference_infrastructure_pass"
                if passed
                else "r1_reference_infrastructure_fail"
            ),
            "r2_development_benchmark_authorized": passed,
            "formal_reference_complete": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/g11_v8/reference_infrastructure_contract_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    contract, digest = load_contract(arguments.contract)
    report = audit_reference_infrastructure(contract, digest)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
