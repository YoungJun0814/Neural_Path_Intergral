"""Strict schema, hash, provenance, and gate audit for G11 development evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.provenance import runtime_provenance, source_provenance


def _strict_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    payload = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact root is not an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _boolean_gate_failures(payload: dict[str, Any]) -> list[str]:
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        return ["missing gates object"]
    return [name for name, value in gates.items() if isinstance(value, bool) and not value]


def _provenance_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    environment = payload.get("environment")
    required_environment = {
        "python",
        "pytorch",
        "cuda",
        "os",
        "processor",
        "torch_threads",
        "dtype",
        "deterministic_algorithms",
        "packages",
    }
    if not isinstance(environment, dict) or not required_environment.issubset(environment):
        failures.append("incomplete environment provenance")
    work = payload.get("work_ledger")
    if not isinstance(work, dict) or not work:
        failures.append("missing categorized work ledger")
    if not isinstance(payload.get("source_commit"), str):
        failures.append("missing source commit")
    if not isinstance(payload.get("dirty_worktree"), bool):
        failures.append("missing dirty-worktree flag")
    seed_hash = payload.get("seed_ledger_sha256")
    if not isinstance(seed_hash, str) or len(seed_hash) != 64:
        failures.append("invalid seed-ledger hash")
    return failures


def run(manifest_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    raw_manifest = manifest_path.read_bytes()
    manifest = yaml.safe_load(raw_manifest)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported G11 artifact manifest")
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in manifest["artifacts"]:
        config_path = Path(entry["config"])
        result_path = Path(entry["result"])
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        payload, result_hash = _strict_json(result_path)
        local_failures: list[str] = []
        if payload.get(entry["config_hash_field"]) != config_hash:
            local_failures.append("config byte hash mismatch")
        gate_failures = _boolean_gate_failures(payload)
        expected_pass = bool(entry.get("expected_pass", True))
        if expected_pass:
            local_failures.extend(gate_failures)
        else:
            expected_failed_gates = sorted(entry.get("expected_failed_gates", []))
            if payload.get("passed") is not False:
                local_failures.append("expected falsification artifact passed")
            if sorted(gate_failures) != expected_failed_gates:
                local_failures.append(
                    "falsification gate signature differs from the manifest"
                )
        local_failures.extend(_provenance_failures(payload))
        if local_failures:
            failures.extend(f"{result_path}: {item}" for item in local_failures)
        records.append(
            {
                "config": str(config_path),
                "config_sha256": config_hash,
                "result": str(result_path),
                "result_sha256": result_hash,
                "expected_pass": expected_pass,
                "observed_failed_gates": gate_failures,
                "failures": local_failures,
            }
        )
    aggregate_records: list[dict[str, Any]] = []
    for raw_path in manifest["aggregate_artifacts"]:
        path = Path(raw_path)
        payload, digest = _strict_json(path)
        local_failures = _boolean_gate_failures(payload)
        if payload.get("passed") is not True:
            local_failures.append("aggregate passed flag is false")
        if local_failures:
            failures.extend(f"{path}: {item}" for item in local_failures)
        aggregate_records.append(
            {"result": str(path), "result_sha256": digest, "failures": local_failures}
        )
    return {
        "schema": "npi.g11.artifact-audit.v1",
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "artifacts": records,
        "aggregate_artifacts": aggregate_records,
        "failures": failures,
        "passed": not failures,
        "work_ledger": {"audit_seconds": time.perf_counter() - started},
        "environment": runtime_provenance(dtype="not_applicable"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/g11_artifact_manifest.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "failures": result["failures"]}))


if __name__ == "__main__":
    main()
