"""Post-confirmation audit of qualification/confirmation seed disjointness."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from src.path_integral.provenance import runtime_provenance, source_provenance


def _load(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _seed_set(payload: dict[str, Any], name: str) -> set[int]:
    records = payload.get("seed_ledger", {}).get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name} lacks a seed ledger")
    seeds = []
    for record in records:
        seed = record.get("seed") if isinstance(record, dict) else None
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{name} contains a malformed seed")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{name} contains duplicate seeds")
    return set(seeds)


def audit_payloads(
    payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    expected_phases = {
        "qualification_probe": "qualification",
        "qualification_fixed": "qualification",
        "confirmation_probe": "confirmation",
        "confirmation_fixed": "confirmation",
    }
    failures = []
    seed_sets = {}
    protocols = {}
    for name, expected_phase in expected_phases.items():
        payload = payloads[name]
        if payload.get("phase") != expected_phase:
            failures.append(f"{name} phase mismatch")
        if bool(payload.get("dirty_worktree")):
            failures.append(f"{name} was generated from a dirty tree")
        protocol = payload.get("protocol_id")
        if not isinstance(protocol, str) or not protocol:
            failures.append(f"{name} lacks a protocol ID")
        else:
            protocols[name] = protocol
        try:
            seed_sets[name] = _seed_set(payload, name)
        except ValueError as error:
            failures.append(str(error))
    if len(set(protocols.values())) != len(protocols):
        failures.append("phase artifacts reuse a protocol ID")

    intersections = {}
    for left, right in itertools.combinations(sorted(seed_sets), 2):
        label = f"{left}__{right}"
        intersections[label] = len(seed_sets[left] & seed_sets[right])
        if intersections[label]:
            failures.append(f"{label} seed intersection is nonempty")
    return (
        {name: len(seeds) for name, seeds in seed_sets.items()},
        intersections,
        failures,
    )


def audit(
    qualification_probe_path: Path,
    qualification_fixed_path: Path,
    confirmation_probe_path: Path,
    confirmation_fixed_path: Path,
) -> dict[str, Any]:
    paths = {
        "qualification_probe": (
            qualification_probe_path,
            "npi.g11.v7-mechanism-probe.v1",
        ),
        "qualification_fixed": (
            qualification_fixed_path,
            "npi.g11.v6-secondary-baselines.v1",
        ),
        "confirmation_probe": (
            confirmation_probe_path,
            "npi.g11.v7-mechanism-probe.v1",
        ),
        "confirmation_fixed": (
            confirmation_fixed_path,
            "npi.g11.v6-secondary-baselines.v1",
        ),
    }
    payloads = {}
    hashes = {}
    for name, (path, schema) in paths.items():
        payloads[name], hashes[name] = _load(path, schema)
    counts, intersections, failures = audit_payloads(payloads)
    return {
        "schema": "npi.g11.v7-phase-seed-audit.v1",
        "source_artifact_sha256": hashes,
        "seed_counts": counts,
        "pairwise_intersection_counts": intersections,
        "failures": failures,
        "phase_seed_audit_passed": not failures,
        "environment": runtime_provenance(dtype="serialized-int64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-probe", type=Path, required=True)
    parser.add_argument("--qualification-fixed", type=Path, required=True)
    parser.add_argument("--confirmation-probe", type=Path, required=True)
    parser.add_argument("--confirmation-fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(
        arguments.qualification_probe,
        arguments.qualification_fixed,
        arguments.confirmation_probe,
        arguments.confirmation_fixed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase_seed_audit_passed": result["phase_seed_audit_passed"],
                "failure_count": len(result["failures"]),
                "seed_counts": result["seed_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
