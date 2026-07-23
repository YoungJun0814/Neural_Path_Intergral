"""Audit seed-block separation across V6 research stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from src.path_integral import SeedLedger
from src.path_integral.provenance import source_provenance


def _cell_ids(artifact: dict[str, Any]) -> set[str]:
    identifiers = {
        str(record["cell_id"])
        for record in artifact.get("records", [])
        if isinstance(record, dict) and isinstance(record.get("cell_id"), str)
    }
    candidate = artifact.get("candidate_manifest")
    if isinstance(candidate, dict):
        identifiers.update(
            str(cell["cell_id"])
            for cell in candidate.get("cells", [])
            if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)
        )
    return identifiers


def audit_stage_splits(stage_paths: dict[str, Path]) -> dict[str, Any]:
    """Require independent canonical seed namespaces for every declared stage."""

    if len(stage_paths) < 2:
        raise ValueError("split audit requires at least two stages")
    if any(not name or name.strip() != name for name in stage_paths):
        raise ValueError("stage names must be nonempty and stripped")
    stages: dict[str, dict[str, Any]] = {}
    for name, path in sorted(stage_paths.items()):
        raw = path.read_bytes()
        artifact = json.loads(raw)
        if not isinstance(artifact, dict):
            raise ValueError(f"stage {name} artifact must be a JSON object")
        ledger_payload = artifact.get("seed_ledger")
        if not isinstance(ledger_payload, dict):
            raise ValueError(f"stage {name} lacks a seed ledger")
        ledger = SeedLedger.from_dict(ledger_payload)
        if not len(ledger):
            raise ValueError(f"stage {name} seed ledger is empty")
        declared_hash = artifact.get("seed_ledger_sha256")
        if declared_hash is not None and declared_hash != ledger.sha256:
            raise ValueError(f"stage {name} seed-ledger hash mismatch")
        stages[name] = {
            "artifact_sha256": hashlib.sha256(raw).hexdigest(),
            "ledger_sha256": ledger.sha256,
            "seed_count": len(ledger),
            "seeds": {record.seed for record in ledger.records},
            "protocols": {record.key.protocol for record in ledger.records},
            "cell_ids": _cell_ids(artifact),
        }

    pairwise = []
    all_disjoint = True
    for left, right in combinations(sorted(stages), 2):
        left_stage = stages[left]
        right_stage = stages[right]
        shared_seeds = sorted(left_stage["seeds"] & right_stage["seeds"])
        shared_protocols = sorted(left_stage["protocols"] & right_stage["protocols"])
        disjoint = not shared_seeds and not shared_protocols
        all_disjoint = all_disjoint and disjoint
        pairwise.append(
            {
                "left": left,
                "right": right,
                "disjoint_seed_values": not shared_seeds,
                "disjoint_protocol_namespaces": not shared_protocols,
                "shared_seed_count": len(shared_seeds),
                "shared_protocols": shared_protocols,
                "shared_cell_ids": sorted(
                    left_stage["cell_ids"] & right_stage["cell_ids"]
                ),
                "passed": disjoint,
            }
        )
    if not all_disjoint:
        raise ValueError("research stages reuse a seed value or protocol namespace")
    provenance = source_provenance()
    return {
        "schema": "npi.g11.v6-split-audit.v1",
        "separation_basis": "disjoint_canonical_seed_blocks",
        "stage_count": len(stages),
        "stages": {
            name: {
                key: value
                for key, value in stage.items()
                if key not in {"seeds", "protocols", "cell_ids"}
            }
            | {
                "protocols": sorted(stage["protocols"]),
                "cell_ids": sorted(stage["cell_ids"]),
            }
            for name, stage in stages.items()
        },
        "pairwise": pairwise,
        "passed": True,
        "formal_readiness": {
            "clean_source": not bool(provenance["dirty_worktree"]),
            "all_seed_blocks_disjoint": True,
        },
        **provenance,
    }


def _parse_stage(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("stage must be NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="append", type=_parse_stage, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    stage_paths = dict(arguments.stage)
    if len(stage_paths) != len(arguments.stage):
        raise ValueError("split-audit stage names must be unique")
    result = audit_stage_splits(stage_paths)
    if arguments.output.exists():
        raise FileExistsError("split audit refuses to overwrite an existing artifact")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
