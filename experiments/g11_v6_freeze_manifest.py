"""Outcome-blind promotion of a clean V6 calibration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from src.path_integral import V6CellManifest
from src.path_integral.provenance import source_provenance

FrozenPhase = Literal["qualification", "confirmation"]


def freeze_manifest(
    calibration_path: Path,
    *,
    phase: FrozenPhase,
    protocol: str | None = None,
) -> tuple[V6CellManifest, dict[str, object]]:
    """Freeze calibrated cells without changing an estimand or threshold."""

    if phase not in ("qualification", "confirmation"):
        raise ValueError("manifest phase must be qualification or confirmation")
    raw = calibration_path.read_bytes()
    calibration = json.loads(raw)
    if (
        not isinstance(calibration, dict)
        or calibration.get("schema") != "npi.g11.v6-rarity-calibration.v1"
        or calibration.get("passed") is not True
        or calibration.get("smoke") is not False
    ):
        raise ValueError("manifest freeze requires a passing non-smoke calibration")
    candidate_payload = calibration.get("candidate_manifest")
    if not isinstance(candidate_payload, dict):
        raise ValueError("calibration does not contain a candidate manifest")
    candidate = V6CellManifest.from_dict(candidate_payload)
    if candidate.sha256 != calibration.get("candidate_manifest_sha256"):
        raise ValueError("calibration candidate-manifest hash mismatch")
    if (
        candidate.phase != "development"
        or candidate.frozen
        or candidate.dirty_tree
        or candidate.smoke
        or candidate.source_commit == "uncommitted"
    ):
        raise ValueError("only a clean non-smoke committed development manifest can be frozen")
    if calibration.get("source_commit") != candidate.source_commit:
        raise ValueError("calibration and candidate manifest source commits differ")
    if calibration.get("dirty_worktree") is not False:
        raise ValueError("calibration artifact was not generated from a clean source tree")

    frozen = V6CellManifest(
        schema=candidate.schema,
        protocol=protocol or f"g11-v6-{phase}-manifest-v1",
        phase=phase,
        frozen=True,
        source_commit=candidate.source_commit,
        dirty_tree=False,
        config_sha256=candidate.config_sha256,
        smoke=False,
        cells=candidate.cells,
    )
    tool_provenance = source_provenance()
    receipt: dict[str, object] = {
        "schema": "npi.g11.v6-manifest-freeze-receipt.v1",
        "phase": phase,
        "calibration_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_manifest_sha256": candidate.sha256,
        "frozen_manifest_sha256": frozen.sha256,
        "source_commit": candidate.source_commit,
        "freeze_tool_source_commit": tool_provenance["source_commit"],
        "freeze_tool_dirty_worktree": tool_provenance["dirty_worktree"],
        "formal_freeze_tool_readiness": not bool(
            tool_provenance["dirty_worktree"]
        ),
        "cell_count": len(frozen.cells),
        "estimands_unchanged": all(
            before.to_dict() == after.to_dict()
            for before, after in zip(candidate.cells, frozen.cells, strict=True)
        ),
    }
    if not receipt["estimands_unchanged"]:
        raise AssertionError("manifest freeze changed a calibrated estimand")
    return frozen, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("qualification", "confirmation"), required=True
    )
    parser.add_argument("--protocol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    manifest, receipt = freeze_manifest(
        arguments.calibration,
        phase=arguments.phase,
        protocol=arguments.protocol,
    )
    receipt_path = arguments.receipt or arguments.output.with_suffix(
        arguments.output.suffix + ".receipt.json"
    )
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError("manifest freeze refuses to overwrite an existing artifact")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
