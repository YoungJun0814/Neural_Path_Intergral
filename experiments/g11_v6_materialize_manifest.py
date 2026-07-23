"""Materialize and optionally prespecify a primary subset of a V6 calibration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.path_integral import V6CellManifest


def run(
    calibration_path: Path,
    output_path: Path,
    *,
    minimum_nominal_probability: float | None = None,
    selected_cell_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    raw = calibration_path.read_bytes()
    calibration = json.loads(raw)
    if (
        not isinstance(calibration, dict)
        or calibration.get("schema") != "npi.g11.v6-rarity-calibration.v1"
        or not calibration.get("passed")
    ):
        raise ValueError("manifest materialization requires a passing calibration artifact")
    manifest_payload = calibration.get("candidate_manifest")
    if not isinstance(manifest_payload, dict):
        raise ValueError("calibration artifact lacks a candidate manifest")
    manifest = V6CellManifest.from_dict(manifest_payload)
    if manifest.sha256 != calibration.get("candidate_manifest_sha256"):
        raise ValueError("calibration candidate-manifest hash mismatch")
    if minimum_nominal_probability is not None:
        if (
            not math.isfinite(minimum_nominal_probability)
            or not 0.0 < minimum_nominal_probability <= 1.0
        ):
            raise ValueError("minimum nominal probability must lie in (0, 1]")
        cells = tuple(
            cell
            for cell in manifest.cells
            if cell.nominal_probability >= minimum_nominal_probability
        )
        if not cells:
            raise ValueError("primary manifest filter removed every cell")
        manifest = V6CellManifest(
            schema=manifest.schema,
            protocol=f"{manifest.protocol}-primary-ge-{minimum_nominal_probability:.0e}",
            phase=manifest.phase,
            frozen=manifest.frozen,
            source_commit=manifest.source_commit,
            dirty_tree=manifest.dirty_tree,
            config_sha256=manifest.config_sha256,
            smoke=manifest.smoke,
            cells=cells,
        )
    if selected_cell_ids is not None:
        if not selected_cell_ids:
            raise ValueError("selected cell IDs must not be empty")
        if len(set(selected_cell_ids)) != len(selected_cell_ids):
            raise ValueError("selected cell IDs must be unique")
        requested = set(selected_cell_ids)
        available = {cell.cell_id for cell in manifest.cells}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"selected cell IDs are absent from the manifest: {unknown}")
        cells = tuple(cell for cell in manifest.cells if cell.cell_id in requested)
        manifest = V6CellManifest(
            schema=manifest.schema,
            protocol=f"{manifest.protocol}-cells-{'-'.join(selected_cell_ids)}",
            phase=manifest.phase,
            frozen=manifest.frozen,
            source_commit=manifest.source_commit,
            dirty_tree=manifest.dirty_tree,
            config_sha256=manifest.config_sha256,
            smoke=manifest.smoke,
            cells=cells,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "schema": "npi.g11.v6-manifest-materialization.v1",
        "calibration_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": manifest.sha256,
        "cell_count": len(manifest.cells),
        "minimum_nominal_probability": minimum_nominal_probability,
        "selected_cell_ids": list(selected_cell_ids) if selected_cell_ids is not None else None,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-nominal-probability", type=float)
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    arguments = parser.parse_args()
    receipt = run(
        arguments.calibration,
        arguments.output,
        minimum_nominal_probability=arguments.minimum_nominal_probability,
        selected_cell_ids=tuple(arguments.cell_ids) if arguments.cell_ids is not None else None,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
