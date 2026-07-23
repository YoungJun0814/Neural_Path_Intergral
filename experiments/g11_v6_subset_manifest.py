"""Create a hash-audited phase-preserving subset of a V6 cell manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.path_integral import V6CellManifest
from src.path_integral.provenance import source_provenance


def subset_manifest(
    manifest_path: Path,
    selected_cell_ids: tuple[str, ...],
    *,
    protocol: str,
) -> tuple[V6CellManifest, dict[str, object]]:
    """Preserve every selected estimand and all phase/freeze metadata."""

    if not selected_cell_ids or len(set(selected_cell_ids)) != len(
        selected_cell_ids
    ):
        raise ValueError("selected cell ids must be nonempty and unique")
    if not protocol or protocol.strip() != protocol:
        raise ValueError("subset protocol must be a nonempty stripped string")
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("V6 subset source must be a manifest object")
    manifest = V6CellManifest.from_dict(payload)
    by_id = {cell.cell_id: cell for cell in manifest.cells}
    missing = sorted(set(selected_cell_ids) - set(by_id))
    if missing:
        raise ValueError(f"subset cells are absent from the manifest: {missing}")
    cells = tuple(by_id[cell_id] for cell_id in selected_cell_ids)
    subset = V6CellManifest(
        schema=manifest.schema,
        protocol=protocol,
        phase=manifest.phase,
        frozen=manifest.frozen,
        source_commit=manifest.source_commit,
        dirty_tree=manifest.dirty_tree,
        config_sha256=manifest.config_sha256,
        smoke=manifest.smoke,
        cells=cells,
    )
    provenance = source_provenance()
    receipt: dict[str, object] = {
        "schema": "npi.g11.v6-manifest-subset-receipt.v1",
        "source_manifest_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_sha256": manifest.sha256,
        "subset_manifest_sha256": subset.sha256,
        "source_cell_count": len(manifest.cells),
        "subset_cell_count": len(subset.cells),
        "selected_cell_ids": list(selected_cell_ids),
        "phase_preserved": subset.phase == manifest.phase,
        "frozen_status_preserved": subset.frozen == manifest.frozen,
        "estimands_unchanged": all(
            by_id[cell.cell_id].to_dict() == cell.to_dict()
            for cell in subset.cells
        ),
        "formal_readiness": {
            "clean_source": not bool(provenance["dirty_worktree"]),
            "clean_non_smoke_source_manifest": (
                not manifest.dirty_tree
                and not manifest.smoke
                and manifest.source_commit != "uncommitted"
            ),
        },
        **provenance,
    }
    if not all(
        bool(receipt[key])
        for key in (
            "phase_preserved",
            "frozen_status_preserved",
            "estimands_unchanged",
        )
    ):
        raise AssertionError("manifest subset changed phase or an estimand")
    return subset, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-id", action="append", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    receipt_path = arguments.receipt or arguments.output.with_suffix(
        arguments.output.suffix + ".receipt.json"
    )
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError("manifest subset refuses existing outputs")
    subset, receipt = subset_manifest(
        arguments.manifest,
        tuple(arguments.cell_id),
        protocol=arguments.protocol,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            subset.to_dict(), indent=2, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
