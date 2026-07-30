"""Build the exact R2 aggregate from the complete frozen final-shard set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_sharded_reference_common import load_context
from src.path_integral.reference_aggregation import (
    aggregate_final_shards,
    validate_allocation_manifest,
)
from src.path_integral.reference_protocol import canonical_sha256
from src.path_integral.reference_shards import (
    find_completed_shards,
    write_json_atomic_nonoverwriting,
)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="ascii"))
    validate_allocation_manifest(payload)
    return payload, canonical_sha256(payload)


def aggregate_reference(
    config_path: Path,
    manifest_path: Path,
    final_directory: Path,
    output_path: Path,
) -> tuple[dict[str, Any], str]:
    context = load_context(config_path)
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    if (
        manifest["config_sha256"] != context.config_sha256
        or manifest["threshold_manifest_sha256"]
        != context.binding["threshold_manifest_sha256"]
    ):
        raise ValueError("aggregate manifest disagrees with execution config")
    completed = find_completed_shards(final_directory)
    final_records = [
        (payload, digest) for _, payload, digest in completed.values()
    ]
    aggregate = aggregate_final_shards(
        manifest, manifest_sha256, final_records
    )
    digest = write_json_atomic_nonoverwriting(output_path, aggregate)
    return aggregate, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument("--final-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    aggregate, digest = aggregate_reference(
        arguments.config,
        arguments.allocation_manifest,
        arguments.final_directory,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "aggregate_sha256": digest,
                "cell_method_count": len(aggregate["cells"]),
                "reference_acceptance_pass": aggregate[
                    "reference_acceptance_pass"
                ],
                "performance_claim_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
