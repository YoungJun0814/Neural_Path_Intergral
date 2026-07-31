"""Freeze exact R2 final counts from the complete independent pilot roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_sharded_reference_common import load_context
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_aggregation import build_allocation_manifest
from src.path_integral.reference_protocol import REFERENCE_METHODS
from src.path_integral.reference_shards import (
    find_completed_shards,
    write_json_atomic_nonoverwriting,
)


def freeze_allocation(
    config_path: Path,
    pilot_directory: Path,
    output_path: Path,
) -> tuple[dict[str, Any], str]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("allocation freeze requires a clean Git worktree")
    context = load_context(config_path)
    sampling = context.config["sampling"]
    contract = context.config["reference_contract"]
    completed = find_completed_shards(pilot_directory)
    pilot_records = [
        (payload, digest) for _, payload, digest in completed.values()
    ]
    method_relative_targets = contract.get(
        "method_relative_standard_error_targets"
    )
    if method_relative_targets is None:
        target_fraction = float(
            contract["maximum_reference_se_fraction_of_final_target"]
        )
        relative_rmse = float(contract["final_relative_rmse_design_target"])
        target_standard_errors: dict[str | tuple[str, str], float] = {
            cell_id: target_fraction
            * relative_rmse
            * float(cell["nominal_probability"])
            for cell_id, cell in context.cells_by_id.items()
        }
    else:
        target_standard_errors = {
            (cell_id, method): float(method_relative_targets[method])
            * float(cell["nominal_probability"])
            for cell_id, cell in context.cells_by_id.items()
            for method in REFERENCE_METHODS
        }
    manifest = build_allocation_manifest(
        protocol_id=context.config["protocol_id"],
        config_sha256=context.config_sha256,
        threshold_manifest_sha256=context.binding["threshold_manifest_sha256"],
        pilot_parent_sha256=context.reference_parent_sha256,
        pilot_namespace=sampling["pilot_namespace"],
        final_namespace=sampling["final_namespace"],
        expected_cells=list(context.cells_by_id),
        expected_methods=REFERENCE_METHODS,
        pilot_replicates=int(sampling["pilot_replicates"]),
        pilot_shards=pilot_records,
        target_standard_errors=target_standard_errors,
        allocation_safety_factor=float(sampling["allocation_safety_factor"]),
        minimum_final_samples=int(sampling["minimum_final_samples"]),
        maximum_final_samples=int(sampling["maximum_final_samples"]),
        final_chunk_size=int(sampling["final_chunk_size"]),
        source_commit=provenance["source_commit"],
        environment_sha256=context.environment_sha256,
        estimand=contract["estimand"],
        dtype=contract["dtype"],
        device=contract["device"],
        design_informed_by_prior_development_outcomes=context.config[
            "design_informed_by_prior_development_outcomes"
        ],
        current_namespace_outcomes_inspected_before_freeze=context.config[
            "current_namespace_outcomes_inspected_before_freeze"
        ],
        maximum_final_samples_by_method=sampling.get(
            "maximum_final_samples_by_method"
        ),
    )
    digest = write_json_atomic_nonoverwriting(output_path, manifest)
    return manifest, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest, digest = freeze_allocation(
        arguments.config, arguments.pilot_directory, arguments.output
    )
    print(
        json.dumps(
            {
                "allocation_manifest_sha256": digest,
                "all_resources_feasible": manifest["all_resources_feasible"],
                "final_execution_authorized": manifest[
                    "final_execution_authorized"
                ],
                "entry_count": len(manifest["entries"]),
                "performance_claim_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
