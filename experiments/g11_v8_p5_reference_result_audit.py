"""Independent recomputation audit of the complete sharded R2 reference."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_sharded_reference_common import load_context
from src.path_integral.reference_aggregation import (
    aggregate_final_shards,
    validate_allocation_manifest,
)
from src.path_integral.reference_protocol import canonical_sha256
from src.path_integral.reference_shards import find_completed_shards

REPORT_SCHEMA = "npi.g11.v8-p5-sharded-reference-audit.v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("reference audit input must be a mapping")
    return payload


def _commit_is_ancestor(commit: Any) -> bool:
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    try:
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def audit_reference_result(
    config_path: Path,
    manifest_path: Path,
    final_directory: Path,
    aggregate_path: Path,
) -> dict[str, Any]:
    context = load_context(config_path)
    manifest = _load_json(manifest_path)
    validate_allocation_manifest(manifest)
    manifest_sha256 = canonical_sha256(manifest)
    aggregate = _load_json(aggregate_path)
    completed = find_completed_shards(final_directory)
    recomputed = aggregate_final_shards(
        manifest,
        manifest_sha256,
        [(payload, digest) for _, payload, digest in completed.values()],
    )
    cell_ids = {cell["cell_id"] for cell in aggregate.get("cells", [])}
    agreement_ids = {
        cell["cell_id"] for cell in aggregate.get("method_agreements", [])
    }
    checks = {
        "config_hash_exact": manifest.get("config_sha256")
        == aggregate.get("config_sha256")
        == context.config_sha256,
        "threshold_hash_exact": manifest.get("threshold_manifest_sha256")
        == aggregate.get("threshold_manifest_sha256")
        == context.binding["threshold_manifest_sha256"],
        "allocation_hash_exact": aggregate.get("allocation_manifest_sha256")
        == manifest_sha256,
        "aggregate_exactly_recomputed": canonical_sha256(aggregate)
        == canonical_sha256(recomputed),
        "complete_48_method_cell_matrix": len(aggregate.get("cells", [])) == 48
        and len(cell_ids) == 24
        and cell_ids == set(context.cells_by_id),
        "complete_24_agreement_matrix": len(
            aggregate.get("method_agreements", [])
        )
        == 24
        and agreement_ids == set(context.cells_by_id),
        "complete_shard_roster": aggregate.get("complete_reference_matrix") is True,
        "all_precision_targets_pass": aggregate.get(
            "all_target_standard_errors"
        )
        is True,
        "all_normalizations_pass": aggregate.get(
            "all_likelihood_normalizations"
        )
        is True,
        "all_independent_methods_agree": aggregate.get(
            "all_independent_methods_agree"
        )
        is True,
        "reference_acceptance_pass": aggregate.get("reference_acceptance_pass")
        is True,
        "source_commit_traceable": aggregate.get("source_commit")
        == manifest.get("source_commit")
        and _commit_is_ancestor(manifest.get("source_commit")),
        "performance_claim_refused": aggregate.get(
            "performance_claim_authorized"
        )
        is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    passed = not failures
    return {
        "schema": REPORT_SCHEMA,
        "config_sha256": context.config_sha256,
        "threshold_binding_sha256": context.binding_sha256,
        "allocation_manifest_sha256": manifest_sha256,
        "aggregate_sha256": canonical_sha256(aggregate),
        "checks": checks,
        "failures": failures,
        "passed": passed,
        "decision": {
            "status": (
                "r2_development_reference_pass"
                if passed
                else "r2_development_reference_fail"
            ),
            "development_reference_complete": passed,
            "fresh_qualification_reference_required": True,
            "p8_qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument("--final-directory", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"refusing to overwrite reference audit: {arguments.output}"
        )
    report = audit_reference_result(
        arguments.config,
        arguments.allocation_manifest,
        arguments.final_directory,
        arguments.aggregate,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
