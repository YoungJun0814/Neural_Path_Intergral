"""Materialize a deterministic V3 proposal bank from clean pure-CEM training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import yaml

from experiments.g11_v6_proposal_source import (
    task_conditioned_training_source_audit,
    task_conditioned_training_source_summary,
)
from src.path_integral.provenance import source_provenance

Phase = Literal["development", "qualification"]


def materialize_proposal_policy(
    template_path: Path,
    training_source_path: Path,
    *,
    protocol_id: str,
    phase: Phase,
    manifest_cell_count: int,
    clusters: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Insert a source-derived proposal bank without copying measured values by hand."""

    raw = template_path.read_bytes()
    template = yaml.safe_load(raw)
    if (
        not isinstance(template, dict)
        or template.get("schema")
        not in {
            "npi.g11.v6-routed-policy.config.v2",
            "npi.g11.v6-routed-policy.config.v3",
        }
        or template.get("phase") != "development"
        or template.get("frozen") is not False
    ):
        raise ValueError(
            "proposal materialization requires an unfrozen V2/V3 development template"
        )
    if phase not in ("development", "qualification"):
        raise ValueError("materialized phase must be development or qualification")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("protocol id must be a nonempty string")
    if (
        isinstance(manifest_cell_count, bool)
        or not isinstance(manifest_cell_count, int)
        or manifest_cell_count < 1
        or isinstance(clusters, bool)
        or not isinstance(clusters, int)
        or clusters < 1
    ):
        raise ValueError("manifest cell and cluster counts must be positive integers")

    summary = task_conditioned_training_source_summary(training_source_path)
    if phase == "qualification" and not summary["formal_training_source_readiness"]:
        raise ValueError(
            "qualification proposal bank requires clean committed non-smoke training"
        )
    proposal = dict(template["proposal"])
    proposal.update(
        {
            "task_controls": summary["task_controls"],
            "training_source_artifact_sha256": summary[
                "source_artifact_sha256"
            ],
            "training_derivation": summary["derivation"],
            "training_source_record_count": summary["source_record_count"],
            "training_total_samples": summary["total_samples"],
            "training_total_work_units": summary["total_work_units"],
            "training_total_wall_seconds": summary["total_wall_seconds"],
            "training_total_cpu_seconds": summary["total_cpu_seconds"],
            "training_amortization_record_count": manifest_cell_count * clusters,
        }
    )
    output = json.loads(json.dumps(template))
    output["schema"] = "npi.g11.v6-routed-policy.config.v3"
    output["protocol_id"] = protocol_id
    output["phase"] = phase
    output["frozen"] = phase == "qualification"
    output["proposal"] = proposal
    output["sampling"]["clusters"] = clusters
    audit = task_conditioned_training_source_audit(
        output["proposal"], training_source_path
    )
    provenance = source_provenance()
    receipt = {
        "schema": "npi.g11.v6-proposal-bank-materialization-receipt.v1",
        "protocol_id": protocol_id,
        "phase": phase,
        "manifest_cell_count": manifest_cell_count,
        "clusters": clusters,
        "training_amortization_record_count": manifest_cell_count * clusters,
        "proposal_training_audit": audit,
        "formal_readiness": {
            "clean_source": not bool(provenance["dirty_worktree"]),
            "nondevelopment_output": phase == "qualification",
            "formal_training_source": summary[
                "formal_training_source_readiness"
            ],
        },
        **provenance,
    }
    return output, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--proposal-training-source", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument(
        "--phase", choices=("development", "qualification"), required=True
    )
    parser.add_argument("--manifest-cell-count", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    receipt_path = arguments.receipt or arguments.output.with_suffix(
        arguments.output.suffix + ".receipt.json"
    )
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError(
            "proposal materialization refuses to overwrite an output or receipt"
        )
    policy, receipt = materialize_proposal_policy(
        arguments.template,
        arguments.proposal_training_source,
        protocol_id=arguments.protocol_id,
        phase=arguments.phase,
        manifest_cell_count=arguments.manifest_cell_count,
        clusters=arguments.clusters,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        yaml.safe_dump(
            policy,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
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
