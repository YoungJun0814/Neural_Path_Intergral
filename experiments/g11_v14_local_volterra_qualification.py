"""Run the hash-bound V14 local-Volterra qualification matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v14_local_volterra_development import (
    ROOT,
    _bound_cells,
    _problem,
    _record,
    _sha256,
    load_config,
)
from src.path_integral.ecrpt_protocol import aggregate_local_volterra_development
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.seed_ledger import SeedLedger


def validate_qualification_config(config: dict[str, Any], *, root: Path = ROOT) -> None:
    if config.get("stage") != "qualification":
        raise ValueError("qualification runner requires qualification stage")
    for name, binding in config.get("bindings", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid qualification binding: {name}")
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"qualification binding mismatch: {name}")
    development = yaml.safe_load(
        (root / config["bindings"]["development_config"]["path"]).read_text(encoding="utf-8")
    )
    frozen_fields = (
        "model",
        "cells",
        "clusters",
        "relative_rmse_target",
        "primary_query_count",
        "primary_comparators",
        "candidate_training",
        "final_budget",
        "tail_safe_policy",
        "gate",
    )
    if any(config[field] != development[field] for field in frozen_fields):
        raise ValueError("qualification design differs from authorized development")
    result = json.loads(
        (root / config["bindings"]["development_result"]["path"]).read_text(encoding="utf-8")
    )
    audit = json.loads(
        (root / config["bindings"]["development_audit"]["path"]).read_text(encoding="utf-8")
    )
    if not bool(result["decision"]["qualification_authorized"]):
        raise ValueError("development did not authorize qualification")
    if not bool(audit["passed"]):
        raise ValueError("development audit did not pass")


def run_qualification(
    config: dict[str, Any], config_sha256: str, *, root: Path = ROOT
) -> dict[str, Any]:
    validate_qualification_config(config, root=root)
    cells = _bound_cells(config, root=root)
    bank = json.loads(
        (root / config["bindings"]["v10r1_proposal_bank"]["path"]).read_text(encoding="utf-8")
    )
    bank_index = {
        (str(entry["cell_id"]), int(entry["replicate"])): entry for entry in bank["entries"]
    }
    ledger = SeedLedger()
    records = []
    for cell in cells:
        problem = _problem(config, cell)
        for cluster in range(int(config["clusters"])):
            key = (problem.task_id, cluster)
            if key not in bank_index:
                raise ValueError(f"V10R1 bank lacks {key}")
            records.append(
                _record(
                    problem,
                    cell,
                    cluster,
                    config,
                    ledger,
                    bank_index[key],
                )
            )
    aggregate = aggregate_local_volterra_development(config=config, records=records)
    return {
        "schema": "npi.g11.v14-local-volterra-qualification-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "stage": "qualification",
        "config_path": str(config["config_path"]),
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "bindings": config["bindings"],
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "records": records,
        "aggregate": aggregate,
        "decision": {
            "software_correctness_evidence_pass": bool(aggregate["correctness_pass"]),
            "development_performance_gate_pass": bool(aggregate["performance_pass"]),
            "qualification_authorized": False,
            "qualification_executed": True,
            "distribution_free_tail_claim_authorized": bool(
                aggregate["distribution_free_tail_certificate_pass"]
            ),
            "broad_performance_claim_authorized": False,
            "top_journal_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, digest = load_config(args.config)
    result = run_qualification(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
