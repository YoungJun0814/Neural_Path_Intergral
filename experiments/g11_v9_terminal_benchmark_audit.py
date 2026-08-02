"""Write the independent V9 terminal benchmark audit artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.path_integral.d1_audit import load_standard_json
from src.path_integral.provenance import source_provenance
from src.path_integral.v9_terminal_benchmark_audit import audit_v9_terminal_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    audit = audit_v9_terminal_benchmark(
        config_path=args.config,
        result=load_standard_json(args.result),
        root=ROOT,
    )
    stage = audit.recomputed_aggregate["stage"]
    payload = {
        "schema": "npi.g11.v9-terminal-benchmark-audit.v1",
        **source_provenance(),
        **asdict(audit),
        "stage": stage,
        "qualification_authorized": (
            audit.passed and stage == "development" and audit.recomputed_aggregate["stage_pass"]
        ),
        "regime_conditional_empirical_claim_authorized": (
            audit.passed and stage == "qualification" and audit.recomputed_aggregate["stage_pass"]
        ),
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": audit.passed,
                "stage_pass": audit.recomputed_aggregate["stage_pass"],
                "failures": audit.failures,
            }
        )
    )


if __name__ == "__main__":
    main()
