"""Write the independent D1 Stage B audit artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.path_integral.d1_audit import load_standard_json
from src.path_integral.d1_stage_b_audit import audit_d1_stage_b
from src.path_integral.provenance import source_provenance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    audit = audit_d1_stage_b(
        config_path=args.config,
        result=load_standard_json(args.result),
        root=ROOT,
    )
    payload = {
        "schema": "npi.g11.v8-d1-stage-b-audit.v1",
        **source_provenance(),
        **asdict(audit),
        "stage_c_authorized": audit.recomputed_aggregate["stage_b_complete"],
        "p8_qualification_authorized": False,
        "performance_claim_authorized": False,
        "submission_authorized": False,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": audit.passed, "failures": audit.failures}))


if __name__ == "__main__":
    main()
