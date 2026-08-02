"""Write an independent audit artifact for a D1 Stage A result."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.path_integral.d1_audit import audit_d1_stage_a, load_standard_json
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
    result = load_standard_json(args.result)
    audit = audit_d1_stage_a(config_path=args.config, result=result, root=ROOT)
    payload = {
        "schema": "npi.g11.v8-d1-p7-falsification-stage-a-audit.v1",
        "config_path": args.config.as_posix(),
        "result_path": args.result.as_posix(),
        **source_provenance(),
        **asdict(audit),
        "p8_qualification_authorized": False,
        "performance_claim_authorized": False,
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": audit.passed, "failures": audit.failures}))


if __name__ == "__main__":
    main()
