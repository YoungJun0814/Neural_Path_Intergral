"""Write the semantic audit of the frozen V9 terminal claim contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.path_integral.provenance import source_provenance
from src.path_integral.v9_terminal_contract import audit_v9_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    audit = audit_v9_contract(args.contract)
    payload = {
        "schema": "npi.g11.v9-terminal-claim-contract-audit.v1",
        "contract_path": args.contract.as_posix(),
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        **source_provenance(),
        **asdict(audit),
        "reference_execution_authorized": audit.passed,
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
