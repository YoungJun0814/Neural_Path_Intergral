"""Write the final independent V9 completion-status audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.path_integral.provenance import source_provenance
from src.path_integral.v9_completion_audit import audit_v9_completion

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    audit = audit_v9_completion(ledger_path=args.ledger, root=ROOT)
    payload = {
        "schema": "npi.g11.v9-completion-status-audit.v1",
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        **source_provenance(),
        **asdict(audit),
        "qualification_authorized": False,
        "top_journal_claim_authorized": False,
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
