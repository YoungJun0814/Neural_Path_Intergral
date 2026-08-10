"""CLI for the independent V10R1 proposal-bank audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.path_integral.v10r1_bank_audit import audit_v10r1_bank

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-training", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V10R1 bank-audit outputs are immutable")
    result_raw = args.result.read_bytes()
    result = json.loads(result_raw.decode("utf-8"))
    audit = audit_v10r1_bank(
        config_path=args.config,
        result=result,
        root=ROOT,
        replay_training=args.replay_training,
    )
    payload = {
        "schema": "npi.g11.v10r1-full-latent-proposal-bank-audit.v1",
        "result_path": args.result.resolve().relative_to(ROOT).as_posix(),
        "result_sha256": hashlib.sha256(result_raw).hexdigest(),
        "checks": dict(audit.checks),
        "failures": list(audit.failures),
        "replay_performed": audit.replay_performed,
        "passed": audit.passed,
        "decision": {
            "development_authorized": audit.passed and audit.replay_performed,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"audit_passed": audit.passed}, sort_keys=True))


if __name__ == "__main__":
    main()
