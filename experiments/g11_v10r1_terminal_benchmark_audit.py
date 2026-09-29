"""CLI for the independent corrected V10R1 benchmark audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.path_integral.v10r1_benchmark_audit import audit_v10r1_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V10R1 benchmark-audit outputs are immutable")
    raw = args.result.read_bytes()
    result = json.loads(raw.decode("utf-8"))
    audit = audit_v10r1_benchmark(
        config_path=args.config,
        result=result,
        root=ROOT,
    )
    payload = {
        "schema": "npi.g11.v10r1-terminal-benchmark-audit.v1",
        "result_path": args.result.resolve().relative_to(ROOT).as_posix(),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "checks": dict(audit.checks),
        "failures": list(audit.failures),
        "recomputed_aggregate": audit.recomputed_aggregate,
        "passed": audit.passed,
        "decision": {
            "qualification_authorized": audit.passed
            and bool(audit.recomputed_aggregate["stage_pass"]),
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
