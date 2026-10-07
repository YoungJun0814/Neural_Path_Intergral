"""Audit a V14 local-Volterra result."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.g11_v14_local_volterra_development import load_config
from src.path_integral.v14_local_volterra_audit import audit_v14_local_volterra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "qualification"), default="development")
    args = parser.parse_args()
    config, digest = load_config(args.config)
    raw = args.result.read_bytes()
    result = json.loads(raw.decode("utf-8"))
    audit = asdict(
        audit_v14_local_volterra(
            config=config,
            config_sha256=digest,
            result=result,
            result_bytes=raw,
            expected_stage=args.stage,
        )
    )
    payload = {
        "schema": "npi.g11.v14-local-volterra-audit.v1",
        "result_path": str(args.result),
        "stage": args.stage,
        **audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
