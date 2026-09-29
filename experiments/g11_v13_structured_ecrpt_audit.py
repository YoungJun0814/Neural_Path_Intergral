"""Audit a V13 structured ECRPT development result."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.g11_v13_structured_ecrpt_development import load_config
from src.path_integral.v13_development_audit import audit_v13_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, digest = load_config(args.config)
    result_bytes = args.result.read_bytes()
    result = json.loads(result_bytes.decode("utf-8"))
    audit = asdict(
        audit_v13_development(
            config=config,
            config_sha256=digest,
            result=result,
            result_bytes=result_bytes,
        )
    )
    payload = {
        "schema": "npi.g11.v13-structured-ecrpt-development-audit.v1",
        "result_path": str(args.result),
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
