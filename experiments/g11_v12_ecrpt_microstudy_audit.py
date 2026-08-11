"""CLI for the independent V12 ECRPT development-result audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.g11_v12_ecrpt_microstudy import load_config, validate_config
from src.path_integral.ecrpt_microstudy_audit import audit_ecrpt_microstudy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha256 = load_config(args.config)
    validate_config(config)
    raw = args.result.read_bytes()
    result = json.loads(raw.decode("utf-8"))
    audit = audit_ecrpt_microstudy(
        config=config,
        config_sha256=config_sha256,
        result=result,
        result_bytes=raw,
    )
    payload = asdict(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
