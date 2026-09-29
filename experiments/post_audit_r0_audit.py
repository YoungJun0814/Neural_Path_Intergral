"""Read-only command-line entry point for new and historical R0 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.path_integral.legacy_v16_semantic_adapter import audit_legacy_v16
from src.path_integral.research_result_audit import _load_bound_file, audit_measurement

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--measurement", type=Path)
    source.add_argument("--legacy-config", type=Path)
    args = parser.parse_args()
    if args.measurement is not None:
        result = audit_measurement(_load_bound_file(args.measurement.resolve()))
    else:
        result = audit_legacy_v16(ROOT, args.legacy_config.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["integrity"] != "pass" or result["semantic_validity"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
