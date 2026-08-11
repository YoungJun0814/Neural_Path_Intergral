"""Re-audit a serialized V15 artifact without rerunning experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.path_integral.v15_result_audit import load_and_audit_v15_result

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    audit = load_and_audit_v15_result(args.result.resolve(), root=ROOT)
    print(json.dumps(asdict(audit), indent=2))


if __name__ == "__main__":
    main()
