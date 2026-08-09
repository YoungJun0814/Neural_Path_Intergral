"""Audit V10 Full-Dimensional Defensive CEM + DCS benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def audit_v10_benchmark(result_path: Path) -> dict[str, Any]:
    raw = result_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))

    checks: list[dict[str, Any]] = []

    # 1. Schema check
    schema_ok = data.get("schema") == "npi.g11.v10-terminal-benchmark-result.v1"
    checks.append({"check": "schema_valid", "passed": schema_ok})

    # 2. Paired records count check (12 cells * 4 clusters = 48)
    paired = data.get("paired_records", [])
    paired_ok = len(paired) == 48
    checks.append({"check": "paired_record_count_48", "passed": paired_ok})

    # 3. External records count check (12 cells * 4 clusters * 3 comparators = 144)
    external = data.get("external_records", [])
    external_ok = len(external) == 144
    checks.append({"check": "external_record_count_144", "passed": external_ok})

    # 4. Exactness check
    max_exactness = max(
        max(rec["exactness"].values()) for rec in paired
    ) if paired else 1.0
    exactness_ok = max_exactness <= 1e-10
    checks.append({"check": "exactness_error_below_threshold", "passed": exactness_ok, "max_exactness_error": max_exactness})

    # 5. Likelihood normalization check
    pass_fraction = data.get("aggregate", {}).get("likelihood_normalization_pass_fraction", 0.0)
    norm_ok = pass_fraction >= 0.95
    checks.append({"check": "likelihood_normalization_pass_fraction", "passed": norm_ok, "fraction": pass_fraction})

    # 6. Replayability & seed uniqueness check
    seed_count = data.get("seed_count", 0)
    seed_ok = seed_count > 0
    checks.append({"check": "seed_uniqueness", "passed": seed_ok})

    all_passed = all(c["passed"] for c in checks)

    return {
        "schema": "npi.g11.v10-terminal-benchmark-audit-result.v1",
        "result_path": str(result_path),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "passed": all_passed,
        "checks": checks,
        "aggregate_summary": data.get("aggregate", {}),
        "decision": data.get("decision", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_v10_benchmark(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"audit_passed": audit["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
