"""Cross-platform comparison of two frozen V5 confirmatory artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load_verified(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("hardware-reproduction input hash mismatch")
    payload = json.loads(raw)
    if payload.get("schema") != "npi.g11.v5-confirmatory.v1":
        raise ValueError("hardware reproduction requires V5 confirmatory artifacts")
    return payload, digest


def compare(
    primary_path: Path,
    primary_sha256: str,
    reproduction_path: Path,
    reproduction_sha256: str,
    *,
    maximum_combined_z_score: float = 4.0,
    operation_work_relative_tolerance: float = 1e-12,
) -> dict[str, Any]:
    if maximum_combined_z_score <= 0.0 or operation_work_relative_tolerance < 0.0:
        raise ValueError("invalid reproduction tolerances")
    primary, primary_hash = _load_verified(primary_path, primary_sha256)
    reproduction, reproduction_hash = _load_verified(reproduction_path, reproduction_sha256)
    primary_records = {item["cell_id"]: item for item in primary["records"]}
    reproduction_records = {item["cell_id"]: item for item in reproduction["records"]}
    failures: list[str] = []
    if set(primary_records) != set(reproduction_records):
        failures.append("cell identity sets differ")
    comparisons: list[dict[str, Any]] = []
    for cell_id in sorted(set(primary_records) & set(reproduction_records)):
        left = primary_records[cell_id]
        right = reproduction_records[cell_id]
        left_result = left["result"]
        right_result = right["result"]
        if not left_result["complete"] or not right_result["complete"]:
            failures.append(f"{cell_id}: both artifacts must be complete")
            continue
        left_se = float(left_result["standard_error"])
        right_se = float(right_result["standard_error"])
        denominator = math.hypot(left_se, right_se)
        difference = abs(float(left_result["estimate"]) - float(right_result["estimate"]))
        z_score = (
            difference / denominator
            if denominator > 0.0
            else (0.0 if difference == 0.0 else math.inf)
        )
        left_work = math.fsum(float(item["work_units"]) for item in left_result["work"]["entries"])
        right_work = math.fsum(
            float(item["work_units"]) for item in right_result["work"]["entries"]
        )
        work_agrees = math.isclose(
            left_work,
            right_work,
            rel_tol=operation_work_relative_tolerance,
            abs_tol=operation_work_relative_tolerance,
        )
        selected_agrees = (
            left["preparation"]["selected_candidate"] == right["preparation"]["selected_candidate"]
        )
        if z_score > maximum_combined_z_score:
            failures.append(f"{cell_id}: estimates disagree beyond combined uncertainty")
        if not work_agrees:
            failures.append(f"{cell_id}: operation work differs")
        if not selected_agrees:
            failures.append(f"{cell_id}: selected construction differs")
        comparisons.append(
            {
                "cell_id": cell_id,
                "combined_z_score": z_score,
                "maximum_combined_z_score": maximum_combined_z_score,
                "primary_operation_work": left_work,
                "reproduction_operation_work": right_work,
                "operation_work_agrees": work_agrees,
                "selected_candidate_agrees": selected_agrees,
            }
        )
    return {
        "schema": "npi.g11.v5-hardware-reproduction.v1",
        "primary": {"path": str(primary_path), "sha256": primary_hash},
        "reproduction": {
            "path": str(reproduction_path),
            "sha256": reproduction_hash,
        },
        "comparisons": comparisons,
        "failures": failures,
        "passed": not failures,
        "scope": (
            "Operation-work and statistical agreement are primary; wall-time "
            "differences require separate hardware profiling."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-sha256", required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--reproduction-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare(
        arguments.primary,
        arguments.primary_sha256,
        arguments.reproduction,
        arguments.reproduction_sha256,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
