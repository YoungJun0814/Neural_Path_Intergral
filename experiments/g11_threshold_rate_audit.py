"""Aggregate predeclared M4 gates across independently configured model regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != "npi.g11.threshold-rate-pilot.v1":
        raise ValueError(f"unsupported threshold-rate artifact: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def run(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 3:
        raise ValueError("M4 audit requires at least three model-regime artifacts")
    artifacts = [_load(path) for path in paths]
    protocol_ids = [payload[0]["protocol_id"] for payload in artifacts]
    if len(set(protocol_ids)) != len(protocol_ids):
        raise ValueError("M4 artifacts must have distinct protocol namespaces")
    cells = [
        (payload["protocol_id"], task, analysis)
        for payload, _digest in artifacts
        for task, analysis in payload["rate_analyses"].items()
    ]
    identified = [cell for cell in cells if bool(cell[2].get("identified"))]
    positive_dcs = [
        cell for cell in identified if cell[2]["exponents"]["dcs_variance"] > 0.0
    ]
    equivalence_margin = 0.25
    equivalent = [
        cell
        for cell in identified
        if cell[2]["confidence_intervals_95"][
            "dcs_second_minus_threshold_l2"
        ][0]
        >= -equivalence_margin
        and cell[2]["confidence_intervals_95"][
            "dcs_second_minus_threshold_l2"
        ][1]
        <= equivalence_margin
    ]
    positive_threshold = [
        cell
        for cell in identified
        if cell[2]["confidence_intervals_95"]["threshold_l2"][0] > 0.0
    ]
    exact = all(
        payload["gates"]["exactness"] and payload["gates"]["no_failures"]
        for payload, _digest in artifacts
    )
    positive_fraction = len(positive_dcs) / len(cells)
    gates = {
        "at_least_three_regimes": len(artifacts) >= 3,
        "all_exactness_and_failure_gates": exact,
        "all_cells_rate_identified": len(identified) == len(cells),
        "all_threshold_l2_lower_95_positive": len(positive_threshold) == len(cells),
        "all_dcs_second_moment_equivalent_to_2r": len(equivalent) == len(cells),
        "positive_dcs_variance_slope_fraction": positive_fraction,
        "positive_dcs_variance_slope_at_least_80_percent": positive_fraction >= 0.80,
    }
    boolean_gates = [value for value in gates.values() if isinstance(value, bool)]
    return {
        "schema": "npi.g11.threshold-rate-audit.v1",
        "input_artifacts": [
            {
                "path": str(path),
                "sha256": digest,
                "protocol_id": payload["protocol_id"],
            }
            for path, (payload, digest) in zip(paths, artifacts, strict=True)
        ],
        "cell_count": len(cells),
        "identified_cell_count": len(identified),
        "gates": gates,
        "passed": all(boolean_gates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.inputs)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], **result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
