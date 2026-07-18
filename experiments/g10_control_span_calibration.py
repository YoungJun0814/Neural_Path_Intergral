"""Calibration-only natural-weight selection for the G10 frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g10_control_span_development import _candidate, _source


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G10 calibration schema-version-1 config")
    if payload.get("frozen") is not True or payload.get("calibration_only") is not True:
        raise ValueError("G10 calibration protocol must be frozen and calibration-only")
    return payload, hashlib.sha256(raw).hexdigest()


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    regimes: list[tuple[str, dict[str, Any]]] = []
    source_hashes: dict[str, str] = {}
    for source in config["calibration_sources"]:
        payload = _source(source)
        source_hashes[str(source["path"])] = str(source["canonical_json_sha256"])
        regimes.extend((str(source["group"]), regime) for regime in payload["regimes"])
    if smoke:
        regimes = regimes[:1]
    candidates = [float(value) for value in config["mixture"]["natural_weight_candidates"]]
    if smoke:
        candidates = candidates[:2]
    evaluation = config["evaluation"]
    steps = 64 if smoke else int(evaluation["fine_steps"])
    paths = 500 if smoke else int(evaluation["paths_per_seed"])
    seeds = [int(value) for value in evaluation["calibration_seeds"]]
    if smoke:
        seeds = seeds[:2]
    torch.set_num_threads(int(evaluation["thread_count"]))
    outputs: list[dict[str, Any]] = []
    for regime_index, (group, regime) in enumerate(regimes):
        reports = [
            _candidate(
                regime,
                alpha=alpha,
                alpha_index=alpha_index,
                steps=steps,
                paths=paths,
                seeds=[seed + 100_000 * regime_index for seed in seeds],
                label_seed_base=int(evaluation["label_seed_base"]) + 100_000 * regime_index,
                gates=config["gates"],
            )
            for alpha_index, alpha in enumerate(candidates)
        ]
        admissible = [report for report in reports if report["admissible"]]
        selected = (
            max(
                admissible,
                key=lambda report: (
                    float(report["geometric_raw_over_marginalized_work_ratio"]),
                    -float(report["natural_weight"]),
                ),
            )
            if admissible
            else None
        )
        outputs.append(
            {
                "regime_id": regime["regime_id"],
                "group": group,
                "model": regime["model"],
                "task": regime["task"],
                "control": regime["control"],
                "candidates": reports,
                "selected_natural_weight": (
                    float(selected["natural_weight"]) if selected is not None else None
                ),
                "selected_calibration_work_ratio": (
                    float(selected["geometric_raw_over_marginalized_work_ratio"])
                    if selected is not None
                    else None
                ),
                "passed": selected is not None,
            }
        )
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "smoke": smoke,
        "calibration_only": True,
        "source_hashes": source_hashes,
        "steps": steps,
        "paths_per_seed": paths,
        "seeds": seeds,
        "regimes": outputs,
        "passed_regimes": sum(bool(output["passed"]) for output in outputs),
        "total_regimes": len(outputs),
        "passed": all(bool(output["passed"]) for output in outputs),
        "restriction": "weights selected on calibration paths only and frozen for validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g10_control_span_calibration.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_regimes": result["passed_regimes"],
                "total_regimes": result["total_regimes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
