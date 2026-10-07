"""Replay-based integrity audit for R1 development diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.post_audit_r1_bank_followup import run as run_followup
from experiments.post_audit_r1_diagnostics import run as run_diagnostics
from src.path_integral.research_result_contract import (
    canonical_digest,
    source_tree_digest,
)
from src.path_integral.seed_ledger import SeedLedger

ROOT = Path(__file__).resolve().parents[1]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if not key.endswith("wall_seconds") and not key.endswith("cpu_seconds")
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def audit(path: Path, *, replay: bool = True) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON constant {value}")),
    )
    schema = payload.get("schema")
    if schema not in ("npi.post-audit.r1-diagnostic.v1",
                      "npi.post-audit.r1-bank-followup.v1"):
        raise ValueError("unsupported R1 schema")
    if payload.get("role") != "development_not_confirmation":
        raise ValueError("R1 result cannot be labeled confirmation")
    if payload["source"]["config_digest"] != canonical_digest(payload["config"]):
        raise ValueError("R1 config digest mismatch")
    if payload["source"]["source_tree_digest"] != source_tree_digest(ROOT):
        raise ValueError("R1 runtime source tree differs from the recorded execution")
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    if not payload["cells"] or len(ledger) < 1:
        raise ValueError("R1 result is empty")
    if replay:
        runner = run_diagnostics if schema.endswith("r1-diagnostic.v1") else run_followup
        fresh = runner(payload["config"])
        if canonical_digest(fresh["seed_ledger"]) != canonical_digest(payload["seed_ledger"]):
            raise ValueError("R1 seed ledger changed on replay")
        if canonical_digest(_without_timing(fresh["cells"])) != canonical_digest(
            _without_timing(payload["cells"])
        ):
            raise ValueError("R1 numerical results or proposals changed on replay")
    return {
        "status": "pass",
        "schema": schema,
        "cell_count": len(payload["cells"]),
        "seed_count": len(ledger),
        "full_numerical_replay": replay,
        "scope": "integrity_and_reproducibility_not_statistical_qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(audit(path, replay=not args.no_replay), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
