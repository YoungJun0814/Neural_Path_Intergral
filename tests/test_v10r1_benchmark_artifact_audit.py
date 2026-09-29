from __future__ import annotations

import copy
import json
from pathlib import Path

from src.path_integral.v10r1_benchmark_audit import audit_v10r1_benchmark

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v10r1/development_v1.yaml"
RESULT = ROOT / "results/g11_v10r1_terminal_development_v1_2026-08-11.json"


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_v10r1_development_audit_reconstructs_execution_seed_order() -> None:
    audit = audit_v10r1_benchmark(
        config_path=CONFIG,
        result=_result(),
        root=ROOT,
    )
    assert audit.passed, audit.failures


def test_v10r1_development_audit_rejects_seed_and_aggregate_mutations() -> None:
    seed_mutation = copy.deepcopy(_result())
    seed_mutation["paired_records"][0]["gaussian_seed"] += 1  # type: ignore[index]
    seed_audit = audit_v10r1_benchmark(
        config_path=CONFIG,
        result=seed_mutation,
        root=ROOT,
    )
    assert not seed_audit.passed
    assert "seed_uniqueness" in seed_audit.failures

    aggregate_mutation = copy.deepcopy(_result())
    aggregate_mutation["aggregate"]["stage_pass"] = True  # type: ignore[index]
    aggregate_audit = audit_v10r1_benchmark(
        config_path=CONFIG,
        result=aggregate_mutation,
        root=ROOT,
    )
    assert not aggregate_audit.passed
    assert "aggregate_recomputation" in aggregate_audit.failures
