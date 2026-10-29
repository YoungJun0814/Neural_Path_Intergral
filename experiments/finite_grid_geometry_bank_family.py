"""Bounded FR2 bank/family factorial; no production adoption or confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.finite_grid_geometry_mechanism_study import ROOT, read_protocol, save
from experiments.post_audit_r2_family_diagnosis import freeze_source
from src.path_integral.finite_grid_geometry_controls import CONTRASTS, PER_WHOLE, whole
from src.path_integral.finite_grid_geometry_protocol import (
    EvaluationBudget,
    GeometryProtocol,
    PhaseLimit,
)
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = ROOT/"results/post_audit/finite_grid_bank_family_v1/session_20261028"
PREVIOUS = ROOT/"results/post_audit/finite_grid_geometry_reset_v1/session_20261028"
PLAN = "docs/plans/FINITE_GRID_FR2_BANK_FAMILY_PROTOCOL_2026-10-28_KO.md"
JOBS = [(task, rep) for task in ("D02", "D04") for rep in range(3)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [(r["task_id"], r["whole"]) for r in rows] != JOBS:
        raise ValueError("exact planned whole matrix required")
    counts = {f"{a}-vs-{b}": {task: 0 for task in ("D02", "D04")} for a, b in CONTRASTS}
    for row in rows:
        if row.get("failed"):
            continue
        for contrast in row["contrasts"]:
            counts[f"{contrast['candidate']}-vs-{contrast['baseline']}"][row["task_id"]] += int(contrast["signal"])
    eligible = [name for name, cells in counts.items() if all(n >= 2 for n in cells.values())]
    decision = "GO_FRESH_FR3_HYPOTHESIS_TEST_ONLY" if eligible else "STOP_NO_REPLICATED_BANK_OR_FAMILY_MECHANISM"
    if any(r.get("failed") for r in rows):
        decision = "UNRESOLVED_MECHANISM_RUN_FAILURE"
        eligible = []
    return {"signal_counts": counts, "eligible_mechanism_hypotheses": eligible, "decision": decision,
            "production_adopted": False, "confirmation_started": False, "P2_qualified": False}


def run() -> dict[str, Any]:
    protocol = read_protocol()
    previous = json.loads((PREVIOUS/"result.json").read_text(encoding="utf-8"))
    receipt = json.loads((PREVIOUS/"audit.json").read_text(encoding="utf-8"))
    if (previous["protocol_digest"] != protocol.digest
            or receipt["result_sha256"] != sha(PREVIOUS/"result.json")):
        raise ValueError("audited reference-pilot binding required")
    if any(row["feasibility"]["status"] != "REFERENCE_FEASIBLE_DEVELOPMENT_ONLY" for row in previous["cells"]):
        raise ValueError("this frozen job matrix requires all four feasible development tasks")
    spent = previous["physics_evaluations"]+receipt["physics_evaluations"]
    config = {"protocol": protocol.to_dict(), "jobs": JOBS, "evaluations_per_whole": PER_WHOLE,
              "previous_result_sha256": sha(PREVIOUS/"result.json"),
              "previous_audit_sha256": sha(PREVIOUS/"audit.json"), "already_spent": spent,
              "prior_wall_budget_reservation_seconds": 600., "child_wall_cap_seconds": 1800.}
    if spent+(len(JOBS)+2)*PER_WHOLE > protocol.limit("FR2").max_evaluations:
        raise ValueError("production and planned replay exceed cumulative FR2 cap")
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = freeze_source(DIRECTORY/"run.json", config, extra_snapshot_paths=(PLAN,))
    save(DIRECTORY/"protocol.json", {"config": config, "source": source})
    budget = EvaluationBudget(PhaseLimit("FR2", protocol.limit("FR2").max_evaluations, 2400.),
                              already_spent=spent, already_elapsed_seconds=600.)
    started = time.perf_counter()
    rows = []
    for task, rep in JOBS:
        before = budget.evaluations
        try:
            row = whole(protocol, task, rep, budget)
        except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as error:
            row = {"task_id": task, "whole": rep, "failed": True,
                   "failure_type": type(error).__name__, "message": str(error)}
        row["evaluations"] = budget.evaluations-before
        if not row.get("failed") and row["evaluations"] != PER_WHOLE:
            raise ValueError("whole accounting mismatch")
        save(DIRECTORY/f"{task}-whole-{rep}.json", row)
        rows.append(row)
        print(task, rep, "failed" if row.get("failed") else [v["rse"] for v in row["finals"]], flush=True)
    result = {"config": config, "source": source, "rows": rows, "summary": summary(rows),
              "new_physics_evaluations": budget.evaluations-spent,
              "cumulative_FR2_evaluations": budget.evaluations,
              "wall_seconds": time.perf_counter()-started,
              "confirmation_task_evaluations": 0, "not_fixed_precision_timing_benchmark": True}
    save(DIRECTORY/"result.json", result)
    return result


def audit() -> dict[str, Any]:
    result = json.loads((DIRECTORY/"result.json").read_text(encoding="utf-8"))
    config = result["config"]
    protocol = GeometryProtocol.from_dict(config["protocol"])
    audit_snapshot(ROOT, result["source"], config)
    if summary(result["rows"]) != result["summary"]:
        raise ValueError("decision recomputation changed")
    if [(r["task_id"], r["whole"]) for r in result["rows"]] != JOBS:
        raise ValueError("planned whole missing")
    torch.set_num_threads(1)
    budget = EvaluationBudget(protocol.limit("FR2"), already_spent=result["cumulative_FR2_evaluations"],
                              already_elapsed_seconds=600.+result["wall_seconds"])
    before = budget.evaluations
    for row in result["rows"]:
        if json.loads((DIRECTORY/f"{row['task_id']}-whole-{row['whole']}.json").read_text(encoding="utf-8")) != row:
            raise ValueError("whole record changed")
        if row["whole"] == 0 and not row.get("failed"):
            replay = whole(protocol, row["task_id"], 0, budget)
            if replay != {k: v for k, v in row.items() if k != "evaluations"}:
                raise ValueError("same-seed whole replay changed")
    receipt = {"status": "DECISIONS_AND_TWO_REPRESENTATIVE_WHOLES_REPLAYED",
               "result_sha256": sha(DIRECTORY/"result.json"),
               "new_physics_evaluations": budget.evaluations-before,
               "cumulative_FR2_evaluations": budget.evaluations,
               "not_new_independent_wholes": True, "P2_qualified": False}
    save(DIRECTORY/"audit.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    output = audit() if args.audit else run()
    print(json.dumps(output.get("summary", output), indent=2, allow_nan=False))
