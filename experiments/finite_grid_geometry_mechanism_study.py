"""Seal FR0 and run only the prespecified FR2 reference-feasibility pilot.

No new event proposal is trained here. A feasible reference is not a successful
mechanism study, a confirmation result, or a passed historical P2 gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.post_audit_r2_family_diagnosis import freeze_source
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.finite_grid_geometry_benchmark import (
    bounded_payoff,
    feasibility,
    normal_coordinates,
    reference_units,
    verify_reference_record,
)
from src.path_integral.finite_grid_geometry_protocol import EvaluationBudget, GeometryProtocol
from src.path_integral.research_result_contract import source_tree_digest
from src.path_integral.structural_v2_inventory import audit_snapshot

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/finite_grid_geometry/reset_v1.json"
CONTRACT = "docs/theory/FINITE_GRID_GEOMETRY_FR0_CONTRACT_2026-10-28_KO.md"
PLAN = "docs/plans/FINITE_GRID_VOLTERRA_RESEARCH_RESET_ROADMAP_2026-10-28_KO.md"
DEFAULT_OUTPUT = ROOT/"results/post_audit/finite_grid_geometry_reset_v1/session_20261028"


def save(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_protocol() -> GeometryProtocol:
    return GeometryProtocol.from_dict(json.loads((ROOT/CONFIG).read_text(encoding="utf-8")))


def run(directory: Path) -> dict[str, Any]:
    protocol = read_protocol()
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = freeze_source(directory/"run.json", protocol.to_dict(),
                           extra_snapshot_paths=(PLAN, CONTRACT))
    save(directory/"protocol.json", {"config": protocol.to_dict(),
                                     "protocol_digest": protocol.digest, "source": source})
    budget = EvaluationBudget(protocol.limit("FR2"))
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        for task in protocol.tasks:
            if task.split != "development":
                continue
            methods = []
            for method in ("conditional_mc", "conditional_rqmc"):
                record = reference_units(protocol, task.task_id, method, budget)
                save(directory/f"{task.task_id}-{method}.json", record)
                methods.append(record)
                print(task.task_id, method, record["mean"], record["rse"], flush=True)
            row = {"task_id": task.task_id, "methods": methods,
                   "feasibility": feasibility(protocol, *methods)}
            rows.append(row)
        seeds = [s for row in rows for method in row["methods"] for s in method["seeds"]]
        if len(seeds) != len(set(seeds)):
            raise ValueError("registered reference stream collision")
        if source_tree_digest(ROOT) != source["source_tree_digest"]:
            raise ValueError("runtime source changed after sealing")
        result = {"protocol_digest": protocol.digest, "source": source, "config": protocol.to_dict(),
                  "phase": "FR2_REFERENCE_ONLY", "cells": rows,
                  "physics_evaluations": budget.evaluations,
                  "planned_evaluations": protocol.planned_pilot_evaluations,
                  "wall_seconds": time.perf_counter()-started, "torch_threads": 1,
                  "confirmation_task_evaluations": 0, "candidate_training_runs": 0,
                  "P2_qualified": False, "mechanism_gate_passed": False,
                  "next": "bank/family mechanism experiment requires its own frozen job matrix"}
        save(directory/"result.json", result)
        return result
    except Exception as error:
        save(directory/"failure.json", {"type": type(error).__name__, "message": str(error),
                                       "reserved_evaluations": budget.evaluations,
                                       "completed_cells": [r["task_id"] for r in rows],
                                       "not_promoted_to_success": True})
        raise


def audit(directory: Path) -> dict[str, Any]:
    result = json.loads((directory/"result.json").read_text(encoding="utf-8"))
    protocol = GeometryProtocol.from_dict(result["config"])
    if protocol.digest != result["protocol_digest"]:
        raise ValueError("protocol digest changed")
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, digest in result["source"]["snapshot_file_hashes"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError("current replay source differs from frozen archive")
    if (result["physics_evaluations"] != protocol.planned_pilot_evaluations
            or result["confirmation_task_evaluations"] != 0 or result["candidate_training_runs"] != 0
            or result["P2_qualified"] or result["mechanism_gate_passed"]):
        raise ValueError("work, holdout or research-claim contract changed")
    torch.set_num_threads(1)
    replay_evaluations = 0
    expected_cells = [t.task_id for t in protocol.tasks if t.split == "development"]
    if [r["task_id"] for r in result["cells"]] != expected_cells:
        raise ValueError("missing or reordered planned cells")
    for row in result["cells"]:
        task = protocol.authorize(row["task_id"], "FR2", "reference_pilot")
        mc, rqmc = row["methods"]
        if feasibility(protocol, mc, rqmc) != row["feasibility"]:
            raise ValueError("stored feasibility changed")
        cache = CachedVolterraTerminal.create(task.problem())
        for method in row["methods"]:
            verify_reference_record(protocol, task.task_id, method)
            saved = json.loads((directory/f"{task.task_id}-{method['method']}.json").read_text(encoding="utf-8"))
            if saved != method:
                raise ValueError("individual unit-record file changed")
            expected_seeds = [protocol.seed(task.task_id, "reference_pilot", i, method["method"])
                              for i in range(protocol.pilot_units)]
            if method["seeds"] != expected_seeds or len(method["unit_means"]) != protocol.pilot_units:
                raise ValueError("seed role or unit count changed")
            z = normal_coordinates(method["method"], 2**protocol.pilot_power,
                                   2*task.steps, expected_seeds[0])
            value = float(bounded_payoff(task, cache, z).mean())
            if value != method["unit_means"][0]:
                raise ValueError("same-source first-unit replay changed")
            replay_evaluations += len(z)
    receipt = {"status": "SOURCE_BINDING_DECISION_AND_FIRST_UNITS_REPLAYED",
               "result_sha256": hashlib.sha256((directory/"result.json").read_bytes()).hexdigest(),
               "replayed_units": 2*len(expected_cells), "physics_evaluations": replay_evaluations,
               "not_new_independent_reference_samples": True, "P2_qualified": False}
    save(directory/"audit.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    output = audit(args.output) if args.audit else run(args.output)
    print(json.dumps({k: v for k, v in output.items() if k not in {"source", "config", "cells"}},
                     indent=2, allow_nan=False))
