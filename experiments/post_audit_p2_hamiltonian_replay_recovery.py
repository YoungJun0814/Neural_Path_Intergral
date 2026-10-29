"""Whole-family same-seed operational replay, preserving the original timeout."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from experiments import post_audit_p2_hamiltonian_tail as original
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/p2_hamiltonian_operational_replay_v1/session_20261028"
PLAN = "docs/plans/P2_HAMILTONIAN_OPERATIONAL_REPLAY_2026-10-28_KO.md"


def numeric_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return {k: v for k, v in left.items() if k != "wall_seconds"} == {
        k: v for k, v in right.items() if k != "wall_seconds"}


def load_original() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    base = ROOT/original.DIRECTORY
    p = json.loads((base/"protocol.json").read_text())
    failure = json.loads((base/"failure.json").read_text())
    audit_snapshot(ROOT, p["source"], p["config"])
    if (failure["completed_pilot"] != 20 or failure["completed_precision"] != 0
            or not failure["error"].startswith("TimeoutError") or (base/"result.json").exists()):
        raise ValueError("preserved original timeout contract changed")
    rows = [json.loads(f.read_text()) for f in sorted(base.glob("pilot-*.json")) if f.name != "pilot-result.json"]
    pre = [json.loads(f.read_text()) for f in sorted(base.glob("preflight-*.json"))]
    if len(rows) != 20 or len(pre) != 4:
        raise ValueError("original completed membership changed")
    cells = [c["id"] for c in p["config"]["cells"]]
    if {original.ident(r) for r in rows} != {original.ident(j) for j in original.make_jobs(cells, "pilot")[:20]}:
        raise ValueError("original prefix not fixed first20")
    for n, h in p["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/n) != h:
            raise ValueError("frozen numerical source changed: "+n)
    old = {original.ident(r): r for r in [*rows, *pre]}
    original.validate_rows([*rows, *pre], p["seeds"]["mapping"])
    return p, failure, old


def config(p: dict[str, Any], old: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paths = [original.DIRECTORY+"/"+n for n in ("protocol.json", "failure.json", "run.source.zip", "continuum.json")]
    paths += [original.DIRECTORY+"/"+name+".json" for name in old]
    return {"original_config": p["config"], "same_seed_same_statistical_experiment": True,
        "original_timeout_preserved": True, "anchors": {n: sha(ROOT/n) for n in paths},
        "pilot_physical_cap": 1050000, "pilot_wall_cap": 7200,
        "precision_physical_cap": 8200000, "precision_wall_cap": 10800}


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    p, failure, old = load_original()
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("one nonoverwriting operational attempt required")
    c, mapping = config(p, old), p["seeds"]["mapping"]
    models = {x["id"]: CachedVolterraTerminal.create(_problem(p["config"]["model"], x, 4096)) for x in p["config"]["cells"]}
    cells = list(models)
    path.mkdir(parents=True, exist_ok=True)
    source = freeze_source(path/"run.json", c, extra_snapshot_paths=(PLAN, *c["anchors"]))
    write(path/"protocol.json", {"source": source, "config": c, "original_seeds": p["seeds"]})
    budgets = {"pilot": original.Budget(1050000, 7200)}
    rows: dict[str, list[dict[str, Any]]] = {"preflight": [], "pilot": [], "precision": []}
    checked = 0
    try:
        for phase in ("preflight", "pilot"):
            jobs = original.make_jobs(cells, phase)
            for job in jobs:
                row = original.unit(job, models[job["cell"]], mapping, budgets["pilot"])
                name = original.ident(row)
                if name in old:
                    if not numeric_equal(row, old[name]):
                        raise ValueError("same-seed original completed whole mismatch: "+name)
                    checked += 1
                write(path/(name+".json"), row)
                rows[phase].append(row)
                print(f"{phase} {len(rows[phase])}/{len(jobs)} {name}", flush=True)
        decision = original.pilot_decision(rows["pilot"], mapping, cells)
        pilot_work = budgets["pilot"].record()
        write(path/"pilot-result.json", {"decision": decision, "work": pilot_work,
            "original_complete_records_matched": checked})
        if decision["fresh_precision_authorized"]:
            budgets["precision"] = original.Budget(8200000, 10800)
            counts = {cell: decision["fresh_allocations"][cell]["required_count"] for cell in cells}
            jobs = original.make_jobs(cells, "precision", counts=counts)
            for job in jobs:
                row = original.unit(job, models[job["cell"]], mapping, budgets["precision"])
                write(path/(original.ident(row)+".json"), row)
                rows["precision"].append(row)
                print(f"precision {len(rows['precision'])}/{len(jobs)} {original.ident(row)}", flush=True)
        original.validate_rows([r for group in rows.values() for r in group], mapping)
        result = {"source": source, "config": c, "original_failure": failure, "records": rows,
            "original_complete_records_matched": checked, "pilot": decision,
            "precision": original.precision_summary(rows["precision"], mapping, cells,
                json.loads((ROOT/original.ANCHOR).read_text()), decision),
            "work": {**original.work_records(budgets), "pilot": pilot_work},
            "P2_qualified": False, "production_adoption": False, "large_training_authorized": False,
            "original_samples_not_counted_again_as_independent": True}
        write(path/"result.json", result)
        return result
    except Exception as error:
        write(path/"failure.json", {"error": type(error).__name__+": "+str(error),
            "completed": {k: len(v) for k, v in rows.items()}, "original_complete_records_matched": checked,
            "work": original.work_records(budgets), "P2_qualified": False})
        raise


def audit() -> dict[str, Any]:
    torch.set_num_threads(8)
    path = ROOT/DIRECTORY
    value, protocol = [json.loads((path/n).read_text()) for n in ("result.json", "protocol.json")]
    p, failure, old = load_original()
    c = config(p, old)
    if (value["config"] != c or protocol != {"source": value["source"], "config": c, "original_seeds": p["seeds"]}
            or value["original_failure"] != failure):
        raise ValueError("recovery contract/source/original-failure binding mismatch")
    snapshot = audit_snapshot(ROOT, value["source"], c)
    for n, h in value["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/n) != h:
            raise ValueError("recovery frozen source changed: "+n)
    mapping, rows = p["seeds"]["mapping"], value["records"]
    models = {x["id"]: CachedVolterraTerminal.create(_problem(p["config"]["model"], x, 4096)) for x in p["config"]["cells"]}
    cells = list(models)
    original.validate_rows([r for group in rows.values() for r in group], mapping)
    if len(rows["preflight"]) != 4 or {original.ident(r) for r in rows["preflight"]} != {original.ident(j) for j in original.make_jobs(cells, "preflight")}:
        raise ValueError("recovery preflight membership mismatch")
    decision = original.pilot_decision(rows["pilot"], mapping, cells)
    if (decision != value["pilot"] or original.precision_summary(rows["precision"], mapping, cells,
            json.loads((ROOT/original.ANCHOR).read_text()), decision) != value["precision"]):
        raise ValueError("recovery statistical decision replay mismatch")
    budget, matched, replayed = original.Budget(200000, 1800), 0, 0
    for group in rows.values():
        for row in group:
            name = original.ident(row)
            if json.loads((path/(name+".json")).read_text()) != row:
                raise ValueError("recovery whole file/result mismatch")
            if name in old:
                if not numeric_equal(row, old[name]):
                    raise ValueError("original timeout-prefix numerical mismatch")
                matched += 1
            if row["phase"] != "preflight" and row["rep"] == 0:
                repeated = original.unit(row, models[row["cell"]], mapping, budget)
                if not numeric_equal(repeated, row):
                    raise ValueError("recovery representative whole replay mismatch")
                replayed += 1
    if matched != 24 or matched != value["original_complete_records_matched"]:
        raise ValueError("original24 completed records not all matched")
    for phase, group in (("pilot", [*rows["preflight"], *rows["pilot"]]), ("precision", rows["precision"])):
        expected = sum(r["physical_forwards"] for r in group)
        w = value["work"][phase]
        if w["physical_forwards"] != expected or w["gradient_callback_requests"] != expected or w["time_step_forwards"] != 4096*expected:
            raise ValueError("recovery physical work mismatch")
    result = {"status": "RECOVERY_SOURCE_ORIGINAL24_ALL_WHOLES_DECISIONS_AND_REPLAYS_PASSED",
        "result_sha256": sha(path/"result.json"), "source": snapshot, "original_complete_records_matched": matched,
        "representative_wholes_replayed": replayed, "work": budget.record(),
        "original_timeout_preserved": True, "not_independent_confirmation": True, "P2_qualified": False}
    write(path/"audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "audit"))
    mode = parser.parse_args().mode
    output = run() if mode == "run" else audit()
    print(json.dumps({"mode": mode, "status": output.get("status", output.get("pilot", {}).get("decision"))}))
