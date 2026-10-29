"""Prespecified independent Hamiltonian reference and tail-aware proof evaluation."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import psutil
import torch

from experiments.post_audit_fresh_ensemble_fine_smc import precision_gate
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import configuration
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.hamiltonian_probability_smc import estimate_hamiltonian_probability
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    precision_count,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.tail_weighted_continuum_bound import (
    observed_probability_root_diagnostic,
    probability_root_budget,
)
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

NAMESPACE = "p2-hamiltonian-tail-bound-v1"
DIRECTORY = "results/post_audit/p2_hamiltonian_tail_v1/session_20261028"
PLAN = "docs/plans/P2_HAMILTONIAN_AND_TAIL_BOUND_2026-10-28_KO.md"
THEORY = "docs/theory/TAIL_WEIGHTED_ROOT_BOUND_AND_SPLIT_HMC_2026-10-28_KO.md"
ANCHOR = "results/post_audit/fresh_ensemble_fine_smc_v1/session_20261020/result.json"
ANCHOR_AUDIT = "results/post_audit/fresh_ensemble_fine_smc_v1/session_20261020/audit.json"
PREVIOUS = "results/post_audit/p2_direct_closure_v1/session_20261027/result.json"
PREVIOUS_AUDIT = "results/post_audit/p2_direct_closure_v1/session_20261027/audit.json"
ARMS = ("gradient-control", "gaussian-hmc")
WORK = 128*(1+62*4)


def fixed_config() -> dict[str, Any]:
    basic = configuration()
    return {"namespace": NAMESPACE, "model": basic["model"], "cells": basic["cells"],
        "steps": 4096, "particles": 128, "points": 64, "power": 3, "moves": 4,
        "scale": .35, "hmc_step": .35, "drift_cap": 20., "wholes": 8,
        "arms": list(ARMS), "resample_every": 4, "resampling": "stratified", "threads": 8,
        "maximum_rse": .015, "maximum_unit_share": .15, "maximum_loo": .05,
        "equivalence_margin": .1, "comparisons": 2, "pilot_physical_cap": 1050000,
        "pilot_wall_cap": 7200, "precision_physical_cap": 8200000, "precision_wall_cap": 10800,
        "whole_wall_cap": 300, "precision_minimum": 32, "precision_maximum": 128,
        "precision_multiple": 8, "allocation_safety": 3,
        "holder_r": [1.05, 1.1, 1.25, 1.5, 2., 3., 4., 8.],
        "continuum_steps": [4096, 8192, 16384],
        "anchors": {p: sha(ROOT/p) for p in (ANCHOR, ANCHOR_AUDIT, PREVIOUS, PREVIOUS_AUDIT)}}


def ident(job: dict[str, Any]) -> str:
    return f"{job['phase']}-{job['cell']}-{job['arm']}-{job['rep']}"


def make_jobs(cells: list[str], phase: str, *, counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    arms = ("gaussian-hmc",) if phase == "precision" else ARMS
    return [{"phase": phase, "cell": c, "arm": a, "rep": r} for c in cells for a in arms
            for r in range((counts or {})[c] if counts else (1 if phase == "preflight" else 8))]


def streams(cells: list[str]) -> tuple[dict[str, int], dict[str, Any]]:
    ledger, mapping = SeedLedger(), {}
    for phase in ("pilot", "preflight", "precision"):
        jobs = make_jobs(cells, phase, counts={c: 128 for c in cells} if phase == "precision" else None)
        for j in jobs:
            mapping[ident(j)] = ledger.allocate(SeedKey(NAMESPACE, phase, "development", j["cell"],
                                                        4096, j["rep"], j["arm"]))
    for c in cells:
        for phase, arms in (("pilot", ARMS), ("precision", ("gaussian-hmc",))):
            for arm in arms:
                key = f"bootstrap-{phase}-{c}-{arm}"
                mapping[key] = ledger.allocate(SeedKey(NAMESPACE, "bootstrap-"+phase, "development", c, 4096, 0, arm))
    historical: set[int] = set()
    def scan(v: Any, context: bool = False) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                scan(x, context or "seed" in k.lower())
        elif isinstance(v, list):
            for x in v:
                scan(x, context)
        elif context and type(v) is int and 0 < v < 2**63:
            historical.add(v)
    paths = [p for p in sorted((ROOT/"results/post_audit").glob("*/session_202610*/protocol.json"))
             if p != ROOT/DIRECTORY/"protocol.json"]
    for path in paths:
        scan(json.loads(path.read_text(encoding="utf-8")))
    if set(mapping.values()) & historical:
        raise ValueError("scoped new/historical seed collision")
    return mapping, {"mapping": mapping, "ledger": ledger.to_dict(), "sha256": ledger.sha256,
        "scoped_historical_paths": [p.relative_to(ROOT).as_posix() for p in paths],
        "historical_seed_values": len(historical), "collision_count": 0,
        "unused_reserved_precision_seeds_are_not_samples": True,
        "not_exhaustive_history_or_PRNG_independence_proof": True}


class Budget:
    def __init__(self, physical: int, wall: float):
        self.cap, self.wall, self.count, self.started = physical, wall, 0, time.perf_counter()

    def charge(self, count: int, unit_start: float) -> None:
        if (self.count+count > self.cap or time.perf_counter()-self.started > self.wall
                or time.perf_counter()-unit_start > 300):
            raise TimeoutError("fixed whole/phase/physical budget exceeded; no partial estimate")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("fixed RSS4GiB cap exceeded")
        self.count += count

    def record(self) -> dict[str, Any]:
        return {"physical_forwards": self.count, "gradient_callback_requests": self.count,
            "time_step_forwards": self.count*4096, "wall_seconds": time.perf_counter()-self.started,
            "backward_requests_not_equal_cost_to_forward": True}


def work_records(budgets: dict[str, Budget]) -> dict[str, Any]:
    return {phase: budgets[phase].record() if phase in budgets else {
        "physical_forwards": 0, "gradient_callback_requests": 0, "time_step_forwards": 0,
        "wall_seconds": 0., "backward_requests_not_equal_cost_to_forward": True}
        for phase in ("pilot", "precision")}


def unit(job: dict[str, Any], cache: CachedVolterraTerminal, mapping: dict[str, int], budget: Budget) -> dict[str, Any]:
    start = time.perf_counter()
    small = job["phase"] == "preflight"
    particles, points = (16, 16) if small else (128, 64)
    cfg = WeightedSMCConfig(particles, tuple((i/(points-1))**3 for i in range(points)), 4, .35, 1,
        mapping[ident(job)], resample_every=4, resampling_scheme="stratified")
    def callback(z):
        budget.charge(len(z), start)
        return cache.log_probability(z)[:, None]
    r = estimate_hamiltonian_probability(callback, dimension=cache.problem.local_dimension,
        config=cfg, use_hamiltonian=job["arm"] == "gaussian-hmc")
    expected = particles*(1+(points-2)*4)
    if r.callback_rows != expected or r.gradient_requests != expected:
        raise ValueError("equal-callback-work contract failed")
    budget.charge(0, start)
    return {**job, "seed": cfg.seed, "log_mu": r.log_normalizer, "physical_forwards": expected,
        "gradient_requests": expected, "time_step_forwards": expected*4096, "dimension": cache.problem.local_dimension,
        "stages": list(r.stages), "threads": torch.get_num_threads(), "wall_seconds": time.perf_counter()-start}


def stats(records: list[dict[str, Any]], mapping: dict[str, int], phase: str, cell: str, arm: str) -> dict[str, Any]:
    selected = sorted([r for r in records if r["cell"] == cell and r["arm"] == arm], key=lambda r: r["rep"])
    if len(selected) < 2 or [r["rep"] for r in selected] != list(range(len(selected))):
        raise ValueError("complete unique fixed independent wholes required")
    logs = [r["log_mu"] for r in selected]
    summary = log_moments(torch.tensor(logs, dtype=torch.float64))
    stable = sensitivity(logs, bootstrap_seed=mapping[f"bootstrap-{phase}-{cell}-{arm}"])
    return {"cell": cell, "arm": arm, "summary": summary, "sensitivity": stable,
        "precision": precision_gate(summary, stable, iid=False),
        "final_initial_ancestors": [r["stages"][-1]["initial_ancestors"] for r in selected],
        "mean_acceptance": [sum(s["mean_acceptance"] for s in r["stages"][:-1])/62 for r in selected],
        "mean_displacement_per_coordinate": [sum(s["mean_squared_displacement_per_coordinate"] for s in r["stages"][:-1])/62 for r in selected],
        "wall_seconds": sum(r["wall_seconds"] for r in selected)}


def pilot_decision(records: list[dict[str, Any]], mapping: dict[str, int], cells: list[str]) -> dict[str, Any]:
    if len(records) != 32 or {ident(r) for r in records} != {ident(j) for j in make_jobs(cells, "pilot")}:
        raise ValueError("all32 pilot wholes required; no prefix allocation")
    rows = [stats(records, mapping, "pilot", c, a) for c in cells for a in ARMS]
    primary = [r for r in rows if r["arm"] == "gaussian-hmc"]
    allocations = {r["cell"]: precision_count(r["summary"], target_rse=.015, safety_factor=3,
                                            minimum=32, maximum=128, multiple=8) for r in primary}
    forecast = sum(allocations[r["cell"]]["required_count"]*r["wall_seconds"]/8 for r in primary)
    proceed = all(a["status"] == "allocated" for a in allocations.values()) and forecast <= 10800
    return {"rows": rows, "primary": "gaussian-hmc", "fresh_allocations": allocations,
        "forecast_seconds_not_guaranteed": forecast, "fresh_precision_authorized": proceed,
        "decision": "RUN_FRESH_PRECISION_DEVELOPMENT" if proceed else "STOP_PRIMARY_PRECISION_FORECAST_BUDGET",
        "pilot_not_promoted_to_confirmation": True, "P2_qualified": False}


def continuum(config: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cell in config["cells"]:
        log_mu = next(r["summary"]["log_mean"] for r in anchor["summary"]["iid_rows"] if r["cell"] == cell["id"])
        for n in config["continuum_steps"]:
            for r in config["holder_r"]:
                bound = probability_root_budget(eta=cell.get("eta", config["model"]["eta"]),
                    xi=config["model"]["xi"], rho=config["model"]["rho"], maturity=config["model"]["maturity"],
                    hurst=config["model"]["hurst"], steps=n, holder_r=r)
                diagnostic = observed_probability_root_diagnostic(log_mu, exponent=float(bound["root_exponent"]),
                    log_root_budget=float(bound["log_root_error_upper"]))
                rows.append({"cell": cell["id"], **bound, **diagnostic})
    return {"rows": rows, "analytic_probability_root_inequality_derived": True,
        "observed_scale_not_true_probability_certificate": True, "useful_relative_certificate_obtained": False}


def precision_summary(records: list[dict[str, Any]], mapping: dict[str, int], cells: list[str],
                      anchor: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    if not pilot["fresh_precision_authorized"]:
        if records:
            raise ValueError("unauthorized precision data")
        return {"executed": False, "qualified_independent_reference": False, "rows": []}
    expected = make_jobs(cells, "precision", counts={c: pilot["fresh_allocations"][c]["required_count"] for c in cells})
    if len(records) != len(expected) or {ident(r) for r in records} != {ident(j) for j in expected}:
        raise ValueError("complete fixed fresh precision family required")
    rows = []
    for c in cells:
        row = stats(records, mapping, "precision", c, "gaussian-hmc")
        ref = next(r for r in anchor["summary"]["iid_rows"] if r["cell"] == c)
        qualified = row["precision"]["pass"] and ref["precision_and_sensitivity"]["pass"]
        eq = relative_equivalence(row["summary"], ref["summary"], comparisons=2) if qualified else {
            "evaluated": False, "pass": False, "reason": "one_or_both_precision_unqualified"}
        rows.append({**row, "equivalence": eq, "qualified_corroboration": bool(qualified and eq["pass"])})
    return {"executed": True, "qualified_independent_reference": all(r["qualified_corroboration"] for r in rows),
        "rows": rows, "not_journal_confirmation_or_continuum_proof": True}


def validate_rows(records: list[dict[str, Any]], mapping: dict[str, int]) -> None:
    for row in records:
        pre = row["phase"] == "preflight"
        expected = 912 if pre else WORK
        if (row["seed"] != mapping[ident(row)] or row["physical_forwards"] != expected
                or row["gradient_requests"] != expected or row["time_step_forwards"] != expected*4096
                or row["dimension"] != 8192 or row["threads"] != 8 or len(row["stages"]) != (15 if pre else 63)):
            raise ValueError("all-whole seed/work/source contract mismatch")


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    path, config = ROOT/DIRECTORY, fixed_config()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("frozen nonoverwriting study required")
    anchor = json.loads((ROOT/ANCHOR).read_text())
    for a, b in ((ANCHOR, ANCHOR_AUDIT), (PREVIOUS, PREVIOUS_AUDIT)):
        value, receipt = json.loads((ROOT/a).read_text()), json.loads((ROOT/b).read_text())
        if receipt["result_sha256"] != sha(ROOT/a):
            raise ValueError("prior audit anchor changed")
        audit_snapshot(ROOT, value["source"], value["config"])
    models = {c["id"]: CachedVolterraTerminal.create(_problem(config["model"], c, 4096)) for c in config["cells"]}
    cells = list(models)
    mapping, seeds = streams(cells)
    path.mkdir(parents=True, exist_ok=True)
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, *config["anchors"]))
    write(path/"protocol.json", {"source": source, "config": config, "seeds": seeds})
    bound = continuum(config, anchor)
    write(path/"continuum.json", bound)
    budgets = {"pilot": Budget(1050000, 7200)}
    preflight: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    fresh: list[dict[str, Any]] = []
    try:
        for job in make_jobs(cells, "preflight"):
            row = unit(job, models[job["cell"]], mapping, budgets["pilot"])
            write(path/(ident(job)+".json"), row)
            preflight.append(row)
            print("preflight "+ident(job), flush=True)
        for job in make_jobs(cells, "pilot"):
            row = unit(job, models[job["cell"]], mapping, budgets["pilot"])
            write(path/(ident(job)+".json"), row)
            records.append(row)
            print(f"pilot {len(records)}/32 {ident(job)} physical={budgets['pilot'].count}", flush=True)
        decision = pilot_decision(records, mapping, cells)
        pilot_work = budgets["pilot"].record()
        write(path/"pilot-result.json", {"source": source, "config": config, "decision": decision,
            "records": records, "preflight": preflight, "work": pilot_work})
        if decision["fresh_precision_authorized"]:
            budgets["precision"] = Budget(8200000, 10800)
            counts = {c: decision["fresh_allocations"][c]["required_count"] for c in cells}
            jobs = make_jobs(cells, "precision", counts=counts)
            for job in jobs:
                row = unit(job, models[job["cell"]], mapping, budgets["precision"])
                write(path/(ident(job)+".json"), row)
                fresh.append(row)
                print(f"fresh precision {len(fresh)}/{len(jobs)} {ident(job)}", flush=True)
        validate_rows([*preflight, *records, *fresh], mapping)
        result = {"source": source, "config": config, "preflight": preflight, "pilot_records": records,
            "precision_records": fresh, "pilot": decision, "precision": precision_summary(fresh, mapping, cells, anchor, decision),
            "continuum": bound, "work": {**work_records(budgets), "pilot": pilot_work},
            "P2_qualified": False, "production_adoption": False, "large_training_authorized": False}
        write(path/"result.json", result)
        return result
    except Exception as error:
        write(path/"failure.json", {"error": type(error).__name__+": "+str(error),
            "completed_pilot": len(records), "completed_precision": len(fresh),
            "work": {k: b.record() for k, b in budgets.items()}, "P2_qualified": False})
        raise


def audit() -> dict[str, Any]:
    torch.set_num_threads(8)
    path = ROOT/DIRECTORY
    value, protocol = [json.loads((path/n).read_text()) for n in ("result.json", "protocol.json")]
    if value["source"] != protocol["source"] or value["config"] != protocol["config"]:
        raise ValueError("source/config binding failed")
    pilot_file = json.loads((path/"pilot-result.json").read_text())
    if pilot_file != {"source": value["source"], "config": value["config"], "decision": value["pilot"],
            "records": value["pilot_records"], "preflight": value["preflight"], "work": value["work"]["pilot"]}:
        raise ValueError("pilot/final source/decision/work binding failed")
    snapshot = audit_snapshot(ROOT, value["source"], value["config"])
    if fixed_config() != value["config"]:
        raise ValueError("changed fixed contract or prior anchors")
    for n, h in value["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/n) != h:
            raise ValueError("frozen source changed: "+n)
    seeds, config = protocol["seeds"], value["config"]
    mapping = seeds["mapping"]
    ledger = SeedLedger.from_dict(seeds["ledger"])
    if ledger.sha256 != seeds["sha256"] or len(mapping) != len(set(mapping.values())) or set(mapping.values()) != {r.seed for r in ledger.records}:
        raise ValueError("seed ledger binding failed")
    for record in ledger.records:
        k = record.key
        key = (f"{k.role}-{k.task}-{k.stream}" if k.role.startswith("bootstrap-") else
               ident({"phase": k.role, "cell": k.task, "arm": k.stream, "rep": k.replicate}))
        if mapping.get(key) != record.seed or k.protocol != NAMESPACE or k.regime != "development" or k.level != 4096:
            raise ValueError("seed mapping semantic binding failed")
    models = {c["id"]: CachedVolterraTerminal.create(_problem(config["model"], c, 4096)) for c in config["cells"]}
    cells, anchor = list(models), json.loads((ROOT/ANCHOR).read_text())
    rows = [*value["preflight"], *value["pilot_records"], *value["precision_records"]]
    validate_rows(rows, mapping)
    if len(value["preflight"]) != 4 or {ident(r) for r in value["preflight"]} != {ident(j) for j in make_jobs(cells, "preflight")}:
        raise ValueError("incomplete preflight family")
    budget, replayed = Budget(200000, 1800), 0
    for row in rows:
        if json.loads((path/(ident(row)+".json")).read_text()) != row:
            raise ValueError("all-whole file/result mismatch")
        if row["phase"] != "preflight" and row["rep"] == 0:
            repeated = unit(row, models[row["cell"]], mapping, budget)
            if any(repeated[k] != v for k, v in row.items() if k != "wall_seconds"):
                raise ValueError("representative whole replay failed")
            replayed += 1
    pilot = pilot_decision(value["pilot_records"], mapping, cells)
    if (pilot != value["pilot"] or continuum(config, anchor) != value["continuum"]
            or precision_summary(value["precision_records"], mapping, cells, anchor, pilot) != value["precision"]
            or json.loads((path/"continuum.json").read_text()) != value["continuum"]):
        raise ValueError("decision or proof calculation replay failed")
    for phase, records in (("pilot", [*value["preflight"], *value["pilot_records"]]), ("precision", value["precision_records"])):
        work = value["work"][phase]
        expected = sum(r["physical_forwards"] for r in records)
        if (work["physical_forwards"] != expected or work["gradient_callback_requests"] != expected
                or work["time_step_forwards"] != expected*4096):
            raise ValueError("phase work mismatch")
    result = {"status": "SOURCE_ALL_WHOLES_DECISIONS_AND_REPRESENTATIVE_REPLAYS_PASSED",
        "source": snapshot, "result_sha256": sha(path/"result.json"), "all_whole_records_checked": len(rows),
        "representative_wholes_replayed": replayed, "work": budget.record(), "P2_qualified": False,
        "not_independent_confirmation_or_relative_bias_certificate": True}
    write(path/"audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "audit"))
    mode = parser.parse_args().mode
    output = run() if mode == "run" else audit()
    print(json.dumps({"mode": mode, "status": output.get("status", output.get("pilot", {}).get("decision"))}))
