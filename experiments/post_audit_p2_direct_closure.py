"""Direct bounded P2 reference/mesh study with an explicit continuum-proof audit."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import psutil
import torch

from experiments.post_audit_fresh_ensemble_fine_smc import precision_gate
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import configuration
from src.path_integral.continuum_probability_bound import continuum_bound_diagnostic
from src.path_integral.cyclic_umbrella_smc import estimate_cyclic_umbrella
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    precision_count,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

NAMESPACE = "p2-direct-cyclic-reference-v1"
DIRECTORY = "results/post_audit/p2_direct_closure_v1/session_20261027"
PLAN = "docs/plans/P2_DIRECT_CLOSURE_EXECUTION_2026-10-27_KO.md"
ANCHOR = "results/post_audit/fresh_ensemble_fine_smc_v1/session_20261020/result.json"
ANCHOR_AUDIT = "results/post_audit/fresh_ensemble_fine_smc_v1/session_20261020/audit.json"
ARMS = ("gradient-control", "gradient-cyclic")


class Budget:
    def __init__(self, physical: int, wall: float):
        self.physical_cap, self.wall_cap = physical, wall
        self.physical = self.joint = self.timesteps = 0
        self.started = time.perf_counter()

    def charge(self, count: int, steps: int, unit_started: float) -> None:
        if (self.physical+2*count > self.physical_cap or time.perf_counter()-self.started > self.wall_cap
                or time.perf_counter()-unit_started > 300):
            raise TimeoutError("fixed physical/phase/unit cap exceeded; no partial normalizer accepted")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("fixed RSS4GiB cap exceeded")
        self.joint += count
        self.physical += 2*count
        self.timesteps += count*(steps+steps//2)

    def to_dict(self) -> dict[str, Any]:
        return {"joint_component_callbacks": self.joint, "physical_forward_evaluations": self.physical,
            "time_step_forward_evaluations": self.timesteps, "wall_seconds": time.perf_counter()-self.started,
            "backward_requests_are_joint_not_physical_call_equivalents": True}


def jobs(cells: list[str]) -> list[dict[str, Any]]:
    return [{"cell": c, "arm": a, "rep": r} for c in cells for a in ARMS for r in range(8)]


def identity(job: dict[str, Any]) -> str:
    return f"{job['cell']}-{job['arm']}-{job['rep']}"


def seed_key(job: dict[str, Any], phase: str) -> str:
    return phase+"/"+identity(job)


def streams(cells: list[str]) -> tuple[dict[str, int], dict[str, Any]]:
    ledger, mapping = SeedLedger(), {}
    requests = [(j, "final") for j in jobs(cells)]
    requests += [({"cell": c, "arm": a, "rep": 0}, p) for c in cells for a in ARMS
                 for p in ("preflight", "bootstrap-fine", "bootstrap-coarse")]
    for job, phase in requests:
        mapping[seed_key(job, phase)] = ledger.allocate(SeedKey(
            NAMESPACE, phase, "development", job["cell"], 4096, job["rep"], job["arm"]))
    old: set[int] = set()

    def scan(v: Any, context: bool = False) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                scan(x, context or "seed" in k.lower())
        elif isinstance(v, list):
            for x in v:
                scan(x, context)
        elif context and type(v) is int and 0 < v < 2**63:
            old.add(v)

    paths = sorted((ROOT/"results/post_audit").glob("*/session_2026102*/protocol.json"))
    paths = [p for p in paths if p != ROOT/DIRECTORY/"protocol.json"]
    for path in paths:
        scan(json.loads(path.read_text(encoding="utf-8")))
    if old & set(mapping.values()):
        raise ValueError("new/scoped historical seed collision")
    return mapping, {"mapping": mapping, "ledger": ledger.to_dict(), "ledger_sha256": ledger.sha256,
        "historical_paths": [p.relative_to(ROOT).as_posix() for p in paths],
        "scoped_historical_seed_values": len(old), "collision_count": 0,
        "not_exhaustive_history_or_PRNG_independence_proof": True}


def prepare(config: dict[str, Any]) -> dict[str, PairedVolterraProbability]:
    return {c["id"]: PairedVolterraProbability.create(_problem(config["model"], c, 4096)) for c in config["cells"]}


def unit(job: dict[str, Any], model: PairedVolterraProbability, mapping: dict[str, int],
         budget: Budget, *, preflight: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    phase = "preflight" if preflight else "final"
    points, particles = (16, 16) if preflight else (64, 128)
    cfg = WeightedSMCConfig(particles, tuple((k/(points-1))**3 for k in range(points)), 2, .35, 1,
        mapping[seed_key(job, phase)], resample_every=4, resampling_scheme="stratified")

    def callback(z):
        budget.charge(len(z), 4096, started)
        return model.log_components(z)

    result = estimate_cyclic_umbrella(callback, dimension=model.dimension,
        log_family_weights=torch.full((2,), -math.log(2), dtype=torch.float64), config=cfg,
        use_cyclic=job["arm"] == "gradient-cyclic")
    expected = particles*(1+(points-2)*2)
    if result.joint_evaluations != expected or result.joint_gradient_requests != expected:
        raise ValueError("unexpected fixed callback work")
    if abs(float(torch.logsumexp(result.log_component_normalizers, 0))-math.log(2)-result.log_normalizer) > 1e-10:
        raise ValueError("unnormalized endpoint family identity failed")
    budget.charge(0, 4096, started)
    return {**job, "preflight": preflight, "seed": cfg.seed,
        "log_normalizer": result.log_normalizer, "log_component_normalizers": result.log_component_normalizers.tolist(),
        "joint_callbacks": expected, "joint_gradient_requests": expected, "physical_forward": 2*expected,
        "time_step_forward": expected*6144, "stages": list(result.stages), "wall_seconds": time.perf_counter()-started,
        "guide_free_proposal": True, "torch_threads": torch.get_num_threads(), "augmentation_dimension": model.dimension}


def summarize(records: list[dict[str, Any]], anchor: dict[str, Any], mapping: dict[str, int]) -> dict[str, Any]:
    cells = [c["id"] for c in configuration()["cells"]]
    actual = [identity(r) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != {identity(j) for j in jobs(cells)}:
        raise ValueError("all32 complete unique independent wholes required")
    rows = []
    for cell in cells:
        ref = next(r for r in anchor["summary"]["iid_rows"] if r["cell"] == cell)
        for arm in ARMS:
            selected = sorted([r for r in records if r["cell"] == cell and r["arm"] == arm], key=lambda r: r["rep"])
            logs = torch.tensor([r["log_component_normalizers"] for r in selected], dtype=torch.float64)
            components: list[dict[str, Any]] = []
            for i, grid in enumerate(("fine", "coarse")):
                stats = log_moments(logs[:, i])
                stable = sensitivity(logs[:, i].tolist(), bootstrap_seed=mapping[seed_key(
                    {"cell": cell, "arm": arm, "rep": 0}, "bootstrap-"+grid)])
                components.append({"grid": grid, "summary": stats, "sensitivity": stable,
                    "precision": precision_gate(stats, stable, iid=False)})
            qualified = bool(ref["precision_and_sensitivity"]["pass"] and components[0]["precision"]["pass"])
            comparison = (relative_equivalence(ref["summary"], components[0]["summary"], comparisons=4) if qualified
                else {"evaluated": False, "pass": False, "reason": "one_or_both_precision_unqualified"})
            forecast = precision_count(components[0]["summary"], target_rse=.015, safety_factor=3,
                                       minimum=32, maximum=256, multiple=8)
            rows.append({"cell": cell, "arm": arm, "components": components,
                "reference_equivalence": comparison, "both_accuracy_qualified": qualified,
                "signed_adjacent_correction": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=4),
                "fine_reference_development_qualified": bool(qualified and comparison["pass"]),
                "fresh_whole_forecast": forecast, "not_a_guaranteed_allocation": True,
                "final_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in selected],
                "minimum_stage_ess": [min(s["ess_before_resampling"] for s in r["stages"]) for r in selected],
                "mean_second_mutation_acceptance": [sum(s["second_mutation_acceptance"] for s in r["stages"][:-1])/62 for r in selected],
                "wall_seconds": sum(r["wall_seconds"] for r in selected)})
    primary = [r for r in rows if r["arm"] == "gradient-cyclic"]
    return {"rows": rows, "primary": "gradient-cyclic",
        "primary_fine_reference_development_qualified": all(r["fine_reference_development_qualified"] for r in primary),
        "primary_adjacent_development_qualified": all(r["signed_adjacent_correction"]["pass"] for r in primary),
        "decision": "P2_UNRESOLVED_THEORETICAL_REMAINDER_AND_ANY_FAILED_EMPIRICAL_GATES",
        "P2_qualified": False, "p1_all_q_risk_measured": False, "production_adoption": False,
        "large_training_authorized": False, "independent_fresh_confirmation_completed": False}


def validate_work(records: list[dict[str, Any]], preflight: list[dict[str, Any]],
                  mapping: dict[str, int], work: dict[str, Any]) -> None:
    for rows, phase, count, stages in ((records, "final", 16000, 63), (preflight, "preflight", 464, 15)):
        for row in rows:
            if (row["seed"] != mapping[seed_key(row, phase)] or row["joint_callbacks"] != count
                    or row["joint_gradient_requests"] != count or row["physical_forward"] != 2*count
                    or row["time_step_forward"] != 6144*count or row["augmentation_dimension"] != 10240
                    or len(row["stages"]) != stages or row["torch_threads"] != 8):
                raise ValueError("all-whole seed/work/dimension/stage contract mismatch")
            vector = torch.tensor(row["log_component_normalizers"], dtype=torch.float64)
            if abs(float(torch.logsumexp(vector, 0))-math.log(2)-row["log_normalizer"]) > 1e-10:
                raise ValueError("all-whole unnormalized family identity mismatch")
    expected = 32*16000+4*464
    if (len(records) != 32 or len(preflight) != 4 or work["joint_component_callbacks"] != expected
            or work["physical_forward_evaluations"] != 2*expected
            or work["time_step_forward_evaluations"] != 6144*expected):
        raise ValueError("complete phase work accounting mismatch")


def continuum_rows(config: dict[str, Any], anchor: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"cell": c["id"], **continuum_bound_diagnostic(eta=c.get("eta", config["model"]["eta"]),
        xi=config["model"]["xi"], rho=config["model"]["rho"], maturity=config["model"]["maturity"],
        hurst=config["model"]["hurst"], steps=n, observed_log_mu=next(r["summary"]["log_mean"]
        for r in anchor["summary"]["iid_rows"] if r["cell"] == c["id"]))}
        for c in config["cells"] for n in (4096, 8192, 16384)]


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("one frozen nonoverwriting direct study required")
    anchor = json.loads((ROOT/ANCHOR).read_text(encoding="utf-8"))
    receipt = json.loads((ROOT/ANCHOR_AUDIT).read_text(encoding="utf-8"))
    if receipt["result_sha256"] != sha(ROOT/ANCHOR):
        raise ValueError("old audited reference changed")
    audit_snapshot(ROOT, anchor["source"], anchor["config"])
    config: dict[str, Any] = {"namespace": NAMESPACE, "model": configuration()["model"], "cells": configuration()["cells"],
        "fine_steps": 4096, "coarse_steps": 2048, "particles": 128, "points": 64, "power": 3,
        "moves": 2, "scale": .35, "drift_cap": 20., "wholes": 8, "arms": list(ARMS),
        "resample_every": 4, "resampling": "stratified", "torch_threads": 8,
        "maximum_rse": .015, "maximum_unit_share": .15, "maximum_loo": .05,
        "equivalence_margin": .1, "comparisons": 4, "physical_cap": 1050000,
        "phase_wall": 7200, "whole_wall": 300,
        "anchors": {n: sha(ROOT/n) for n in (ANCHOR, ANCHOR_AUDIT)}}
    models = prepare(config)
    mapping, seeds = streams(list(models))
    path.mkdir(parents=True, exist_ok=True)
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR, ANCHOR_AUDIT,
        "docs/theory/CONDITIONAL_CDF_ABSOLUTE_BIAS_BOUND_2026-10-16_KO.md",
        "docs/theory/ORIGINAL_BLP_INTEGRATED_DRIVER_MSE_BOUND_2026-10-19_KO.md"))
    write(path/"protocol.json", {"source": source, "config": config, "seeds": seeds})
    theoretical = continuum_rows(config, anchor)
    write(path/"continuum.json", {"rows": theoretical, "relative_bias_certificate_obtained": False,
        "cannot_substitute_adjacent_difference_or_asymptotic_rate": True})
    budget = Budget(1050000, 7200)
    records: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    try:
        for cell in models:
            for arm in ARMS:
                job: dict[str, Any] = {"cell": cell, "arm": arm, "rep": 0}
                row = unit(job, models[cell], mapping, budget, preflight=True)
                write(path/("preflight-"+identity(job)+".json"), row)
                preflight.append(row)
                print("preflight "+identity(job), flush=True)
        for job in jobs(list(models)):
            row = unit(job, models[job["cell"]], mapping, budget)
            write(path/(identity(job)+".json"), row)
            records.append(row)
            print(f"complete {len(records)}/32 {identity(job)} physical={budget.physical}", flush=True)
        validate_work(records, preflight, mapping, budget.to_dict())
        value = {"source": source, "config": config, "records": records, "preflight": preflight,
            "continuum": theoretical, "work": budget.to_dict(), "summary": summarize(records, anchor, mapping)}
        write(path/"result.json", value)
        return value
    except Exception as error:
        write(path/"failure.json", {"error": type(error).__name__+": "+str(error),
            "complete_units": len(records), "work": budget.to_dict(), "P2_qualified": False})
        raise


def audit() -> dict[str, Any]:
    torch.set_num_threads(8)
    path = ROOT/DIRECTORY
    value = json.loads((path/"result.json").read_text(encoding="utf-8"))
    protocol = json.loads((path/"protocol.json").read_text(encoding="utf-8"))
    if protocol["source"] != value["source"] or protocol["config"] != value["config"]:
        raise ValueError("protocol/result binding mismatch")
    source = audit_snapshot(ROOT, value["source"], value["config"])
    for n, h in value["config"]["anchors"].items():
        if sha(ROOT/n) != h:
            raise ValueError("reference anchor changed")
    anchor = json.loads((ROOT/ANCHOR).read_text(encoding="utf-8"))
    mapping = protocol["seeds"]["mapping"]
    ledger = SeedLedger.from_dict(protocol["seeds"]["ledger"])
    if ledger.sha256 != protocol["seeds"]["ledger_sha256"] or set(mapping.values()) != {r.seed for r in ledger.records}:
        raise ValueError("seed ledger mismatch")
    models = prepare(value["config"])
    validate_work(value["records"], value["preflight"], mapping, value["work"])
    for row in value["preflight"]:
        if json.loads((path/("preflight-"+identity(row)+".json")).read_text()) != row:
            raise ValueError("preflight/result binding mismatch")
    budget, replayed = Budget(130000, 1800), 0
    for row in value["records"]:
        if json.loads((path/(identity(row)+".json")).read_text()) != row:
            raise ValueError("whole/result binding mismatch")
        if row["rep"] == 0:
            repeated = unit(row, models[row["cell"]], mapping, budget)
            if any(repeated[k] != v for k, v in row.items() if k != "wall_seconds"):
                raise ValueError("representative independent whole replay mismatch")
            replayed += 1
    if summarize(value["records"], anchor, mapping) != value["summary"] or continuum_rows(value["config"], anchor) != value["continuum"]:
        raise ValueError("statistical/theoretical decision replay mismatch")
    for n, h in value["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/n) != h:
            raise ValueError("frozen runtime source changed: "+n)
    output = {"status": "SOURCE_SEED_ALL_WHOLES_DECISIONS_AND_FOUR_REPRESENTATIVE_REPLAYS_PASSED",
        "source": source, "result_sha256": sha(path/"result.json"), "whole_units_checked": len(value["records"]),
        "whole_replays": replayed, "work": budget.to_dict(), "P2_qualified": False,
        "not_independent_confirmation_or_continuum_proof": True}
    write(path/"audit.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "audit"))
    args = parser.parse_args()
    output = run() if args.mode == "run" else audit()
    print(json.dumps({"mode": args.mode, "status": output.get("status", output.get("summary", {}).get("decision"))}))
