"""Frozen multi-time covariance proposals; development, never P2 approval."""

from __future__ import annotations

import argparse
import json
import math
import time
from statistics import NormalDist
from typing import Any

import numpy as np
import psutil
import torch

from experiments.post_audit_conditional_intervention import store_array
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import configuration, prepare
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.residual_brownian_covariance import ARMS, ResidualBrownianCovarianceFamily
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.stratified_paired_risk import (
    estimate,
    merge_sufficient,
    paired,
    sufficient,
)
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/residual_brownian_covariance_v1/session_20261026"
PLAN = "docs/plans/MULTITIME_RESIDUAL_BROWNIAN_PROPOSAL_2026-10-26_KO.md"
NAMESPACE = "multitime-residual-brownian-covariance-v1"
ANCHOR = "results/post_audit/conditional_intervention_v1/session_20261026/result.json"
ANCHOR_SHA = "d4d1215077eef40fc842f0f78668bd7ab742cc79379646b7c3779826d55b0619"
CHUNK = 256


class Budget:
    def __init__(self, physical: int = 730000, scores: int = 26000000000, wall: float = 1800):
        self.physical_cap, self.score_cap, self.wall = physical, scores, wall
        self.physical = self.scores = 0
        self.started = time.perf_counter()

    def charge(self, count: int, components: int, times: int) -> None:
        if any(type(v) is not int or v < 1 for v in (count, components, times)):
            raise ValueError("positive integer evaluation dimensions required")
        scores = count*(components+3*(components-1)*times+times)
        if (self.physical+count > self.physical_cap or self.scores+scores > self.score_cap
                or time.perf_counter()-self.started > self.wall):
            raise TimeoutError("fixed physical/density/wall cap exceeded")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("fixed RSS4GiB cap exceeded")
        self.physical += count
        self.scores += scores

    def to_dict(self) -> dict[str, Any]:
        return {"physical_evaluations": self.physical, "gaussian_component_time_scores": self.scores,
                "wall_seconds": time.perf_counter()-self.started,
                "density_score_count_is_not_total_FLOP": True}


def key(cell: str, phase: str, block: int, stratum: str, role: str) -> str:
    return f"{cell}/{phase}/{block}/{stratum}/{role}"


def jobs(cells: list[str]) -> list[dict[str, Any]]:
    direct = [{"cell": c, "phase": "direct", "arm": a, "block": b}
              for c in cells for a in ARMS for b in range(8)]
    evaluation = [{"cell": c, "phase": d, "arm": "all", "block": b}
                  for c in cells for d in ("heldout", "reference") for b in range(8)]
    return direct+evaluation


def unit_id(job: dict[str, Any]) -> str:
    return f"{job['cell']}-{job['phase']}-{job['arm']}-{job['block']}"


def streams(cells: list[str]) -> tuple[dict[str, int], dict[str, Any]]:
    ledger, mapping = SeedLedger(), {}
    requests = [(c, "preflight", 0, a) for c in cells for a in ARMS]
    for job in jobs(cells):
        strata = ARMS if job["phase"] == "heldout" else (job["arm"],)
        requests += [(job["cell"], job["phase"], job["block"], a) for a in strata]
    for cell, phase, block, stratum in requests:
        for role in ("path", "label"):
            mapping[key(cell, phase, block, stratum, role)] = ledger.allocate(
                SeedKey(NAMESPACE, phase, "development", cell, 32, block, stratum+"-"+role))
    old: set[int] = set()

    def scan(value: Any, context: bool = False) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                scan(v, context or "seed" in k.lower())
        elif isinstance(value, list):
            for v in value:
                scan(v, context)
        elif context and type(value) is int and 0 < value < 2**63:
            old.add(value)

    history = sorted((ROOT/"results/post_audit").glob("*/session_2026102[1236]/protocol.json"))
    for path in history:
        scan(json.loads(path.read_text(encoding="utf-8")))
    if old & set(mapping.values()):
        raise ValueError("new/scoped historical seed collision")
    return mapping, {"ledger": ledger.to_dict(), "ledger_sha256": ledger.sha256,
        "historical_paths": [p.relative_to(ROOT).as_posix() for p in history],
        "scoped_historical_seed_integer_values": len(old), "collision_count": 0,
        "not_exhaustive_history_or_PRNG_independence_proof": True}


def sample(family: ResidualBrownianCovarianceFamily, count: int, job: dict[str, Any],
           stratum: str, mapping: dict[str, int]) -> torch.Tensor:
    kwargs = {role+"_seed": mapping[key(job["cell"], job["phase"], job["block"], stratum, role)]
              for role in ("path", "label")}
    if job["phase"] == "reference":
        return family.reference_sample(count, **kwargs)
    return family.sample(count, stratum, **kwargs)


def scores(cache: Any, family: ResidualBrownianCovarianceFamily, z: torch.Tensor,
           budget: Budget) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    density_parts, g_parts, r_parts = [], [], []
    density_wall = physics_wall = 0.
    for start in range(0, len(z), CHUNK):
        points = z[start:start+CHUNK]
        budget.charge(len(points), family.base.components, family.times)
        clock = time.perf_counter()
        d, r = family.log_densities(points), family.reference_log_density(points)
        density_wall += time.perf_counter()-clock
        clock = time.perf_counter()
        g = cache.log_probability(points)
        physics_wall += time.perf_counter()-clock
        if not torch.isfinite(g).all() or (g > 1e-12).any():
            raise FloatingPointError("finite bounded conditional payoff required")
        density_parts.append(d)
        r_parts.append(r)
        g_parts.append(g)
    return torch.cat(g_parts), torch.cat(density_parts), torch.cat(r_parts), {
        "density_all_four_arms_and_reference": density_wall, "physics": physics_wall}


def values(job: dict[str, Any], prepared: dict[str, Any], mapping: dict[str, int],
           budget: Budget) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache, family = prepared[job["cell"]]
    started = time.perf_counter()
    strata = ARMS if job["phase"] == "heldout" else (job["arm"],)
    count = {"direct": 8192, "heldout": 2048, "reference": 4096}[job["phase"]]
    moments, events, arrays, event_arrays, witnesses = [], [], [], [], []
    sample_wall = density_wall = physics_wall = 0.
    for stratum in strata:
        clock = time.perf_counter()
        z = sample(family, count, job, stratum, mapping)
        sample_wall += time.perf_counter()-clock
        g, d, r, costs = scores(cache, family, z, budget)
        density_wall += costs["density_all_four_arms_and_reference"]
        physics_wall += costs["physics"]
        if job["phase"] == "direct":
            denominator = d[:, ARMS.index(job["arm"])]
        elif job["phase"] == "heldout":
            denominator = torch.logsumexp(d, 1)-math.log(4)
        else:
            denominator = r
        event = (g-denominator)[:, None]
        risk = 2*g[:, None]-denominator[:, None]-d
        logs = event if job["phase"] == "direct" else risk
        moments.append(sufficient(logs))
        events.append(sufficient(event))
        arrays.append(logs.numpy())
        event_arrays.append(event.numpy())
        for column in range(logs.shape[1]):
            index = int(torch.argmax(logs[:, column]))
            witnesses.append({"stratum": stratum, "column": column, "index": index,
                "coordinates": z[index].tolist(), "log_g": float(g[index]),
                "log_density_qs": d[index].tolist(), "log_density_R": float(r[index]),
                "log_contribution": float(logs[index, column]), "post_hoc_not_confirmation": True})
    record = {**job, "strata": moments, "event_strata": events,
        "sampling_wall": sample_wall, "density_all_arms_wall": density_wall,
        "physics_wall": physics_wall, "unit_wall": time.perf_counter()-started,
        "counts_per_stratum": count, "witnesses": witnesses}
    if record["unit_wall"] > 60:
        raise TimeoutError("fixed complete-unit60sec cap exceeded")
    return record, {"logs": np.stack(arrays), "event": np.stack(event_arrays)}


def family_summary(rows: list[dict[str, Any]], field: str = "strata") -> dict[str, Any]:
    if len(rows) != 8 or sorted(r["block"] for r in rows) != list(range(8)):
        raise ValueError("eight complete blocks required")
    rows = sorted(rows, key=lambda r: r["block"])
    strata = len(rows[0][field])
    if any(len(r[field]) != strata for r in rows):
        raise ValueError("complete same stratum family required")
    masses = torch.full((strata,), 1/strata, dtype=torch.float64)
    raw = estimate([merge_sufficient([r[field][s] for r in rows]) for s in range(strata)], masses)
    logs = torch.tensor([estimate(row[field], masses)["log_mean"] for row in rows], dtype=torch.float64)
    block = estimate([sufficient(logs)], torch.ones(1, dtype=torch.float64))
    scaled = (logs-logs.max(0).values).exp()
    total = scaled.sum(0)
    loo = ((total[None, :]-scaled)/7/(total[None, :]/8)-1).abs().max(0).values.tolist()
    return {"raw": raw, "blocks": block, "loo": loo}


def risk_qualified(summary: dict[str, Any]) -> bool:
    return (max(summary["raw"]["relative_se"]+summary["blocks"]["relative_se"]) <= .2
            and max(summary["blocks"]["maximum_fraction"]) <= .25 and max(summary["loo"]) <= .15)


def agreement(h: dict[str, Any], r: dict[str, Any], index: int, *, qualified: bool) -> dict[str, Any]:
    if not qualified:
        return {"evaluated": False, "development_equivalence": False,
                "reason": "at_least_one_reference_precision_unqualified"}
    lh, lr = h["log_mean"][index], r["log_mean"][index]
    scale = max(lh, lr)
    a, b = math.exp(lh-scale), math.exp(lr-scale)
    difference = 2*(a-b)/(a+b)
    se = 4*a*b/(a+b)**2*math.hypot(h["relative_se"][index], r["relative_se"][index])
    z = -NormalDist().inv_cdf(.05/(2*8))
    return {"evaluated": True, "symmetric_relative_difference": difference, "se": se,
        "lower": difference-z*se, "upper": difference+z*se,
        "development_equivalence": abs(difference)+z*se <= .1,
        "not_finite_sample_truth_certificate": True}


def summarize(records: list[dict[str, Any]], cells: list[str]) -> dict[str, Any]:
    actual = [unit_id(r) for r in records]
    expected = [unit_id(j) for j in jobs(cells)]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("all96 fixed unique units required")
    result = []
    for cell in cells:
        direct = {arm: family_summary([r for r in records if r["cell"] == cell
                  and r["phase"] == "direct" and r["arm"] == arm]) for arm in ARMS}
        references: dict[str, Any] = {}
        for phase in ("heldout", "reference"):
            rows = [r for r in records if r["cell"] == cell and r["phase"] == phase]
            risk = family_summary(rows)
            contrasts = [paired(s, 0, 3, comparisons=4) for s in (risk["raw"], risk["blocks"])]
            references[phase] = {"risk": risk, "event": family_summary(rows, "event_strata"),
                "risk_precision_qualified": risk_qualified(risk), "contrasts": contrasts,
                "coupled_decrease_qualified": all(c["absolute_lower"] > 0 and c["relative_lower"] > .2 for c in contrasts)}
        candidate = direct["coupled"]
        event_qualified = (candidate["raw"]["relative_se"][0] <= .05
            and candidate["blocks"]["relative_se"][0] <= 2*candidate["raw"]["relative_se"][0]
            and candidate["raw"]["maximum_fraction"][0] <= .1 and candidate["loo"][0] <= .15)
        reference_precision = all(v["risk_precision_qualified"] for v in references.values())
        contrasts = [agreement(references["heldout"]["risk"]["raw"], references["reference"]["risk"]["raw"],
                               i, qualified=reference_precision) for i in range(4)]
        qualified = (event_qualified and reference_precision and all(c["development_equivalence"] for c in contrasts)
            and all(v["coupled_decrease_qualified"] for v in references.values()))
        result.append({"cell": cell, "direct": direct, "references": references,
            "event_candidate_development_qualified": bool(event_qualified),
            "reference_agreement": contrasts, "separate_confirmation_candidate": bool(qualified)})
    return {"cells": result, "decision": "SEPARATE_CONFIRMATION_CANDIDATE" if all(
        r["separate_confirmation_candidate"] for r in result) else "NO_ADOPTION_COVARIANCE_PRECISION_OR_COVERAGE_UNRESOLVED",
        "event_proposal_candidates_changed": True, "fitting_or_selection_runs": 0,
        "primary_arm": "coupled", "production_adoption": False, "P2_qualified": False,
        "large_training_authorized": False, "not_learned_model_or_population_efficiency_certificate": True}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    if sha(ROOT/ANCHOR) != ANCHOR_SHA:
        raise ValueError("frozen failed intervention anchor changed")
    anchor = json.loads((ROOT/ANCHOR).read_text())
    audit_snapshot(ROOT, anchor["source"], anchor["config"])
    config: dict[str, Any] = {"namespace": NAMESPACE, "model": configuration()["model"],
        "cells": configuration()["cells"], "steps": 32, "variance": 5., "alpha": .5,
        "density_chunk": 256, "arms": list(ARMS), "primary": "coupled", "blocks": 8,
        "direct_count": 8192, "heldout_per_stratum": 2048, "reference_count": 4096,
        "natural_mass": .1, "event_rse": .05, "risk_rse": .2, "maximum_block_fraction": .25,
        "maximum_loo": .15, "paired_relative_lower": .2, "agreement_margin": .1,
        "paired_comparisons": 4, "agreement_comparisons": 8,
        "physical_cap": 730000, "density_score_cap": 26000000000, "phase_wall": 1800,
        "preflight_prediction_safety": 1.5, "source_anchor_sha256": ANCHOR_SHA}
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("fresh nonoverwriting one-shot covariance study")
    path.mkdir(parents=True, exist_ok=True)
    ready, setup = prepare(config)
    prepared = {cell: (cache, ResidualBrownianCovarianceFamily(spec,
        price_sign=-1 if cache.problem.rho > 0 else 1)) for cell, (cache, spec) in ready.items()}
    mapping, ledger = streams(list(prepared))
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write(path/"protocol.json", {"config": config, "source": source, "seeds": ledger, "setup": setup})
    budget = Budget()
    records: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    try:
        preflight_started = time.perf_counter()
        for cell, (cache, family) in prepared.items():
            for arm in ARMS:
                job = {"cell": cell, "phase": "preflight", "block": 0}
                z = sample(family, 128, job, arm, mapping)
                g, d, r, timing = scores(cache, family, z, budget)
                preflight.append({"cell": cell, "arm": arm, "costs": timing,
                    "minimum_log_density": float(d.min()), "minimum_log_reference_density": float(r.min()),
                    "finite_payoffs": bool(torch.isfinite(g).all())})
        elapsed = time.perf_counter()-preflight_started
        forecast = elapsed/1024*720896*1.5
        write(path/"preflight.json", {"rows": preflight, "elapsed": elapsed,
            "production_wall_forecast_with_safety": forecast, "not_runtime_guarantee": True})
        if forecast > config["phase_wall"]:
            raise TimeoutError("prespecified cost forecast excludes production")
        for job in jobs(list(prepared)):
            row, arrays = values(job, prepared, mapping, budget)
            name = unit_id(job)
            row["array_path"] = name+".npz"
            row["array_sha256"] = store_array(path/row["array_path"], arrays)
            write(path/(name+".json"), row)
            records.append(row)
            print(f"complete {name} physical={budget.physical}", flush=True)
        summary = summarize(records, list(prepared))
        value: dict[str, Any] = {"source": source, "config": config, "setup": setup, "records": records,
            "preflight": preflight, "work": budget.to_dict(), "summary": summary}
        write(path/"result.json", value)
    except Exception as error:
        write(path/"failure.json", {"error": type(error).__name__+": "+str(error),
            "complete_units": len(records), "work": budget.to_dict(), "P2_qualified": False})
        raise
    return value


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    path = ROOT/DIRECTORY
    value = json.loads((path/"result.json").read_text())
    snapshot = audit_snapshot(ROOT, value["source"], value["config"])
    ready, _ = prepare(value["config"])
    prepared = {cell: (cache, ResidualBrownianCovarianceFamily(spec,
        price_sign=-1 if cache.problem.rho > 0 else 1)) for cell, (cache, spec) in ready.items()}
    # Use the frozen ledger rather than rescanning a history now containing this run.
    protocol = json.loads((path/"protocol.json").read_text())
    ledger = SeedLedger.from_dict(protocol["seeds"]["ledger"])
    mapping = {key(record.key.task, record.key.role, record.key.replicate,
                   record.key.stream.rsplit("-", 1)[0], record.key.stream.rsplit("-", 1)[1]): record.seed
               for record in ledger.records}
    budget = Budget(45000, 1500000000, 600)
    replayed = 0
    for row in value["records"]:
        name = unit_id(row)
        if json.loads((path/(name+".json")).read_text()) != row or sha(path/row["array_path"]) != row["array_sha256"]:
            raise ValueError("unit/array/result binding mismatch")
        with np.load(path/row["array_path"], allow_pickle=False) as arrays:
            if len(arrays["logs"]) != len(row["strata"]):
                raise ValueError("known stratum completion mismatch")
            for field, name_array in (("strata", "logs"), ("event_strata", "event")):
                calculated = [sufficient(torch.from_numpy(part)) for part in arrays[name_array]]
                if calculated != row[field]:
                    raise ValueError("complete array moment recomputation mismatch")
            if row["block"] == 0 and (row["phase"] != "direct" or row["arm"] == "coupled"):
                fresh, replay_arrays = values(row, prepared, mapping, budget)
                for k in replay_arrays:
                    if not np.array_equal(replay_arrays[k], arrays[k]):
                        raise ValueError("representative physics replay mismatch")
                if fresh["witnesses"] != row["witnesses"]:
                    raise ValueError("representative witness mismatch")
                replayed += 1
    if summarize(value["records"], list(prepared)) != value["summary"]:
        raise ValueError("complete scientific decision replay mismatch")
    changed = [name for name, digest in value["source"]["snapshot_file_hashes"].items()
               if sha(ROOT/name) != digest]
    if changed:
        raise ValueError("frozen source changed: "+str(changed))
    result = {"status": "ALL_ARRAY_STATISTICS_AND_REPRESENTATIVE_PHYSICS_REPLAY_PASSED",
        "result_sha256": sha(path/"result.json"), "source": snapshot,
        "array_units_checked": len(value["records"]), "representative_replays": replayed,
        "frozen_current_files_checked": len(value["source"]["snapshot_file_hashes"]),
        "work": budget.to_dict(), "not_independent_confirmation": True, "P2_qualified": False}
    write(path/"audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "audit"))
    args = parser.parse_args()
    result = run() if args.mode == "run" else audit()
    print(json.dumps({"mode": args.mode, "status": result.get("status", result.get("summary", {}).get("decision"))}))
