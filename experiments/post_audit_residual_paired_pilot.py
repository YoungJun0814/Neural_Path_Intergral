"""Fresh paired residual pilot; no matched-wall or production speedup claim."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from functools import partial
from typing import Any

import torch

from experiments.post_audit_bounded_bank_runtime import key
from experiments.post_audit_bounded_frozen_bank_contrast import DIRECTORY as D2
from experiments.post_audit_exact_residual_structure import anchors
from experiments.post_audit_frozen_q_path_attribution import COUNT, phase_key
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_residual_log_moments import DIRECTORY as Q0
from experiments.post_audit_residual_log_moments import PLAN, ScalarBudget
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.balanced_mixture_bank import stream_layout
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    sample_gaussian_mixture,
)
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.residual_log_quadrature import log_gaussian_moments, slice_log_probability
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.stratified_paired_risk import estimate, sufficient
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/residual_paired_pilot_v1/session_20261021"
NAMESPACE = "residual-log-moment-pilot-v1"
ENDPOINTS = ["raw_event", "conditional_event", "raw_event_M2", "original_q_conditional_M2", "conditional_event_M2"]


class PilotBudget(ScalarBudget):
    def __init__(self, scalar: int, wall: float, physical: int):
        super().__init__(scalar, wall)
        self.physical, self.physical_cap = 0, physical

    def charge_physical(self, n: int) -> None:
        self.charge(0)
        if type(n) is not int or n < 1 or self.physical+n > self.physical_cap:
            raise TimeoutError("fixed physical budget exceeded")
        self.physical += n


def seeds(cells: list[str], prepared: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = SeedLedger()
    mapping = {}
    for c in cells:
        for b in range(8):
            mapping[c+"/"+str(b)] = {role: ledger.allocate(SeedKey(NAMESPACE, "pilot", "development", c, 32, b, role))
                for role in ("path", "label")}
    old: set[int] = set()
    for cell in cells:
        spec = prepared[cell][1]
        requests = [(1240, d, key("preflight", cell, d)) for d in ("iid", "stratified")]
        requests += [(29760, d, key("D2", cell, d, block=b)) for d in ("iid", "stratified") for b in range(8)]
        requests += [(COUNT, "stratified", phase_key("confirmation", cell, b)) for b in range(8)]
        for n, design, k in requests:
            for row in stream_layout(spec, n, design, k):
                old.update(row[x] for x in ("path_seed", "label_seed") if x in row)
    new = {x.seed for x in ledger.records}
    if len(new) != 32 or new & old:
        raise ValueError("new/historical scoped seed collision")
    return mapping, {"ledger": ledger.to_dict(), "ledger_sha256": ledger.sha256,
        "historical_distinct_seeds_compared": len(old), "collision_count": 0,
        "scope": "D2 IID/stratified preflight and production, plus feature confirmation; not exhaustive whole-repository history"}


def scores(log_g: float, log_d: float, quadrature: dict[str, Any]) -> list[float]:
    m1, m2 = [m["log_estimate"] for m in quadrature["moments"]]
    return [log_g-log_d, m1-log_d, 2*(log_g-log_d), m2-2*log_d, 2*(m1-log_d)]


def block(cache: Any, spec: GaussianMixtureShiftSpec, cell: str, index: int,
          streams: dict[str, int], budget: PilotBudget) -> dict[str, Any]:
    started = time.perf_counter()
    if not bool((spec.means[:, 61] == 0).all()) or spec.dimension != 64:
        raise ValueError("fixed blind local residual law required")
    clock = time.perf_counter()
    draw = sample_gaussian_mixture(spec, 16, gaussian_generator=torch.Generator().manual_seed(streams["path"]),
        label_generator=torch.Generator().manual_seed(streams["label"]))
    z = draw.target_coordinates
    draw_wall = time.perf_counter()-clock
    clock = time.perf_counter()
    d = torch.logsumexp(z @ spec.means.T-.5*spec.means.square().sum(1)+spec.weights.log(), 1)
    density_wall = time.perf_counter()-clock
    clock = time.perf_counter()
    budget.charge_physical(16)
    g = cache.log_probability(z)
    g_wall = time.perf_counter()-clock
    rows: list[dict[str, Any]] = []
    profile_wall, quadrature_wall = 0., 0.
    for i in range(16):
        clock = time.perf_counter()
        budget.charge_physical(1)
        profile = LastVarianceResidualSlice.create(cache, z[i:i+1])
        profile_wall += time.perf_counter()-clock
        clock = time.perf_counter()
        moments = log_gaussian_moments(partial(slice_log_probability, profile), charge=budget.charge)
        quadrature_wall += time.perf_counter()-clock
        rows.append({"coordinates": z[i].tolist(), "profile": dataclasses.asdict(profile),
            "log_g": float(g[i]), "log_density": float(d[i]), "moments": moments,
            "scores": scores(float(g[i]), float(d[i]), moments)})
    return {"cell": cell, "block": index, "seeds": streams, "rows": rows,
        "draw_wall": draw_wall, "density_wall": density_wall, "raw_payoff_wall": g_wall,
        "profile_wall": profile_wall, "quadrature_wall": quadrature_wall,
        "wall": time.perf_counter()-started, "physical_evaluations": 32,
        "scalar_evaluations": sum(r["moments"]["scalar_evaluations"] for r in rows)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    cells = sorted({r["cell"] for r in records})
    if len(records) != 16 or len(cells) != 2 or any(sorted(r["block"] for r in records if r["cell"] == c) != list(range(8)) for c in cells):
        raise ValueError("two complete eight-block pilots required")
    summaries = []
    for cell in cells:
        group = sorted((r for r in records if r["cell"] == cell), key=lambda r: r["block"])
        if any(len(r["rows"]) != 16 for r in group):
            raise ValueError("all contexts required")
        logs = torch.tensor([p["scores"] for r in group for p in r["rows"]], dtype=torch.float64)
        raw = estimate([sufficient(logs)], torch.ones(1, dtype=torch.float64))
        blocklogs = logs.reshape(8, 16, 5).logsumexp(1)-math.log(16)
        blocks = estimate([sufficient(blocklogs)], torch.ones(1, dtype=torch.float64))
        scaled = (blocklogs-blocklogs.max(0).values).exp()
        total = scaled.sum(0)
        loo = ((total[None, :]-scaled)/7/(total[None, :]/8)-1).abs().max(0).values.tolist()
        resolved = all(p["moments"]["relative_numerically_resolved"] for r in group for p in r["rows"])
        gates = [raw["relative_se"][i] <= .2 and blocks["relative_se"][i] <= .2
            and raw["maximum_fraction"][i] <= .1 and loo[i] <= .15 for i in range(5)]
        mu2log = 2*raw["log_mean"][1]
        scale = max(raw["log_mean"][3:5])
        original = math.exp(raw["log_mean"][3]-scale)-math.exp(mu2log-scale)
        conditional = math.exp(raw["log_mean"][4]-scale)-math.exp(mu2log-scale)
        if min(original, conditional) < -1e-10 or conditional > original+1e-10:
            raise FloatingPointError("paired conditional moment/Jensen identity violated")
        raw_cost = sum(r["draw_wall"]+r["density_wall"]+r["raw_payoff_wall"] for r in group)
        conditional_cost = sum(r["draw_wall"]+r["density_wall"]+r["profile_wall"]+r["quadrature_wall"] for r in group)
        variance_ratio = conditional/original if original > 0 else None
        cost_ratio = conditional_cost/raw_cost if raw_cost > 0 else None
        proxy = variance_ratio*cost_ratio if variance_ratio is not None and cost_ratio is not None else None
        approximate_errors = []
        for k in (1, 3, 4):
            numerators = []
            for r in group:
                for p in r["rows"]:
                    error = p["moments"]["moments"][0 if k in (1, 4) else 1]["relative_error_proxy_plus_tail"]
                    if error is None:
                        error = 1e300
                    if k == 4:
                        error = 2*error+error**2 if error < 1e100 else 1e300
                    numerators.append(p["scores"][k]+math.log(error) if error > 0 else -math.inf)
            ratio_log = float(torch.logsumexp(torch.tensor(numerators, dtype=torch.float64), 0))-math.log(128)-raw["log_mean"][k]
            approximate_errors.append({"endpoint": ENDPOINTS[k], "relative_quadrature_error_proxy": math.exp(min(690., ratio_log)),
                "not_sampling_se_or_rigorous_bias_bound": True})
        qualified = resolved and all(gates)
        summaries.append({"cell": cell, "count": 128, "endpoints": ENDPOINTS, "raw": raw,
            "blocks": blocks, "loo_relative_shift": loo, "all_profile_numerics_resolved": resolved,
            "precision_gates": gates, "qualified_pilot_cell": qualified,
            "event_variance_ratio_proxy": variance_ratio, "computation_cost_ratio": cost_ratio,
            "work_normalized_variance_proxy": proxy, "conditional_variance_loss_fraction_proxy": 1-math.exp(raw["log_mean"][4]-raw["log_mean"][3]),
            "quadrature_error_propagation": approximate_errors,
            "not_matched_wall_or_fixed_precision_benchmark": True})
    qualified = all(c["qualified_pilot_cell"] for c in summaries)
    if not qualified:
        decision = "NO_ADOPTION_PRECISION_UNRESOLVED"
    elif not all(c["work_normalized_variance_proxy"] is not None and c["work_normalized_variance_proxy"] <= .8 for c in summaries):
        decision = "STOP_SINGLE_COORDINATE_PRACTICAL_ADOPTION"
    else:
        decision = "CANDIDATE_FOR_SEPARATE_MATCHED_WALL_STUDY_ONLY"
    return {"cells": summaries, "decision": decision, "P2_qualified": False, "D3_authorized": False,
        "model_or_G_changed": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    q0path = ROOT/Q0
    q0, receipt = [json.loads((q0path/n).read_text()) for n in ("result.json", "audit.json")]
    if sha(q0path/"result.json") != receipt["result_sha256"] or not q0["q1_pilot_numerics_gate"]:
        raise ValueError("complete audited Q0 numerical gate required")
    audit_snapshot(ROOT, q0["source"], q0["config"])
    parent, _ = anchors()
    config = {"namespace": NAMESPACE, "model": parent["config"]["parent_model"], "cells": parent["config"]["cells"],
        "blocks": 8, "block_count": 16, "Q0_sha256": sha(q0path/"result.json"),
        "physics_cap": 512, "scalar_cap": 2500000, "wall_cap": 900, "P2_qualified": False}
    prepared, setup = prepare(config)
    mapping, ledger = seeds(list(prepared), prepared)
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("fresh one-shot pilot; no overwrite")
    path.mkdir(parents=True, exist_ok=True)
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, Q0+"/result.json", Q0+"/audit.json", D2+"/result.json"))
    write_json_atomic_nonoverwriting(path/"protocol.json", {"config": config, "source": source, "seeds": ledger})
    budget = PilotBudget(2500000, 900, 512)
    records, io_wall = [], 0.
    try:
        for cell, (cache, spec) in prepared.items():
            for b in range(8):
                row = block(cache, spec, cell, b, mapping[cell+"/"+str(b)], budget)
                clock = time.perf_counter()
                write_json_atomic_nonoverwriting(path/(cell+"-"+str(b)+".json"), row)
                io_wall += time.perf_counter()-clock
                records.append(row)
                print(f"Completed paired pilot {len(records)}/16", flush=True)
        clock = time.perf_counter()
        summary = summarize(records)
        aggregation_wall = time.perf_counter()-clock
    except Exception as e:
        write_json_atomic_nonoverwriting(path/"failure.json", {"error": type(e).__name__+": "+str(e),
            "physical_evaluations": budget.physical, "scalar_evaluations": budget.count, "complete_blocks": len(records)})
        raise
    output = {"source": source, "config": config, "seeds": ledger, "setup": setup, "records": records,
        "summary": summary, "physical_evaluations": budget.physical, "scalar_evaluations": budget.count,
        "block_io_wall": io_wall, "aggregation_wall": aggregation_wall,
        "total_compute_and_block_io_wall": time.perf_counter()-budget.started,
        "source_and_final_io_not_in_speed_proxy": True, "offline_fit_selection_work": 0}
    write_json_atomic_nonoverwriting(path/"result.json", output)
    return output


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, digest in result["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("frozen pilot source changed")
    prepared, _ = prepare(result["config"])
    mapping, ledger = seeds(list(prepared), prepared)
    if ledger != result["seeds"] or summarize(result["records"]) != result["summary"]:
        raise ValueError("complete pilot seed/statistic/decision recomputation failed")
    budget = PilotBudget(500000, 180, 64)
    for saved in result["records"]:
        if json.loads((path/(saved["cell"]+"-"+str(saved["block"])+".json")).read_text()) != saved:
            raise ValueError("saved complete block mismatch")
        if saved["block"] == 0:
            cache, spec = prepared[saved["cell"]]
            rebuilt = block(cache, spec, saved["cell"], 0, mapping[saved["cell"]+"/0"], budget)
            if rebuilt["rows"] != saved["rows"] or rebuilt["scalar_evaluations"] != saved["scalar_evaluations"]:
                raise ValueError("prespecified full paired replay failed")
    if result["physical_evaluations"] != 512 or budget.physical != 64 or result["physical_evaluations"]+budget.physical > 600:
        raise ValueError("pilot physical accounting mismatch")
    return {"status": "all_statistics_seeds_and_two_blocks_replayed", "result_sha256": sha(path/"result.json"),
        "production_physical": 512, "audit_physical": 64, "audit_scalar": budget.count,
        "combined_scalar": result["scalar_evaluations"]+budget.count, "decision": result["summary"]["decision"],
        "P2_qualified": False, "not_independent_confirmation_or_speedup": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        value = audit()
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", value)
        print(json.dumps(value))
    else:
        value = run()
        print(json.dumps({"decision": value["summary"]["decision"], "physical": value["physical_evaluations"], "scalar": value["scalar_evaluations"]}))
