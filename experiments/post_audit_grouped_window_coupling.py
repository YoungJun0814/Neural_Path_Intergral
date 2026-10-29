"""One frozen, bounded exact-density and equal-trace window development study."""

from __future__ import annotations

import argparse
import json
import math
import time
from statistics import NormalDist, median
from typing import Any

import numpy as np
import psutil
import torch

from experiments.post_audit_conditional_intervention import store_array
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_residual_brownian_covariance import (
    family_summary,
    key,
    risk_qualified,
    unit_id,
)
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import configuration, prepare
from src.path_integral.grouped_covariance_density import GroupedCovarianceDensity
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.residual_brownian_covariance import ARMS as OLD_ARMS
from src.path_integral.residual_brownian_covariance import ResidualBrownianCovarianceFamily
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.stratified_paired_risk import paired, sufficient
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.window_coupled_covariance import WindowCoupledCovariance

DIRECTORY = "results/post_audit/grouped_window_coupling_v1/session_20261027"
PLAN = "docs/plans/GROUPED_DENSITY_AND_WINDOW_COUPLING_2026-10-27_KO.md"
NAMESPACE = "grouped-window-coupling-v1"
ANCHOR = "results/post_audit/residual_brownian_covariance_v1/session_20261026/result.json"
ANCHOR_SHA = "5dc431315907c8c645cd2eff6ea59ee995caec5c912aa57f3820b5881e666994"
ANCHOR_AUDIT = "results/post_audit/residual_brownian_covariance_v1/session_20261026/audit.json"
ANCHOR_AUDIT_SHA = "c101fe08776e66869ddbfcebfc0f8ba1140fc5a5eab324f8d017b176792ed3eb"
ARMS = ("base", "local", "window")
CHUNK = 256


class Budget:
    def __init__(self, physical: int, wall: float, old_passes: int = 50000):
        self.physical_cap, self.wall_cap, self.old_cap = physical, wall, old_passes
        self.physical = self.old_passes = 0
        self.started = time.perf_counter()

    def charge(self, physical: int = 0, old: int = 0) -> None:
        if any(type(v) is not int or v < 0 for v in (physical, old)):
            raise ValueError("nonnegative integer work required")
        if (self.physical+physical > self.physical_cap or self.old_passes+old > self.old_cap
                or time.perf_counter()-self.started > self.wall_cap):
            raise TimeoutError("prespecified work/wall budget exceeded")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("prespecified RSS4GiB cap exceeded")
        self.physical += physical
        self.old_passes += old

    def to_dict(self) -> dict[str, Any]:
        return {"physical_evaluations": self.physical, "old_density_coordinate_passes": self.old_passes,
                "wall_seconds": time.perf_counter()-self.started}


def jobs(cells: list[str]) -> list[dict[str, Any]]:
    return ([{"cell": c, "phase": "direct", "arm": a, "block": b}
             for c in cells for a in ARMS for b in range(8)]
            +[{"cell": c, "phase": p, "arm": "all", "block": b}
              for c in cells for p in ("heldout", "reference") for b in range(8)])


def streams(cells: list[str]) -> tuple[dict[str, int], dict[str, Any]]:
    ledger, mapping = SeedLedger(), {}
    requests = [(c, p, 0, a) for c in cells for p, arms in
                (("kernel", ("base", "coupled", "R")), ("preflight", ARMS)) for a in arms]
    for job in jobs(cells):
        requests += [(job["cell"], job["phase"], job["block"], s)
                     for s in (ARMS if job["phase"] == "heldout" else (job["arm"],))]
    for cell, phase, block, stratum in requests:
        for role in ("path", "label"):
            name = key(cell, phase, block, stratum, role)
            if name in mapping:
                raise ValueError("duplicate seed role")
            mapping[name] = ledger.allocate(SeedKey(NAMESPACE, phase, "development", cell,
                                                     32, block, stratum+"-"+role))
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

    history = [p for p in sorted((ROOT/"results/post_audit").glob("*/session_2026102*/protocol.json"))
               if NAMESPACE not in p.read_text(encoding="utf-8")]
    for path in history:
        scan(json.loads(path.read_text(encoding="utf-8")))
    if old & set(mapping.values()):
        raise ValueError("scoped historical seed collision")
    return mapping, {"ledger": ledger.to_dict(), "ledger_sha256": ledger.sha256,
        "historical_paths": [p.relative_to(ROOT).as_posix() for p in history],
        "scoped_historical_seed_integer_values": len(old), "collision_count": 0,
        "not_exhaustive_history_or_PRNG_independence_proof": True}


def seed_kwargs(mapping: dict[str, int], cell: str, phase: str, block: int, arm: str) -> dict[str, int]:
    return {r+"_seed": mapping[key(cell, phase, block, arm, r)] for r in ("path", "label")}


def prepared_models(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ready, setup = prepare(config)
    models = {}
    for cell, (cache, spec) in ready.items():
        family = ResidualBrownianCovarianceFamily(spec, price_sign=-1 if cache.problem.rho > 0 else 1)
        models[cell] = (cache, family, GroupedCovarianceDensity.create(family),
                        WindowCoupledCovariance.create(family, config["window_width"]))
    return models, setup


def kernel_decision(rows: list[dict[str, Any]], max_error: float) -> dict[str, Any]:
    if len(rows) != 3 or [r["order"] for r in rows] != ["AB", "BA", "AB"]:
        raise ValueError("complete fixed AB/BA/AB timing required")
    ratios = [r["old_all_seconds"]/r["grouped_all_seconds"] for r in rows]
    passed = max_error <= 1e-9 and median(ratios) >= 1.2
    return {"maximum_log_error": max_error, "paired_speed_ratios": ratios,
        "median_paired_speed_ratio": median(ratios), "kernel_candidate": bool(passed),
        "fresh_window_pilot_allowed": bool(passed), "not_accuracy_matched_total_wall_or_population_gain": True}


def kernel_check(models: dict[str, Any], mapping: dict[str, int], path: Any,
                 budget: Budget) -> dict[str, Any]:
    fixtures, max_error = {}, 0.
    for cell, (_, family, grouped, _) in models.items():
        points = []
        for arm in ("base", "coupled", "R"):
            kwargs = seed_kwargs(mapping, cell, "kernel", 0, arm)
            points.append(family.reference_sample(512, **kwargs) if arm == "R"
                          else family.sample(512, arm, **kwargs))
        z = torch.cat(points)
        fixtures[cell] = z
        budget.charge(old=len(z))
        expected = family.log_densities(z)
        actual = grouped.log_densities(z)
        max_error = max(max_error, float((expected-actual).abs().max()))
        for i, arm in enumerate(OLD_ARMS):
            max_error = max(max_error, float((grouped.log_density(z, arm)-expected[:, i]).abs().max()))
        budget.charge(old=128)
        family.log_densities(z[:128])
        grouped.log_densities(z[:128])
    rows = []
    for order in ("AB", "BA", "AB"):
        row: dict[str, Any] = {"order": order}
        for name in order:
            elapsed = 0.
            for cell, (_, family, grouped, _) in models.items():
                z = fixtures[cell]
                budget.charge(old=len(z) if name == "A" else 0)
                start = time.perf_counter()
                (family.log_densities if name == "A" else grouped.log_densities)(z)
                elapsed += time.perf_counter()-start
            row["old_all_seconds" if name == "A" else "grouped_all_seconds"] = elapsed
        rows.append(row)
    per_arm = {}
    for arm in OLD_ARMS:
        start = time.perf_counter()
        for cell, (_, _, grouped, _) in models.items():
            budget.charge()
            grouped.log_density(fixtures[cell], arm)
        per_arm[arm] = time.perf_counter()-start
    digest = store_array(path/"kernel_coordinates.npz", {c: z.numpy() for c, z in fixtures.items()})
    return {"decision": kernel_decision(rows, max_error), "timings": rows,
        "grouped_single_arm_seconds": per_arm, "fresh_coordinate_count": sum(map(len, fixtures.values())),
        "array_path": "kernel_coordinates.npz", "array_sha256": digest,
        "group_counts": {c: m[2].group_counts() for c, m in models.items()},
        "work": budget.to_dict(), "density_only_no_new_event_performance_samples": True}


def witness_check(anchor: dict[str, Any], models: dict[str, Any], budget: Budget) -> dict[str, Any]:
    groups: dict[tuple[str, bytes], dict[str, Any]] = {}
    slots = 0
    for row in anchor["records"]:
        for witness in row["witnesses"]:
            slots += 1
            coordinate = np.array(witness["coordinates"], dtype=np.float64)
            group = groups.setdefault((row["cell"], coordinate.tobytes()),
                                      {"cell": row["cell"], "coordinates": coordinate.tolist(), "slots": []})
            group["slots"].append({"unit": unit_id(row), **{k: v for k, v in witness.items() if k != "coordinates"}})
    if len(groups) > 500:
        raise ValueError("fixed deduplicated witness cap exceeded")
    output, max_g, max_d, max_grouped, max_contribution = [], 0., 0., 0., 0.
    for cell, (cache, family, grouped, window) in models.items():
        selected = [r for r in groups.values() if r["cell"] == cell]
        z = torch.tensor([r["coordinates"] for r in selected], dtype=torch.float64)
        budget.charge(physical=len(z), old=len(z))
        terminal, integrated, variance = cache.quantities(z)
        g = torch.special.log_ndtr((math.log(cache.problem.task.level)-terminal)
            /torch.sqrt((1-cache.problem.rho**2)*integrated))
        d, dg, dr = family.log_densities(z), grouped.log_densities(z), family.reference_log_density(z)
        max_grouped = max(max_grouped, float((d-dg).abs().max()))
        share = variance[:, :-1]/variance[:, :-1].sum(1, keepdim=True)
        local = z.reshape(-1, cache.problem.steps, 2) @ cache.kernel.local_cholesky.T
        price = cache.problem.rho*torch.sqrt(variance[:, :-1])*local[:, :, 0]
        local_projection = (z[:, 1:-1:2]+family.price_sign*z[:, 2::2])/math.sqrt(2)
        window_projection = z @ window.directions.T
        for i, group in enumerate(selected):
            for slot in group["slots"]:
                max_g = max(max_g, abs(float(g[i])-slot["log_g"]))
                max_d = max(max_d, float((d[i]-torch.tensor(slot["log_density_qs"], dtype=torch.float64)).abs().max()),
                            abs(float(dr[i])-slot["log_density_R"]))
                phase = slot["unit"].rsplit("-", 2)[0].rsplit("-", 1)[-1]
                if phase == "direct":
                    arm = slot["stratum"]
                    contribution = g[i]-d[i, OLD_ARMS.index(arm)]
                else:
                    denominator = torch.logsumexp(d[i], 0)-math.log(4) if phase == "heldout" else dr[i]
                    contribution = 2*g[i]-denominator-d[i, slot["column"]]
                max_contribution = max(max_contribution, abs(float(contribution)-slot["log_contribution"]))
            hhi = float(share[i].square().sum())
            output.append({**group, "log_g_replayed": float(g[i]), "log_density_qs_replayed": d[i].tolist(),
                "variance_peak_index": int(share[i].argmax()), "variance_share": share[i].tolist(),
                "variance_hhi": hhi, "effective_duration_steps": 1/hhi,
                "top1_variance_share": float(share[i].max()), "top4_variance_share": float(share[i].topk(4).values.sum()),
                "rho_sqrt_variance_dw": price[i].tolist(), "local_projection": local_projection[i].tolist(),
                "window_projection": window_projection[i].tolist(), "posthoc_selected_extreme_not_population": True})
    if max(max_g, max_d, max_grouped, max_contribution) > 1e-9:
        raise ValueError("saved witness physics/density replay mismatch")
    return {"selected_slots": slots, "unique_cell_coordinate_pairs": len(groups),
        "maximum_log_g_error": max_g, "maximum_saved_density_error": max_d,
        "maximum_grouped_density_error": max_grouped, "rows": output,
        "maximum_saved_contribution_error": max_contribution,
        "work": budget.to_dict(), "not_mode_frequency_causal_effect_or_confirmation": True}


def sample(model: Any, count: int, job: dict[str, Any], arm: str,
           mapping: dict[str, int]) -> torch.Tensor:
    _, family, _, window = model
    kwargs = seed_kwargs(mapping, job["cell"], job["phase"], job["block"], arm)
    if job["phase"] == "reference":
        return window.reference_sample(count, **kwargs)
    if arm == "window":
        return window.sample(count, **kwargs)
    return family.sample(count, "coupled" if arm == "local" else arm, **kwargs)


def scores(model: Any, z: torch.Tensor, budget: Budget) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    cache, _, grouped, window = model
    gs, ds, rs = [], [], []
    density_wall = physics_wall = 0.
    for begin in range(0, len(z), CHUNK):
        points = z[begin:begin+CHUNK]
        budget.charge(physical=len(points))
        start = time.perf_counter()
        old = grouped.log_densities(points, ("base", "coupled"))
        d = torch.cat((old, window.log_density(points)[:, None]), 1)
        r = window.reference_log_density(points)
        density_wall += time.perf_counter()-start
        start = time.perf_counter()
        g = cache.log_probability(points)
        physics_wall += time.perf_counter()-start
        if not torch.isfinite(g).all() or (g > 1e-12).any():
            raise FloatingPointError("finite bounded conditional probability required")
        gs.append(g)
        ds.append(d)
        rs.append(r)
    return torch.cat(gs), torch.cat(ds), torch.cat(rs), {"density_all_arms_and_R": density_wall, "physics": physics_wall}


def values(job: dict[str, Any], models: dict[str, Any], mapping: dict[str, int],
           budget: Budget) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    start = time.perf_counter()
    strata = ARMS if job["phase"] == "heldout" else (job["arm"],)
    count = {"direct": 4096, "heldout": 1024, "reference": 2048}[job["phase"]]
    moments, events, arrays, event_arrays = [], [], [], []
    sample_wall = density_wall = physics_wall = 0.
    for stratum in strata:
        clock = time.perf_counter()
        z = sample(models[job["cell"]], count, job, stratum, mapping)
        sample_wall += time.perf_counter()-clock
        g, d, r, costs = scores(models[job["cell"]], z, budget)
        density_wall += costs["density_all_arms_and_R"]
        physics_wall += costs["physics"]
        if job["phase"] == "direct":
            denominator = d[:, ARMS.index(job["arm"])]
        elif job["phase"] == "heldout":
            denominator = torch.logsumexp(d, 1)-math.log(len(ARMS))
        else:
            denominator = r
        event = (g-denominator)[:, None]
        logs = event if job["phase"] == "direct" else 2*g[:, None]-denominator[:, None]-d
        moments.append(sufficient(logs))
        events.append(sufficient(event))
        arrays.append(logs.numpy())
        event_arrays.append(event.numpy())
    result = {**job, "strata": moments, "event_strata": events, "counts_per_stratum": count,
        "sampling_wall": sample_wall, "density_all_arms_wall": density_wall,
        "physics_wall": physics_wall, "unit_wall": time.perf_counter()-start}
    if result["unit_wall"] > 60:
        raise TimeoutError("prespecified complete-unit60sec cap exceeded")
    return result, {"logs": np.stack(arrays), "event": np.stack(event_arrays)}


def agreement(h: dict[str, Any], r: dict[str, Any], index: int, qualified: bool) -> dict[str, Any]:
    if not qualified:
        return {"evaluated": False, "development_equivalence": False, "reason": "reference_precision_unqualified"}
    lh, lr = h["log_mean"][index], r["log_mean"][index]
    a, b = math.exp(lh-max(lh, lr)), math.exp(lr-max(lh, lr))
    difference = 2*(a-b)/(a+b)
    se = 4*a*b/(a+b)**2*math.hypot(h["relative_se"][index], r["relative_se"][index])
    z = -NormalDist().inv_cdf(.05/(2*6))
    return {"evaluated": True, "symmetric_relative_difference": difference, "se": se,
        "lower": difference-z*se, "upper": difference+z*se,
        "development_equivalence": abs(difference)+z*se <= .1,
        "not_finite_sample_truth_certificate": True}


def summarize(records: list[dict[str, Any]], cells: list[str]) -> dict[str, Any]:
    actual, expected = [unit_id(r) for r in records], [unit_id(j) for j in jobs(cells)]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("all80 fixed unique units required")
    output = []
    for cell in cells:
        direct = {arm: family_summary([r for r in records if r["cell"] == cell
                  and r["phase"] == "direct" and r["arm"] == arm]) for arm in ARMS}
        references: dict[str, Any] = {}
        for phase in ("heldout", "reference"):
            rows = [r for r in records if r["cell"] == cell and r["phase"] == phase]
            risk = family_summary(rows)
            contrasts = {a: [paired(s, index, 2, comparisons=8) for s in (risk["raw"], risk["blocks"])]
                         for index, a in enumerate(ARMS[:2])}
            decrease = all(c["absolute_lower"] > 0 and c["relative_lower"] > (.2 if a == "base" else 0.)
                           for a, pair in contrasts.items() for c in pair)
            references[phase] = {"risk": risk, "event": family_summary(rows, "event_strata"),
                "risk_precision_qualified": bool(risk_qualified(risk)), "contrasts": contrasts,
                "window_decrease_qualified": bool(decrease)}
        w = direct["window"]
        event = (w["raw"]["relative_se"][0] <= .05 and w["blocks"]["relative_se"][0] <= 2*w["raw"]["relative_se"][0]
                 and w["raw"]["maximum_fraction"][0] <= .1 and w["loo"][0] <= .15)
        qualified = all(r["risk_precision_qualified"] for r in references.values())
        comparisons = [agreement(references["heldout"]["risk"]["raw"], references["reference"]["risk"]["raw"],
                                 i, qualified) for i in range(3)]
        candidate = (event and qualified and all(c["development_equivalence"] for c in comparisons)
                     and all(r["window_decrease_qualified"] for r in references.values()))
        output.append({"cell": cell, "direct": direct, "references": references,
            "event_window_development_qualified": bool(event), "reference_agreement": comparisons,
            "separate_confirmation_candidate": bool(candidate)})
    return {"cells": output, "decision": "SEPARATE_CONFIRMATION_CANDIDATE" if all(
        c["separate_confirmation_candidate"] for c in output) else "NO_ADOPTION_WINDOW_PRECISION_OR_COVERAGE_UNRESOLVED",
        "primary_arm": "window", "production_adoption": False, "P2_qualified": False,
        "large_training_authorized": False, "fitting_or_selection_runs": 0,
        "not_population_efficiency_or_continuum_certificate": True}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    if sha(ROOT/ANCHOR) != ANCHOR_SHA or sha(ROOT/ANCHOR_AUDIT) != ANCHOR_AUDIT_SHA:
        raise ValueError("frozen prior result/audit changed")
    anchor = json.loads((ROOT/ANCHOR).read_text(encoding="utf-8"))
    audit_snapshot(ROOT, anchor["source"], anchor["config"])
    config = {"namespace": NAMESPACE, "model": configuration()["model"], "cells": configuration()["cells"],
        "steps": 32, "window_width": 4, "variance": 5., "alpha": .5, "chunk": CHUNK,
        "arms": list(ARMS), "blocks": 8, "direct_count": 4096, "heldout_per_stratum": 1024,
        "reference_count": 2048, "kernel_minimum_speed_ratio": 1.2, "kernel_log_tolerance": 1e-9,
        "event_rse": .05, "risk_rse": .2, "maximum_loo": .15, "maximum_block_fraction": .25,
        "paired_comparisons": 8, "agreement_comparisons": 6, "agreement_margin": .1,
        "physical_cap": 280000, "phase_wall": 1800, "unit_wall": 60,
        "source_anchor_sha256": ANCHOR_SHA, "source_anchor_audit_sha256": ANCHOR_AUDIT_SHA}
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("fresh nonoverwriting one-shot study required")
    models, setup = prepared_models(config)
    mapping, ledger = streams(list(models))
    path.mkdir(parents=True, exist_ok=True)
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR, ANCHOR_AUDIT))
    write(path/"protocol.json", {"source": source, "config": config, "setup": setup, "seeds": ledger})
    records: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    budget = Budget(280000, 1800)
    try:
        kernel = kernel_check(models, mapping, path, Budget(0, 600))
        write(path/"kernel.json", kernel)
        witnesses = witness_check(anchor, models, Budget(500, 600))
        write(path/"witnesses.json", witnesses)
        print("kernel "+json.dumps(kernel["decision"]), flush=True)
        if not kernel["decision"]["fresh_window_pilot_allowed"]:
            summary: dict[str, Any] = {"decision": "STOP_EXACT_KERNEL_NUMERICAL_OR_SPEED_GATE",
                "P2_qualified": False, "production_adoption": False, "large_training_authorized": False,
                "fresh_window_pilot_started": False, "window_proposal_performance_not_evaluated": True}
            preflight = []
        else:
            preflight = []
            preflight_start = time.perf_counter()
            for cell, model in models.items():
                for arm in ARMS:
                    job = {"cell": cell, "phase": "preflight", "block": 0}
                    z = sample(model, 64, job, arm, mapping)
                    g, d, r, cost = scores(model, z, budget)
                    preflight.append({"cell": cell, "arm": arm, "costs": cost,
                        "minimum_log_density": float(d.min()), "finite_probability": bool(torch.isfinite(g).all()),
                        "minimum_log_reference": float(r.min())})
            forecast = (time.perf_counter()-preflight_start)/384*278528*1.5
            write(path/"preflight.json", {"rows": preflight, "production_wall_forecast_with_safety": forecast,
                                           "not_runtime_guarantee": True})
            if forecast > 1800:
                raise TimeoutError("prespecified cost forecast excludes production")
            for job in jobs(list(models)):
                record, arrays = values(job, models, mapping, budget)
                name = unit_id(job)
                record["array_path"] = name+".npz"
                record["array_sha256"] = store_array(path/record["array_path"], arrays)
                write(path/(name+".json"), record)
                records.append(record)
                print(f"complete {name} physical={budget.physical}", flush=True)
            summary = summarize(records, list(models))
        value = {"source": source, "config": config, "setup": setup, "kernel": kernel,
            "witnesses_sha256": sha(path/"witnesses.json"), "records": records,
            "preflight": preflight, "work": budget.to_dict(), "summary": summary}
        write(path/"result.json", value)
        return value
    except Exception as error:
        write(path/"failure.json", {"error": type(error).__name__+": "+str(error),
            "complete_units": len(records), "work": budget.to_dict(), "P2_qualified": False})
        raise


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    path = ROOT/DIRECTORY
    value = json.loads((path/"result.json").read_text(encoding="utf-8"))
    snapshot = audit_snapshot(ROOT, value["source"], value["config"])
    models, _ = prepared_models(value["config"])
    protocol = json.loads((path/"protocol.json").read_text(encoding="utf-8"))
    ledger = SeedLedger.from_dict(protocol["seeds"]["ledger"])
    mapping = {key(r.key.task, r.key.role, r.key.replicate,
                   r.key.stream.rsplit("-", 1)[0], r.key.stream.rsplit("-", 1)[1]): r.seed for r in ledger.records}
    kernel = value["kernel"]
    if (json.loads((path/"kernel.json").read_text()) != kernel
            or sha(path/kernel["array_path"]) != kernel["array_sha256"]
            or kernel_decision(kernel["timings"], kernel["decision"]["maximum_log_error"]) != kernel["decision"]):
        raise ValueError("kernel binding/decision mismatch")
    kernel_budget = Budget(0, 600)
    error = 0.
    with np.load(path/kernel["array_path"], allow_pickle=False) as arrays:
        for cell, (_, family, grouped, _) in models.items():
            z = torch.from_numpy(arrays[cell])
            fresh = torch.cat([family.reference_sample(512, **seed_kwargs(mapping, cell, "kernel", 0, a))
                if a == "R" else family.sample(512, a, **seed_kwargs(mapping, cell, "kernel", 0, a))
                for a in ("base", "coupled", "R")])
            if not torch.equal(z, fresh):
                raise ValueError("fresh kernel coordinate stream replay mismatch")
            kernel_budget.charge(old=len(z))
            error = max(error, float((family.log_densities(z)-grouped.log_densities(z)).abs().max()))
    if error > 1e-9:
        raise ValueError("kernel numerical replay failed")
    anchor = json.loads((ROOT/ANCHOR).read_text(encoding="utf-8"))
    witnesses = json.loads((path/"witnesses.json").read_text(encoding="utf-8"))
    if sha(path/"witnesses.json") != value["witnesses_sha256"]:
        raise ValueError("witness hash mismatch")
    repeated = witness_check(anchor, models, Budget(500, 600))
    if any(repeated[k] != v for k, v in witnesses.items() if k != "work"):
        raise ValueError("all selected witness physics/structure replay failed")
    budget, replayed = Budget(20000, 600), 0
    for row in value["records"]:
        if (json.loads((path/(unit_id(row)+".json")).read_text()) != row
                or sha(path/row["array_path"]) != row["array_sha256"]):
            raise ValueError("unit/array/result binding mismatch")
        with np.load(path/row["array_path"], allow_pickle=False) as arrays:
            for field, name in (("strata", "logs"), ("event_strata", "event")):
                if [sufficient(torch.from_numpy(a)) for a in arrays[name]] != row[field]:
                    raise ValueError("complete array moment replay mismatch")
            if row["block"] == 0 and (row["phase"] != "direct" or row["arm"] == "window"):
                _, replay_arrays = values(row, models, mapping, budget)
                if any(not np.array_equal(a, arrays[k]) for k, a in replay_arrays.items()):
                    raise ValueError("representative physics replay mismatch")
                replayed += 1
    if value["records"]:
        if summarize(value["records"], list(models)) != value["summary"]:
            raise ValueError("complete scientific decision replay mismatch")
    elif kernel["decision"]["fresh_window_pilot_allowed"] or value["summary"]["decision"] != "STOP_EXACT_KERNEL_NUMERICAL_OR_SPEED_GATE":
        raise ValueError("empty study without valid prespecified stop")
    changed = [n for n, d in value["source"]["snapshot_file_hashes"].items() if sha(ROOT/n) != d]
    if changed:
        raise ValueError("frozen source changed: "+str(changed))
    result = {"status": "ALL_BINDINGS_STATISTICS_WITNESSES_AND_REPRESENTATIVE_REPLAYS_PASSED",
        "result_sha256": sha(path/"result.json"), "source": snapshot,
        "all_unit_arrays_checked": len(value["records"]), "representative_replays": replayed,
        "unique_selected_witnesses_replayed": witnesses["unique_cell_coordinate_pairs"],
        "kernel_maximum_log_error_replayed": error, "kernel_work": kernel_budget.to_dict(),
        "witness_work": repeated["work"], "physics_replay_work": budget.to_dict(),
        "not_independent_confirmation_or_timing_reproduction": True, "P2_qualified": False}
    write(path/"audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "audit"))
    args = parser.parse_args()
    result = run() if args.mode == "run" else audit()
    print(json.dumps({"mode": args.mode, "status": result.get("status", result.get("summary", {}).get("decision"))}))
