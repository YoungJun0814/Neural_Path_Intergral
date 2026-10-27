"""Bounded N32 interventions and actual sign-projected event estimators."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import psutil
import torch

from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import configuration, prepare
from src.path_integral.conditional_intervention import (
    conditional_replace,
    log_half_squared_difference,
    orbit_endpoints,
    require_blind_residuals,
    sign_orbit_log_probabilities,
)
from src.path_integral.gaussian_span_marginalization import sample_gaussian_mixture
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.risk_aware_model_correction import full_mixture_log_q_over_p
from src.path_integral.risk_path_attribution import estimate, merge, moments
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.stratified_paired_risk import paired
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/conditional_intervention_v1/session_20261026"
PLAN = "docs/plans/TWO_TRACK_CONDITIONAL_INTERVENTION_2026-10-26_KO.md"
NAMESPACE = "conditional-intervention-sign-orbit-v1"
ENDPOINTS = ["raw", "sign57", "sign61", "joint_sign"]
GROUPS = {"r57": [57], "r61": [61], "r57_r61": [57, 61],
          "used_residuals": list(range(1, 62, 2)),
          "brownian": list(range(0, 64, 2)), "unused63": [63]}
ANCHOR = "results/post_audit/genealogy_error_calibration_v1/session_20261023/result.json"
ANCHOR_SHA = "30f93470c1118881240b5845336107b1e97ff9af62ac5226d88688e80d3ed638"


class Work:
    def __init__(self, full: int = 550000, scalar: int = 1400000, wall: float = 1800):
        self.full_cap, self.scalar_cap, self.wall = full, scalar, wall
        self.full, self.scalar, self.started = 0, 0, time.perf_counter()

    def charge(self, full: int, scalar: int = 0) -> None:
        if any(type(n) is not int or n < 0 for n in (full, scalar)):
            raise ValueError("nonnegative integer work required")
        if (self.full+full > self.full_cap or self.scalar+scalar > self.scalar_cap
                or time.perf_counter()-self.started > self.wall):
            raise TimeoutError("fixed intervention budget exceeded")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("intervention RSS4GiB cap exceeded")
        self.full += full
        self.scalar += scalar

    def result(self) -> dict[str, Any]:
        return {"full_equivalent_contexts": self.full, "orbit_scalar_CDFs": self.scalar,
                "wall_seconds": time.perf_counter()-self.started}


def stream_key(cell: str, phase: str, block: int, role: str) -> str:
    return f"{cell}/{phase}/{block}/{role}"


def streams(cells: list[str]) -> tuple[dict[str, int], dict[str, Any]]:
    ledger, mapping = SeedLedger(), {}
    for cell in cells:
        phases = [("preflight", 0), ("timing", 0)]
        phases += [("assay", b) for b in range(8)]
        phases += [(design, b) for design in ("q", "p") for b in range(16)]
        for phase, block in phases:
            roles = ["path", "label"]
            if phase == "assay":
                roles += [group+"-"+role for group in GROUPS for role in ("path", "label")]
            for role in roles:
                mapping[stream_key(cell, phase, block, role)] = ledger.allocate(
                    SeedKey(NAMESPACE, phase, "development", cell, 32, block, role))
    old: set[int] = set()

    def scan(obj: Any, seed_context: bool = False) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                scan(value, seed_context or "seed" in key.lower())
        elif isinstance(obj, list):
            for value in obj:
                scan(value, seed_context)
        elif seed_context and type(obj) is int and 0 < obj < 2**63:
            old.add(obj)

    # Prespecified recent protocol scope; not every PRNG state in repository history.
    history = sorted((ROOT/"results/post_audit").glob("*/session_2026102[123]/protocol.json"))
    for path in history:
        scan(json.loads(path.read_text()))
    if set(mapping.values()) & old:
        raise ValueError("new/scoped historical seed collision")
    return mapping, {"ledger": ledger.to_dict(), "ledger_sha256": ledger.sha256,
        "recent_protocol_paths": [p.relative_to(ROOT).as_posix() for p in history],
        "scoped_historical_integer_seed_values": len(old), "collision_count": 0,
        "not_exhaustive_history_or_PRNG_independence_proof": True}


def draw(spec: Any, count: int, cell: str, phase: str, block: int,
         mapping: dict[str, int]) -> torch.Tensor:
    path = torch.Generator().manual_seed(mapping[stream_key(cell, phase, block, "path")])
    if phase == "p":
        return torch.randn((count, 64), dtype=torch.float64, generator=path)
    return sample_gaussian_mixture(spec, count, gaussian_generator=path,
        label_generator=torch.Generator().manual_seed(mapping[stream_key(cell, phase, block, "label")])).target_coordinates


def summary(parts: list[dict[str, Any]]) -> dict[str, Any]:
    raw = estimate([merge(parts)], torch.ones(1, dtype=torch.float64))
    mean = torch.tensor(raw["mean"], dtype=torch.float64)
    covariance = torch.tensor(raw["covariance"], dtype=torch.float64)
    rse = [(math.sqrt(max(0., float(covariance[i, i])))/float(v)) if v > 0 else None
           for i, v in enumerate(mean)]
    logs = [float(s+math.log(v)) if v > 0 else None for s, v in zip(raw["scale"], mean, strict=True)]
    return {**raw, "log_mean": logs, "relative_se": rse,
        "maximum_fraction": [float(m/v) if v > 0 else 0.
                             for m, v in zip(raw["maximum_term"], mean, strict=True)],
        "log_scale": raw["scale"], "scaled_mean": raw["mean"],
        "scaled_covariance_of_mean": raw["covariance"]}


def blocks_summary(records: list[dict[str, Any]], field: str) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    raw = summary([row[field] for row in records])
    logs = torch.tensor([summary([row[field]])["log_mean"] for row in records], dtype=torch.float64)
    block = summary([moments(logs)])
    scaled = (logs-logs.max(0).values).exp()
    total = scaled.sum(0)
    loo = ((total[None, :]-scaled)/(len(records)-1)/(total[None, :]/len(records))-1).abs().max(0).values.tolist()
    return raw, block, loo


def assay_values(cache: Any, spec: Any, cell: str, block: int,
                 mapping: dict[str, int], work: Work) -> dict[str, np.ndarray]:
    z = draw(spec, 1024, cell, "assay", block, mapping)
    work.charge(len(z))
    base = cache.log_probability(z)-full_mixture_log_q_over_p(z, spec)
    values = [base, 2*base]
    replaced_scores = []
    for name, group in GROUPS.items():
        replaced = conditional_replace(spec, z, group,
            label_generator=torch.Generator().manual_seed(mapping[stream_key(cell, "assay", block, name+"-label")]),
            gaussian_generator=torch.Generator().manual_seed(mapping[stream_key(cell, "assay", block, name+"-path")]))
        work.charge(len(z))
        score = cache.log_probability(replaced)-full_mixture_log_q_over_p(replaced, spec)
        values.append(log_half_squared_difference(base, score))
        replaced_scores.append(score)
    return {"logs": torch.stack(values, 1).numpy(),
            "replacement_log_events": torch.stack(replaced_scores, 1).numpy()}


def final_values(cache: Any, spec: Any, cell: str, design: str, block: int,
                 mapping: dict[str, int], work: Work) -> dict[str, np.ndarray]:
    count = 8192 if design == "q" else 2048
    z = draw(spec, count, cell, design, block, mapping)
    work.charge(count, 4*count)
    orbit = sign_orbit_log_probabilities(cache, spec, z)
    d = full_mixture_log_q_over_p(z, spec)
    endpoints = orbit_endpoints(orbit, d, design=design)
    return {name: value.numpy() for name, value in endpoints.items()}


def store_array(path: Path, values: dict[str, np.ndarray]) -> str:
    # x mode preserves failed/completed files; a partial file is never a completed unit.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return sha(path)


def make_record(cell: str, phase: str, block: int, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {"cell": cell, "phase": phase, "block": block,
            **{name: moments(torch.from_numpy(value)) for name, value in arrays.items()}}


def preflight(prepared: dict[str, Any], mapping: dict[str, int], work: Work) -> list[dict[str, Any]]:
    rows = []
    for cell, (cache, spec) in prepared.items():
        require_blind_residuals(spec)
        z = draw(spec, 16, cell, "preflight", 0, mapping)
        work.charge(16, 64)
        fast = sign_orbit_log_probabilities(cache, spec, z)
        differences = []
        density_difference = []
        d = full_mixture_log_q_over_p(z, spec)
        for k, signs in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
            changed = z.clone()
            changed[:, [57, 61]] *= torch.tensor(signs, dtype=torch.float64)
            work.charge(16)
            differences.append(float((cache.log_probability(changed)-fast[:, k]).abs().max()))
            density_difference.append(float((full_mixture_log_q_over_p(changed, spec)-d).abs().max()))
        if max(differences) > 1e-8 or max(density_difference) != 0:
            raise FloatingPointError("full-law/sign-density preflight failed")
        rows.append({"cell": cell, "full_log_probability_maximum_error": max(differences),
                     "full_mixture_density_sign_error": max(density_difference)})
    return rows


def timing(prepared: dict[str, Any], mapping: dict[str, int], work: Work) -> list[dict[str, Any]]:
    rows = []
    for cell, (cache, spec) in prepared.items():
        clock = time.perf_counter()
        z = draw(spec, 4096, cell, "timing", 0, mapping)
        sample_wall = time.perf_counter()-clock
        timings = []
        for method in ("raw", "joint_sign"):
            work.charge(256, 1024 if method == "joint_sign" else 0)
            if method == "raw":
                cache.log_probability(z[:256])-full_mixture_log_q_over_p(z[:256], spec)
            else:
                orbit_endpoints(sign_orbit_log_probabilities(cache, spec, z[:256]),
                    full_mixture_log_q_over_p(z[:256], spec), design="q")["event"][:, 3]
        for order in (("raw", "joint_sign"), ("joint_sign", "raw"), ("raw", "joint_sign")):
            entry: dict[str, Any] = {"order": list(order)}
            for method in order:
                work.charge(4096, 16384 if method == "joint_sign" else 0)
                clock = time.perf_counter()
                if method == "raw":
                    value = cache.log_probability(z)-full_mixture_log_q_over_p(z, spec)
                else:
                    value = orbit_endpoints(sign_orbit_log_probabilities(cache, spec, z),
                        full_mixture_log_q_over_p(z, spec), design="q")["event"][:, 3]
                entry[method+"_wall"] = time.perf_counter()-clock
                entry[method+"_log_mean"] = float(torch.logsumexp(value, 0)-math.log(4096))
            timings.append(entry)
        rows.append({"cell": cell, "sample_wall": sample_wall, "repeats": timings,
            "not_accuracy_matched_speedup_or_exclusive_host_benchmark": True})
    return rows


def comparison(q: dict[str, Any], p: dict[str, Any], index: int) -> dict[str, Any]:
    lq, lp = q["log_mean"][index], p["log_mean"][index]
    scale = max(lq, lp)
    x, y = math.exp(lq-scale), math.exp(lp-scale)
    difference = 2*(x-y)/(x+y)
    se = 4*x*y/(x+y)**2*math.hypot(q["relative_se"][index], p["relative_se"][index])
    z = -NormalDist().inv_cdf(.05/(2*8))
    return {"symmetric_relative_difference": difference, "se": se,
        "lower": difference-z*se, "upper": difference+z*se,
        "development_equivalence": abs(difference)+z*se <= .1,
        "not_finite_sample_coverage_or_truth_certificate": True}


def summarize(records: list[dict[str, Any]], cells: list[str]) -> dict[str, Any]:
    identities = [(r["cell"], r["phase"], r["block"]) for r in records]
    expected = [(c, "assay", b) for c in cells for b in range(8)]
    expected += [(c, d, b) for c in cells for d in ("q", "p") for b in range(16)]
    if len(identities) != len(set(identities)) or set(identities) != set(expected):
        raise ValueError("complete fixed assay/final units required")
    result = []
    for cell in cells:
        subset = sorted((r for r in records if r["cell"] == cell and r["phase"] == "assay"), key=lambda r: r["block"])
        assay = summary([r["logs"] for r in subset])
        # Descriptive removable variance / M2, not an additive causal attribution.
        fractions = [math.exp(value-assay["log_mean"][1]) if value is not None else 0.
                     for value in assay["log_mean"][2:]]
        assay_block_fractions = []
        for row in subset:
            per_block = summary([row["logs"]])
            assay_block_fractions.append([math.exp(v-per_block["log_mean"][1]) if v is not None else 0.
                                          for v in per_block["log_mean"][2:]])
        designs: dict[str, Any] = {}
        for design in ("q", "p"):
            subset = sorted((r for r in records if r["cell"] == cell and r["phase"] == design), key=lambda r: r["block"])
            raw, block, loo = blocks_summary(subset, "risk")
            risk_qualified = (all(v is not None and v <= .2 for v in raw["relative_se"]+block["relative_se"])
                and max(block["maximum_fraction"]) <= .25 and max(loo) <= .15)
            contrasts = [paired(s, 0, 3, comparisons=4) for s in (raw, block)]
            event_raw, event_block, _ = blocks_summary(subset, "event")
            event_qualified = (event_raw["relative_se"][3] <= .05
                and event_block["relative_se"][3] <= 2*event_raw["relative_se"][3]
                and event_raw["maximum_fraction"][3] <= .1)
            designs[design] = {"risk": raw, "risk_blocks": block, "risk_loo": loo,
                "risk_development_qualified": risk_qualified, "joint_risk_contrasts": contrasts,
                "risk_reduction_lower_over_20pct": all(c["relative_lower"] > .2 for c in contrasts),
                "event": event_raw, "event_blocks": event_block,
                "q_event_development_qualified": event_qualified if design == "q" else None}
        contrasts = [comparison(designs["q"]["risk"], designs["p"]["risk"], i) for i in range(4)]
        qualified = (designs["q"]["q_event_development_qualified"]
            and all(v["risk_development_qualified"] and v["risk_reduction_lower_over_20pct"] for v in designs.values())
            and all(v["development_equivalence"] for v in contrasts))
        result.append({"cell": cell, "assay": assay, "groups": list(GROUPS),
            "removable_variance_over_M2_descriptive": fractions,
            "per_block_removable_variance_over_M2": assay_block_fractions,
            "assay_group_count_not_normalized_or_additive": True, "designs": designs,
            "q_p_risk_comparison": contrasts, "separate_confirmation_candidate": bool(qualified)})
    return {"cells": result, "decision": "CANDIDATE_FOR_SEPARATE_CONFIRMATION" if all(
        r["separate_confirmation_candidate"] for r in result) else "NO_ADOPTION_COVERAGE_OR_PRECISION_UNRESOLVED",
        "event_estimator_candidate_implemented": True, "event_proposal_q_changed": False,
        "new_G_fits": 0, "production_adoption": False, "P2_qualified": False,
        "large_training_authorized": False, "diagnostic_intervals_not_tail_certificates": True}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    if sha(ROOT/ANCHOR) != ANCHOR_SHA:
        raise ValueError("original calibration anchor changed")
    old = json.loads((ROOT/ANCHOR).read_text())
    audit_snapshot(ROOT, old["source"], old["config"])
    config: dict[str, Any] = {"namespace": NAMESPACE, "model": configuration()["model"], "cells": configuration()["cells"],
        "groups": GROUPS, "endpoints": ENDPOINTS, "assay_blocks": 8, "assay_count": 1024,
        "final_blocks": 16, "q_block_count": 8192, "p_block_count": 2048,
        "event_rse": .05, "risk_rse": .2, "risk_block_maximum_fraction": .25, "loo": .15,
        "minimum_risk_reduction_lower": .2, "agreement_margin": .1, "risk_comparisons": 4,
        "agreement_comparisons": 8, "full_cap": 550000, "scalar_cap": 1600000,
        "phase_wall": 1800, "warmup_contexts_per_method_cell": 256,
        "anchor_sha256": ANCHOR_SHA}
    # Scalar cap includes timing, as explicitly corrected before any sampling.
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("fresh protocol only; do not overwrite or extend")
    path.mkdir(parents=True, exist_ok=True)
    prepared, setup = prepare(config)
    mapping, ledger = streams(list(prepared))
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write(path/"protocol.json", {"config": config, "source": source, "seeds": ledger, "setup": setup})
    work = Work(config["full_cap"], config["scalar_cap"], config["phase_wall"])
    structural = Work(160, 128, 120)
    records = []
    structural_work = None
    try:
        structural_result = preflight(prepared, mapping, structural)
        structural_work = structural.result()
        write(path/"preflight.json", {"rows": structural_result, "work": structural_work})
        for cell, (cache, spec) in prepared.items():
            jobs = [("assay", b) for b in range(8)]
            jobs += [(d, b) for d in ("q", "p") for b in range(16)]
            for phase, block in jobs:
                started = time.perf_counter()
                arrays = (assay_values(cache, spec, cell, block, mapping, work) if phase == "assay"
                          else final_values(cache, spec, cell, phase, block, mapping, work))
                record = make_record(cell, phase, block, arrays)
                name = f"{cell}-{phase}-{block}"
                record["array_path"] = name+".npz"
                record["array_sha256"] = store_array(path/(name+".npz"), arrays)
                record["wall_seconds"] = time.perf_counter()-started
                write(path/(name+".json"), record)
                records.append(record)
                print(f"complete {name} full={work.full}", flush=True)
        timing_result = timing(prepared, mapping, work)
        scientific_summary = summarize(records, list(prepared))
        value: dict[str, Any] = {"source": source, "config": config, "setup": setup, "records": records,
            "preflight": structural_result, "timing": timing_result, "work": work.result(),
            "preflight_work": structural_work, "summary": scientific_summary}
        write(path/"result.json", value)
    except Exception as error:
        write(path/"failure.json", {"error": type(error).__name__+": "+str(error),
            "complete_units": len(records), "work": work.result(),
            "preflight_work": structural_work or structural.result()})
        raise
    print(value["summary"]["decision"], flush=True)
    return value


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    path = ROOT/DIRECTORY
    value = json.loads((path/"result.json").read_text())
    snapshot = audit_snapshot(ROOT, value["source"], value["config"])
    prepared, _ = prepare(value["config"])
    mapping, _ = streams(list(prepared))
    work = Work(60000, 120000, 600)
    records = []
    maximum = 0.
    for stored in value["records"]:
        name = f"{stored['cell']}-{stored['phase']}-{stored['block']}"
        if json.loads((path/(name+".json")).read_text()) != stored:
            raise ValueError("unit/result mismatch")
        if sha(path/stored["array_path"]) != stored["array_sha256"]:
            raise ValueError("array hash mismatch")
        with np.load(path/stored["array_path"], allow_pickle=False) as arrays:
            arrays_dict = dict(arrays)
        derived = make_record(stored["cell"], stored["phase"], stored["block"], arrays_dict)
        for key in derived:
            if derived[key] != stored[key]:
                raise ValueError("full-array moment recomputation mismatch")
        if stored["block"] == 0:
            cache, spec = prepared[stored["cell"]]
            replay = (assay_values(cache, spec, stored["cell"], 0, mapping, work) if stored["phase"] == "assay"
                      else final_values(cache, spec, stored["cell"], stored["phase"], 0, mapping, work))
            for key in replay:
                original, fresh = arrays_dict[key], replay[key]
                if not np.array_equal(np.isneginf(original), np.isneginf(fresh)):
                    raise ValueError("exact-zero pattern replay mismatch")
                finite = np.isfinite(original) & np.isfinite(fresh)
                error = float(np.max(np.abs(original[finite]-fresh[finite]))) if finite.any() else 0.
                maximum = max(maximum, error)
                if error > 1e-10:
                    raise ValueError("numerical representative replay mismatch")
        records.append(stored)
    recalculated = summarize(records, list(prepared))
    if recalculated != value["summary"]:
        raise ValueError("complete decision replay mismatch")
    changed = [name for name, digest in value["source"]["snapshot_file_hashes"].items()
               if sha(ROOT/name) != digest]
    if changed:
        raise ValueError("frozen source changed: "+str(changed))
    receipt = {"status": "FULL_ARRAY_STATISTICS_AND_REPRESENTATIVE_PHYSICS_REPLAY_PASSED",
        "result_sha256": sha(path/"result.json"), "source": snapshot,
        "all_units": len(records), "all_arrays_checked": len(records),
        "representative_full_replays": 6, "maximum_log_error": maximum,
        "frozen_current_files_unchanged": len(value["source"]["snapshot_file_hashes"]),
        "work": work.result(), "not_independent_confirmation": True,
        "P2_qualified": False, "production_adoption": False}
    write(path/"audit.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "audit"))
    args = parser.parse_args()
    output = run() if args.mode == "run" else audit()
    print(json.dumps({"mode": args.mode, "status": output.get("status", output.get("summary", {}).get("decision"))}))
