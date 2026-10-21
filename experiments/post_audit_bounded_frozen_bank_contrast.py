"""D2 complete frozen-family contrast, with a hard conditional D3 gate."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_bounded_bank_runtime import (
    CONFIG,
    anchors,
    begin_run,
    key,
    risk_block,
    summarize_blocks,
    validate_streams,
)
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.balanced_mixture_bank import balanced_counts, stream_layout
from src.path_integral.bounded_decision_protocol import DESIGNS, d2_decision
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.stratified_paired_risk import (
    estimate,
    independent_contrast,
    validate_sufficient,
)
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/bounded_frozen_bank_contrast_v1/session_20261020"


def check_policy(config: dict[str, Any]) -> None:
    expected = {"namespace": "bounded-bank-decision-v1", "steps": 32, "threads": 1, "repetitions": 5,
        "training_objective": "event-kl", "arms": ["U", "E-IID", "E-STRAT"], "designs": list(DESIGNS),
        "natural_mass": .1, "shifted_floor_mass": .1, "adaptive_mass": .8,
        "maximum_iterations": 256, "line_search_iterations": 40, "gap_tolerance": 1e-6, "alphas": [0, .25, .5],
        "preflight_per_cell_design": 1240, "d2_blocks": 8, "d2_block_size": 29760,
        "d3_train_count_per_design": 19840, "d3_common_stratified_selection": 19840,
        "probability_pilot": 16384, "risk_pilot_per_design": 19840,
        "allocation_safety_factor": 2, "primary_risk_design": "stratified",
        "minimum_point_m2_reduction": .2, "minimum_workflow_wall_reduction": .2,
        "minimum_successes_per_cell": 4, "comparison_intervals": 240, "family_alpha": .05,
        "production_cap": 13479520, "audit_cap": 1000000, "d2_work": 952320,
        "preflight_work": 4960, "d3_maximum_work": 12522240,
        "maximum_block_to_raw_rse": 2, "maximum_loo_shift": .15, "maximum_single_fraction": .1,
        "probability_confirmation": {"minimum": 65536, "maximum": 262144, "multiple": 16384, "target_rse": .05},
        "risk_confirmation": {"minimum": 39680, "maximum": 158720, "multiple": 9920, "target_rse": .2},
        "wall_caps": {"preflight": 300, "D2": 900, "D3": 3600, "total_production": 7200, "audit": 1800},
        "rss_cap_bytes": 4294967296, "cells": ["canonical-h005-k1", "high-eta200-k1"],
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}
    if config != expected:
        raise ValueError("declared bounded policy differs from implementation")


def families(parent: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for cell, (_, spec) in prepared.items():
        fits = sorted((r for r in parent["summary"]["fits"] if r["cell"] == cell), key=lambda r: r["rep"])
        if [f["rep"] for f in fits] != list(range(5)):
            raise ValueError("all five historical fits required")
        weights = [spec.weights.tolist()]+[f["weights"][arm] for arm in ("event-kl", "risk-m2") for f in fits]
        for w in weights:
            v = torch.tensor(w, dtype=torch.float64)
            if (v.shape != (373,) or not torch.isfinite(v).all() or (v <= 0).any()
                    or abs(float(v.sum())-1) > 1e-12 or abs(float(v[0])-.1) > 1e-12):
                raise ValueError("bound historical normalized dictionary required")
        out[cell] = {"weights": weights, "ids": ["U"]+[arm+"-"+str(rep) for arm in ("event-kl", "risk-m2") for rep in range(5)]}
    return out


def jobs(cells: tuple[str, ...]) -> list[dict[str, Any]]:
    return [key("D2", cell, design, block=b) for cell in cells for design in DESIGNS for b in range(8)]


def identity(j: dict[str, Any]) -> str:
    return j["cell"]+"-"+j["design"]+"-"+str(j["block"])


def summarize(records: list[dict[str, Any]], prepared: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    expected = {identity(j) for j in jobs(tuple(prepared))}
    if len(records) != 32 or {identity(r) for r in records} != expected:
        raise ValueError("complete D2 family; no prefix or omitted proposal")
    cells = []
    for cell in prepared:
        methods = {}
        for design in DESIGNS:
            group = sorted((r for r in records if (r["cell"], r["design"]) == (cell, design)), key=lambda r: r["block"])
            methods[design] = summarize_blocks(group, prepared[cell][1], design,
                key("D2", cell, design), probability_first=True)
            if methods[design]["estimator"]["count"] != 238080:
                raise ValueError("D2 total count changed")
        contrasts = [independent_contrast(methods["iid"]["estimator"], methods["stratified"]["estimator"],
            target=k, comparisons=24) for k in range(12)]
        s = methods["stratified"]
        masses = prepared[cell][1].weights
        m = torch.tensor(s["estimator"]["scaled_mean"], dtype=torch.float64)
        means = torch.tensor(s["estimator"]["stratum_scaled_means"], dtype=torch.float64)
        scales = torch.tensor(s["estimator"]["log_scale"], dtype=torch.float64)
        between = (masses[:, None]*(means-m).square()).sum(0)
        within = torch.zeros_like(m)
        for mass, part, mean in zip(masses, s["strata"], means, strict=True):
            cross = torch.tensor(part["log_cross_sum"], dtype=torch.float64).diag()
            within += mass*((cross-2*scales).exp()/part["count"]-mean.square()).clamp_min(0)
        cells.append({"cell": cell, "target_ids": ["mu"]+family[cell]["ids"], "methods": methods, "contrasts": contrasts,
            "observed_variation_decomposition": {"between_label_scaled_variance": between.tolist(),
                "within_component_scaled_variance": within.tolist(), "log_scale": scales.tolist(),
                "component_contribution_fractions": (masses[:, None]*means/m).tolist(),
                "plug_in_description_not_population_decomposition_or_mode_attribution": True}})
    return {"cells": cells, "decision": d2_decision(cells)}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    parent = anchors()
    declared = yaml.safe_load((ROOT/CONFIG).read_text())
    check_policy(declared)
    config = {"declared": declared, "parent_model": parent["config"]["model"], "cells": parent["config"]["cells"],
        "parent_source": parent["source"]["snapshot_sha256"], "D3_policy_frozen_before_D2": True}
    directory, source = begin_run(DIRECTORY, config)
    production_started = time.time()
    preparation_started = time.perf_counter()
    prepared, setup = prepare({"model": config["parent_model"], "cells": config["cells"]})
    if time.perf_counter()-preparation_started > 120:
        raise TimeoutError("fixed preparation wall cap exceeded before sampling")
    family = families(parent, prepared)
    cells = tuple(prepared)
    requests = [(1240, design, key("preflight", cell, design)) for cell in cells for design in DESIGNS]
    requests += [(29760, j["design"], j) for j in jobs(cells)]
    unique_seeds = validate_streams(prepared, requests)
    manifest = {"source": source["snapshot_sha256"], "config": config, "family": family,
        "units": ["preflight-"+c+"-"+d for c in cells for d in DESIGNS]+[identity(j) for j in jobs(cells)],
        "limits": {"P": {"work": 4960, "wall": 300.}, "D2": {"work": 952320, "wall": 900.}}}
    journal = RecoveryJournal(directory, manifest)
    preflight = []
    for cell in cells:
        for design in DESIGNS:
            k = key("preflight", cell, design)
            preflight.append(journal.unit("preflight-"+cell+"-"+design, phase="P", work=1240, maximum_wall=60.,
                operation=partial(risk_block, prepared[cell], family[cell]["weights"], design, 1240, k)))
    if any(p["wall"]*24 > 90 for p in preflight):
        write_json_atomic_nonoverwriting(directory/"resource_stop.json", {"status": "preflight_resource_forecast_failed", "physical_evaluations": 4960})
        raise TimeoutError("D2 full-block forecast exceeds reservation")
    start = time.perf_counter()
    records = []
    for j in jobs(cells):
        record = journal.unit(identity(j), phase="D2", work=29760, maximum_wall=90.,
            operation=partial(risk_block, prepared[j["cell"]], family[j["cell"]]["weights"], j["design"], 29760, j))
        records.append(record)
        print(f"D2 complete balanced bank {len(records)}/32", flush=True)
    summary = summarize(records, prepared, family)
    output = {"config": config, "source": source, "manifest": manifest, "setup": setup, "family": family,
        "preflight": preflight, "records": records, "summary": summary,
        "physical_evaluations": journal.accounting("P")[0]+journal.accounting("D2")[0],
        "unique_seed_roles": unique_seeds, "d2_wall": time.perf_counter()-start}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-D2.clock.json")["started_unix"] > 900:
        journal._write(directory/"phase-D2.violation.json", {"manifest": journal.digest, "phase": "D2", "status": "final_io_wall_failure"})
        raise TimeoutError("D2 final I/O wall exceeded")
    # Audit D2 before authorizing any conditional learning work.
    receipt = audit(directory/"result.json")
    write_json_atomic_nonoverwriting(directory/"audit.json", receipt)
    if summary["decision"]["d2_gate"]:
        from experiments.post_audit_bounded_bank_learning import audit as audit_learning
        from experiments.post_audit_bounded_bank_learning import run as learn
        d3 = learn(directory, source, prepared, setup, summary)
        write_json_atomic_nonoverwriting(directory/"D3/result.json", d3)
        d3_journal = RecoveryJournal(directory/"D3", d3["manifest"])
        if time.time()-d3_journal._read(directory/"D3/phase-D3.clock.json")["started_unix"] > 3600:
            d3_journal._write(directory/"D3/phase-D3.violation.json", {"manifest": d3_journal.digest, "phase": "D3", "status": "final_io_wall_failure"})
            raise TimeoutError("D3 final I/O cap exceeded")
        d3_receipt = audit_learning(directory/"D3", d3, source, prepared, setup, summary)
        d3_receipt["result_sha256"] = sha(directory/"D3/result.json")
        write_json_atomic_nonoverwriting(directory/"D3/audit.json", d3_receipt)
    else:
        write_json_atomic_nonoverwriting(directory/"D3_NOT_RUN.json", {"reason": "frozen_D2_gate_failed",
            "decision": "STOP_UNRESOLVED_BUDGET", "D3_evaluations": 0,
            "unspent_D3_budget_not_transferred": 12522240})
    if time.time()-production_started > 7200:
        write_json_atomic_nonoverwriting(directory/"total_production_wall_failure.json", {"status": "INVALID_PROTOCOL_total_wall"})
        raise TimeoutError("total production cap exceeded")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    begin = time.perf_counter()
    saved = json.loads(path.read_text())
    parent = anchors()
    check_policy(saved["config"]["declared"])
    if saved["config"]["declared"] != yaml.safe_load((ROOT/CONFIG).read_text()):
        raise ValueError("frozen policy file changed")
    audit_snapshot(ROOT, saved["source"], saved["config"])
    for name, digest in saved["source"]["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/")) and sha(ROOT/name) != digest:
            raise ValueError("frozen numerical source changed")
    prepared, costs = prepare({"model": saved["config"]["parent_model"], "cells": saved["config"]["cells"]})
    family = families(parent, prepared)
    if family != saved["family"] or saved["manifest"]["family"] != family:
        raise ValueError("frozen all-proposal family changed")
    for c in prepared:
        if any(costs[c][k] != saved["setup"][c][k] for k in ("static_components", "static_digest")):
            raise ValueError("geometry digest mismatch")
    journal = RecoveryJournal(path.parent, saved["manifest"])
    expected = set(saved["manifest"]["units"])
    for suffix in ("start", "finish"):
        if {p.name.removesuffix("."+suffix+".json") for p in path.parent.glob("*."+suffix+".json")} != expected:
            raise ValueError("incomplete/unexpected D2 journal family")
    records: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = []
    replay_calls = 0
    expected_keys = {identity(j): j for j in jobs(tuple(prepared))}
    expected_keys.update({"preflight-"+cell+"-"+design: key("preflight", cell, design)
        for cell in prepared for design in DESIGNS})
    for ident in saved["manifest"]["units"]:
        start, finish = [journal._read(path.parent/(ident+"."+s+".json")) for s in ("start", "finish")]
        record = finish["result"]
        is_preflight = ident.startswith("preflight-")
        n, cap, phase = (1240, 60., "P") if is_preflight else (29760, 90., "D2")
        k = {name: record[name] for name in ("phase", "cell", "design", "rep", "role", "block")}
        if (k != expected_keys[ident] or finish["status"] != "complete" or finish["work"] != n or record["actual_work"] != n
                or start["reserved_work"] != n or start["reserved_wall"] != cap
                or finish["phase"] != phase or start["phase"] != phase or finish["wall"] > cap
                or record["torch_threads"] != 1 or record["weights_digest"] != canonical_digest(family[k["cell"]]["weights"])
                or record["seed_digest"] != canonical_digest(stream_layout(prepared[k["cell"]][1], n, k["design"], k))):
            raise ValueError("unit source/identity/seed/work binding failed")
        parts = record["strata"]
        counts = (n,) if k["design"] == "iid" else balanced_counts(prepared[k["cell"]][1].weights, n)
        if len(parts) != len(counts) or any(p["count"] != cnt for p, cnt in zip(parts, counts, strict=True)):
            raise ValueError("saved stratum counts changed")
        for part in parts:
            validate_sufficient(part)
            if len(part["log_sum"]) != 12:
                raise ValueError("missing target moments")
        rebuilt = estimate(parts, torch.ones(1, dtype=torch.float64) if k["design"] == "iid" else prepared[k["cell"]][1].weights)
        if rebuilt != record["estimator"]:
            raise ValueError("saved estimator/covariance mismatch")
        if is_preflight or k["block"] == 0:
            replay = risk_block(prepared[k["cell"]], family[k["cell"]]["weights"], k["design"], n, k)
            if any(replay[field] != record[field] for field in ("strata", "estimator", "seed_digest", "sample_digest")):
                raise ValueError("complete bank numerical replay mismatch")
            replay_calls += n
        (preflights if is_preflight else records).append(record)
    summary = summarize(records, prepared, family)
    if (summary != saved["summary"] or records != saved["records"] or preflights != saved["preflight"]
            or journal.accounting("P")[0] != 4960 or journal.accounting("D2")[0] != 952320
            or saved["physical_evaluations"] != 957280
            or any(path.parent.glob("*.violation.json"))):
        raise ValueError("complete summary/work/gate mismatch")
    if replay_calls != 124000 or time.perf_counter()-begin > 1800:
        raise ValueError("audit budget exceeded")
    return {"status": "D2_frozen_source_all_counts_cross_moments_and_eight_bank_replays_pass",
        "result_sha256": sha(path), "audit_volterra_evaluations": replay_calls,
        "d2_gate": summary["decision"]["d2_gate"], "p2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(ROOT/DIRECTORY/"result.json")))
    else:
        result = run()
        print(json.dumps(result["summary"]["decision"]))
