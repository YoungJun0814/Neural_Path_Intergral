"""One-shot frozen-q path-feature discovery and independent confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.post_audit_bounded_bank_runtime import group_moments, key, validate_streams
from experiments.post_audit_bounded_frozen_bank_contrast import DIRECTORY as PARENT
from experiments.post_audit_defensive_allocation import guard
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.balanced_mixture_bank import draw_bank, stream_layout
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.risk_path_attribution import (
    FEATURES,
    candidate_rules,
    contrasts,
    diagnostic_logs,
    estimate,
    merge,
    moments,
    rule_mask,
)
from src.path_integral.stratified_paired_risk import estimate as old_estimate
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import sensitivity

DIRECTORY = "results/post_audit/frozen_q_path_attribution_v1/session_20261021"
PLAN = "docs/plans/FROZEN_Q_RISK_PATH_ATTRIBUTION_2026-10-21_KO.md"
EXPECTED = "323b00268d797b6e70222a748ce500ee4c5f7f1ff9da1e6724acd750cb3ee1bb"
COUNT = 29760


def evidence() -> dict[str, Any]:
    path = ROOT/PARENT/"result.json"
    parent, receipt = [json.loads(p.read_text()) for p in (path, path.with_name("audit.json"))]
    if sha(path) != EXPECTED or receipt["result_sha256"] != EXPECTED or receipt["d2_gate"]:
        raise ValueError("audited frozen D2 STOP evidence required")
    audit_snapshot(ROOT, parent["source"], parent["config"])
    return parent


def physical_features(cache: Any, guide: PhysicalWindowGuide, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One terminal-law calculation; do not call log_probability again."""
    terminal, integrated, variance = cache.quantities(z)
    p = cache.problem
    g = torch.special.log_ndtr((math.log(p.task.level)-terminal)/torch.sqrt((1-p.rho**2)*integrated))
    v = variance[:, :-1]
    mass = v/v.sum(1, keepdim=True)
    times = torch.arange(p.steps, dtype=torch.float64)/p.steps
    total_energy = z.square().sum(1)
    projected = (z @ guide.vol_directions.T).square()+(z @ guide.price_directions.T).square()
    residual = 1-projected.max(1).values/total_energy.clamp_min(torch.finfo(torch.float64).tiny)
    if (residual < -1e-10).any() or (residual > 1+1e-10).any():
        raise FloatingPointError("orthogonal pair projection bound failed")
    features = torch.stack((integrated.log(), (v.max(1).values/p.xi).log(),
        v.argmax(1).double()/p.steps, mass.square().sum(1), mass @ times, residual.clamp(0, 1)), 1)
    if not torch.isfinite(features).all() or not torch.isfinite(g).all() or (g > 1e-12).any():
        raise FloatingPointError("invalid feature/conditional probability")
    return g, features, variance


def component_description(parent: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in parent["summary"]["cells"]:
        summary = cell["methods"]["stratified"]
        pi = torch.tensor(parent["family"][cell["cell"]]["weights"][0], dtype=torch.float64)
        for k, target in enumerate(cell["target_ids"]):
            terms = torch.tensor([p["log_sum"][k]-math.log(p["count"])+float(pi[j].log())
                for j, p in enumerate(summary["strata"])], dtype=torch.float64)
            log_total = float(torch.logsumexp(terms, 0))
            largest = int(terms.argmax())
            part = summary["strata"][largest]
            component = None if largest == 0 else {"template": (largest-1)//12,
                "kind": "point" if (largest-1)//12 < 15 else "window",
                "vol_amplitude": (2, 4, 6, 8)[((largest-1)%12)//3],
                "price_amplitude": (0, 2, 4)[(largest-1)%3]}
            rows.append({"cell": cell["cell"], "target": target, "largest_label": largest,
                "label_metadata": component, "largest_label_contribution": math.exp(float(terms[largest])-log_total),
                "maximum_within_largest_label": math.exp(part["log_maximum"][k]-part["log_sum"][k]),
                "not_path_mode_or_geometry_proof": True})
    return rows


def draw_trace(prepared: Any, guide: PhysicalWindowGuide, weights: list[list[float]],
               k: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    cache, spec = prepared
    z, labels, sd = draw_bank(spec, COUNT, "stratified", k)
    gs, fs, vs, cs = [], [], [], []
    for batch in z.split(2048):
        with torch.no_grad():
            g, f, v = physical_features(cache, guide, batch)
            c = batch @ spec.means.T-.5*spec.means.square().sum(1)
        gs.append(g)
        fs.append(f)
        vs.append(v)
        cs.append(c)
        guard()
    g, features, variance, c = torch.cat(gs), torch.cat(fs), torch.cat(vs), torch.cat(cs)
    d = torch.logsumexp(c+spec.weights.log(), 1)
    columns = [g-d]
    for w in weights:
        columns.append(2*g-d-torch.logsumexp(c+torch.tensor(w, dtype=torch.float64).log(), 1))
    all_logs = torch.stack(columns, 1)
    if not torch.isfinite(all_logs).all() or (all_logs[:, 0] > math.log(10)+1e-10).any() or (all_logs[:, 1:] > math.log(100)+1e-10).any():
        raise FloatingPointError("defensive event/risk bound failed")
    digest = hashlib.sha256()
    for value in (z, labels, g, d):
        digest.update(value.contiguous().numpy().tobytes())
    # All eleven old q's remain bound; only U/event-kl-0 are feature endpoints.
    parts = group_moments(all_logs, labels, spec, "stratified")
    logs = all_logs[:, :3]
    tops = []
    for q in range(2):
        for index in torch.argsort(logs[:, q+1], descending=True, stable=True)[:8].tolist():
            tops.append({"q": q, "index": index, "label": int(labels[index]),
                "features": features[index].tolist(), "z": z[index].tolist(), "variance": variance[index].tolist(),
                "log_event": float(logs[index, 0]), "log_risk": float(logs[index, q+1])})
    return {"features": features, "logs": logs, "labels": labels,
        "metadata": {**k, "actual_work": COUNT, "seed_digest": sd, "sample_digest": digest.hexdigest(),
            "old_strata": parts, "old_estimator": old_estimate(parts, spec.weights), "tops": tops,
            "wall": time.perf_counter()-start, "threads": torch.get_num_threads()}}


def save_trace(directory: Path, ident: str, trace: dict[str, Any]) -> dict[str, Any]:
    path = directory/(ident+".npz")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **{k: trace[k].numpy() for k in ("features", "logs", "labels")})
    return {**trace["metadata"], "array": path.name, "array_sha256": sha(path)}


def load_trace(directory: Path, record: dict[str, Any]) -> dict[str, torch.Tensor]:
    name = record["array"]
    if Path(name).name != name or not name.endswith(".npz") or sha(directory/name) != record["array_sha256"]:
        raise ValueError("trace file identity/hash mismatch")
    with np.load(directory/name, allow_pickle=False) as raw:
        if set(raw.files) != {"features", "logs", "labels"}:
            raise ValueError("complete trace array family required")
        data = {k: torch.from_numpy(raw[k].copy()) for k in raw.files}
    if (data["features"].shape != (COUNT, 6) or data["logs"].shape != (COUNT, 3)
            or data["labels"].shape != (COUNT,) or not torch.isfinite(data["features"]).all()
            or not torch.isfinite(data["logs"]).all()
            or data["features"].dtype != torch.float64 or data["logs"].dtype != torch.float64
            or data["labels"].dtype != torch.int64):
        raise ValueError("trace dtype/count/nonfinite failure")
    return data


def phase_key(phase: str, cell: str, block: int) -> dict[str, Any]:
    if phase not in ("discovery", "confirmation"):
        raise ValueError("fixed discovery/confirmation phases required")
    return key("D2" if phase == "discovery" else "frozen-q-path-attribution-v1", cell,
        "stratified", role="risk" if phase == "discovery" else "feature-confirmation", block=block)


def select(directory: Path, records: list[dict[str, Any]], cells: tuple[str, ...]) -> dict[str, Any]:
    fs, logs = {}, {}
    for cell in cells:
        group = sorted((r for r in records if r["cell"] == cell), key=lambda r: r["block"])
        if [r["block"] for r in group] != list(range(8)):
            raise ValueError("eight complete discovery blocks required")
        arrays = [load_trace(directory, r) for r in group]
        fs[cell], logs[cell] = [torch.cat([a[k] for a in arrays]) for k in ("features", "logs")]
    return candidate_rules(fs, logs)


def confirmation_summary(directory: Path, records: list[dict[str, Any]], prepared: dict[str, Any],
                         rule: dict[str, Any]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell, (_, spec) in prepared.items():
        group = sorted((r for r in records if r["cell"] == cell), key=lambda r: r["block"])
        if [r["block"] for r in group] != list(range(8)):
            raise ValueError("eight complete confirmation blocks required")
        block_parts, block_stats = [], []
        for r in group:
            array = load_trace(directory, r)
            logs = diagnostic_logs(array["logs"], rule_mask(array["features"], rule, cell))
            parts = [moments(logs[array["labels"] == j]) for j in range(spec.components)]
            block_parts.append(parts)
            block_stats.append(estimate(parts, spec.weights))
        merged = [merge([parts[j] for parts in block_parts]) for j in range(spec.components)]
        raw = estimate(merged, spec.weights)
        block_values = torch.stack([torch.tensor(b["mean"], dtype=torch.float64)
            *torch.exp(torch.tensor(b["scale"], dtype=torch.float64)-torch.tensor(raw["scale"], dtype=torch.float64))
            for b in block_stats])
        # Work in raw's scale: this avoids exp(log H) underflow in block means.
        block = estimate([moments(block_values.log())], torch.ones(1, dtype=torch.float64))
        block["scale"] = (torch.tensor(block["scale"], dtype=torch.float64)+torch.tensor(raw["scale"], dtype=torch.float64)).tolist()
        outcomes: list[dict[str, Any]] = []
        for q in range(2):
            k = 1+q
            mean, covariance = raw["mean"][k], raw["covariance"][k][k]
            rse = math.sqrt(max(0., covariance))/mean
            maximum = raw["maximum_term"][k]/mean
            log_means = [math.log(b["mean"][k])+b["scale"][k] for b in block_stats]
            sens = sensitivity(log_means, bootstrap_seed=2187000+q+(0 if cell == tuple(prepared)[0] else 10))
            precision = (rse <= .2 and sens["between_unit_relative_se"] <= 2*rse
                and sens["maximum_leave_one_out_relative_shift"] <= .15 and maximum <= .1)
            raw_c, block_c = contrasts(raw, q), contrasts(block, q)
            individual = [contrasts(b, q) for b in block_stats]
            repeated = sum(b["risk_mass"] >= .2 and b["enrichment"] is not None and b["enrichment"] >= 2
                and b["risk_mass"] > b["event_mass"] for b in individual)
            outcomes.append({"q": ("U", "event-kl-0")[q], "rse": rse, "maximum_fraction": maximum,
                "sensitivity": sens, "precision": precision, "raw": raw_c, "block": block_c,
                "individual_blocks": individual, "repeated_blocks": repeated,
                "confirmed": bool(precision and raw_c["all_lower_positive"] and block_c["all_lower_positive"] and repeated >= 6)})
        cells.append({"cell": cell, "raw": raw, "block": block, "outcomes": outcomes})
    passed = all(c["outcomes"][0]["confirmed"] for c in cells)
    return {"cells": cells, "decision": "FEATURE_CONFIRMED_DEVELOPMENT_ONLY" if passed else "STOP_FEATURE_NOT_INDEPENDENTLY_CONFIRMED",
        "primary_confirmed": passed, "secondary_transfer_confirmed": all(c["outcomes"][1]["confirmed"] for c in cells),
        "D3_authorized": False, "p2_qualified": False, "model_changed": False}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    parent = evidence()
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("one-shot: nonempty evidence directory cannot be resumed/replaced")
    path.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"parent": EXPECTED, "features": list(FEATURES), "quantiles": [.1, .9], "count": COUNT,
        "phases": ["discovery", "confirmation"], "blocks": 8, "risk_q": ["U", "event-kl-0"],
        "model": parent["config"]["parent_model"], "cells": parent["config"]["cells"],
        "production_cap": 952320, "audit_cap": 119040, "D3_authorized": False}
    source = freeze_source(path/"run.json", config,
        extra_snapshot_paths=(PLAN, PARENT+"/result.json", PARENT+"/audit.json"))
    write_json_atomic_nonoverwriting(path/"protocol.json", {"config": config, "source": source})
    started = time.perf_counter()
    prepared, _ = prepare(config)
    guides = {c: PhysicalWindowGuide.create(cache.problem, kind="mixed-window", amplitudes=[2., 4., 6., 8.],
        price_amplitudes=[0., 2., 4.]) for c, (cache, _) in prepared.items()}
    if any(not torch.equal(guides[c].spec.means, prepared[c][1].means) for c in prepared):
        raise ValueError("physical feature geometry changed")
    cells = tuple(prepared)
    roles = [(COUNT, "stratified", phase_key(phase, c, b)) for phase in config["phases"] for c in cells for b in range(8)]
    validate_streams(prepared, roles)
    manifest = {"source": source["snapshot_sha256"], "config": config,
        "limits": {phase: {"work": 476160, "wall": 900.} for phase in config["phases"]}}
    journal = RecoveryJournal(path, manifest)
    write_json_atomic_nonoverwriting(path/"component_description.json", {"rows": component_description(parent), "physics": 0})
    discovery: list[dict[str, Any]] = []
    confirmation: list[dict[str, Any]] = []
    selection: dict[str, Any] | None = None
    for phase, records in (("discovery", discovery), ("confirmation", confirmation)):
        if phase == "confirmation" and (selection is None or selection["selected"] is None):
            break
        for c in cells:
            for b in range(8):
                ident = phase+"-"+c+"-"+str(b)
                def operation(cell=c, block=b, name=ident, stage=phase):
                    trace = draw_trace(prepared[cell], guides[cell], parent["family"][cell]["weights"], phase_key(stage, cell, block))
                    if stage == "discovery":
                        old = next(r for r in parent["records"] if (r["cell"], r["design"], r["block"]) == (cell, "stratified", block))
                        for field, saved_field in (("old_strata", "strata"), ("old_estimator", "estimator"),
                                ("seed_digest", "seed_digest"), ("sample_digest", "sample_digest")):
                            if trace["metadata"][field] != old[saved_field]:
                                raise ValueError("discovery old D2 numerical binding failed: "+field)
                    return save_trace(path, name, trace)
                records.append(journal.unit(ident, phase=phase, work=COUNT, maximum_wall=90., operation=operation))
                print(f"{phase} completed {len(records)}/16", flush=True)
        if phase == "discovery":
            selection = select(path, discovery, cells)
            write_json_atomic_nonoverwriting(path/"selection.json", selection)
            clock = journal._read(path/"phase-discovery.clock.json")
            if time.time()-clock["started_unix"] > 900:
                journal._write(path/"phase-discovery.violation.json", {"manifest": journal.digest, "status": "selection_io_wall_failure"})
                raise TimeoutError("discovery selection/I/O wall exceeded cap")
    assert selection is not None
    summary: dict[str, Any] = (confirmation_summary(path, confirmation, prepared, selection["selected"]) if confirmation else
        {"decision": "STOP_NO_ELIGIBLE_DISCOVERY_FEATURE", "primary_confirmed": False,
            "D3_authorized": False, "p2_qualified": False, "model_changed": False})
    output: dict[str, Any] = {"config": config, "source": source, "manifest": manifest, "discovery": discovery,
        "confirmation": confirmation, "selection": selection, "summary": summary,
        "actual_work": sum(journal.accounting(phase)[0] for phase in config["phases"]),
        "wall": time.perf_counter()-started}
    write_json_atomic_nonoverwriting(path/"result.json", output)
    for phase in config["phases"]:
        clock = path/("phase-"+phase+".clock.json")
        # Discovery clock ends at selection freeze; confirmation includes final statistics/I/O.
        if phase == "confirmation" and clock.exists() and time.time()-journal._read(clock)["started_unix"] > 900:
            journal._write(path/("phase-"+phase+".violation.json"), {"manifest": journal.digest, "status": "final_io_wall_failure"})
            raise TimeoutError("confirmation final I/O exceeded cap")
    if output["wall"] > 3600:
        raise TimeoutError("total production wall cap exceeded")
    return output


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    parent = evidence()
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, value in result["source"]["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/")) and sha(ROOT/name) != value:
            raise ValueError("frozen numerical source changed")
    prepared, _ = prepare(result["config"])
    guides = {c: PhysicalWindowGuide.create(cache.problem, kind="mixed-window", amplitudes=[2., 4., 6., 8.],
        price_amplitudes=[0., 2., 4.]) for c, (cache, _) in prepared.items()}
    journal = RecoveryJournal(path, result["manifest"])
    expected, work, replay = set(), 0, 0
    for phase, records in (("discovery", result["discovery"]), ("confirmation", result["confirmation"])):
        if records and (len(records) != 16 or {(r["cell"], r["block"]) for r in records} != {(c, b) for c in prepared for b in range(8)}):
            raise ValueError("complete phase family required")
        for r in records:
            ident = phase+"-"+r["cell"]+"-"+str(r["block"])
            expected.add(ident)
            start, finish = [journal._read(path/(ident+"."+s+".json")) for s in ("start", "finish")]
            if (finish["status"] != "complete" or finish["result"] != r or finish["work"] != COUNT
                    or r["actual_work"] != COUNT or r["threads"] != 1 or finish["wall"] > 90
                    or start["reserved_work"] != COUNT or start["reserved_wall"] != 90
                    or any(r[k] != v for k, v in phase_key(phase, r["cell"], r["block"]).items())):
                raise ValueError("journal work/identity binding failed")
            array = load_trace(path, r)
            spec = prepared[r["cell"]][1]
            if r["seed_digest"] != canonical_digest(stream_layout(spec, COUNT, "stratified", phase_key(phase, r["cell"], r["block"]))):
                raise ValueError("complete per-block seed layout binding failed")
            counts = array["labels"].bincount(minlength=373)
            if counts.tolist() != [COUNT//10]+[72]*372:
                raise ValueError("trace component count mismatch")
            subset = group_moments(array["logs"], array["labels"], spec, "stratified")
            for fresh, original in zip(subset, r["old_strata"], strict=True):
                expected_part = {"count": original["count"], "log_sum": original["log_sum"][:3],
                    "log_cross_sum": [row[:3] for row in original["log_cross_sum"][:3]],
                    "log_maximum": original["log_maximum"][:3]}
                if fresh != expected_part:
                    raise ValueError("all-array event/risk moments do not match the trace")
            if phase == "discovery":
                old = next(v for v in parent["records"] if (v["cell"], v["design"], v["block"]) == (r["cell"], "stratified", r["block"]))
                if any(r[a] != old[b] for a, b in (("old_strata", "strata"), ("old_estimator", "estimator"), ("sample_digest", "sample_digest"), ("seed_digest", "seed_digest"))):
                    raise ValueError("old D2 replay binding invalid")
            if r["block"] == 0:
                t = draw_trace(prepared[r["cell"]], guides[r["cell"]], parent["family"][r["cell"]]["weights"], phase_key(phase, r["cell"], 0))
                if (any(not torch.equal(t[k], array[k]) for k in array)
                        or any(t["metadata"][k] != r[k] for k in ("old_strata", "old_estimator", "sample_digest", "seed_digest", "tops"))):
                    raise ValueError("representative numerical trace replay failed")
                replay += COUNT
            work += COUNT
    for suffix in ("start", "finish"):
        if {p.name.removesuffix("."+suffix+".json") for p in path.glob("*."+suffix+".json")} != expected:
            raise ValueError("extra or incomplete journal units")
    selection = select(path, result["discovery"], tuple(prepared))
    if selection != result["selection"] or selection != json.loads((path/"selection.json").read_text()):
        raise ValueError("frozen selection changed")
    component = json.loads((path/"component_description.json").read_text())
    if component != {"rows": component_description(parent), "physics": 0}:
        raise ValueError("zero-physics component description changed")
    if result["confirmation"]:
        summary = confirmation_summary(path, result["confirmation"], prepared, selection["selected"])
        if summary != result["summary"]:
            raise ValueError("all confirmation statistics or decision changed")
    elif selection["selected"] is not None:
        raise ValueError("eligible selection without complete confirmation")
    if (work != result["actual_work"] or work > 952320 or replay > 119040 or time.perf_counter()-started > 900
            or any(path.glob("*.violation.json"))):
        raise ValueError("work/wall/violation failure")
    return {"status": "frozen_q_feature_all_arrays_statistics_and_prespecified_trace_replay_pass",
        "result_sha256": sha(path/"result.json"), "audit_physical_evaluations": replay,
        "production_physical_evaluations": work, "decision": result["summary"]["decision"], "p2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        receipt = audit()
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        result = run()
        print(json.dumps(result["summary"]))
