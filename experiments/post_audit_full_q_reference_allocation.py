"""Fresh all-q pilot; no parent0 variability substitution or silent gate change."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.coverage_envelope_reference import CoverageEnvelopeReference
from src.path_integral.guide_free_block_smc import estimate_guide_free_block_normalizer
from src.path_integral.proposal_bridge_smc import estimate_proposal_bridge
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    merge_moments,
    precision_count,
    sensitivity,
)
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/FULL_FIXED_Q_REFERENCE_ALLOCATION_PILOT_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/full_q_reference_allocation_v1/session_20261010"
ANCHOR = "results/post_audit/guide_free_block_recovery_v1/session_20261010/result.json"
ANCHOR_HASH = "d42c882bb7e11d0db9c8a3e0a61b6831bd7be9eae40102413451dd0e80e0a24f"


def seed(identity: str, stream: str) -> int:
    return int(hashlib.sha256(("full-fixed-q-allocation-v1/"+identity+"/"+stream).encode()).hexdigest()[:15], 16)


def selections(anchor: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for cell in sorted({s["cell"] for s in anchor["summaries"]}):
        result[cell] = {}
        for target in ("mu", "risk"):
            rows = [s for s in anchor["summaries"] if s["cell"] == cell and s["target"] == target]
            if {s["method"] for s in rows} != {"full-pcn", "multiscale-block-pcn"} or len(rows) != 2:
                raise ValueError("invalid fixed selection anchor")
            result[cell][target] = min(rows, key=lambda s: (s["observed_cv2_wall_per_whole"], s["method"]))["method"]
    return result


def jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grid = []
    for r in rows:
        base = {"cell": r["cell"]["id"], "parent_training_rep": r["parent_training_rep"]}
        grid += [{**base, "method": "guide-iid", "target": "paired", "rep": rep, "batch": b}
                 for rep in range(8) for b in range(8)]
        grid += [{**base, "method": method, "target": "risk", "rep": rep, "batch": 0}
                 for method in ("guide-free-smc", "envelope-smc") for rep in range(8)]
        if r["parent_training_rep"] == 0:
            grid += [{**base, "method": "guide-free-smc", "target": "mu", "rep": rep, "batch": 0}
                     for rep in range(8)]
    return grid


def identity(j: dict[str, Any]) -> str:
    return "-".join(str(j[k]) for k in ("cell", "parent_training_rep", "method", "target", "rep", "batch"))


def expected_work(job: dict[str, Any]) -> int:
    return {"guide-iid": 4096, "guide-free-smc": 508928, "envelope-smc": 254976}[job["method"]]


def prepare(metadata: dict[str, Any], rows: list[dict[str, Any]], guide_spec: dict[str, Any]) -> dict[str, Any]:
    cells, qcache = {}, {}
    for row in rows:
        cell = row["cell"]["id"]
        if cell not in cells:
            p = _problem(metadata["config"]["model"], row["cell"], 32)
            guide = build_volterra_excursion_guide(p, **guide_spec)
            cells[cell] = {"evaluator": CachedVolterraTerminal.create(p), "guide": guide,
                           "gdensity": CachedShiftDensity.create(guide), "dimension": p.local_dimension,
                           "reference": CoverageEnvelopeReference.create(p)}
        q = proposal_from_parameters(row["q_parameters"])
        qcache[(cell, row["parent_training_rep"])] = {"qdensity": CachedShiftDensity.create(q),
                                                    "delta": q.defensive_mass, "q_digest": row["q_digest"]}
    return {"cells": cells, "qs": qcache, "preparation_work": 256}


def unit_values(j: dict[str, Any], cache: dict[str, Any], chosen: dict[str, dict[str, str]]) -> dict[str, Any]:
    begin = time.perf_counter()
    cell, q = cache["cells"][j["cell"]], cache["qs"][(j["cell"], j["parent_training_rep"])]
    streams = {s: seed(identity(j), s) for s in ("path", "label", "whole")}
    if j["method"] == "guide-iid":
        z = cell["guide"].sample(4096, path_seed=streams["path"], label_seed=streams["label"]).samples
        g, d = cell["evaluator"].log_probability(z), cell["gdensity"].log_q_over_p(z)
        values = {"mu": g-d, "risk": 2*g-q["qdensity"].log_q_over_p(z)-d}
        payload: dict[str, Any] = {"moments": {t: log_moments(v) for t, v in values.items()}}
        calls, stages = 4096, None
    else:
        def potential(z):
            g = cell["evaluator"].log_probability(z)
            return g if j["target"] == "mu" else log_risk_potential(g, q["qdensity"].log_q_over_p(z), defensive_mass=q["delta"])
        free = j["method"] == "guide-free-smc"
        cfg = WeightedSMCConfig(1024, tuple((k/63)**2 for k in range(64)), 8 if free else 4, .35, 1,
                               streams["whole"], resample_every=4, resampling_scheme="stratified",
                               independence_every=0 if free else 2)
        if free:
            out = estimate_guide_free_block_normalizer(potential, dimension=cell["dimension"], config=cfg,
                use_blocks=chosen[j["cell"]][j["target"]] == "multiscale-block-pcn", block_scale=.7)
            normalizer, calls, stages = out.log_normalizer, out.potential_evaluations, list(out.stages)
        else:
            r = cell["reference"]
            guided = estimate_proposal_bridge(potential, proposal=r, config=cfg, base="proposal",
                                              directional_units=r.units, directional_price_indices=r.price_indices)
            normalizer, calls, stages = guided.log_normalizer, guided.potential_evaluations, list(guided.stages)
        payload = {"log_normalizer": normalizer-(math.log(q["delta"]) if j["target"] == "risk" else 0.)}
    if calls != expected_work(j):
        raise ValueError("unexpected unit work")
    return {**j, **payload, "seeds": streams, "q_digest": q["q_digest"], "actual_work": calls,
            "stages": stages, "wall": time.perf_counter()-begin,
            "guide_independent_kernel": j["method"] == "guide-free-smc"}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cell in sorted({r["cell"] for r in records}):
        for target, parents, methods in (("risk", range(5), ("guide-iid", "guide-free-smc", "envelope-smc")),
                                         ("mu", (0,), ("guide-iid", "guide-free-smc"))):
            for parent in parents:
                for method in methods:
                    group = [r for r in records if r["cell"] == cell and r["method"] == method
                             and r["target"] in (target, "paired")
                             and (target == "mu" and method == "guide-iid" or r["parent_training_rep"] == parent)]
                    if method == "guide-iid":
                        stats = merge_moments([r["moments"][target] for r in group])
                        blocks = sorted({(r["parent_training_rep"], r["rep"]) for r in group})
                        logs = [merge_moments([r["moments"][target] for r in group
                                if (r["parent_training_rep"], r["rep"]) == b])["log_mean"] for b in blocks]
                        expected = 320 if target == "mu" else 64
                    else:
                        logs = [r["log_normalizer"] for r in group]
                        stats = log_moments(torch.tensor(logs, dtype=torch.float64))
                        expected = 8
                    if len(group) != expected:
                        raise ValueError("incomplete estimator grid")
                    wall = sum(r["wall"] for r in group)
                    count = stats["count"]
                    allocation = precision_count(stats, target_rse=.015, safety_factor=2,
                        minimum=65536 if method == "guide-iid" else 32,
                        maximum=8388608 if method == "guide-iid" else 256,
                        multiple=4096 if method == "guide-iid" else 1)
                    unit_work = 1 if method == "guide-iid" else expected_work(group[0])
                    output.append({"cell": cell, "target": target, "parent_training_rep": parent,
                        "method": method, "summary": stats, "whole_or_equal_size_block_means": logs,
                        "sensitivity": sensitivity(logs, bootstrap_seed=seed(cell+str(parent)+target+method, "bootstrap")),
                        "allocation": allocation, "forecast_work": allocation["required_count"]*unit_work,
                        "forecast_wall": allocation["required_count"]*wall/count, "wall": wall,
                        "guide_independent_kernel": method == "guide-free-smc",
                        "mu_iid_pools_five_disjoint_q_risk_streams": target == "mu" and method == "guide-iid"})
    return output


def allocation_decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    work, wall = sum(s["forecast_work"] for s in summaries), sum(s["forecast_wall"] for s in summaries)
    failures = [f"{s['cell']}/q{s['parent_training_rep']}/{s['target']}/{s['method']}:sample_cap"
                for s in summaries if s["allocation"]["status"] != "allocated"]
    if work > 160000000:
        failures.append("original_production_work_cap")
    if 2*wall > 3600:
        failures.append("original_production_wall_cap_with_safety2")
    return {"status": "allocation_feasible_not_precision_qualified" if not failures else "allocation_blocked",
            "failure_reasons": failures, "forecast_work": work, "forecast_wall": wall,
            "forecast_wall_with_safety2": 2*wall, "p1_qualified": False, "p2_qualified": False,
            "large_training_authorized": False, "scope": "fresh_all_q_development_allocation_only"}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    if hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != ANCHOR_HASH:
        raise ValueError("changed development selection anchor")
    chosen = selections(json.loads((ROOT/ANCHOR).read_text()))
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"input": input_config, "selected_guide_free": chosen, "anchor": ANCHOR, "anchor_sha256": ANCHOR_HASH,
              "steps": 32, "whole_repetitions": 8, "iid_per_q": 262144,
              "scope": "full_two_cell_five_q_fresh_allocation_pilot"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs(rows)
    streams = [seed(identity(j), s) for j in grid for s in ("path", "label", "whole")]
    if len(grid) != 816 or len(streams) != len(set(streams)):
        raise ValueError("unexpected grid/seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "q_bindings": [{"cell": r["cell"]["id"], "parent_training_rep": r["parent_training_rep"],
                                "q_digest": r["q_digest"]} for r in rows],
                "limits": {"A": {"work": 85000000, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    saved: dict[str, Any] = {}
    def preparation():
        saved.update(prepare(metadata, rows, input_config["guide"]))
        return {"actual_work": saved["preparation_work"]}
    journal.unit("preparation", phase="A", work=1024, maximum_wall=30., operation=preparation)
    records = []
    for j in grid:
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            return unit_values(job, saved, chosen)
        records.append(journal.unit(identity(j), phase="A", work=expected_work(j),
                                    maximum_wall=5. if j["method"] == "guide-iid" else 30., operation=operation))
        if len(records) % 80 == 0:
            print(f"full-q reference allocation {len(records)}/{len(grid)}", flush=True)
    summaries = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "decision": allocation_decision(summaries), "units": len(grid), "work": journal.accounting("A")[0],
              "phase_wall": time.time()-journal._read(directory/"phase-A.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-A.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary and final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["anchor"] != ANCHOR or config["anchor_sha256"] != ANCHOR_HASH
            or hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != ANCHOR_HASH
            or config["selected_guide_free"] != selections(json.loads((ROOT/ANCHOR).read_text()))
            or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()):
        raise ValueError("changed source/input/selection anchor")
    metadata, rows = inputs(config["input"])
    grid = jobs(rows)
    bindings = [{"cell": r["cell"]["id"], "parent_training_rep": r["parent_training_rep"],
                 "q_digest": r["q_digest"]} for r in rows]
    if manifest["expected_units"] != [identity(j) for j in grid] or manifest["q_bindings"] != bindings:
        raise ValueError("changed grid/q binding")
    journal = RecoveryJournal(path.parent, manifest)
    records = []
    for j in grid:
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed unit")
        record = finish["result"]
        digest = next(r["q_digest"] for r in rows if (r["cell"]["id"], r["parent_training_rep"])
                      == (j["cell"], j["parent_training_rep"]))
        if (any(record[k] != v for k, v in j.items()) or record["q_digest"] != digest
                or record["seeds"] != {s: seed(identity(j), s) for s in ("path", "label", "whole")}
                or record["actual_work"] != expected_work(j)
                or record["guide_independent_kernel"] != (j["method"] == "guide-free-smc")):
            raise ValueError("changed unit identity")
        records.append(record)
    summaries = summarize(records)
    if (summaries != output["summaries"] or allocation_decision(summaries) != output["decision"]
            or journal.accounting("A")[0] != output["work"] or output["work"] != 256+sum(expected_work(j) for j in grid)
            or output["units"] != len(grid)):
        raise ValueError("changed summary/accounting/decision")
    cache = prepare(metadata, rows, config["input"]["guide"])
    replayed_draws, replayed_wholes, audit_work = 0, 0, cache["preparation_work"]
    chosen = config["selected_guide_free"]
    subset = [r for r in records if r["rep"] == 0 and r["batch"] == 0 and
              ((r["method"] == "guide-iid" and r["parent_training_rep"] in (0, 4)) or
               (r["method"] != "guide-iid" and r["parent_training_rep"] == 0 and r["target"] == "risk"))]
    for record in subset:
        replay = unit_values(record, cache, chosen)
        field = "moments" if record["method"] == "guide-iid" else "log_normalizer"
        if replay[field] != record[field] or replay["stages"] != record["stages"]:
            raise ValueError("selected unit numerical replay mismatch")
        audit_work += replay["actual_work"]
        replayed_draws += 4096 if record["method"] == "guide-iid" else 0
        replayed_wholes += int(record["method"] != "guide-iid")
    if (path.parent/"phase-A.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_q_seed_moment_arithmetic_replay_pass_not_production_certificate",
            "units": len(grid), "replayed_draws": replayed_draws, "replayed_wholes": replayed_wholes,
            "audit_potential_calls": audit_work, "decision": output["decision"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "phase_wall", "units", "decision")}))
