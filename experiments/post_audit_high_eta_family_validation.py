"""Fresh, scoped five-q validation; preserves unresolved canonical and mesh gates."""

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

from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_umbrella_reference_pilot import (
    expected_work as umbrella_work,
)
from experiments.post_audit_umbrella_reference_pilot import (
    identity as umbrella_identity,
)
from experiments.post_audit_umbrella_reference_pilot import (
    prepare,
)
from experiments.post_audit_umbrella_reference_pilot import (
    seed as umbrella_seed,
)
from experiments.post_audit_umbrella_reference_pilot import (
    unit_values as umbrella_unit,
)
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    merge_moments,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/HIGH_ETA_FIVE_Q_FRESH_REFERENCE_VALIDATION_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/high_eta_family_validation_v1/session_20261010"
ANCHOR = "results/post_audit/umbrella_reference_pilot_v1/session_20261010/result.json"
CELL = "high-eta200-k1"
NAMESPACE = "high-family-precision-v1-"
TARGETS = ["mu"]+[f"risk-q{j}" for j in range(5)]


def jobs() -> list[dict[str, Any]]:
    common = {"cell": CELL, "namespace": NAMESPACE}
    return ([{**common, "method": "guide-iid", "rep": rep, "batch": b} for rep in range(40) for b in range(8)]
            +[{**common, "method": method, "rep": rep, "batch": 0}
              for method, count in (("free-full", 69), ("envelope", 52)) for rep in range(count)])


def identity(j: dict[str, Any]) -> str:
    return umbrella_identity(j)+f"-{j['batch']}"


def seed(j: dict[str, Any], stream: str) -> int:
    if stream == "whole":
        return umbrella_seed(NAMESPACE+umbrella_identity(j))
    return int(hashlib.sha256((NAMESPACE+identity(j)+"/"+stream).encode()).hexdigest()[:15], 16)


def unit_values(j: dict[str, Any], cells: dict[str, Any]) -> dict[str, Any]:
    if j["method"] != "guide-iid":
        return umbrella_unit(j, cells)
    begin = time.perf_counter()
    c = cells[CELL]
    streams = {s: seed(j, s) for s in ("path", "label")}
    z = c["guide"].sample(4096, path_seed=streams["path"], label_seed=streams["label"]).samples
    g, density = c["evaluator"].log_probability(z), c["gdensity"].log_q_over_p(z)
    values = [g-density]+[2*g-q.log_q_over_p(z)-density for q in c["qdensities"]]
    return {**j, "seeds": streams, "actual_work": 4096, "moments": [log_moments(v) for v in values],
            "q_digests": c["q_digests"], "wall": time.perf_counter()-begin,
            "guide_independent_kernel": False}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in ("guide-iid", "free-full", "envelope"):
        group = [r for r in records if r["method"] == method]
        expected = {"guide-iid": 320, "free-full": 69, "envelope": 52}[method]
        if len(group) != expected:
            raise ValueError("incomplete fixed-count validation grid")
        for j, target in enumerate(TARGETS):
            if method == "guide-iid":
                stats = merge_moments([r["moments"][j] for r in group])
                logs = [merge_moments([r["moments"][j] for r in group if r["rep"] == rep])["log_mean"] for rep in range(40)]
            else:
                logs = [r["log_estimand_values"][j] for r in group]
                stats = log_moments(torch.tensor(logs, dtype=torch.float64))
            result.append({"method": method, "target": target, "summary": stats,
                           "whole_or_equal_size_block_log_means": logs,
                           "sensitivity": sensitivity(logs, bootstrap_seed=umbrella_seed(NAMESPACE+method+target+"bootstrap")),
                           "guide_independent_kernel": method == "free-full"})
    return result


def decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    for target in TARGETS:
        group = {s["method"]: s for s in summaries if s["target"] == target}
        for method in ("guide-iid", "free-full", "envelope"):
            s = group[method]
            iid = method == "guide-iid"
            good = (s["summary"]["relative_se"] <= .015
                    and s["summary"]["maximum_fraction"] <= (.01 if iid else .15)
                    and s["sensitivity"]["maximum_leave_one_out_relative_shift"] <= .05
                    and (not iid or s["sensitivity"]["between_unit_relative_se"] <= 1.5*s["summary"]["relative_se"]))
            checks.append({"target": target, "method": method, "pass": good})
        pairs = [("guide-iid", "free-full")] if target == "mu" else [
            ("guide-iid", "free-full"), ("guide-iid", "envelope"), ("free-full", "envelope")]
        for a, b in pairs:
            agreements.append({"target": target, "methods": [a, b], "equivalence": relative_equivalence(
                group[a]["summary"], group[b]["summary"], comparisons=2 if target == "mu" else 30)})
    return {"high_eta_five_q_development_reference_pass": all(s["pass"] for s in checks)
            and all(s["equivalence"]["pass"] for s in agreements), "precision_checks": checks,
            "agreements": agreements, "p1_qualified": False, "p2_qualified": False,
            "large_training_authorized": False, "scope": "high_eta_five_original_qs_finite_N32_development_only"}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    config_input = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(config_input)
    log_weights = {CELL: anchor["config"]["log_mixture_weights"][CELL]}
    config = {"input": config_input, "anchor": ANCHOR, "anchor_sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest(),
              "cell": CELL, "log_mixture_weights": log_weights, "iid_blocks": 40, "iid_per_block": 32768,
              "smc_wholes": {"free-full": 69, "envelope": 52}, "scope": "scoped_high_eta_five_q_reference_validation"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    streams = [seed(j, s) for j in grid for s in (("path", "label") if j["method"] == "guide-iid" else ("whole",))]
    if len(grid) != 441 or len(streams) != len(set(streams)):
        raise ValueError("unexpected grid/seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/config_input["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"F": {"work": 60000000, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    cells: dict[str, Any] = {}
    def preparation():
        cells.update(prepare(metadata, rows, log_weights))
        c = cells[CELL]
        guide = build_volterra_excursion_guide(c["evaluator"].problem, **config_input["guide"])
        c.update(guide=guide, gdensity=CachedShiftDensity.create(guide))
        return {"actual_work": 128}
    journal.unit("preparation", phase="F", work=1024, maximum_wall=30., operation=preparation)
    records = []
    for j in grid:
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            return unit_values(job, cells)
        calls = 4096 if j["method"] == "guide-iid" else umbrella_work(j)
        records.append(journal.unit(identity(j), phase="F", work=calls,
                                    maximum_wall=5. if j["method"] == "guide-iid" else 30., operation=operation))
        if len(records) % 64 == 0:
            print(f"high-eta five-q validation {len(records)}/{len(grid)}", flush=True)
    summaries = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "decision": decision(summaries), "units": len(grid), "work": journal.accounting("F")[0],
              "phase_wall": time.time()-journal._read(directory/"phase-F.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-F.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["anchor"] != ANCHOR or hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != config["anchor_sha256"]
            or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()
            or manifest["expected_units"] != [identity(j) for j in jobs()]):
        raise ValueError("changed source/input/anchor/grid")
    anchor = json.loads((ROOT/ANCHOR).read_text())
    weights = {CELL: anchor["config"]["log_mixture_weights"][CELL]}
    if config["log_mixture_weights"] != weights:
        raise ValueError("changed weights")
    metadata, rows = inputs(config["input"])
    cells = prepare(metadata, rows, weights)
    c = cells[CELL]
    c["guide"] = build_volterra_excursion_guide(c["evaluator"].problem, **config["input"]["guide"])
    c["gdensity"] = CachedShiftDensity.create(c["guide"])
    journal = RecoveryJournal(path.parent, manifest)
    records, replayed_draws, replayed_wholes, audit_work = [], 0, 0, 128
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed unit")
        record = finish["result"]
        iid = j["method"] == "guide-iid"
        if (any(record[k] != v for k, v in j.items()) or record["q_digests"] != c["q_digests"]
                or record["actual_work"] != (4096 if iid else umbrella_work(j))
                or record["guide_independent_kernel"] != (j["method"] == "free-full")):
            raise ValueError("changed unit binding")
        if iid:
            if record["seeds"] != {s: seed(j, s) for s in ("path", "label")}:
                raise ValueError("changed IID seeds")
        else:
            if record["seed"] != seed(j, "whole") or record["log_mixture_weights"] != weights[CELL]:
                raise ValueError("changed whole seed/weights")
            vector = torch.tensor(record["log_scaled_component_normalizers"], dtype=torch.float64)
            if abs(float(torch.logsumexp(vector+torch.tensor(weights[CELL], dtype=torch.float64), 0))-record["log_normalizer"]) > 1e-10:
                raise ValueError("broken unnormalized family identity")
            correction = torch.tensor([0.]+[math.log(d) for d in c["deltas"]], dtype=torch.float64)
            if (vector-correction).tolist() != record["log_estimand_values"]:
                raise ValueError("incorrect scaled-risk component correction")
        records.append(record)
        if (iid and j["rep"] in (0, 39) and j["batch"] == 0) or (not iid and j["rep"] == 0):
            replay = unit_values(j, cells)
            fields = ("moments",) if iid else ("log_normalizer", "log_estimand_values", "stages")
            if any(replay[k] != record[k] for k in fields):
                raise ValueError("selected numerical replay mismatch")
            replayed_draws += 4096 if iid else 0
            replayed_wholes += int(not iid)
            audit_work += replay["actual_work"]
    summaries = summarize(records)
    if (summaries != output["summaries"] or decision(summaries) != output["decision"]
            or journal.accounting("F")[0] != output["work"] or output["work"] != 49685632 or output["units"] != 441):
        raise ValueError("changed summary/decision/accounting")
    if (path.parent/"phase-F.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_anchor_q_seed_moment_identity_arithmetic_replay_pass_not_continuum_certificate",
            "replayed_draws": replayed_draws, "replayed_wholes": replayed_wholes, "audit_potential_calls": audit_work,
            "decision": output["decision"]["high_eta_five_q_development_reference_pass"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "phase_wall", "decision", "units")}))
