"""Fresh fixed guide-independent pCN versus multiscale block-pCN pilot."""

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
from src.path_integral.guide_free_block_smc import estimate_guide_free_block_normalizer
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, precision_count, sensitivity
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/GUIDE_FREE_MULTISCALE_BLOCK_RECOVERY_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/guide_free_block_recovery_v1/session_20261010"
CALLS = 508928


def seed(identity: str) -> int:
    return int(hashlib.sha256(("guide-free-block-v1/"+identity).encode()).hexdigest()[:15], 16)


def jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"cell": r["cell"]["id"], "method": method, "target": target, "rep": rep}
            for r in rows if r["parent_training_rep"] == 0
            for method in ("full-pcn", "multiscale-block-pcn") for target in ("mu", "risk") for rep in range(8)]


def identity(job: dict[str, Any]) -> str:
    return "guide-free-block-v1-"+job["cell"]+"-"+job["method"]+"-"+job["target"]+f"-{job['rep']}"


def prepare(metadata: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    p = _problem(metadata["config"]["model"], row["cell"], 32)
    q = proposal_from_parameters(row["q_parameters"])
    return {"evaluator": CachedVolterraTerminal.create(p), "qdensity": CachedShiftDensity.create(q),
            "delta": q.defensive_mass, "dimension": p.local_dimension}


def unit_values(job: dict[str, Any], cache: dict[str, Any], q_digest: str) -> dict[str, Any]:
    start = time.perf_counter()
    delta = cache["delta"]
    def potential(z):
        g = cache["evaluator"].log_probability(z)
        return g if job["target"] == "mu" else log_risk_potential(g, cache["qdensity"].log_q_over_p(z), defensive_mass=delta)
    cfg = WeightedSMCConfig(1024, tuple((k/63)**2 for k in range(64)), 8, .35, 1, seed(identity(job)),
                           resample_every=4, resampling_scheme="stratified")
    result = estimate_guide_free_block_normalizer(potential, dimension=cache["dimension"], config=cfg,
                                                 use_blocks=job["method"] == "multiscale-block-pcn", block_scale=.7)
    if result.potential_evaluations != CALLS:
        raise ValueError("unexpected potential work")
    return {**job, "q_digest": q_digest, "seed": seed(identity(job)), "actual_work": CALLS,
            "log_normalizer": result.log_normalizer-(math.log(delta) if job["target"] == "risk" else 0.),
            "stages": list(result.stages), "wall": time.perf_counter()-start}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for cell in sorted({r["cell"] for r in records}):
        for method in ("full-pcn", "multiscale-block-pcn"):
            for target in ("mu", "risk"):
                group = [r for r in records if (r["cell"], r["method"], r["target"]) == (cell, method, target)]
                if len(group) != 8 or {r["rep"] for r in group} != set(range(8)):
                    raise ValueError("incomplete whole-replication grid")
                logs = [r["log_normalizer"] for r in group]
                stats = log_moments(torch.tensor(logs, dtype=torch.float64))
                wall = sum(r["wall"] for r in group)
                summaries.append({"cell": cell, "method": method, "target": target, "summary": stats,
                    "sensitivity": sensitivity(logs, bootstrap_seed=seed(cell+method+target+"bootstrap")),
                    "whole_replication_log_means": logs, "wall": wall,
                    "observed_cv2_wall_per_whole": stats["count"]*stats["relative_se"]**2*wall/len(group),
                    "allocation_forecast": precision_count(stats, target_rse=.015, safety_factor=2,
                                                            minimum=32, maximum=256, multiple=1),
                    "terminal_initial_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in group],
                    "terminal_ess": [r["stages"][-1]["ess_before_resampling"] for r in group],
                    "scope": "guide_independent_eight_whole_development_not_precision_certificate"})
    return summaries


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"input": input_config, "steps": 32, "particles": 1024, "levels": 64, "mutations": 8,
              "full_scale": .35, "block_scale": .7, "blocks": [2, 8, 32, 64], "whole_repetitions": 8,
              "scope": "guide_free_parent0_two_cell_pilot"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs(rows)
    if len(grid) != 64 or len({seed(identity(j)) for j in grid}) != len(grid):
        raise ValueError("unexpected grid/seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "q_digests": {r["cell"]["id"]: r["q_digest"] for r in rows if r["parent_training_rep"] == 0},
                "limits": {"G": {"work": 40000000, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    caches: dict[str, Any] = {}
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        def preparation(r=row):
            caches[r["cell"]["id"]] = prepare(metadata, r)
            return {"actual_work": 0}
        journal.unit(row["cell"]["id"]+"-prepare", phase="G", work=0, maximum_wall=30., operation=preparation)
    records = []
    for j in grid:
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            return unit_values(job, caches[job["cell"]], manifest["q_digests"][job["cell"]])
        records.append(journal.unit(identity(j), phase="G", work=CALLS, maximum_wall=30., operation=operation))
        if len(records) % 8 == 0:
            print(f"guide-free block pilot {len(records)}/{len(grid)}", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records),
              "work": journal.accounting("G")[0], "units": len(grid), "p1_qualified": False,
              "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-G.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-G.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary and final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()):
        raise ValueError("changed source/input binding")
    metadata, rows = inputs(config["input"])
    grid = jobs(rows)
    if manifest["expected_units"] != [identity(j) for j in grid]:
        raise ValueError("changed grid")
    caches = {r["cell"]["id"]: prepare(metadata, r) for r in rows if r["parent_training_rep"] == 0}
    digests = {r["cell"]["id"]: r["q_digest"] for r in rows if r["parent_training_rep"] == 0}
    if manifest["q_digests"] != digests:
        raise ValueError("changed frozen q")
    journal = RecoveryJournal(path.parent, manifest)
    records, replayed = [], 0
    for j in grid:
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed unit cannot be used")
        record = finish["result"]
        if (any(record[k] != v for k, v in j.items()) or record["q_digest"] != digests[j["cell"]]
                or record["seed"] != seed(identity(j)) or record["actual_work"] != CALLS):
            raise ValueError("changed unit binding")
        records.append(record)
        if j["rep"] == 0:
            replay = unit_values(j, caches[j["cell"]], digests[j["cell"]])
            if replay["log_normalizer"] != record["log_normalizer"] or replay["stages"] != record["stages"]:
                raise ValueError("whole-run numerical replay mismatch")
            replayed += 1
    if (summarize(records) != output["summaries"] or journal.accounting("G")[0] != output["work"]
            or output["work"] != len(grid)*CALLS or output["units"] != len(grid)
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("changed summary/accounting/qualification")
    if (path.parent/"phase-G.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_q_seed_arithmetic_whole_replay_pass_not_precision_certificate",
            "units": len(grid), "replayed_wholes": replayed, "audit_potential_calls": replayed*CALLS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "phase_wall", "p1_qualified", "units")}))
