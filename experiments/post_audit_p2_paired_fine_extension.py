"""Fresh finer-grid paired SMC pilot; never authorizes continuum or training."""

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

from experiments.post_audit_p2_paired_gradient import CELLS, JOINT_WORK
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, sensitivity
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "results/post_audit/p2_paired_gradient_v1/session_20261010/result.json"
PLAN = "docs/plans/P2_PAIRED_GRADIENT_FINE_EXTENSION_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/p2_paired_fine_extension_v1/session_20261010"
GRIDS = (2048, 4096)


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "fine_steps": n, "rep": r} for c in CELLS for n in GRIDS for r in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['fine_steps']}-{j['rep']}"


def seed(j: dict[str, Any]) -> int:
    return int(hashlib.sha256(("paired-fine-gradient-v1/"+identity(j)).encode()).hexdigest()[:15], 16)


def prepare(config: dict[str, Any]) -> dict[tuple[str, int], PairedVolterraProbability]:
    return {(c["id"], n): PairedVolterraProbability.create(_problem(config["model"], c, n))
            for c in config["cells"] for n in GRIDS}


def unit_values(j: dict[str, Any], paired: PairedVolterraProbability) -> dict[str, Any]:
    begin = time.perf_counter()
    cfg = WeightedSMCConfig(128, tuple((k/63)**2 for k in range(64)), 4, .35, 1, seed(j),
                            resample_every=4, resampling_scheme="stratified")
    r = estimate_gradient_umbrella_normalizers(paired.log_components, dimension=paired.dimension,
                                               log_mixture_weights=torch.full((2,), .5, dtype=torch.float64).log(),
                                               config=cfg, maximum_drift_norm=20.)
    return {**j, "seed": seed(j), "actual_work": 2*r.potential_evaluations,
            "joint_forward_evaluations": r.potential_evaluations, "joint_backward_evaluations": r.gradient_evaluations,
            "time_step_forward_evaluations": r.potential_evaluations*(j["fine_steps"]+j["fine_steps"]//2),
            "joint_dimension": paired.dimension, "log_normalizer": r.log_normalizer,
            "log_component_normalizers": r.log_component_normalizers.tolist(), "stages": list(r.stages),
            "wall": time.perf_counter()-begin, "guide_independent_kernel": True}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) != 32 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete fixed fresh grid required")
    out = []
    for cell in CELLS:
        for n in GRIDS:
            group = sorted((r for r in records if r["cell"] == cell and r["fine_steps"] == n), key=lambda r: r["rep"])
            logs = torch.tensor([r["log_component_normalizers"] for r in group], dtype=torch.float64)
            out.append({"cell": cell, "fine_steps": n,
                        "components": [{"grid": name, "summary": log_moments(logs[:, k]),
                                        "sensitivity": sensitivity(logs[:, k].tolist(), bootstrap_seed=seed({"cell": cell+name, "fine_steps": n, "rep": 200}))}
                                       for k, name in enumerate(("fine", "coarse"))],
                        "paired_difference": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=4),
                        "initial_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in group],
                        "scope": "finer_paired_finite_grid_development_not_continuum_certificate"})
    return out


def check_unit(j: dict[str, Any], r: dict[str, Any]) -> None:
    if (any(r[k] != v for k, v in j.items()) or r["seed"] != seed(j)
            or r["actual_work"] != 2*JOINT_WORK or r["joint_forward_evaluations"] != JOINT_WORK
            or r["joint_backward_evaluations"] != JOINT_WORK or not r["guide_independent_kernel"]
            or r["joint_dimension"] != 5*j["fine_steps"]//2
            or r["time_step_forward_evaluations"] != JOINT_WORK*(j["fine_steps"]+j["fine_steps"]//2)):
        raise ValueError("changed unit law/binding/cost")
    v = torch.tensor(r["log_component_normalizers"], dtype=torch.float64)
    if not torch.isfinite(v).all() or abs(float(torch.logsumexp(v-math.log(2), 0))-r["log_normalizer"]) > 1e-10:
        raise ValueError("broken paired unnormalized identity")


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    config = {"model": anchor["config"]["model"], "cells": anchor["config"]["cells"],
              "anchor": {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()},
              "fine_grids": list(GRIDS), "wholes": 8, "joint_backward_cap": 1200000}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    if len({seed(j) for j in grid}) != 32:
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "limits": {"F": {"work": 3000000, "wall": 4800.}}}
    journal = RecoveryJournal(directory, manifest)
    prepared: dict[tuple[str, int], PairedVolterraProbability] = {}
    def preparation():
        prepared.update(prepare(config))
        return {"actual_work": 0}
    journal.unit("preparation", phase="F", work=0, maximum_wall=30., operation=preparation)
    records, backward = [], 0
    for j in grid:
        if backward+JOINT_WORK > config["joint_backward_cap"]:
            raise RuntimeError("backward reservation exhausted")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard")
        def operation(job=j):
            r = unit_values(job, prepared[(job["cell"], job["fine_steps"])])
            check_unit(job, r)
            if psutil.Process().memory_info().rss > 4*1024**3:
                raise MemoryError("post-unit RSS guard")
            return r
        records.append(journal.unit(identity(j), phase="F", work=2*JOINT_WORK, maximum_wall=120., operation=operation))
        backward += JOINT_WORK
        if len(records) % 8 == 0:
            print(f"paired finer-grid diagnostic {len(records)}/32", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records),
              "units": len(records), "work": journal.accounting("F")[0], "joint_backward_evaluations": backward,
              "time_step_forward_evaluations": sum(r["time_step_forward_evaluations"] for r in records),
              "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-F.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-F.clock.json")["started_unix"] > 4800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["anchor"] != {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()}
            or manifest["source"] != output["source"]["snapshot_sha256"] or manifest["config"] != config
            or manifest["expected_units"] != [identity(j) for j in jobs()]):
        raise ValueError("changed source/anchor/grid")
    prepared, journal = prepare(config), RecoveryJournal(path.parent, manifest)
    records, replayed = [], 0
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed whole")
        r = finish["result"]
        check_unit(j, r)
        if j["rep"] == 0:
            replay = unit_values(j, prepared[(j["cell"], j["fine_steps"])])
            if any(replay[k] != r[k] for k in ("log_normalizer", "log_component_normalizers", "stages")):
                raise ValueError("whole numerical replay mismatch")
            replayed += 1
        records.append(r)
    if (summarize(records) != output["summaries"] or output["units"] != 32
            or journal.accounting("F")[0] != output["work"] or output["work"] != 2039808
            or output["joint_backward_evaluations"] != 1019904
            or output["time_step_forward_evaluations"] != 4699717632
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
            or (path.parent/"phase-F.violation.json").exists() or (path.parent/"final_io_failure.json").exists()):
        raise ValueError("changed summaries/cost/gates or operational failure")
    return {"status": "finer_joint_source_anchor_seed_identity_cost_replay_pass_not_continuum_certificate",
            "replayed_wholes": replayed, "audit_physical_forward_evaluations": 2*JOINT_WORK*replayed,
            "audit_joint_backward_evaluations": JOINT_WORK*replayed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("units", "work", "phase_wall", "joint_backward_evaluations")}))
