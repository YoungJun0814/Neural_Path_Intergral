"""Bounded full versus exactly driver-conditioned probability SMC pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.driver_conditioned_terminal import DriverConditionedTerminal
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/ALL_DRIVER_CONDITIONING_PROBABILITY_PILOT_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/driver_conditioned_probability_v1/session_20261010"
WORK = 254976


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "method": m, "rep": r} for c in CELLS for m in ("full", "driver-conditioned") for r in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['method']}-{j['rep']}"


def seed(j: dict[str, Any]) -> int:
    return int(hashlib.sha256(("driver-conditioned-probability-v1/"+identity(j)).encode()).hexdigest()[:15], 16)


def prepare(config: dict[str, Any]) -> dict[tuple[str, str], Any]:
    return {(c["id"], m): cls.create(_problem(config["model"], c, 32)) for c in config["cells"]
            for m, cls in (("full", CachedVolterraTerminal), ("driver-conditioned", DriverConditionedTerminal))}


def unit_values(j: dict[str, Any], evaluator: Any) -> dict[str, Any]:
    begin = time.perf_counter()
    dimension = 64 if j["method"] == "full" else evaluator.dimension
    cfg = WeightedSMCConfig(1024, tuple((k/63)**2 for k in range(64)), 4, .35, 1, seed(j),
                            resample_every=4, resampling_scheme="stratified")
    r = estimate_gradient_umbrella_normalizers(lambda z: evaluator.log_probability(z)[:, None], dimension=dimension,
                                               log_mixture_weights=torch.zeros(1, dtype=torch.float64),
                                               config=cfg, maximum_drift_norm=20.)
    return {**j, "seed": seed(j), "dimension": dimension, "actual_work": r.potential_evaluations,
            "backward_evaluations": r.gradient_evaluations, "log_normalizer": r.log_normalizer,
            "log_components": r.log_component_normalizers.tolist(), "stages": list(r.stages),
            "wall": time.perf_counter()-begin, "guide_independent_kernel": True}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 32 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete fixed comparison required")
    rows = []
    for c in CELLS:
        for m in ("full", "driver-conditioned"):
            group = sorted((r for r in records if r["cell"] == c and r["method"] == m), key=lambda r: r["rep"])
            logs = [r["log_normalizer"] for r in group]
            rows.append({"cell": c, "method": m, "dimension": group[0]["dimension"],
                         "summary": log_moments(torch.tensor(logs, dtype=torch.float64)),
                         "sensitivity": sensitivity(logs, bootstrap_seed=seed({"cell": c, "method": m, "rep": 1000})),
                         "initial_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in group]})
    agreements = []
    for c in CELLS:
        group = [r for r in rows if r["cell"] == c]
        agreements.append({"cell": c, "equivalence": relative_equivalence(group[0]["summary"], group[1]["summary"], comparisons=2)})
    return {"rows": rows, "agreements": agreements, "scope": "same_finite_grid_mu_only_development_pilot",
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def check_unit(j: dict[str, Any], r: dict[str, Any]) -> None:
    if (any(r[k] != v for k, v in j.items()) or r["seed"] != seed(j)
            or r["actual_work"] != WORK or r["backward_evaluations"] != WORK
            or r["dimension"] != (64 if j["method"] == "full" else 31)
            or not r["guide_independent_kernel"] or r["log_components"] != [r["log_normalizer"]]):
        raise ValueError("changed whole binding/cost/endpoint")


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, source_rows = inputs(input_config)
    config = {"input": input_config, "model": metadata["config"]["model"],
              "cells": [next(r["cell"] for r in source_rows if r["cell"]["id"] == c) for c in CELLS]}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    if len({seed(j) for j in grid}) != 32:
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"D": {"work": 10000000, "wall": 1800.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}
    def preparation():
        prepared.update(prepare(config))
        return {"actual_work": 0}
    journal.unit("preparation", phase="D", work=0, maximum_wall=30., operation=preparation)
    records = []
    for j in grid:
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard")
        def operation(job=j):
            r = unit_values(job, prepared[(job["cell"], job["method"])])
            check_unit(job, r)
            return r
        records.append(journal.unit(identity(j), phase="D", work=WORK, maximum_wall=30., operation=operation))
        if len(records) % 8 == 0:
            print(f"driver-conditioned probability pilot {len(records)}/32", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records),
              "units": 32, "work": journal.accounting("D")[0], "backward_evaluations": 32*WORK,
              "phase_wall": time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (manifest["source"] != output["source"]["snapshot_sha256"] or manifest["config"] != config
            or manifest["expected_units"] != [identity(j) for j in jobs()]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()):
        raise ValueError("changed source/input/grid")
    prepared, journal = prepare(config), RecoveryJournal(path.parent, manifest)
    records, replayed = [], 0
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed whole")
        r = finish["result"]
        check_unit(j, r)
        if j["rep"] == 0:
            replay = unit_values(j, prepared[(j["cell"], j["method"])])
            if any(replay[k] != r[k] for k in ("log_normalizer", "log_components", "stages")):
                raise ValueError("whole numerical replay mismatch")
            replayed += 1
        records.append(r)
    if (summarize(records) != output["summaries"] or output["units"] != 32 or output["work"] != 32*WORK
            or journal.accounting("D")[0] != output["work"] or output["backward_evaluations"] != 32*WORK
            or (path.parent/"phase-D.violation.json").exists() or (path.parent/"final_io_failure.json").exists()):
        raise ValueError("changed summaries/cost or operational failure")
    return {"status": "same_law_reduced_driver_source_seed_endpoint_cost_replay_pass",
            "replayed_wholes": replayed, "audit_forward_evaluations": WORK*replayed,
            "audit_backward_evaluations": WORK*replayed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("units", "work", "phase_wall")}))
