"""Fixed N4096 bridge/particle allocation pilot; no accuracy authorization."""

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

from experiments.post_audit_p2_paired_gradient import CELLS
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
ANCHOR = "results/post_audit/p2_paired_fine_extension_v1/session_20261010/result.json"
PLAN = "docs/plans/POST_RECOVERY_NEXT_PRIORITIES_2026-10-16_KO.md"
DIRECTORY = "results/post_audit/p2_fine_resource_v1/session_20261016"
DESIGNS = {"more-temperatures": (128, 128), "more-particles": (256, 64)}


def jobs(namespace: str | None = None) -> list[dict[str, Any]]:
    grid = [{"cell": c, "design": d, "rep": r} for c in CELLS for d in DESIGNS for r in range(8)]
    return grid if namespace is None else [{**j, "seed_namespace": namespace} for j in grid]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['design']}-{j['rep']}"


def seed(j: dict[str, Any]) -> int:
    return int(hashlib.sha256((j.get("seed_namespace", "paired-fine-resources-v1")+"/"+identity(j)).encode()).hexdigest()[:15], 16)


def joint_work(j: dict[str, Any]) -> int:
    p, points = DESIGNS[j["design"]]
    return p*(1+4*(points-2))


def prepare(config: dict[str, Any]) -> dict[str, PairedVolterraProbability]:
    return {c["id"]: PairedVolterraProbability.create(_problem(config["model"], c, 4096)) for c in config["cells"]}


def unit_values(j: dict[str, Any], paired: PairedVolterraProbability) -> dict[str, Any]:
    begin = time.perf_counter()
    particles, points = DESIGNS[j["design"]]
    cfg = WeightedSMCConfig(particles, tuple((k/(points-1))**2 for k in range(points)), 4, .35, 1, seed(j),
                            resample_every=4, resampling_scheme="stratified")
    r = estimate_gradient_umbrella_normalizers(paired.log_components, dimension=paired.dimension,
                                               log_mixture_weights=torch.full((2,), .5, dtype=torch.float64).log(),
                                               config=cfg, maximum_drift_norm=20.)
    return {**j, "seed": seed(j), "actual_work": 2*r.potential_evaluations,
            "torch_threads": torch.get_num_threads(),
            "joint_forward_evaluations": r.potential_evaluations, "joint_backward_evaluations": r.gradient_evaluations,
            "time_step_forward_evaluations": r.potential_evaluations*6144,
            "joint_dimension": paired.dimension, "log_normalizer": r.log_normalizer,
            "log_component_normalizers": r.log_component_normalizers.tolist(), "stages": list(r.stages),
            "wall": time.perf_counter()-begin, "guide_independent_kernel": True}


def check_unit(j: dict[str, Any], r: dict[str, Any]) -> None:
    expected = joint_work(j)
    if (any(r[k] != v for k, v in j.items()) or r["seed"] != seed(j) or r["actual_work"] != 2*expected
            or r["joint_forward_evaluations"] != expected or r["joint_backward_evaluations"] != expected
            or r["time_step_forward_evaluations"] != expected*6144 or r["joint_dimension"] != 10240
            or not r["guide_independent_kernel"]):
        raise ValueError("changed paired-grid law/binding/cost")
    v = torch.tensor(r["log_component_normalizers"], dtype=torch.float64)
    if not torch.isfinite(v).all() or abs(float(torch.logsumexp(v-math.log(2), 0))-r["log_normalizer"]) > 1e-10:
        raise ValueError("broken unnormalized component identity")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 32 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete fixed resource grid required")
    namespaces = {r.get("seed_namespace", "paired-fine-resources-v1") for r in records}
    if len(namespaces) != 1:
        raise ValueError("mixed protocol namespaces")
    namespace = next(iter(namespaces))
    rows: list[dict[str, Any]] = []
    for c in CELLS:
        for design in DESIGNS:
            group = sorted((r for r in records if r["cell"] == c and r["design"] == design), key=lambda r: r["rep"])
            logs = torch.tensor([r["log_component_normalizers"] for r in group], dtype=torch.float64)
            components: list[dict[str, Any]] = [{"grid": name, "summary": log_moments(logs[:, k]),
                           "sensitivity": sensitivity(logs[:, k].tolist(), bootstrap_seed=seed({"cell": c+name, "design": design, "rep": 1000, "seed_namespace": namespace}))}
                          for k, name in enumerate(("fine", "coarse"))]
            worst = max(r["summary"]["relative_se"] for r in components)
            required = max(32, math.ceil(3*8*(worst/.015)**2))
            work = joint_work({"design": design})
            rows.append({"cell": c, "design": design, "particles": DESIGNS[design][0], "temperature_points": DESIGNS[design][1],
                         "components": components, "paired_difference": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=4),
                         "initial_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in group],
                         "fresh_precision_forecast": {"target_rse": .015, "safety": 3, "required_wholes": required,
                                                      "physical_forward": 2*work*required, "joint_backward": work*required,
                                                      "time_step_forward": 6144*work*required,
                                                      "wall_shared_laptop_seconds": required*sum(r["wall"] for r in group)/8},
                         "scope": "pilot_allocation_forecast_not_actual_precision_or_continuum_certificate"})
    selection = {c: min((r for r in rows if r["cell"] == c),
                        key=lambda r: (r["fresh_precision_forecast"]["time_step_forward"], r["design"]))["design"] for c in CELLS}
    return {"rows": rows, "selected_development_design": selection,
            "selection_rule": "lowest_safety3_timestep_forecast_per_cell_before_fresh_confirmation",
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run(*, directory_name: str = DIRECTORY, namespace: str | None = None,
        phase_wall: float = 7200., unit_wall: float = 240., plan: str = PLAN,
        failure_anchor: str | None = None, torch_threads: int = 1,
        backend_anchor: str | None = None) -> dict[str, Any]:
    if (isinstance(torch_threads, bool) or not isinstance(torch_threads, int) or torch_threads not in (1, 2, 4, 8)
            or (namespace is None and torch_threads != 1)):
        raise ValueError("frozen versioned CPU backend required")
    torch.set_num_threads(torch_threads)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    config: dict[str, Any] = {"model": anchor["config"]["model"], "cells": anchor["config"]["cells"], "designs": DESIGNS,
              "anchor": {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()},
              "backward_cap": 2500000}
    extra: tuple[str, ...] = (plan, ANCHOR)
    if namespace is not None:
        config.update({"seed_namespace": namespace, "phase_wall": phase_wall, "unit_wall": unit_wall,
                       "torch_threads": torch_threads})
    if backend_anchor is not None:
        backend = json.loads((ROOT/backend_anchor).read_text())
        audit_snapshot(ROOT, backend["source"], backend["config"])
        if backend["selected_threads"] != torch_threads or not all(r["numerical_equivalence_pass"] for r in backend["rows"]):
            raise ValueError("backend calibration mismatch")
        config["backend_anchor"] = {"path": backend_anchor, "sha256": hashlib.sha256((ROOT/backend_anchor).read_bytes()).hexdigest()}
        extra += (backend_anchor,)
    if failure_anchor is not None:
        failed = json.loads((ROOT/failure_anchor).read_text())
        if failed["payload"]["status"] != "protocol_failure":
            raise ValueError("operational failure anchor required")
        config["failure_anchor"] = {"path": failure_anchor, "sha256": hashlib.sha256((ROOT/failure_anchor).read_bytes()).hexdigest()}
        extra += (failure_anchor,)
    # JSON round trip fixes tuple/list representation before all manifest hashes.
    config = json.loads(json.dumps(config))
    directory = ROOT/directory_name
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=extra)
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs(namespace)
    if len({seed(j) for j in grid}) != 32:
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "limits": {"R": {"work": 5000000, "wall": phase_wall}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}
    def preparation():
        prepared.update(prepare(config))
        return {"actual_work": 0}
    journal.unit("preparation", phase="R", work=0, maximum_wall=30., operation=preparation)
    records, backward = [], 0
    for j in grid:
        if backward+joint_work(j) > config["backward_cap"]:
            raise RuntimeError("backward reservation exhausted")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard")
        def operation(job=j):
            r = unit_values(job, prepared[job["cell"]])
            check_unit(job, r)
            if r["torch_threads"] != torch_threads:
                raise ValueError("backend changed during whole")
            if psutil.Process().memory_info().rss > 4*1024**3:
                raise MemoryError("post-unit RSS guard")
            return r
        records.append(journal.unit(identity(j), phase="R", work=2*joint_work(j), maximum_wall=unit_wall, operation=operation))
        backward += joint_work(j)
        print(f"fine-grid resource pilot {len(records)}/32", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records),
              "units": len(records), "work": journal.accounting("R")[0], "joint_backward_evaluations": backward,
              "time_step_forward_evaluations": sum(r["time_step_forward_evaluations"] for r in records),
              "phase_wall": time.time()-journal._read(directory/"phase-R.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-R.clock.json")["started_unix"] > phase_wall:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    torch.set_num_threads(config.get("torch_threads", 1))
    audit_snapshot(ROOT, output["source"], config)
    grid = jobs(config.get("seed_namespace"))
    if (config["anchor"] != {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()}
            or manifest["config"] != config or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["expected_units"] != [identity(j) for j in grid]):
        raise ValueError("changed source/anchor/grid")
    if "failure_anchor" in config:
        binding = config["failure_anchor"]
        if hashlib.sha256((ROOT/binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError("changed operational failure anchor")
    if "backend_anchor" in config:
        binding = config["backend_anchor"]
        if hashlib.sha256((ROOT/binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError("changed backend anchor")
        backend = json.loads((ROOT/binding["path"]).read_text())
        audit_snapshot(ROOT, backend["source"], backend["config"])
        if backend["selected_threads"] != config["torch_threads"] or not all(r["numerical_equivalence_pass"] for r in backend["rows"]):
            raise ValueError("backend equivalence binding failed")
    prepared, journal = prepare(config), RecoveryJournal(path.parent, manifest)
    records, audit_work, audit_back = [], 0, 0
    for j in grid:
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed whole")
        r = finish["result"]
        check_unit(j, r)
        if r["torch_threads"] != config.get("torch_threads", 1):
            raise ValueError("changed unit backend")
        if j["rep"] == 0:
            replay = unit_values(j, prepared[j["cell"]])
            if any(replay[k] != r[k] for k in ("log_normalizer", "log_component_normalizers", "stages")):
                raise ValueError("whole numerical replay mismatch")
            audit_work += replay["actual_work"]
            audit_back += replay["joint_backward_evaluations"]
        records.append(r)
    if (summarize(records) != output["summaries"] or output["units"] != 32
            or journal.accounting("R")[0] != output["work"] or output["work"] != 4108288
            or output["joint_backward_evaluations"] != 2054144
            or output["time_step_forward_evaluations"] != 12620660736
            or (path.parent/"phase-R.violation.json").exists() or (path.parent/"final_io_failure.json").exists()):
        raise ValueError("changed summaries/cost or operational failure")
    return {"status": "fine_resource_source_anchor_seed_identity_cost_replay_pass",
            "replayed_wholes": 4, "audit_physical_forward_evaluations": audit_work,
            "audit_joint_backward_evaluations": audit_back}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("units", "work", "phase_wall")}))
