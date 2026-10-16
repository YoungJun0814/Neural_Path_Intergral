"""Exact implicit static-shift paired fine-grid IID reference pilot."""

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
from src.path_integral.fft_static_shift_reference import FFTStaticShiftReference
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, merge_moments, sensitivity

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "results/post_audit/p2_paired_fine_extension_v1/session_20261010/result.json"
PLAN = "docs/plans/FFT_STATIC_FINE_REFERENCE_PILOT_2026-10-16_KO.md"
DIRECTORY = "results/post_audit/fft_static_fine_reference_v1/session_20261016"
CHUNK, DRAW_COUNT, WORK = 512, 32768, 65536


def with_log_maximum(moment: dict[str, Any]) -> dict[str, Any]:
    """Restore metadata without changing raw mean/variance moments."""
    if "log_maximum" in moment:
        return dict(moment)
    fraction = moment["maximum_fraction"]
    if not math.isfinite(fraction) or not 0 < fraction <= 1 or not math.isfinite(moment["log_sum"]):
        raise ValueError("invalid saved maximum contribution metadata")
    return {**moment, "log_maximum": moment["log_sum"]+math.log(fraction)}


def merge_retaining_maximum(parts: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [with_log_maximum(p) for p in parts]
    merged = merge_moments(complete)
    return {**merged, "log_maximum": max(p["log_maximum"] for p in complete)}


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "rep": r} for c in CELLS for r in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['rep']}"


def seed(j: dict[str, Any], chunk: int, role: str) -> int:
    return int(hashlib.sha256(f"fft-static-fine-reference-v1/{identity(j)}/{chunk}/{role}".encode()).hexdigest()[:15], 16)


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for c in config["cells"]:
        p = _problem(config["model"], c, 4096)
        out[c["id"]] = (FFTStaticShiftReference.create(p, **config["guide"]), PairedVolterraProbability.create(p))
    return out


def unit_values(j: dict[str, Any], prepared: Any) -> dict[str, Any]:
    begin = time.perf_counter()
    guide, pair = prepared
    fine, coarse = [], []
    for chunk in range(DRAW_COUNT//CHUNK):
        z = guide.sample(CHUNK, path_seed=seed(j, chunk, "path"), label_seed=seed(j, chunk, "label"))
        epsilon = torch.randn((CHUNK, 2048), dtype=torch.float64, generator=torch.Generator().manual_seed(seed(j, chunk, "augmentation")))
        values = pair.log_components(torch.cat((z, epsilon), 1))-guide.log_q_over_p(z)[:, None]
        fine.append(log_moments(values[:, 0]))
        coarse.append(log_moments(values[:, 1]))
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("chunk RSS guard")
    return {**j, "actual_work": WORK, "density_evaluations": DRAW_COUNT,
            "time_step_forward_evaluations": DRAW_COUNT*6144,
            "moments": [merge_retaining_maximum(fine), merge_retaining_maximum(coarse)], "wall": time.perf_counter()-begin,
            "seed_digest": hashlib.sha256(json.dumps([seed(j, k, role) for k in range(64) for role in ("path", "label", "augmentation")]).encode()).hexdigest()}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) != 16 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete fixed IID block grid required")
    rows = []
    for c in CELLS:
        group = sorted((r for r in records if r["cell"] == c), key=lambda r: r["rep"])
        logs = torch.tensor([[r["moments"][k]["log_mean"] for k in (0, 1)] for r in group], dtype=torch.float64)
        rows.append({"cell": c,
                     "components": [{"grid": name, "summary": merge_retaining_maximum([r["moments"][k] for r in group]),
                                     "block_sensitivity": sensitivity(logs[:, k].tolist(), bootstrap_seed=seed({"cell": c, "rep": 1000}, k, "bootstrap"))}
                                    for k, name in enumerate(("fine", "coarse"))],
                     "paired_difference": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=2),
                     "scope": "fresh_finite_grid_static_IID_pilot_not_continuum_or_tail_certificate"})
    return rows


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    config = {"model": anchor["config"]["model"], "cells": anchor["config"]["cells"],
              "guide": {"amplitudes": [2., 4., 6., 8.], "price_amplitudes": [0., 2., 4.], "defensive_mass": .1},
              "anchor": {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()}}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    seeds = [seed(j, k, role) for j in grid for k in range(64) for role in ("path", "label", "augmentation")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "limits": {"I": {"work": 1200000, "wall": 7200.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}
    def preparation():
        prepared.update(prepare(config))
        return {"actual_work": 0}
    journal.unit("preparation", phase="I", work=0, maximum_wall=30., operation=preparation)
    records = []
    for j in grid:
        def operation(job: dict[str, Any] = j) -> dict[str, Any]:
            return unit_values(job, prepared[job["cell"]])
        records.append(journal.unit(identity(j), phase="I", work=WORK, maximum_wall=240.,
                                    operation=operation))
        print(f"FFT static fine reference {len(records)}/16", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records),
              "units": 16, "work": journal.accounting("I")[0], "density_evaluations": 16*DRAW_COUNT,
              "time_step_forward_evaluations": 16*DRAW_COUNT*6144,
              "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-I.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-I.clock.json")["started_unix"] > 7200:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if "recovery_metadata" in output:
        recovery = output["recovery_metadata"]
        audit_snapshot(ROOT, recovery["source"], recovery["config"])
        for relative, digest in recovery["config"]["immutable_files"].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
                raise ValueError("changed original recovery evidence")
    if (config["anchor"] != {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()}
            or manifest["config"] != config or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["expected_units"] != [identity(j) for j in jobs()]):
        raise ValueError("changed source/anchor/grid")
    prepared, journal = prepare(config), RecoveryJournal(path.parent, manifest)
    records, replayed = [], 0
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed block")
        r = finish["result"]
        digest = hashlib.sha256(json.dumps([seed(j, k, role) for k in range(64) for role in ("path", "label", "augmentation")]).encode()).hexdigest()
        if (any(r[k] != v for k, v in j.items()) or r["seed_digest"] != digest or r["actual_work"] != WORK
                or r["density_evaluations"] != DRAW_COUNT or r["time_step_forward_evaluations"] != DRAW_COUNT*6144):
            raise ValueError("changed block binding/cost")
        if j["rep"] == 0:
            replay = unit_values(j, prepared[j["cell"]])
            for current, original in zip(replay["moments"], r["moments"], strict=True):
                if ({k: current[k] for k in original} != original
                        or abs(current["log_maximum"]-with_log_maximum(original)["log_maximum"]) > 1e-10):
                    raise ValueError("numerical block replay mismatch")
            replayed += 1
        records.append(r)
    if (summarize(records) != output["summaries"] or output["units"] != 16
            or journal.accounting("I")[0] != output["work"] or output["work"] != 1048576
            or output["density_evaluations"] != 524288 or output["time_step_forward_evaluations"] != 3221225472
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
            or (path.parent/"phase-I.violation.json").exists() or (path.parent/"final_io_failure.json").exists()):
        raise ValueError("changed summaries/cost/gates or operational failure")
    return {"status": "static_fine_joint_law_source_seed_moment_cost_replay_pass", "replayed_blocks": replayed,
            "audit_physical_evaluations": replayed*WORK, "audit_density_evaluations": replayed*DRAW_COUNT}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("units", "work", "phase_wall")}))
