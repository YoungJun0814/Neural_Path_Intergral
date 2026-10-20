"""Fresh paired raw/marginal IS pilot; no independent-oracle qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.post_audit_bridge_mutation_allocation import load_anchors
from experiments.post_audit_fft_driver_marginal_calibration import (
    DIRECTORY as CALIBRATION_DIRECTORY,
)
from experiments.post_audit_fft_driver_marginal_calibration import GUIDE, PLAN, THEORY
from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.fft_driver_marginal_reference import FFTDriverMarginalReference
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, sensitivity

DIRECTORY = "results/post_audit/fft_driver_marginal_pilot_v1/session_20261020"
NAMESPACE = "fft-driver-marginal-pilot-v1"
DRAW_COUNT, CHUNK = 1024, 16
CALIBRATION = CALIBRATION_DIRECTORY+"/result.json"


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "rep": rep} for c in CELLS for rep in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['rep']}"


def seed(j: dict[str, Any], chunk: int, role: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}/{identity(j)}/{chunk}/{role}".encode()).hexdigest()[:15], 16)


def prepare(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared, details = {}, []
    for cell in config["original"]["cells"]:
        begin = time.perf_counter()
        p = _problem(config["original"]["model"], cell, 4096)
        guide = PhysicalWindowGuide.create(p, kind="mixed-window", **GUIDE)
        ref = FFTDriverMarginalReference.create(p, guide.spec)
        prepared[cell["id"]] = guide, ref, CachedVolterraTerminal.create(p)
        wall = time.perf_counter()-begin
        if wall > 180. or psutil.Process().memory_info().rss > 4*1024**3:
            raise RuntimeError("preparation time/RSS budget exceeded")
        details.append({"cell": cell["id"], "wall": wall, "iterations": ref.preparation_iterations,
                        "covariance_rows": ref.preparation_covariance_rows})
    return prepared, details


def unit_values(j: dict[str, Any], prepared: Any) -> dict[str, Any]:
    begin = time.perf_counter()
    guide, ref, raw = prepared
    parts: list[list[dict[str, Any]]] = [[], []]
    error, covariance_rows, iterations = 0., 0, 0
    with torch.no_grad():
        for chunk in range(DRAW_COUNT//CHUNK):
            z = guide.sample(CHUNK, path_seed=seed(j, chunk, "path"), label_seed=seed(j, chunk, "label"))
            marginal = ref.evaluate(ref.operator.project(z))
            raw_log = raw.log_probability(z)-guide.log_q_over_p(z)
            parts[0].append(log_moments(raw_log))
            parts[1].append(log_moments(marginal.log_contribution))
            error = max(error, float(marginal.log_contribution_error_upper.max()))
            covariance_rows += marginal.covariance_row_evaluations
            iterations = max(iterations, *marginal.solve_iterations)
            if psutil.Process().memory_info().rss > 4*1024**3:
                raise MemoryError("chunk RSS guard")
    return {**j, "actual_work": DRAW_COUNT, "raw_physical": DRAW_COUNT,
            "conditional_cdf_evaluations": DRAW_COUNT, "density_evaluations": 2*DRAW_COUNT,
            "covariance_rows": covariance_rows, "maximum_solve_iterations": iterations,
            "maximum_log_error_upper": error, "moments": [merge_retaining_maximum(p) for p in parts],
            "torch_threads": torch.get_num_threads(),
            "seed_digest": hashlib.sha256(json.dumps([seed(j, k, role) for k in range(64) for role in ("path", "label")]).encode()).hexdigest(),
            "wall": time.perf_counter()-begin}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 16 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete marginal pilot required")
    rows = []
    for c in CELLS:
        group = sorted((r for r in records if r["cell"] == c), key=lambda r: r["rep"])
        logs = torch.tensor([[r["moments"][k]["log_mean"] for k in (0, 1)] for r in group], dtype=torch.float64)
        components: list[dict[str, Any]] = [{"method": name,
            "summary": merge_retaining_maximum([r["moments"][k] for r in group]),
            "block_sensitivity": sensitivity(logs[:, k].tolist(), bootstrap_seed=seed({"cell": c, "rep": 1000}, k, "bootstrap"))}
            for k, name in enumerate(("raw", "driver-marginal"))]
        a, b = [r["summary"] for r in components]
        ratio = (a["relative_se"]/b["relative_se"])**2 if b["relative_se"] > 0 else None
        rows.append({"cell": c, "components": components,
                     "paired_method_difference": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=2),
                     "observed_rse_squared_ratio_not_efficiency_certificate": ratio,
                     "maximum_log_error_upper": max(r["maximum_log_error_upper"] for r in group),
                     "covariance_rows": sum(r["covariance_rows"] for r in group)})
    return {"rows": rows, "scope": "same_proposal_paired_fine_probability_not_independent_reference_or_mesh_gate",
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, _ = load_anchors()
    calibration = json.loads((ROOT/CALIBRATION).read_text())
    audit_snapshot(ROOT, calibration["source"], calibration["config"])
    if (not calibration["pass"] or (ROOT/CALIBRATION_DIRECTORY/"failure.json").exists()
            or any(max(f["score_wall"] for f in r["fixtures"])*64 > 600. for r in calibration["rows"])):
        raise RuntimeError("large-grid preflight/bounded unit forecast failed; pilot not authorized")
    config = {"original": original, "guide": GUIDE, "namespace": NAMESPACE, "torch_threads": 8,
              "anchors": {CALIBRATION: sha(ROOT/CALIBRATION)}, "draw_count": DRAW_COUNT, "chunk": CHUNK}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, CALIBRATION))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in jobs()],
                "limits": {"D": {"work": 16384, "wall": 10800.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}
    def preparation():
        p, details = prepare(config)
        prepared.update(p)
        return {"actual_work": 0, "details": details}
    prep = journal.unit("preparation", phase="D", work=0, maximum_wall=360., operation=preparation)
    records = []
    for j in jobs():
        def operation(job: dict[str, Any] = j) -> dict[str, Any]:
            return unit_values(job, prepared[job["cell"]])
        records.append(journal.unit(identity(j), phase="D", work=DRAW_COUNT, maximum_wall=600., operation=operation))
        print(f"driver marginal paired pilot {len(records)}/16", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "preparation": prep,
              "summary": summarize(records), "units": 16, "raw_physical": 16384,
              "conditional_cdf_evaluations": 16384, "density_evaluations": 32768,
              "preparation_covariance_rows": sum(r["covariance_rows"] for r in prep["details"]),
              "scoring_covariance_rows": sum(r["covariance_rows"] for r in records),
              "phase_wall": time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"] > 10800.:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["guide"] != GUIDE or config["namespace"] != NAMESPACE or config["torch_threads"] != 8
            or config["draw_count"] != DRAW_COUNT or config["chunk"] != CHUNK
            or any(sha(ROOT/p) != d for p, d in config["anchors"].items())
            or manifest != {"config": config, "source": output["source"]["snapshot_sha256"],
                "expected_units": [identity(j) for j in jobs()], "limits": {"D": {"work": 16384, "wall": 10800.}}}):
        raise ValueError("changed fixed configuration")
    for p, digest in output["source"]["snapshot_file_hashes"].items():
        if (p.startswith("src/") or p == "experiments/post_audit_fft_driver_marginal_pilot.py") and sha(ROOT/p) != digest:
            raise ValueError("changed numerical runtime")
    prepared, _ = prepare(config)
    journal = RecoveryJournal(path.parent, manifest)
    records = []
    for suffix in ("start", "finish"):
        if {p.name for p in path.parent.glob("*."+suffix+".json")} != {"preparation."+suffix+".json", *(identity(j)+"."+suffix+".json" for j in jobs())}:
            raise ValueError("unexpected journal membership")
    for j in jobs():
        f = journal._read(path.parent/(identity(j)+".finish.json"))
        r = f["result"]
        digest = hashlib.sha256(json.dumps([seed(j, k, role) for k in range(64) for role in ("path", "label")]).encode()).hexdigest()
        if (f["status"] != "complete" or f["work"] != DRAW_COUNT or f["wall"] > 600.
                or any(r[k] != v for k, v in j.items()) or r["actual_work"] != DRAW_COUNT
                or r["raw_physical"] != DRAW_COUNT or r["conditional_cdf_evaluations"] != DRAW_COUNT
                or r["density_evaluations"] != 2*DRAW_COUNT or r["seed_digest"] != digest
                or r["torch_threads"] != 8 or r["maximum_log_error_upper"] > 1e-4):
            raise ValueError("changed raw/conditional unit identity/work")
        if j["rep"] == 0:
            replay = unit_values(j, prepared[j["cell"]])
            if any(replay[k] != r[k] for k in ("moments", "covariance_rows", "maximum_solve_iterations", "maximum_log_error_upper")):
                raise ValueError("conditional numerical replay mismatch")
        records.append(r)
    if (summarize(records) != output["summary"] or output["units"] != 16
            or output["raw_physical"] != 16384 or output["conditional_cdf_evaluations"] != 16384
            or output["density_evaluations"] != 32768 or journal.accounting("D")[0] != 16384
            or output["scoring_covariance_rows"] != sum(r["covariance_rows"] for r in records)
            or output["phase_wall"] > 10800. or any((path.parent/p).exists() for p in ("phase-D.violation.json", "final_io_failure.json"))):
        raise ValueError("changed summaries/cost or operational failure")
    return {"status": "fft_driver_marginal_source_seed_moments_solver_replay_pass", "replayed_blocks": 2,
            "audit_raw_physical": 2048, "audit_conditional_cdf": 2048}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        path = ROOT/args.audit
        receipt = {**audit(path), "result_sha256": sha(path)}
        write_json_atomic_nonoverwriting(path.parent/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        print(json.dumps({k: v for k, v in run().items() if k in ("units", "raw_physical", "phase_wall")}))
