"""Fresh fine-grid diagnostic; neither reference nor training gate is waived."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.fft_excursion_reference import (
    AdjacentBlockLatentAdapter,
    FFTExcursionReference,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.stable_gaussian import signed_log_normal_cdf_difference
from src.path_integral.structural_v2_inventory import audit_snapshot

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/P2_FINE_MESH_RECOVERY_PROTOCOL_2026-10-09_KO.md"
DIRECTORY = "results/post_audit/p2_fine_mesh_recovery_v1/session_20261009"


def seed(identity: str, stream: str) -> int:
    return int(hashlib.sha256(("p2-fine-excursion-v1/"+identity+"/"+stream).encode()).hexdigest()[:15], 16)


def batch_values(reference: FFTExcursionReference, coarse: CachedVolterraTerminal,
                 adapter: AdjacentBlockLatentAdapter, identity: str, count: int = 2048) -> dict[str, Any]:
    begin = time.perf_counter()
    p = reference.cache.problem
    assert isinstance(p.task, TerminalThresholdTask)
    streams = {s: seed(identity, s) for s in ("path", "label", "augmentation")}
    z = reference.sample(count, path_seed=streams["path"], label_seed=streams["label"])
    eps = torch.randn((count, p.steps//2), dtype=torch.float64,
                      generator=torch.Generator().manual_seed(streams["augmentation"]))
    density = reference.log_q_over_p(z)
    thresholds, peak = [], []
    for evaluator, coordinates in ((reference.cache, z), (coarse, adapter.coarse(z, eps))):
        terminal, integrated, variance = evaluator.quantities(coordinates)
        thresholds.append((math.log(p.task.level)-terminal)/torch.sqrt((1-p.rho**2)*integrated))
        peak.append((variance[:, :-1].max(1).values/variance[:, :-1].sum(1)).tolist())
    gf, gc = [torch.special.log_ndtr(t) for t in thresholds]
    signs, logs = signed_log_normal_cdf_difference(*thresholds)
    return {"fine": (gf-density).tolist(), "coarse": (gc-density).tolist(),
            "sign": signs.tolist(), "difference_log_absolute": (logs-density).tolist(),
            "peak_cell_share": peak, "seeds": streams, "actual_work": 2*count,
            "time_step_evaluations": count*(p.steps+p.steps//2), "wall": time.perf_counter()-begin}


def summarize(parts: list[dict[str, Any]], comparisons: int = 10) -> dict[str, Any]:
    fine = torch.tensor([x for b in parts for x in b["fine"]], dtype=torch.float64)
    coarse = torch.tensor([x for b in parts for x in b["coarse"]], dtype=torch.float64)
    signs = torch.tensor([x for b in parts for x in b["sign"]], dtype=torch.float64)
    logs = torch.tensor([x for b in parts for x in b["difference_log_absolute"]], dtype=torch.float64)
    if (len(fine) < 2 or len(fine) != len(coarse) or len(logs) != len(fine)
            or not torch.isfinite(fine).all() or not torch.isfinite(coarse).all()
            or bool(((signs != -1) & (signs != 0) & (signs != 1)).any())):
        raise ValueError("invalid paired records")
    scale = float(torch.max(torch.cat((fine, coarse, logs))))
    f, c, d = torch.exp(fine-scale), torch.exp(coarse-scale), signs*torch.exp(logs-scale)
    if not torch.allclose(f-c, d, rtol=1e-10, atol=1e-14):
        raise ValueError("signed CDF difference and paired samples disagree")
    mf, mc, md = float(f.mean()), float(c.mean()), float(d.mean())
    relative = md/mf
    se = float((d-relative*f).std(unbiased=True))/(math.sqrt(len(f))*mf)
    z = NormalDist().inv_cdf(1-.05/(2*comparisons))
    upper = abs(relative)+z*se
    peak = [torch.tensor([x for b in parts for x in b["peak_cell_share"][k]], dtype=torch.float64) for k in (0, 1)]
    return {"fine_mean": math.exp(scale)*mf, "coarse_mean": math.exp(scale)*mc,
            "signed_difference_mean": math.exp(scale)*md,
            "fine_rse": float(f.std(unbiased=True))/(math.sqrt(len(f))*mf),
            "coarse_rse": float(c.std(unbiased=True))/(math.sqrt(len(c))*mc),
            "relative_fine_minus_coarse": relative, "relative_difference_se": se,
            "confidence_z": z, "upper_absolute_relative_difference": upper,
            "small_difference_development_pass": upper <= .1,
            "maximum_fine_contribution_share": float(f.max()/f.sum()),
            "paired_mean_identity_error_scaled": abs(mf-mc-md), "count": len(f),
            "contribution_weighted_peak_cell_share_fine": float((f*peak[0]).sum()/f.sum()),
            "contribution_weighted_peak_cell_share_coarse": float((c*peak[1]).sum()/c.sum()),
            "scope": "paired_delta_method_development_not_continuum_certificate"}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_recovery_p2_parallel_v1.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"input": input_config, "fine_steps": [32, 64, 128, 256, 512], "count": 65536,
              "batch": 2048, "proposal": "exact_conditional_excursion_all_times_4_5_6",
              "scope": "fine_mesh_development_not_confirmation"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"M": {"work": 1400000, "wall": 1800.}},
                "expected_units": [row["cell"]["id"]+f"-{n}-{b}" for row in rows if row["parent_training_rep"] == 0
                                   for n in config["fine_steps"] for b in range(32)]}
    journal = RecoveryJournal(directory, manifest)
    records = []
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        for n in config["fine_steps"]:
            p = _problem(metadata["config"]["model"], row["cell"], n)
            key = row["cell"]["id"]+f"-{n}"
            cache: dict[str, Any] = {}
            def prepare(problem=p, saved=cache):
                saved.update(reference=FFTExcursionReference.create(problem),
                             coarse=CachedVolterraTerminal.create(replace(problem, steps=problem.steps//2)),
                             adapter=AdjacentBlockLatentAdapter.create(problem))
                return {"actual_work": 0}
            journal.unit(key+"-prepare", phase="M", work=0, maximum_wall=20., operation=prepare)
            reference, coarse, adapter = cache["reference"], cache["coarse"], cache["adapter"]
            parts = []
            for b in range(32):
                if psutil.Process().memory_info().rss > 4*1024**3:
                    raise MemoryError("RSS cap exceeded")
                ident = key+f"-{b}"
                def operation(unit: str = ident, r: FFTExcursionReference = reference,
                              c: CachedVolterraTerminal = coarse, a: AdjacentBlockLatentAdapter = adapter) -> dict[str, Any]:
                    return batch_values(r, c, a, unit)
                parts.append(journal.unit(ident, phase="M", work=4096, maximum_wall=20.,
                    operation=operation))
            records.append({"cell": row["cell"], "fine_steps": n, "coarse_steps": n//2, **summarize(parts)})
            print(key+" fine-mesh diagnostic complete", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "records": records,
              "work": journal.accounting("M")[0], "p1_qualified": False, "p2_qualified": False,
              "large_training_authorized": False, "diagnostic_only": True,
              "time_step_evaluations": sum(65536*(n+n//2) for n in config["fine_steps"])*2,
              "phase_wall": time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary and final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    audit_snapshot(ROOT, output["source"], output["config"])
    manifest, config = output["manifest"], output["config"]
    if manifest["source"] != output["source"]["snapshot_sha256"]:
        raise ValueError("source binding mismatch")
    artifact = ROOT/config["input"]["proposal_artifact_path"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise ValueError("changed input artifact")
    metadata, rows = inputs(config["input"])
    expected = [r["cell"]["id"]+f"-{n}-{b}" for r in rows if r["parent_training_rep"] == 0
                for n in config["fine_steps"] for b in range(32)]
    if expected != manifest["expected_units"]:
        raise ValueError("changed grid")
    journal = RecoveryJournal(path.parent, manifest)
    records, replayed = [], 0
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        for n in config["fine_steps"]:
            key = row["cell"]["id"]+f"-{n}"
            parts = [journal._read(path.parent/(key+f"-{b}.finish.json"))["result"] for b in range(32)]
            for b, part in enumerate(parts):
                if part["seeds"] != {s: seed(key+f"-{b}", s) for s in ("path", "label", "augmentation")}:
                    raise ValueError("seed mismatch")
                if part["actual_work"] != 4096 or part["time_step_evaluations"] != 2048*(n+n//2):
                    raise ValueError("work mismatch")
            records.append({"cell": row["cell"], "fine_steps": n, "coarse_steps": n//2, **summarize(parts)})
            p = _problem(metadata["config"]["model"], row["cell"], n)
            replay = batch_values(FFTExcursionReference.create(p), CachedVolterraTerminal.create(replace(p, steps=n//2)),
                                  AdjacentBlockLatentAdapter.create(p), key+"-0")
            for k in ("fine", "coarse", "sign", "difference_log_absolute", "peak_cell_share"):
                torch.testing.assert_close(torch.tensor(replay[k], dtype=torch.float64), torch.tensor(parts[0][k], dtype=torch.float64),
                                           rtol=1e-12, atol=1e-12, equal_nan=False)
            replayed += len(replay["fine"])
    if records != output["records"] or journal.accounting("M")[0] != output["work"]:
        raise ValueError("summary/accounting mismatch")
    if (path.parent/"phase-M.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("failed operational protocol")
    return {"status": "source_input_seed_arithmetic_replay_pass_not_statistical_certificate",
            "units": len(expected), "replayed_paths": replayed, "p2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "phase_wall", "p2_qualified", "time_step_evaluations")}))
