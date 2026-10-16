"""Exploratory samplewise driver/price quadrature mesh decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_p2_fine_mesh_recovery import seed
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
PLAN = "docs/plans/P2_DRIVER_QUADRATURE_DIAGNOSIS_2026-10-09_KO.md"
DIRECTORY = "results/post_audit/p2_driver_quadrature_diagnosis_v1/session_20261009"


def thresholds(reference: FFTExcursionReference, coarse: CachedVolterraTerminal,
               adapter: AdjacentBlockLatentAdapter, z: torch.Tensor, eps: torch.Tensor) -> list[torch.Tensor]:
    p = reference.cache.problem
    assert isinstance(p.task, TerminalThresholdTask)
    terminal_f, integrated_f, variance_f = reference.cache.quantities(z)
    terminal_c, integrated_c, _ = coarse.quantities(adapter.coarse(z, eps))
    dw_f = (z.reshape(-1, p.steps, 2) @ reference.cache.kernel.local_cholesky.T)[:, :, 0]
    dw_m = dw_f.reshape(-1, p.steps//2, 2).sum(2)
    v_m = variance_f[:, :-1:2]
    integrated_m = 2*p.step_dt*v_m.sum(1)
    terminal_m = math.log(p.spot)-.5*integrated_m+p.rho*(v_m.sqrt()*dw_m).sum(1)
    return [(math.log(p.task.level)-t)/torch.sqrt((1-p.rho**2)*i)
            for t, i in ((terminal_f, integrated_f), (terminal_m, integrated_m), (terminal_c, integrated_c))]


def batch_values(reference: FFTExcursionReference, coarse: CachedVolterraTerminal,
                 adapter: AdjacentBlockLatentAdapter, ident: str) -> dict[str, Any]:
    begin = time.perf_counter()
    streams = {s: seed(ident, s) for s in ("path", "label", "augmentation")}
    z = reference.sample(2048, path_seed=streams["path"], label_seed=streams["label"])
    eps = torch.randn((2048, reference.cache.problem.steps//2), dtype=torch.float64,
                      generator=torch.Generator().manual_seed(streams["augmentation"]))
    ts = thresholds(reference, coarse, adapter, z, eps)
    density = reference.log_q_over_p(z)
    logs = [(torch.special.log_ndtr(t)-density).tolist() for t in ts]
    corrections = []
    for a, b in ((0, 1), (1, 2), (0, 2)):
        sign, logabs = signed_log_normal_cdf_difference(ts[a], ts[b])
        corrections.append({"sign": sign.tolist(),
                            "logs": [None if s == 0 else x for s, x in zip(sign.tolist(), (logabs-density).tolist(), strict=True)]})
    return {"logs": logs, "corrections": corrections, "seeds": streams,
            "actual_work": 6144, "time_step_evaluations": 2048*2*reference.cache.problem.steps,
            "wall": time.perf_counter()-begin}


def summarize(parts: list[dict[str, Any]]) -> dict[str, Any]:
    logs = [torch.tensor([x for p in parts for x in p["logs"][k]], dtype=torch.float64) for k in range(3)]
    scale = float(torch.max(torch.cat(logs)))
    values = [torch.exp(x-scale) for x in logs]
    ds = []
    for k in range(3):
        signs = torch.tensor([x for p in parts for x in p["corrections"][k]["sign"]], dtype=torch.float64)
        magnitudes = torch.tensor([-math.inf if x is None else x for p in parts for x in p["corrections"][k]["logs"]], dtype=torch.float64)
        ds.append(signs*torch.exp(magnitudes-scale))
    for d, (a, b) in zip(ds, ((0, 1), (1, 2), (0, 2)), strict=True):
        torch.testing.assert_close(d, values[a]-values[b], rtol=1e-10, atol=1e-14)
    torch.testing.assert_close(ds[0]+ds[1], ds[2], rtol=1e-10, atol=1e-14)
    f = values[0]
    summaries = []
    for label, d in zip(("price_quadrature", "driver_approximation", "total"), ds, strict=True):
        ratio = float(d.mean()/f.mean())
        se = float((d-ratio*f).std(unbiased=True))/(math.sqrt(len(f))*float(f.mean()))
        summaries.append({"component": label, "mean": math.exp(scale)*float(d.mean()),
                          "relative_to_fine_mean": ratio, "paired_ratio_se": se})
    return {"fine_mean": math.exp(scale)*float(values[0].mean()),
            "fine_rse": float(f.std(unbiased=True))/(math.sqrt(len(f))*float(f.mean())),
            "counterfactual_mean": math.exp(scale)*float(values[1].mean()),
            "coarse_mean": math.exp(scale)*float(values[2].mean()), "components": summaries,
            "triangle_identity_max_scaled_error": float((ds[0]+ds[1]-ds[2]).abs().max()), "count": len(f)}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_recovery_p2_parallel_v1.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"input": input_config, "fine_steps": [32, 64, 256, 512], "count": 65536,
              "batch": 2048, "scope": "exploratory_order_dependent_mesh_error_decomposition"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    manifest = {"config": config, "source": source["snapshot_sha256"],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"D": {"work": 2000000, "wall": 600.}}}
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    journal = RecoveryJournal(directory, manifest)
    records = []
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        for n in config["fine_steps"]:
            p = _problem(metadata["config"]["model"], row["cell"], n)
            key = "quadrature-driver-"+row["cell"]["id"]+f"-{n}"
            cache: dict[str, Any] = {}
            def prepare(problem=p, saved=cache):
                saved.update(reference=FFTExcursionReference.create(problem),
                             coarse=CachedVolterraTerminal.create(replace(problem, steps=problem.steps//2)),
                             adapter=AdjacentBlockLatentAdapter.create(problem))
                return {"actual_work": 0}
            journal.unit(key+"-prepare", phase="D", work=0, maximum_wall=20., operation=prepare)
            parts = []
            for b in range(32):
                if psutil.Process().memory_info().rss > 4*1024**3:
                    raise MemoryError("RSS cap")
                def operation(ident=key+f"-{b}", saved=cache):
                    return batch_values(saved["reference"], saved["coarse"], saved["adapter"], ident)
                parts.append(journal.unit(key+f"-{b}", phase="D", work=6144, maximum_wall=20., operation=operation))
            records.append({"cell": row["cell"], "fine_steps": n, "coarse_steps": n//2, **summarize(parts)})
            print(key+" decomposition complete", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "records": records,
              "work": journal.accounting("D")[0], "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"] > 600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    audit_snapshot(ROOT, output["source"], output["config"])
    config, manifest = output["config"], output["manifest"]
    if (manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()):
        raise ValueError("source/input mismatch")
    metadata, rows = inputs(config["input"])
    journal, records, replayed = RecoveryJournal(path.parent, manifest), [], 0
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        for n in config["fine_steps"]:
            key = "quadrature-driver-"+row["cell"]["id"]+f"-{n}"
            parts = [journal._read(path.parent/(key+f"-{b}.finish.json"))["result"] for b in range(32)]
            for b, part in enumerate(parts):
                if part["seeds"] != {s: seed(key+f"-{b}", s) for s in ("path", "label", "augmentation")} or part["actual_work"] != 6144:
                    raise ValueError("seed/cost mismatch")
            records.append({"cell": row["cell"], "fine_steps": n, "coarse_steps": n//2, **summarize(parts)})
            p = _problem(metadata["config"]["model"], row["cell"], n)
            replay = batch_values(FFTExcursionReference.create(p), CachedVolterraTerminal.create(replace(p, steps=n//2)),
                                  AdjacentBlockLatentAdapter.create(p), key+"-0")
            if replay["logs"] != parts[0]["logs"] or replay["corrections"] != parts[0]["corrections"]:
                raise ValueError("numerical replay mismatch")
            replayed += 2048
    if records != output["records"] or journal.accounting("D")[0] != output["work"]:
        raise ValueError("summary/cost mismatch")
    if (path.parent/"phase-D.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_input_seed_signed_identity_replay_pass_not_bias_certificate", "replayed_paths": replayed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else {k: v for k, v in run().items() if k in ("work", "phase_wall")}))
