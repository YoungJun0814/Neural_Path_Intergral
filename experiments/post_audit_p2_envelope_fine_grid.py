"""Fixed-count finer-grid diagnostic under the unchanged exact envelope law."""

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

from experiments.post_audit_p2_fine_mesh_recovery import summarize
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.fft_excursion_reference import (
    AdjacentBlockLatentAdapter,
    FFTCoverageEnvelopeReference,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.stable_gaussian import signed_log_normal_cdf_difference
from src.path_integral.structural_v2_inventory import audit_snapshot

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/P2_ENVELOPE_FINE_GRID_EXTENSION_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/p2_envelope_fine_grid_v1/session_20261010"
STEPS = (1024, 2048, 4096)
BATCH, BATCHES = 512, 128


def seed(identity: str, stream: str) -> int:
    return int(hashlib.sha256(("p2-envelope-fine-grid-v1/"+identity+"/"+stream).encode()).hexdigest()[:15], 16)


def unit_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [r["cell"]["id"]+f"-{n}-{b}" for r in rows if r["parent_training_rep"] == 0
            for n in STEPS for b in range(BATCHES)]


def batch_values(reference: FFTCoverageEnvelopeReference, coarse: CachedVolterraTerminal,
                 adapter: AdjacentBlockLatentAdapter, identity: str, count: int = BATCH) -> dict[str, Any]:
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
    rss = psutil.Process().memory_info().rss
    if rss > 4*1024**3:
        raise MemoryError("after-batch RSS guard exceeded")
    return {"fine": (gf-density).tolist(), "coarse": (gc-density).tolist(),
            "sign": signs.tolist(), "difference_log_absolute": (logs-density).tolist(),
            "peak_cell_share": peak, "seeds": streams, "actual_work": 2*count,
            "time_step_evaluations": count*(p.steps+p.steps//2),
            "observed_after_batch_rss": rss, "wall": time.perf_counter()-begin}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"input": input_config, "fine_steps": list(STEPS), "count": BATCH*BATCHES,
              "batch": BATCH, "proposal": "corrected_f64_exact_coverage_envelope_fft",
              "scope": "finer_grid_development_not_continuum_or_reference_certificate"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    expected = unit_ids(rows)
    seeds = [seed(i, s) for i in expected for s in ("path", "label", "augmentation")]
    if len(set(seeds)) != len(seeds) or len(expected) != 768:
        raise ValueError("unexpected grid or seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": expected,
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"M": {"work": 800000, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    records, max_rss = [], 0
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        for n in STEPS:
            p = _problem(metadata["config"]["model"], row["cell"], n)
            key = row["cell"]["id"]+f"-{n}"
            cache: dict[str, Any] = {}
            def prepare(problem=p, saved=cache):
                saved.update(reference=FFTCoverageEnvelopeReference.create(problem),
                             coarse=CachedVolterraTerminal.create(replace(problem, steps=problem.steps//2)),
                             adapter=AdjacentBlockLatentAdapter.create(problem))
                return {"actual_work": 0}
            journal.unit(key+"-prepare", phase="M", work=0, maximum_wall=60., operation=prepare)
            reference, coarse, adapter = cache["reference"], cache["coarse"], cache["adapter"]
            parts = []
            for b in range(BATCHES):
                if psutil.Process().memory_info().rss > 4*1024**3:
                    raise MemoryError("before-batch RSS guard exceeded")
                ident = key+f"-{b}"
                def operation(unit: str = ident, r: FFTCoverageEnvelopeReference = reference,
                              c: CachedVolterraTerminal = coarse, a: AdjacentBlockLatentAdapter = adapter) -> dict[str, Any]:
                    return batch_values(r, c, a, unit)
                part = journal.unit(ident, phase="M", work=2*BATCH, maximum_wall=20., operation=operation)
                parts.append(part)
                max_rss = max(max_rss, part["observed_after_batch_rss"])
            records.append({"cell": row["cell"], "fine_steps": n, "coarse_steps": n//2, **summarize(parts)})
            print(key+" envelope fine-grid diagnostic complete", flush=True)
            del reference, coarse, adapter, cache, parts, part, prepare, operation
    output = {"config": config, "source": source, "manifest": manifest, "records": records,
              "work": journal.accounting("M")[0], "time_step_evaluations": 2*BATCH*BATCHES*sum(n+n//2 for n in STEPS),
              "observed_max_after_batch_rss": max_rss, "p1_qualified": False, "p2_qualified": False,
              "large_training_authorized": False, "diagnostic_only": True,
              "phase_wall": time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary and final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()
            or config["fine_steps"] != list(STEPS) or config["batch"] != BATCH or config["count"] != BATCH*BATCHES):
        raise ValueError("changed source/config/input binding")
    metadata, rows = inputs(config["input"])
    if unit_ids(rows) != manifest["expected_units"]:
        raise ValueError("changed expected units")
    journal = RecoveryJournal(path.parent, manifest)
    records, replayed, steps_work, max_rss = [], 0, 0, 0
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        for n in STEPS:
            key = row["cell"]["id"]+f"-{n}"
            parts = []
            for b in range(BATCHES):
                finish = journal._read(path.parent/(key+f"-{b}.finish.json"))
                if finish["status"] != "complete":
                    raise ValueError("failed unit cannot be used")
                part = finish["result"]
                if (part["seeds"] != {s: seed(key+f"-{b}", s) for s in ("path", "label", "augmentation")}
                        or part["actual_work"] != 2*BATCH or part["time_step_evaluations"] != BATCH*(n+n//2)):
                    raise ValueError("seed/work mismatch")
                parts.append(part)
                steps_work += part["time_step_evaluations"]
                max_rss = max(max_rss, part["observed_after_batch_rss"])
            records.append({"cell": row["cell"], "fine_steps": n, "coarse_steps": n//2, **summarize(parts)})
            p = _problem(metadata["config"]["model"], row["cell"], n)
            replay = batch_values(FFTCoverageEnvelopeReference.create(p),
                                 CachedVolterraTerminal.create(replace(p, steps=n//2)),
                                 AdjacentBlockLatentAdapter.create(p), key+"-0")
            for k in ("fine", "coarse", "sign", "difference_log_absolute", "peak_cell_share"):
                torch.testing.assert_close(torch.tensor(replay[k], dtype=torch.float64),
                                           torch.tensor(parts[0][k], dtype=torch.float64), rtol=1e-12, atol=1e-12)
            replayed += BATCH
    if (records != output["records"] or journal.accounting("M")[0] != output["work"]
            or output["work"] != len(unit_ids(rows))*2*BATCH
            or steps_work != output["time_step_evaluations"] or max_rss != output["observed_max_after_batch_rss"]
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("changed summary/accounting/qualification")
    if (path.parent/"phase-M.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_input_seed_arithmetic_replay_pass_not_continuum_certificate",
            "replayed_paths": replayed, "units": len(unit_ids(rows)), "p2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "phase_wall", "p2_qualified", "time_step_evaluations")}))
