"""Operational thread calibration with fixed-input paired gradient equivalence."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from src.path_integral.gradient_umbrella_smc import evaluate_family_gradient
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "results/post_audit/p2_paired_fine_extension_v1/session_20261010/result.json"
PLAN = "docs/plans/FINE_BACKEND_THREAD_CALIBRATION_2026-10-16_KO.md"
DIRECTORY = "results/post_audit/fine_backend_calibration_v1/session_20261016"
THREADS = (1, 2, 4, 8)


def run() -> dict[str, Any]:
    begin = time.perf_counter()
    anchor = json.loads((ROOT/ANCHOR).read_text())
    cell = next(c for c in anchor["config"]["cells"] if c["id"] == "high-eta200-k1")
    input_seed = int(hashlib.sha256(b"fine-backend-calibration-v1/input").hexdigest()[:15], 16)
    config = {"model": anchor["config"]["model"], "cell": cell, "seed": input_seed,
              "threads": list(THREADS), "batch": 128, "warmups": 1, "timed": 5,
              "anchor_sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    torch.set_num_threads(1)
    paired = PairedVolterraProbability.create(_problem(config["model"], cell, 4096))
    z = torch.randn((128, paired.dimension), dtype=torch.float64, generator=torch.Generator().manual_seed(input_seed))
    weights = torch.full((2,), .5, dtype=torch.float64).log()
    baseline: tuple[torch.Tensor, ...] | None = None
    rows: list[dict[str, Any]] = []
    for threads in THREADS:
        torch.set_num_threads(threads)
        timings = []
        for rep in range(6):
            tick = time.perf_counter()
            values = evaluate_family_gradient(paired.log_components, z, weights, maximum_drift_norm=20.)
            elapsed = time.perf_counter()-tick
            if baseline is None:
                baseline = tuple(v.clone() for v in values)
            for a, b in zip(values, baseline, strict=True):
                torch.testing.assert_close(a, b, rtol=5e-10, atol=1e-8)
            if rep:
                timings.append(elapsed)
            if time.perf_counter()-begin > 180 or psutil.Process().memory_info().rss > 4*1024**3:
                raise RuntimeError("backend calibration resource cap")
        rows.append({"threads": threads, "seconds": timings, "median_seconds": statistics.median(timings),
                     "numerical_equivalence_pass": True})
    fastest = min(rows, key=lambda r: r["median_seconds"])
    speedup = rows[0]["median_seconds"]/fastest["median_seconds"]
    selected = fastest["threads"] if speedup >= 1.2 else 1
    output = {"config": config, "source": source, "rows": rows, "selected_threads": selected,
              "callback_speedup_observed": speedup, "physical_forward_evaluations": 6144,
              "joint_backward_evaluations": 3072, "wall": time.perf_counter()-begin,
              "scope": "fixed_input_local_backend_calibration_not_estimator_or_method_speedup"}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    return output


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("selected_threads", "callback_speedup_observed", "physical_forward_evaluations", "wall")}))
