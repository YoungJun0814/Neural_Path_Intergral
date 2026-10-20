"""Fixed large-grid covariance/conditional IS preflight, not performance evidence."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import psutil
import torch

from experiments.post_audit_bridge_mutation_allocation import load_anchors
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.fft_driver_marginal_reference import FFTDriverMarginalReference
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting

DIRECTORY = "results/post_audit/fft_driver_marginal_calibration_v1/session_20261020"
PLAN = "docs/plans/P2_REFERENCE_MARGINAL_RECOVERY_2026-10-20_KO.md"
THEORY = "docs/theory/FFT_DRIVER_MARGINAL_REFERENCE_2026-10-20_KO.md"
NAMESPACE = "fft-driver-marginal-calibration-v1"
GUIDE: dict[str, Any] = {"amplitudes": [2., 4., 6., 8.], "price_amplitudes": [0., 2., 4.], "defensive_mass": .1}


def seed(cell: str, rep: int, role: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}/{cell}/{rep}/{role}".encode()).hexdigest()[:15], 16)


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, _ = load_anchors()
    config = {"original": original, "guide": GUIDE, "namespace": NAMESPACE, "torch_threads": 8,
              "draws_per_fixture": 16, "fixtures_per_cell": 4}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY))
    rows = []
    for cell in original["cells"]:
        p = _problem(original["model"], cell, 4096)
        begin = time.perf_counter()
        guide = PhysicalWindowGuide.create(p, kind="mixed-window", **GUIDE)
        ref = FFTDriverMarginalReference.create(p, guide.spec)
        raw = CachedVolterraTerminal.create(p)
        prep = time.perf_counter()-begin
        fixtures = []
        for rep in range(4):
            z = guide.sample(16, path_seed=seed(cell["id"], rep, "path"), label_seed=seed(cell["id"], rep, "label"))
            start = time.perf_counter()
            ev = ref.evaluate(ref.operator.project(z))
            wall = time.perf_counter()-start
            raw_values = raw.log_probability(z)-guide.log_q_over_p(z)
            fixtures.append({"score_wall": wall, "solve_iterations": list(ev.solve_iterations),
                "covariance_rows": ev.covariance_row_evaluations,
                "maximum_log_error_upper": float(ev.log_contribution_error_upper.max()),
                "minimum_variance": float(ev.conditional_variance.min()),
                "raw_log_contributions": raw_values.tolist(), "marginal_log_contributions": ev.log_contribution.tolist(),
                "rss": psutil.Process().memory_info().rss})
        passed = prep <= 180. and max(r["score_wall"] for r in fixtures) <= 10. and max(r["rss"] for r in fixtures) <= 4*1024**3
        rows.append({"cell": cell["id"], "preparation_wall": prep, "preparation_iterations": ref.preparation_iterations,
                     "preparation_covariance_rows": ref.preparation_covariance_rows,
                     "fixtures": fixtures, "pass": passed})
        print(f"driver marginal preflight {cell['id']}: pass={passed}", flush=True)
    output = {"config": config, "source": source, "rows": rows, "pass": all(r["pass"] for r in rows),
              "raw_physical": 128, "conditional_cdf_evaluations": 128,
              "interpretation": "fixed_numerical_resource_fixtures_not_mu_precision_or_efficiency_claim"}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    print(json.dumps({"pass": output["pass"], "raw_physical": 128, "conditional_cdf_evaluations": 128}))
    return output


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        directory = ROOT/DIRECTORY
        if directory.is_dir() and not (directory/"failure.json").exists():
            write_json_atomic_nonoverwriting(directory/"failure.json", {
                "status": "preflight_failure", "error": type(error).__name__+": "+str(error),
                "no_precision_or_training_authorized": True})
        raise
