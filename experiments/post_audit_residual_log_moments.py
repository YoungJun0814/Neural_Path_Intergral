"""Fixed-context log-moment numerical gate, using no new Volterra paths."""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from typing import Any

import numpy as np
import torch

from experiments.post_audit_exact_residual_structure import DIRECTORY as PARENT
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.residual_log_quadrature import log_gaussian_moments, slice_log_probability
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/residual_log_moments_v1/session_20261021"
PLAN = "docs/plans/RESIDUAL_LOG_MOMENT_AND_PAIRED_PILOT_2026-10-21_KO.md"
EXPECTED = "083c03374ee6d81062359db5ac63e669b3de10b84b87c1e1049144af11cf8f0a"
REVIEW = "9fa72501f8e74b60e3c7d47d7e750948a4c2778a4e24eea13054ff62cb8386b0"


class ScalarBudget:
    def __init__(self, cap: int, wall: float):
        self.cap, self.wall, self.count, self.started = cap, wall, 0, time.perf_counter()

    def charge(self, n: int) -> None:
        if type(n) is not int or n < 0:
            raise ValueError("nonnegative integer work required")
        if self.count+n > self.cap or time.perf_counter()-self.started > self.wall:
            raise TimeoutError("fixed scalar/wall budget exceeded")
        self.count += n


def evidence() -> dict[str, Any]:
    path = ROOT/PARENT
    if sha(path/"result.json") != EXPECTED or sha(path/"numerical_review.json") != REVIEW:
        raise ValueError("fixed original and post-hoc evidence required")
    value = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, value["source"], value["config"])
    return value


def known_answers(budget: ScalarBudget) -> list[dict[str, Any]]:
    rows = []
    for center, width in ((0., 1.), (10., .2), (-7., .75)):
        def log_bump(u: np.ndarray, c: float = center, w: float = width) -> np.ndarray:
            return -.5*((u-c)/w)**2
        result = log_gaussian_moments(log_bump, charge=budget.charge)
        exact = [.5*math.log(width**2/p/(1+width**2/p))-center**2/(2*(1+width**2/p)) for p in (1, 2)]
        error = max(abs(m["log_estimate"]-x) for m, x in zip(result["moments"], exact, strict=True))
        rows.append({"center": center, "width": width, "result": result,
            "exact_log_moments": exact, "maximum_log_error": error,
            "pass": result["relative_numerically_resolved"] and error <= 1e-9})
    return rows


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    parent = evidence()
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("one-shot log-moment gate; no overwrite")
    path.mkdir(parents=True, exist_ok=True)
    config = {"cutoff": 16, "panel_width": 1, "orders": [16, 32, 64], "relative_tolerance": 1e-6,
        "scalar_cap": 150000, "wall_cap": 120, "physical_cap": 0, "parent_sha256": EXPECTED}
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, PARENT+"/result.json", PARENT+"/numerical_review.json"))
    write_json_atomic_nonoverwriting(path/"protocol.json", {"config": config, "source": source})
    budget = ScalarBudget(150000, 120)
    rows = []
    try:
        toys = known_answers(budget)
        for cell in parent["summary"]["cells"]:
            for context in cell["contexts"]:
                profile = LastVarianceResidualSlice(**context["profile"])
                result = log_gaussian_moments(partial(slice_log_probability, profile), charge=budget.charge)
                rows.append({"cell": cell["cell"], "role": context["role"], "profile": context["profile"],
                    "previous_moments": context["conditional_moments"], "result": result})
    except Exception as e:
        write_json_atomic_nonoverwriting(path/"failure.json", {"error": type(e).__name__+": "+str(e),
            "scalar_evaluations": budget.count, "completed_contexts": rows, "physical_evaluations": 0})
        raise
    passed = all(r["pass"] for r in toys) and all(r["result"]["relative_numerically_resolved"] for r in rows)
    output = {"config": config, "source": source, "toys": toys, "contexts": rows,
        "decision": "Q0_RELATIVE_NUMERICAL_GATE_PASSED" if passed else "Q0_UNRESOLVED_NO_PILOT",
        "q1_pilot_numerics_gate": passed, "physical_evaluations": 0, "scalar_evaluations": budget.count,
        "wall_computation_only": time.perf_counter()-budget.started, "P2_qualified": False,
        "not_rigorous_or_population_certificate": True}
    write_json_atomic_nonoverwriting(path/"result.json", output)
    return output


def audit() -> dict[str, Any]:
    parent = evidence()
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, digest in result["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("frozen log-moment source changed")
    budget = ScalarBudget(50000, 60)
    if known_answers(budget) != result["toys"]:
        raise ValueError("known-answer numerical replay failed")
    expected = [(c["cell"], r["role"], r["profile"]) for c in parent["summary"]["cells"] for r in c["contexts"]]
    if [(r["cell"], r["role"], r["profile"]) for r in result["contexts"]] != expected:
        raise ValueError("complete fixed-context binding required")
    for row in result["contexts"]:
        if row["role"] == "zero":
            replay = log_gaussian_moments(partial(slice_log_probability, LastVarianceResidualSlice(**row["profile"])), charge=budget.charge)
            if replay != row["result"]:
                raise ValueError("two formerly absolute-only profiles did not replay")
    passed = all(r["pass"] for r in result["toys"]) and all(r["result"]["relative_numerically_resolved"] for r in result["contexts"])
    if result["q1_pilot_numerics_gate"] != passed or result["physical_evaluations"] != 0 or result["P2_qualified"]:
        raise ValueError("numerical gate/accounting changed")
    return {"status": "known_answers_and_two_zero_profiles_replayed", "result_sha256": sha(path/"result.json"),
        "scalar_evaluations": budget.count, "physical_evaluations": 0, "q1_pilot_numerics_gate": passed,
        "P2_qualified": False, "internal_errors_not_rigorous_bounds": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        value = audit()
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", value)
    else:
        value = run()
    print(json.dumps({k: value[k] for k in ("decision", "scalar_evaluations", "physical_evaluations", "q1_pilot_numerics_gate") if k in value}))
