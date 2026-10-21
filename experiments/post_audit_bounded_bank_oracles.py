"""D1: fixed two-dimensional Gaussian oracles, no Volterra payoff calls."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr, roots_hermitenorm

from experiments.post_audit_bounded_direct_m2 import DIRECTORY as D0
from experiments.post_audit_bounded_direct_m2 import PLAN
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.balanced_mixture_bank import draw_bank
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.stratified_paired_risk import estimate, sufficient
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/bounded_bank_oracles_v1/session_20261020"
LIMIT = 4194304


def cases() -> list[tuple[str, GaussianBumpOracle, GaussianMixtureShiftSpec]]:
    common = ((0., 0.), (3., 0.), (-3., 0.), (0., 3.), (0., -3.))
    rows = [
        ("A", GaussianBumpOracle((1.,), ((3., 0.),), (1.,)), common),
        ("B", GaussianBumpOracle((.5, .5), ((3., 0.), (-3., 0.)), (1., 1.)), common),
        ("C", GaussianBumpOracle((.999, .001), ((2., 0.), (0., 5.)), (.75, .75)), common),
        ("D", GaussianBumpOracle((1.,), ((0., 4.),), (.75,)),
            ((0., 0.), (3., 0.), (-3., 0.), (6., 0.), (-6., 0.))),
    ]
    weights = torch.tensor([.1, .225, .225, .225, .225], dtype=torch.float64)
    return [(name, toy, GaussianMixtureShiftSpec(torch.tensor(means, dtype=torch.float64), weights)) for name, toy, means in rows]


class Budget:
    def __init__(self, maximum: int = LIMIT, wall: float = 1800.) -> None:
        self.count, self.maximum, self.wall = 0, maximum, wall
        self.started = time.perf_counter()

    def charge(self, count: int) -> None:
        if self.count+count > self.maximum or time.perf_counter()-self.started > self.wall:
            raise TimeoutError("fixed D1 synthetic work/wall cap exceeded")
        self.count += count


def numpy_logs(z: np.ndarray, toy: GaussianBumpOracle, spec: GaussianMixtureShiftSpec) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(toy.centers)
    scales = np.asarray(toy.scales)
    g = logsumexp(np.log(toy.amplitudes)-np.sum((z[:, None, :]-centers)**2, axis=2)/(2*scales**2), axis=1)
    means = spec.means.numpy()
    d = logsumexp(z @ means.T-.5*np.sum(means**2, axis=1)+np.log(spec.weights.numpy()), axis=1)
    return g, d


def numerical_m2(toy: GaussianBumpOracle, spec: GaussianMixtureShiftSpec, budget: Budget) -> dict[str, Any]:
    gh = []
    for order in (48, 96, 192):
        nodes, weights = roots_hermitenorm(order)
        x, y = np.meshgrid(nodes, nodes, indexing="ij")
        z = np.column_stack((x.ravel(), y.ravel()))
        budget.charge(len(z))
        g, d = numpy_logs(z, toy, spec)
        log_weights = np.log(weights[:, None]*weights[None, :]).ravel()-math.log(2*math.pi)
        gh.append(float(np.exp(logsumexp(log_weights+2*g-d))))
    inner_errors = []

    def integrand(y: float, x: float) -> float:
        budget.charge(1)
        g, d = numpy_logs(np.array([[x, y]], dtype=np.float64), toy, spec)
        return math.exp(float(2*g[0]-d[0])-.5*(x*x+y*y)-math.log(2*math.pi))

    def inner(x: float) -> float:
        value, error = quad(integrand, -12., 12., args=(x,), epsabs=1e-12, epsrel=1e-8, limit=150)
        inner_errors.append(error)
        return float(value)

    value, error = quad(inner, -12., 12., epsabs=1e-12, epsrel=1e-8, limit=150)
    error_gh = abs(gh[-1]/value-1)
    error_convergence = abs(gh[-2]/gh[-1]-1)
    return {"gauss_hermite_orders": [48, 96, 192], "gauss_hermite_m2": gh,
        "adaptive_m2": value, "adaptive_outer_reported_error": error,
        "maximum_inner_reported_error": max(inner_errors),
        "absolute_omitted_tail_upper": float(40*ndtr(-12.)),
        "gh192_vs_adaptive_relative": error_gh, "gh96_vs_192_relative": error_convergence,
        "pass": error_gh <= 1e-5 and error_convergence <= 1e-5,
        "reported_error_not_rigorous_interval_certificate": True}


def compute(budget: Budget) -> dict[str, Any]:
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    for name, toy, spec in cases():
        oracle = numerical_m2(toy, spec, budget)
        repeats = []
        for design in ("iid", "stratified"):
            for rep in range(16):
                key = {"phase": "D1", "case": name, "design": design, "rep": rep}
                z, labels, digest = draw_bank(spec, 16000, design, key)
                budget.charge(len(z))
                g, d = numpy_logs(z.numpy(), toy, spec)
                logs = torch.tensor(np.column_stack((g-d, 2*g-2*d)), dtype=torch.float64)
                parts = [sufficient(logs)] if design == "iid" else [sufficient(logs[labels == j]) for j in range(5)]
                stats = estimate(parts, torch.ones(1, dtype=torch.float64) if design == "iid" else spec.weights)
                exact = [toy.log_endpoint(1), math.log(oracle["adaptive_m2"])]
                repeats.append({"design": design, "rep": rep, "seed_digest": digest, "summary": stats,
                    "relative_error": [math.expm1(v-e) for v, e in zip(stats["log_mean"], exact, strict=True)],
                    "nominal_95_development_contains_oracle": [abs(math.expm1(v-e)) <= 1.96*r*math.exp(v-e)
                        for v, e, r in zip(stats["log_mean"], exact, stats["relative_se"], strict=True)]})
        rows.append({"case": name, "analytic_mu": math.exp(toy.log_endpoint(1)), "oracle": oracle, "repetitions": repeats})
        print(f"D1 completed fixed toy {name}", flush=True)
    return {"cases": rows, "d1_gate": all(r["oracle"]["pass"] for r in rows),
        "synthetic_evaluations": budget.count, "synthetic_cap": LIMIT,
        "additional_volterra_evaluations": 0, "wall": time.perf_counter()-budget.started,
        "coverage_claim": False, "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    d0path = ROOT/D0/"result.json"
    d0, receipt = [json.loads(p.read_text()) for p in (d0path, d0path.parent/"audit.json")]
    if receipt["result_sha256"] != sha(d0path) or not d0["summary"]["d0_evidence_gate"]:
        raise ValueError("audited D0 required")
    config = {"d0_result": sha(d0path), "d0_audit": sha(d0path.parent/"audit.json"),
        "toy_count": 16000, "repetitions": 16, "orders": [48, 96, 192], "cap": LIMIT, "wall": 1800}
    directory = ROOT/DIRECTORY
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("one-shot synthetic family; no overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, D0+"/result.json", D0+"/audit.json"))
    write_json_atomic_nonoverwriting(directory/"protocol.json", {"config": config, "source": source})
    started = time.perf_counter()
    budget = Budget()
    try:
        output = {"config": config, "source": source, "summary": compute(budget)}
    except Exception as error:
        write_json_atomic_nonoverwriting(directory/"failure.json", {"status": "D1_failure", "error": str(error),
            "wall": time.perf_counter()-started, "synthetic_evaluations": budget.count,
            "additional_volterra_evaluations": 0})
        raise
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    return output


def audit(path: Path) -> dict[str, Any]:
    output = json.loads(path.read_text())
    audit_snapshot(ROOT, output["source"], output["config"])
    rows = output["summary"]["cases"]
    if len(rows) != 4 or [r["case"] for r in rows] != ["A", "B", "C", "D"]:
        raise ValueError("complete fixed toy family required")
    for (name, toy, spec), row in zip(cases(), rows, strict=True):
        if len(row["repetitions"]) != 32 or row["case"] != name or row["analytic_mu"] != math.exp(toy.log_endpoint(1)):
            raise ValueError("toy identity/count mismatch")
        for r in row["repetitions"]:
            if r["summary"]["count"] != 16000 or r["design"] not in ("iid", "stratified"):
                raise ValueError("toy sample count/design changed")
        # One prespecified representative per toy/design, no new oracle tuning.
        for design in ("iid", "stratified"):
            z, labels, digest = draw_bank(spec, 16000, design, {"phase": "D1", "case": name, "design": design, "rep": 0})
            g, d = numpy_logs(z.numpy(), toy, spec)
            logs = torch.tensor(np.column_stack((g-d, 2*g-2*d)), dtype=torch.float64)
            parts = [sufficient(logs)] if design == "iid" else [sufficient(logs[labels == j]) for j in range(5)]
            rebuilt = estimate(parts, torch.ones(1, dtype=torch.float64) if design == "iid" else spec.weights)
            saved = next(r for r in row["repetitions"] if (r["design"], r["rep"]) == (design, 0))
            if rebuilt != saved["summary"] or digest != saved["seed_digest"]:
                raise ValueError("toy numerical replay mismatch")
    return {"status": "D1_source_all_toy_count_and_eight_representative_replays_pass",
        "d1_gate": output["summary"]["d1_gate"], "result_sha256": sha(path),
        "audit_synthetic_evaluations": 128000, "audit_volterra_evaluations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.audit:
        receipt = audit(ROOT/DIRECTORY/"result.json")
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        result = run()
        print(json.dumps({k: v for k, v in result["summary"].items() if k != "cases"}))
