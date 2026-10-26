"""Frozen kernel benchmark, joint-residual oracle and marginal-risk contract."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time
from functools import partial
from typing import Any

import numpy as np
import torch

from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_residual_log_moments import DIRECTORY as Q0
from experiments.post_audit_residual_paired_pilot import DIRECTORY as Q1
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.analytic_residual_batch import JointBlindResidualSlice, batched_moments
from src.path_integral.marginal_risk_contract import marginal_risk_known_answer
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.residual_log_quadrature import log_gaussian_moments, slice_log_probability
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/analytic_residual_batch_v1/session_20261021"
PLAN = "docs/plans/ANALYTIC_RESIDUAL_BATCH_AND_MARGINAL_RISK_2026-10-21_KO.md"
THEORY = "docs/theory/ANALYTIC_RESIDUAL_AND_MARGINAL_RISK_CONTRACT_2026-10-21_KO.md"
EXPECTED = {Q0: "53c604c5faafa573ae380605342a130f68d54dcde6cefb30393a380bc482bb6b",
    Q1: "8645b470b638ff9860ea831915c1bf10e2111d66ca1780f31640c2c6c008ade1"}


class Budget:
    def __init__(self, scalar: int, physical: int, wall: float):
        self.scalar_cap, self.physical_cap, self.wall = scalar, physical, wall
        self.scalar, self.physical, self.started = 0, 0, time.perf_counter()

    def charge(self, n: int) -> None:
        if type(n) is not int or n < 0:
            raise ValueError("nonnegative integer scalar work required")
        if self.scalar+n > self.scalar_cap or time.perf_counter()-self.started > self.wall:
            raise TimeoutError("fixed analytic residual scalar/wall budget exceeded")
        self.scalar += n

    def physics(self, n: int) -> None:
        self.charge(0)
        if type(n) is not int or n < 1 or self.physical+n > self.physical_cap:
            raise TimeoutError("fixed joint residual physics cap exceeded")
        self.physical += n


def evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for name, digest in EXPECTED.items():
        path = ROOT/name
        value, audit = [json.loads((path/f).read_text()) for f in ("result.json", "audit.json")]
        if sha(path/"result.json") != digest or audit["result_sha256"] != digest:
            raise ValueError("immutable numerical/failed pilot anchors changed")
        audit_snapshot(ROOT, value["source"], value["config"])
        rows.append(value)
    if rows[1]["summary"]["decision"] != "NO_ADOPTION_PRECISION_UNRESOLVED":
        raise ValueError("original failed statistical decision must be preserved")
    return rows[0], rows[1]


def inputs(q0: dict[str, Any], q1: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"id": r["cell"]+"/Q0/"+r["role"], "profile": r["profile"],
        "reference_log_moments": [m["log_estimate"] for m in r["result"]["moments"]]} for r in q0["contexts"]]
    for b in q1["records"]:
        rows += [{"id": b["cell"]+"/Q1/"+str(b["block"])+"/"+str(i), "profile": r["profile"],
            "reference_log_moments": [m["log_estimate"] for m in r["moments"]["moments"]]} for i, r in enumerate(b["rows"])]
    if len(rows) != 262 or len({r["id"] for r in rows}) != 262:
        raise ValueError("all six and all 256 profiles required")
    return rows


def compare(fast: dict[str, Any], references: list[list[float]]) -> dict[str, Any]:
    if len(fast["rows"]) != len(references):
        raise ValueError("complete comparison required")
    errors = [abs(math.expm1(m["log_estimate"]-reference[k])) for r, reference in zip(fast["rows"], references, strict=True)
        for k, m in enumerate(r["moments"])]
    enclosed = all(m["log_monotone_envelope_lower"]-1e-9 <= reference[k] <= m["log_monotone_envelope_upper"]+1e-9
        for r, reference in zip(fast["rows"], references, strict=True) for k, m in enumerate(r["moments"]))
    maximum = max(errors)
    return {"maximum_relative_disagreement": maximum, "references_inside_numerical_envelopes": enclosed,
        "pass": fast["all_relative_numerically_resolved"] and maximum <= 1e-6 and enclosed}


def kernel_stage(data: list[dict[str, Any]], budget: Budget) -> dict[str, Any]:
    profiles = [LastVarianceResidualSlice(**r["profile"]) for r in data]
    reference = [r["reference_log_moments"] for r in data]
    constants = [LastVarianceResidualSlice(32, 0., 0., -4., 1., .2, 1., -.5, level) for level in (-2., 2.)]
    known = batched_moments(constants, charge=budget.charge)
    known_errors = []
    for p, known_row in zip(constants, known["rows"], strict=True):
        logg = float(slice_log_probability(p, np.array([0.]))[0])
        budget.charge(1)
        known_errors.append(max(abs(m["log_estimate"]-power*logg) for power, m in enumerate(known_row["moments"], 1)))
    fast = batched_moments(profiles, charge=budget.charge)
    validation = compare(fast, reference)
    # Fixed warmup: the first zero profile of each cell; no data-dependent selection.
    warmup_start = budget.scalar
    for j in (0, 3):
        log_gaussian_moments(partial(slice_log_probability, profiles[j]), charge=budget.charge)
    batched_moments([profiles[0], profiles[3]], charge=budget.charge)
    warmup_scalar = budget.scalar-warmup_start
    timings = []
    for order in (("adaptive", "batch"), ("batch", "adaptive"), ("adaptive", "batch")):
        row: dict[str, Any] = {"order": list(order)}
        repeated_fast, repeated_old = None, None
        for method in order:
            clock = time.perf_counter()
            before = budget.scalar
            if method == "adaptive":
                repeated_old = [log_gaussian_moments(partial(slice_log_probability, p), charge=budget.charge) for p in profiles]
            else:
                repeated_fast = batched_moments(profiles, charge=budget.charge)
            row[method+"_wall"] = time.perf_counter()-clock
            row[method+"_scalar_evaluations"] = budget.scalar-before
            budget.charge(0)
        assert repeated_old is not None and repeated_fast is not None
        old_logs = [[m["log_estimate"] for m in r["moments"]] for r in repeated_old]
        row["comparison"] = compare(repeated_fast, old_logs)
        row["all_adaptive_relative_resolved"] = all(r["relative_numerically_resolved"] for r in repeated_old)
        timings.append(row)
    ratio = statistics.median(r["adaptive_wall"] for r in timings)/statistics.median(r["batch_wall"] for r in timings)
    numerical = (validation["pass"] and known["all_relative_numerically_resolved"] and max(known_errors) <= 1e-10
        and all(r["comparison"]["pass"] and r["all_adaptive_relative_resolved"] for r in timings))
    return {"input_binding": [{"id": r["id"], "profile_digest": canonical_digest(r["profile"])} for r in data],
        "fast": fast, "validation": validation, "known_answer_maximum_log_errors": known_errors,
        "known_answers": known, "timings": timings, "median_scalar_kernel_speed_ratio": ratio,
        "warmup_scalar_evaluations": warmup_scalar, "known_answer_reference_evaluations": 2,
        "decision": "ADOPT_BATCH_DIAGNOSTIC_ONLY" if numerical and ratio >= 2 else "NO_ADOPTION_BATCH_KERNEL",
        "not_event_estimator_speedup_or_precision_qualification": True,
        "scalar_evaluations": budget.scalar, "wall": time.perf_counter()-budget.started}


def joint_context(cache: Any, spec: Any, z: torch.Tensor, budget: Budget) -> dict[str, Any]:
    budget.physics(1)
    profile = JointBlindResidualSlice.create(cache, z)
    pairs = np.array([(a, b) for a in (-2., 0., 2.) for b in (-2., 0., 2.)])
    points = z.repeat(9, 1)
    points[:, [57, 61]] = torch.tensor(pairs, dtype=torch.float64)
    budget.physics(9)
    full = cache.log_probability(points)
    sliced = profile.log_probability(pairs)
    budget.charge(9)
    error = float(np.max(np.abs(full.numpy()-sliced)))
    fixed = points.clone()
    fixed[:, [57, 61]] = 0
    support = bool((spec.means[:, [57, 61]] == 0).all()) and torch.equal(fixed, fixed[0].expand_as(fixed))
    return {"coordinates": z[0].tolist(), "profile": dataclasses.asdict(profile), "pairs": pairs.tolist(),
        "full_log_g": full.tolist(), "slice_log_g": sliced.tolist(), "maximum_log_error": error,
        "all_component_means_exactly_zero_in_both_coordinates": support,
        "factorization_for_any_mixture_weights": support, "pass": support and error <= 1e-8}


def joint_stage(q1: dict[str, Any], budget: Budget) -> dict[str, Any]:
    prepared, _ = prepare(q1["config"])
    cells: list[dict[str, Any]] = []
    for cell, (cache, spec) in prepared.items():
        contexts: list[dict[str, Any]] = []
        for role in ("zero", "first", "last"):
            if role == "zero":
                z = torch.zeros((1, 64), dtype=torch.float64)
            else:
                b = next(r for r in q1["records"] if r["cell"] == cell and r["block"] == (0 if role == "first" else 7))
                z = torch.tensor([b["rows"][0 if role == "first" else 15]["coordinates"]], dtype=torch.float64)
            contexts.append({"role": role, **joint_context(cache, spec, z, budget)})
        cells.append({"cell": cell, "contexts": contexts})
    return {"cells": cells, "all_pass": all(r["pass"] for c in cells for r in c["contexts"]),
        "physical_evaluations": budget.physical, "scalar_evaluations": budget.scalar,
        "not_joint_model_adoption_or_continuous_time_claim": True}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    q0, q1 = evidence()
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("one-shot analytic residual study; no overwrite")
    path.mkdir(parents=True, exist_ok=True)
    config = {"anchors": EXPECTED, "profiles": 262, "orders": [16, 32, 64], "cutoff": 16,
        "relative_tolerance": 1e-6, "speed_ratio_minimum": 2, "repetitions": 3,
        "kernel_scalar_cap": 12000000, "kernel_wall_cap": 600, "joint_physical_cap": 60}
    source = freeze_source(path/"run.json", config,
        extra_snapshot_paths=(PLAN, THEORY, Q0+"/result.json", Q0+"/audit.json", Q1+"/result.json", Q1+"/audit.json"))
    write_json_atomic_nonoverwriting(path/"protocol.json", {"config": config, "source": source})
    kernel_budget, joint_budget = Budget(12000000, 0, 600), Budget(1000, 60, 120)
    try:
        kernel = kernel_stage(inputs(q0, q1), kernel_budget)
        # The joint clock begins only when that separate stage starts.
        joint_budget = Budget(1000, 60, 120)
        joint = joint_stage(q1, joint_budget)
        oracle = marginal_risk_known_answer()
    except Exception as e:
        write_json_atomic_nonoverwriting(path/"failure.json", {"error": type(e).__name__+": "+str(e),
            "kernel_scalar": kernel_budget.scalar, "joint_scalar": joint_budget.scalar, "physical": joint_budget.physical})
        raise
    result = {"source": source, "config": config, "kernel": kernel, "joint": joint,
        "marginal_risk_known_answer": oracle, "original_Q1_decision_preserved": q1["summary"]["decision"],
        "P2_qualified": False, "D3_authorized": False, "large_training_authorized": False,
        "new_model_or_G_adopted": False}
    write_json_atomic_nonoverwriting(path/"result.json", result)
    return result


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    q0, q1 = evidence()
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, digest in result["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("frozen analytic residual source changed")
    data = inputs(q0, q1)
    expected_binding = [{"id": r["id"], "profile_digest": canonical_digest(r["profile"])} for r in data]
    if result["kernel"]["input_binding"] != expected_binding:
        raise ValueError("complete fixed profile binding changed")
    budget = Budget(3000000, 0, 120)
    replay = batched_moments([LastVarianceResidualSlice(**r["profile"]) for r in data], charge=budget.charge)
    if replay != result["kernel"]["fast"] or compare(replay, [r["reference_log_moments"] for r in data]) != result["kernel"]["validation"]:
        raise ValueError("complete 262-profile numerical replay failed")
    warmup_old_count = 0
    for j in (0, 3):
        old = log_gaussian_moments(partial(slice_log_probability, LastVarianceResidualSlice(**data[j]["profile"])), charge=budget.charge)
        warmup_old_count += old["scalar_evaluations"]
        if not compare(batched_moments([LastVarianceResidualSlice(**data[j]["profile"])], charge=budget.charge),
                [[m["log_estimate"] for m in old["moments"]]])["pass"]:
            raise ValueError("independent two-zero scalar replay failed")
    warmup_batch = batched_moments([LastVarianceResidualSlice(**data[j]["profile"]) for j in (0, 3)], charge=budget.charge)
    if result["kernel"]["warmup_scalar_evaluations"] != warmup_old_count+warmup_batch["scalar_evaluations_including_padding_and_endpoints"]:
        raise ValueError("warmup scalar accounting changed")
    timings = result["kernel"]["timings"]
    if [r["order"] for r in timings] != [["adaptive", "batch"], ["batch", "adaptive"], ["adaptive", "batch"]]:
        raise ValueError("benchmark order changed")
    if any(not math.isfinite(r[k+"_wall"]) or r[k+"_wall"] <= 0 for r in timings for k in ("adaptive", "batch")):
        raise ValueError("positive benchmark clocks required")
    expected_count = (result["kernel"]["known_answers"]["scalar_evaluations_including_padding_and_endpoints"]+2
        +replay["scalar_evaluations_including_padding_and_endpoints"]+result["kernel"]["warmup_scalar_evaluations"]
        +sum(r[k+"_scalar_evaluations"] for r in timings for k in ("adaptive", "batch")))
    ratio = statistics.median(r["adaptive_wall"] for r in timings)/statistics.median(r["batch_wall"] for r in timings)
    numerical = (result["kernel"]["validation"]["pass"] and result["kernel"]["known_answers"]["all_relative_numerically_resolved"]
        and max(result["kernel"]["known_answer_maximum_log_errors"]) <= 1e-10
        and all(r["comparison"]["pass"] and r["all_adaptive_relative_resolved"] for r in timings))
    decision = "ADOPT_BATCH_DIAGNOSTIC_ONLY" if numerical and ratio >= 2 else "NO_ADOPTION_BATCH_KERNEL"
    if (expected_count != result["kernel"]["scalar_evaluations"] or ratio != result["kernel"]["median_scalar_kernel_speed_ratio"]
            or decision != result["kernel"]["decision"]):
        raise ValueError("kernel accounting or adoption decision changed")
    prepared, _ = prepare(q1["config"])
    jb = Budget(1000, 20, 60)
    for c in result["joint"]["cells"]:
        if [r["role"] for r in c["contexts"]] != ["zero", "first", "last"]:
            raise ValueError("complete fixed joint context family required")
        for row in c["contexts"]:
            if row["role"] == "zero":
                expected_coordinates = [0.]*64
            else:
                old_block = next(b for b in q1["records"] if b["cell"] == c["cell"] and b["block"] == (0 if row["role"] == "first" else 7))
                expected_coordinates = old_block["rows"][0 if row["role"] == "first" else 15]["coordinates"]
            if row["coordinates"] != expected_coordinates:
                raise ValueError("fixed joint context binding changed")
            fields = dict(row["profile"])
            for name in ("coordinates", "variance_indices", "log_variances", "dws"):
                fields[name] = tuple(fields[name])
            jb.charge(9)
            scalar = JointBlindResidualSlice(**fields).log_probability(np.array(row["pairs"]))
            error = float(np.max(np.abs(np.array(row["full_log_g"])-scalar)))
            if error != row["maximum_log_error"] or scalar.tolist() != row["slice_log_g"] or not row["pass"]:
                raise ValueError("all stored joint scalar identities failed")
        saved = next(r for r in c["contexts"] if r["role"] == "zero")
        cache, spec = prepared[c["cell"]]
        fresh = joint_context(cache, spec, torch.zeros((1, 64), dtype=torch.float64), jb)
        if canonical_digest(fresh) != canonical_digest({k: v for k, v in saved.items() if k != "role"}):
            raise ValueError("two-zero joint full-law replay mismatch")
    if marginal_risk_known_answer() != result["marginal_risk_known_answer"] or result["P2_qualified"]:
        raise ValueError("finite risk oracle or decision changed")
    return {"status": "all_batch_profiles_two_adaptive_zeros_two_joint_zeros_and_finite_risk_oracle_replayed",
        "result_sha256": sha(path/"result.json"), "kernel_audit_scalar": budget.scalar,
        "joint_audit_physical": jb.physical, "joint_audit_scalar": jb.scalar,
        "combined_physical": result["joint"]["physical_evaluations"]+jb.physical,
        "kernel_decision": result["kernel"]["decision"], "P2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        value = audit()
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", value)
        print(json.dumps(value))
    else:
        value = run()
        print(json.dumps({"kernel": value["kernel"]["decision"], "joint_pass": value["joint"]["all_pass"],
            "scalar": value["kernel"]["scalar_evaluations"], "speed_ratio": value["kernel"]["median_scalar_kernel_speed_ratio"]}))
