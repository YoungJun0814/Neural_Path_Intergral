"""Explicit serialization-only recovery, preserving failed work and a narrower joint scope."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from functools import partial
from typing import Any

import numpy as np
import torch

from experiments.post_audit_analytic_residual_batch import DIRECTORY as FAILED
from experiments.post_audit_analytic_residual_batch import (
    PLAN,
    Q0,
    Q1,
    THEORY,
    Budget,
    compare,
    evidence,
    inputs,
    joint_context,
    kernel_stage,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.analytic_residual_batch import batched_moments
from src.path_integral.marginal_risk_contract import marginal_risk_known_answer
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.residual_log_quadrature import log_gaussian_moments, slice_log_probability
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/analytic_residual_batch_v1/session_20261021_serialization_recovery"
ADDENDUM = "docs/plans/ANALYTIC_RESIDUAL_SERIALIZATION_RECOVERY_2026-10-21_KO.md"


def json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    return value


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    q0, q1 = evidence()
    failed = json.loads((ROOT/FAILED/"serialization_failure.json").read_text())
    protocol = json.loads((ROOT/FAILED/"protocol.json").read_text())
    audit_snapshot(ROOT, protocol["source"], protocol["config"])
    if failed["status"] != "RESULT_PERSISTENCE_FAILED_AFTER_COMPUTATION" or (ROOT/FAILED/"result.json").exists():
        raise ValueError("specific immutable failed publication attempt required")
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("one-shot serialization recovery; no overwrite")
    path.mkdir(parents=True, exist_ok=True)
    config = {"original_failure_sha256": sha(ROOT/FAILED/"serialization_failure.json"),
        "profiles": 262, "repetitions": 3, "relative_tolerance": 1e-6, "speed_ratio_minimum": 2,
        "recovery_kernel_cap": 12000000, "audit_kernel_cap": 3000000,
        "failed_kernel_upper_reservation": 12000000, "joint_recovery_contexts": "two_zero_only",
        "failed_joint_physical": 60, "recovery_joint_physical": 20, "audit_joint_physical": 20}
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, ADDENDUM,
        FAILED+"/protocol.json", FAILED+"/serialization_failure.json", Q0+"/result.json", Q1+"/result.json"))
    write_json_atomic_nonoverwriting(path/"protocol.json", {"config": config, "source": source})
    kb, jb = Budget(12000000, 0, 600), Budget(1000, 20, 120)
    try:
        kernel = kernel_stage(inputs(q0, q1), kb)
        jb = Budget(1000, 20, 120)
        prepared, _ = prepare(q1["config"])
        cells: list[dict[str, Any]] = [{"cell": cell, "contexts": [{"role": "zero", **joint_context(cache, spec, torch.zeros((1, 64), dtype=torch.float64), jb)}]}
            for cell, (cache, spec) in prepared.items()]
        result = json_compatible({"source": source, "config": config, "kernel": kernel,
            "joint": {"cells": cells, "all_pass": all(r["pass"] for c in cells for r in c["contexts"]),
                "physical_evaluations": jb.physical, "scalar_evaluations": jb.scalar,
                "coverage": "two_zero_contexts_only_not_original_six"},
            "marginal_risk_known_answer": marginal_risk_known_answer(),
            "original_Q1_decision_preserved": q1["summary"]["decision"],
            "first_run_timing_lost_and_not_compared": True, "P2_qualified": False,
            "D3_authorized": False, "large_training_authorized": False, "new_model_or_G_adopted": False})
        write_json_atomic_nonoverwriting(path/"result.json", result)
    except Exception as e:
        write_json_atomic_nonoverwriting(path/"failure.json", {"error": type(e).__name__+": "+str(e),
            "kernel_scalar": kb.scalar, "joint_scalar": jb.scalar, "physical": jb.physical})
        raise
    return result


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    q0, q1 = evidence()
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, digest in result["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("frozen recovery source changed")
    data = inputs(q0, q1)
    binding = [{"id": r["id"], "profile_digest": canonical_digest(r["profile"])} for r in data]
    if result["kernel"]["input_binding"] != binding:
        raise ValueError("complete recovery profile binding failed")
    budget = Budget(3000000, 0, 120)
    profiles = [LastVarianceResidualSlice(**r["profile"]) for r in data]
    replay = json_compatible(batched_moments(profiles, charge=budget.charge))
    validation = compare(replay, [r["reference_log_moments"] for r in data])
    if replay != result["kernel"]["fast"] or validation != result["kernel"]["validation"]:
        raise ValueError("complete recovery batch replay failed")
    warmup_old_count = 0
    for j in (0, 3):
        old = log_gaussian_moments(partial(slice_log_probability, profiles[j]), charge=budget.charge)
        warmup_old_count += old["scalar_evaluations"]
        if not compare(batched_moments([profiles[j]], charge=budget.charge), [[m["log_estimate"] for m in old["moments"]]])["pass"]:
            raise ValueError("two adaptive zero profile replay failed")
    warmup_batch = batched_moments([profiles[0], profiles[3]], charge=budget.charge)
    expected_warmup = warmup_old_count+warmup_batch["scalar_evaluations_including_padding_and_endpoints"]
    timings = result["kernel"]["timings"]
    if [r["order"] for r in timings] != [["adaptive", "batch"], ["batch", "adaptive"], ["adaptive", "batch"]]:
        raise ValueError("recovery timing order changed")
    if any(not math.isfinite(r[k+"_wall"]) or r[k+"_wall"] <= 0 for r in timings for k in ("adaptive", "batch")):
        raise ValueError("positive finite kernel clocks required")
    expected_count = (result["kernel"]["known_answers"]["scalar_evaluations_including_padding_and_endpoints"]+2
        +replay["scalar_evaluations_including_padding_and_endpoints"]+expected_warmup
        +sum(r[k+"_scalar_evaluations"] for r in timings for k in ("adaptive", "batch")))
    ratio = statistics.median(r["adaptive_wall"] for r in timings)/statistics.median(r["batch_wall"] for r in timings)
    numerical = (validation["pass"] and result["kernel"]["known_answers"]["all_relative_numerically_resolved"]
        and max(result["kernel"]["known_answer_maximum_log_errors"]) <= 1e-10
        and all(r["comparison"]["pass"] and r["all_adaptive_relative_resolved"] for r in timings))
    decision = "ADOPT_BATCH_DIAGNOSTIC_ONLY" if numerical and ratio >= 2 else "NO_ADOPTION_BATCH_KERNEL"
    if (expected_warmup != result["kernel"]["warmup_scalar_evaluations"] or expected_count != result["kernel"]["scalar_evaluations"]
            or ratio != result["kernel"]["median_scalar_kernel_speed_ratio"] or decision != result["kernel"]["decision"]):
        raise ValueError("recovery decision/accounting changed")
    prepared, _ = prepare(q1["config"])
    jb = Budget(1000, 20, 60)
    if len(result["joint"]["cells"]) != 2 or {c["cell"] for c in result["joint"]["cells"]} != set(prepared):
        raise ValueError("two complete zero-context cells required")
    for c in result["joint"]["cells"]:
        if len(c["contexts"]) != 1 or c["contexts"][0]["role"] != "zero":
            raise ValueError("narrowed joint recovery scope changed")
        cache, spec = prepared[c["cell"]]
        fresh = json_compatible(joint_context(cache, spec, torch.zeros((1, 64), dtype=torch.float64), jb))
        if fresh != {k: v for k, v in c["contexts"][0].items() if k != "role"}:
            raise ValueError("two joint zero full-law replay failed")
    if (result["joint"]["physical_evaluations"] != 20 or jb.physical != 20
            or marginal_risk_known_answer() != result["marginal_risk_known_answer"] or result["P2_qualified"]):
        raise ValueError("risk oracle or recovery physical accounting failed")
    return {"status": "recovery_all_262_batch_two_adaptive_zeros_two_joint_zeros_and_oracle_replayed",
        "result_sha256": sha(path/"result.json"), "kernel_audit_scalar": budget.scalar,
        "joint_audit_physical": jb.physical, "joint_audit_scalar": jb.scalar,
        "physical_total_including_failed_original_inferred": 100,
        "failed_original_kernel_scalar_upper_reservation": 12000000,
        "recovery_kernel_scalar": result["kernel"]["scalar_evaluations"],
        "kernel_decision": decision, "P2_qualified": False}


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
