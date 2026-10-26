"""Prespecified mixture oracle, then one fixed-work temperature allocation contrast."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import torch

from experiments.post_audit_analytic_residual_batch import Budget
from experiments.post_audit_conditional_risk_target_smc import DIRECTORY as OLD
from experiments.post_audit_conditional_risk_target_smc import PARENT as IID
from experiments.post_audit_conditional_risk_target_smc import (
    VolterraRiskRecorder,
    one_shot,
    tensor_digest,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.conditional_marginal_g import ACTIVE, json_compatible
from src.path_integral.conditional_risk_target_smc import (
    STREAMS,
    Arm,
    TargetSMCConfig,
    island_gate,
    lineage_summary,
    mode_diagnostics,
    normalizer_summary,
    run_target_smc,
)
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.smc_mixture_oracle import (
    final_moments,
    mixture_evaluation,
    mixture_spec,
    oracle_summary,
    reference_suite,
)
from src.path_integral.structural_v2_inventory import audit_snapshot

PLAN = "docs/plans/MIXTURE_ORACLE_AND_LATE_BRIDGE_2026-10-23_KO.md"
ORACLE = "results/post_audit/mixture_smc_oracle_v1/session_20261023"
VOLTERRA = "results/post_audit/late_bridge_schedule_v1/session_20261023"
EXPECTED = "ac83eb5aafca2ca804ea779a9a0165b11299c6f3c20b094221a19fdbe2f4999a"
EXPECTED_AUDIT = "d641979704fdc808c67bb5e394659c8cda1cd31c495cfb83eaea3f1915e10f32"
NAMESPACE = "mixture-oracle-late-bridge-v1"


def parent_evidence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT/OLD
    parent, audit = [json.loads((path/n).read_text()) for n in ("result.json", "audit.json")]
    if (sha(path/"result.json") != EXPECTED or sha(path/"audit.json") != EXPECTED_AUDIT
            or audit["result_sha256"] != EXPECTED
            or parent["summary"]["decision"] != "STOP_GENEALOGY_OR_MUTATION_UNQUALIFIED"):
        raise ValueError("immutable audited preceding SMC STOP required")
    audit_snapshot(ROOT, parent["source"], parent["config"])
    iid = json.loads((ROOT/IID/"result.json").read_text())
    if sha(ROOT/IID/"result.json") != parent["config"]["parent_sha256"]:
        raise ValueError("preceding IID seed-ledger source changed")
    return parent, [parent["seeds"]["ledger"], iid["seeds"]["ledger"]]


def jobs(phase: str, cells: list[str]) -> list[dict[str, Any]]:
    if phase == "oracle":
        return [{"cell": "mixture-1d", "schedule": s, "arm": a, "replicate": r}
                for s in ("quartic", "linear") for a in ("prior", "component") for r in range(16)]
    if phase != "volterra" or len(cells) != 2 or len(set(cells)) != 2:
        raise ValueError("two declared Volterra cells or oracle phase required")
    return [{"cell": c, "schedule": s, "arm": "component", "replicate": r}
            for c in cells for r in range(4) for s in ("quartic", "linear")]


def identity(job: dict[str, Any]) -> str:
    return f"{job['cell']}-{job['schedule']}-{job['arm']}-{job['replicate']}"


def configuration(job: dict[str, Any], particles: int) -> TargetSMCConfig:
    if job["schedule"] not in ("quartic", "linear") or job["arm"] not in ("prior", "component"):
        raise ValueError("declared schedule and arm required")
    exponent = 4 if job["schedule"] == "quartic" else 1
    return TargetSMCConfig(particles, tuple((j/32)**exponent for j in range(33)), cast(Arm, job["arm"]), .35, 4)


def seal(phase: str, planned: list[dict[str, Any]], previous: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = SeedLedger()
    mapping = {identity(job): {stream: ledger.allocate(SeedKey(NAMESPACE+"/"+phase,
        job["schedule"]+"/"+job["arm"], "development", job["cell"], 1 if phase == "oracle" else 32,
        job["replicate"], stream)) for stream in STREAMS} for job in planned}
    old = {r.seed for payload in previous for r in SeedLedger.from_dict(payload).records}
    if old & {r.seed for r in ledger.records}:
        raise ValueError("scoped new/previous stream collision")
    return mapping, {"ledger": ledger.to_dict(), "sha256": ledger.sha256,
        "previous_seeds_compared": len(old), "collision_count": 0, "not_whole_history_or_PRNG_independence_proof": True}


def assert_current(source: dict[str, Any], config: dict[str, Any]) -> None:
    audit_snapshot(ROOT, source, config)
    if any(sha(ROOT/n) != h for n, h in source["snapshot_file_hashes"].items()):
        raise ValueError("frozen source/config/plan/parent changed")


def check_protocol(value: dict[str, Any], protocol: dict[str, Any], mapping: dict[str, Any], ledger: dict[str, Any]) -> None:
    if (value["seeds"] != ledger or protocol["mapping"] != mapping or protocol["seeds"] != ledger
            or protocol["source"] != value["source"] or protocol["config"] != value["config"]):
        raise ValueError("full protocol source/config/stream binding changed")


class GaussianBudget:
    def __init__(self) -> None:
        self.contexts = 0
        self.started = time.perf_counter()

    def charge(self, n: int) -> None:
        if type(n) is not int or n < 0 or self.contexts+n > 3200000 or time.perf_counter()-self.started > 300:
            raise TimeoutError("fixed Gaussian oracle context/wall budget exceeded")
        self.contexts += n


def oracle_island(job: dict[str, Any], streams: dict[str, int], budget: GaussianBudget) -> dict[str, Any]:
    chain = hashlib.sha256(b"mixture-smc-oracle-v1").digest()
    calls = 0
    before = budget.contexts

    def evaluate(x: torch.Tensor):
        nonlocal chain, calls
        budget.charge(len(x))
        v = mixture_evaluation(x)
        chain = hashlib.sha256(chain+bytes.fromhex(tensor_digest(x, v.log_m2, v.log_dq, v.features))).digest()
        calls += 1
        return v

    result = run_target_smc(mixture_spec(), evaluate, config=configuration(job, 512), seeds=streams)
    return {"job": job, "replicate": job["replicate"], "seeds": streams, "log_normalizer": result["log_normalizer"],
        "moments": final_moments(result), "full_result_sha256": canonical_digest(result),
        "potential_trace_sha256": chain.hex(), "contexts": budget.contexts-before, "calls": calls,
        "final_particle_gate_not_used_as_oracle_accuracy_gate": True}


def summarize_oracle(records: list[dict[str, Any]], refs: dict[str, Any]) -> dict[str, Any]:
    if [r["job"] for r in records] != jobs("oracle", []):
        raise ValueError("all64 oracle jobs in fixed order required")
    summaries = [{"schedule": s, "arm": a, **oracle_summary([r for r in records
        if r["job"]["schedule"] == s and r["job"]["arm"] == a], refs["rows"][-1])}
        for s in ("quartic", "linear") for a in ("prior", "component")]
    passed = refs["pass"] and all(r["pass"] for r in summaries)
    return {"conditions": summaries, "pass": passed,
        "decision": "PASS_MIXTURE_ORACLE_LIMITED" if passed else "STOP_MIXTURE_ORACLE_UNRESOLVED",
        "only_authorizes_prespecified_schedule_development": passed, "P2_qualified": False}


def run_oracle() -> dict[str, Any]:
    torch.set_num_threads(1)
    end_started = time.perf_counter()

    def body(path: Path, progress: dict[str, Any]) -> dict[str, Any]:
        _, previous = parent_evidence()
        planned = jobs("oracle", [])
        mapping, ledger = seal("oracle", planned, previous)
        config = {"phase": "oracle", "namespace": NAMESPACE, "particles": 512, "replicates": 16,
            "jobs": planned, "contexts_cap": 3200000, "compute_wall_cap": 300,
            "parent_sha256": EXPECTED, "model": "1D q=.2N0+.8N2; m2=exp(-.4(x-1)^2-.3)"}
        progress["phase"] = "source_freeze"
        clock = time.perf_counter()
        source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, OLD+"/result.json", OLD+"/audit.json"))
        source_wall = time.perf_counter()-clock
        write(path/"protocol.json", json_compatible({"source": source, "config": config, "mapping": mapping, "seeds": ledger}))
        progress["phase"] = "deterministic_reference"
        clock = time.perf_counter()
        refs = reference_suite()
        reference_wall = time.perf_counter()-clock
        write(path/"reference.json", json_compatible(refs))
        budget, records, artifacts = GaussianBudget(), [], []
        if refs["pass"]:
            for job in planned:
                progress["phase"] = identity(job)
                try:
                    record = oracle_island(job, mapping[identity(job)], budget)
                finally:
                    progress["Gaussian_contexts"] = budget.contexts
                name = identity(job)+".json"
                write(path/name, json_compatible(record))
                artifacts.append({"path": name, "sha256": sha(path/name)})
                records.append(record)
            summary = summarize_oracle(records, refs)
        else:
            summary = {"pass": False, "decision": "STOP_DETERMINISTIC_MIXTURE_REFERENCE_UNRESOLVED", "P2_qualified": False}
        value = json_compatible({"config": config, "source": source, "seeds": ledger, "artifacts": artifacts,
            "reference_sha256": sha(path/"reference.json"), "summary": summary, "Gaussian_contexts": budget.contexts,
            "Volterra_physical": 0, "scalar_reference_evaluations": refs["scalar_reference_evaluations"],
            "cost": {"reference_wall": reference_wall, "source_freeze": source_wall,
                "island_compute_and_io": time.perf_counter()-budget.started}})
        progress["phase"] = "result_io"
        clock = time.perf_counter()
        write(path/"result.json", value)
        write(path/"end_to_end.json", {"result_sha256": sha(path/"result.json"),
            "result_io": time.perf_counter()-clock, "wall_before_receipt": time.perf_counter()-end_started})
        return value

    return one_shot(ROOT/ORACLE, body)


def audit_oracle() -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    path = ROOT/ORACLE
    value = json.loads((path/"result.json").read_text())
    assert_current(value["source"], value["config"])
    _, previous = parent_evidence()
    planned = jobs("oracle", [])
    mapping, ledger = seal("oracle", planned, previous)
    protocol = json.loads((path/"protocol.json").read_text())
    check_protocol(value, protocol, mapping, ledger)
    refs = json.loads((path/"reference.json").read_text())
    if sha(path/"reference.json") != value["reference_sha256"] or reference_suite() != refs:
        raise ValueError("independent deterministic oracle reference changed")
    budget, records = GaussianBudget(), []
    if refs["pass"]:
        for job, item in zip(planned, value["artifacts"], strict=True):
            if item["path"] != identity(job)+".json" or sha(path/item["path"]) != item["sha256"]:
                raise ValueError("oracle complete artifact inventory changed")
            record = json.loads((path/item["path"]).read_text())
            if oracle_island(job, mapping[identity(job)], budget) != record:
                raise ValueError("full oracle island replay failed")
            records.append(record)
        if summarize_oracle(records, refs) != value["summary"] or budget.contexts != value["Gaussian_contexts"]:
            raise ValueError("oracle summary/work changed")
    if json.loads((path/"end_to_end.json").read_text())["result_sha256"] != sha(path/"result.json"):
        raise ValueError("oracle end-to-end binding changed")
    return {"status": "all_oracle_references_and_islands_replayed", "result_sha256": sha(path/"result.json"),
        "Gaussian_contexts": budget.contexts, "Volterra_physical": 0,
        "wall": time.perf_counter()-started, "decision": value["summary"]["decision"]}


def qualified_oracle() -> dict[str, Any]:
    path = ROOT/ORACLE
    value, audit = [json.loads((path/n).read_text()) for n in ("result.json", "audit.json")]
    if (not value["summary"]["pass"] or value["summary"]["decision"] != "PASS_MIXTURE_ORACLE_LIMITED"
            or audit["result_sha256"] != sha(path/"result.json")
            or audit["status"] != "all_oracle_references_and_islands_replayed"):
        raise ValueError("passed and fully replayed nontrivial mixture oracle required")
    assert_current(value["source"], value["config"])
    return value


def volterra_island(cache: Any, q: GaussianMixtureShiftSpec, job: dict[str, Any], streams: dict[str, int],
                    budget: Budget) -> dict[str, Any]:
    recorder = VolterraRiskRecorder(cache, q, budget)
    stages: list[dict[str, Any]] = []
    before_p, before_s, started = budget.physical, budget.scalar, time.perf_counter()
    outcome: dict[str, Any]
    try:
        data = run_target_smc(q, recorder, config=configuration(job, 128), seeds=streams, observer=stages.append)
        outcome = {"status": "complete", "result": data}
    except Exception as error:
        category = "RESOURCE" if isinstance(error, TimeoutError) else (
            "NUMERICAL" if isinstance(error, (FloatingPointError, ValueError)) else "TECHNICAL")
        outcome = {"status": "failed", "failure_type": category, "error": type(error).__name__+": "+str(error),
            "completed_stages": stages}
    return json_compatible({"job": job, "seeds": streams, **outcome, "trace": recorder.trace,
        "trace_sha256": recorder.chain.hex(), "physical": budget.physical-before_p,
        "scalar": budget.scalar-before_s, "wall": time.perf_counter()-started, "potential_wall": recorder.wall})


def summarize_volterra(records: list[dict[str, Any]], cells: list[str]) -> dict[str, Any]:
    if [r["job"] for r in records] != jobs("volterra", cells):
        raise ValueError("all sixteen fresh schedule jobs required in fixed order")
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        schedules: dict[str, Any] = {}
        for s in ("quartic", "linear"):
            rows = [r for r in records if r["job"]["cell"] == cell and r["job"]["schedule"] == s]
            if all(r["status"] == "complete" for r in rows):
                schedules[s] = {"complete": True, "normalizer": normalizer_summary([r["result"]["log_normalizer"] for r in rows]),
                    "island_gates": [r["result"]["gate"]["pass"] for r in rows],
                    "modes": mode_diagnostics([r["result"] for r in rows]) if s == "linear" else None}
            else:
                schedules[s] = {"complete": False}
        contrast: dict[str, Any] = {"accuracy_qualified": False, "no_detected_contradiction": None}
        if all(schedules[s]["complete"] and schedules[s]["normalizer"]["qualified"] for s in schedules):
            a, b = [schedules[s]["normalizer"] for s in ("quartic", "linear")]
            scale = max(a["log_mean_normalizer"], b["log_mean_normalizer"])
            x, y = math.exp(a["log_mean_normalizer"]-scale), math.exp(b["log_mean_normalizer"]-scale)
            contrast = {"accuracy_qualified": True, "no_detected_contradiction": abs(x-y) <=
                3.291*math.hypot(x*a["rse"], y*b["rse"])+1e-12*max(x, y),
                "not_equivalence_or_population_accuracy_proof": True}
        summaries.append({"cell": cell, "schedules": schedules, "contrast": contrast})
    if any(r.get("failure_type") == "RESOURCE" for r in records):
        decision = "STOP_RESOURCE_LIMIT"
    elif any(r.get("failure_type") == "TECHNICAL" for r in records):
        decision = "STOP_TECHNICAL_FAILURE"
    elif any(r["status"] != "complete" for r in records):
        decision = "STOP_NUMERICAL_TARGET_UNRESOLVED"
    elif not all(all(s["schedules"]["linear"]["island_gates"]) for s in summaries):
        decision = "STOP_LATE_BRIDGE_BANK_UNQUALIFIED"
    elif not all(s["schedules"]["linear"]["normalizer"]["qualified"] and s["schedules"]["linear"]["modes"]["pass"] for s in summaries):
        decision = "STOP_LATE_BRIDGE_ISLAND_OR_MODE_UNRESOLVED"
    elif not all(s["contrast"]["accuracy_qualified"] and s["contrast"]["no_detected_contradiction"] for s in summaries):
        decision = "STOP_LATE_BRIDGE_REFERENCE_UNRESOLVED"
    else:
        decision = "LATE_BRIDGE_BANK_CANDIDATE_ONLY"
    return {"cells": summaries, "decision": decision, "actual_G_fits": 0, "event_q_changed": False,
        "P2_qualified": False, "D3_authorized": False, "large_training_authorized": False,
        "not_fixed_precision_or_population_speedup": True}


def run_volterra() -> dict[str, Any]:
    torch.set_num_threads(1)
    end_started = time.perf_counter()

    def body(path: Path, progress: dict[str, Any]) -> dict[str, Any]:
        parent, previous = parent_evidence()
        oracle = qualified_oracle()
        previous.append(oracle["seeds"]["ledger"])
        config = {"phase": "volterra", "namespace": NAMESPACE, "model": parent["config"]["model"],
            "cells": parent["config"]["cells"], "particles": 128, "replicates": 4,
            "schedules": {s: list(configuration({"schedule": s, "arm": "component"}, 128).temperatures) for s in ("quartic", "linear")},
            "scale": .35, "resample_every": 4, "feature_variance_floor": 1e-8,
            "physical_cap": 200000, "scalar_cap": 800000000, "wall_cap": 2400,
            "parent_sha256": EXPECTED, "oracle_sha256": sha(ROOT/ORACLE/"result.json")}
        progress["phase"] = "setup"
        clock = time.perf_counter()
        prepared, setup = prepare(config)
        setup_wall = time.perf_counter()-clock
        reduced = {c: GaussianMixtureShiftSpec(q.means[:, ACTIVE].clone(), q.weights.clone()) for c, (_, q) in prepared.items()}
        planned = jobs("volterra", list(prepared))
        mapping, ledger = seal("volterra", planned, previous)
        progress["phase"] = "source_freeze"
        clock = time.perf_counter()
        source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, OLD+"/result.json", OLD+"/audit.json", ORACLE+"/result.json", ORACLE+"/audit.json"))
        source_wall = time.perf_counter()-clock
        clock = time.perf_counter()
        write(path/"protocol.json", json_compatible({"source": source, "config": config, "mapping": mapping, "seeds": ledger}))
        protocol_io = time.perf_counter()-clock
        budget, records, artifacts, io_wall = Budget(800000000, 200000, 2400), [], [], 0.
        for job in planned:
            progress["phase"] = identity(job)
            budget.charge(0)
            record = volterra_island(prepared[job["cell"]][0], reduced[job["cell"]], job, mapping[identity(job)], budget)
            progress.update({"physical": budget.physical, "scalar": budget.scalar})
            clock = time.perf_counter()
            name = identity(job)+".json"
            write(path/name, record)
            io_wall += time.perf_counter()-clock
            artifacts.append({"path": name, "sha256": sha(path/name)})
            records.append(record)
            d = record.get("result", {}).get("diagnostics", {})
            print(f"Completed {len(records)}/16 {identity(job)}: {record['status']}, ancestor ESS={d.get('weighted_ancestor_ess')}", flush=True)
        clock = time.perf_counter()
        summary = summarize_volterra(records, list(prepared))
        aggregation = time.perf_counter()-clock
        result = json_compatible({"source": source, "config": config, "seeds": ledger, "artifacts": artifacts,
            "summary": summary, "physical": budget.physical, "scalar": budget.scalar,
            "cost": {"setup": setup, "setup_wall": setup_wall, "source_freeze": source_wall,
                "protocol_io": protocol_io, "island_io": io_wall, "aggregation": aggregation,
                "compute_and_island_io": time.perf_counter()-budget.started}})
        progress["phase"] = "result_io"
        clock = time.perf_counter()
        write(path/"result.json", result)
        write(path/"end_to_end.json", {"result_sha256": sha(path/"result.json"),
            "result_io": time.perf_counter()-clock, "wall_before_receipt": time.perf_counter()-end_started,
            "audit_and_tests_excluded": True})
        return result

    return one_shot(ROOT/VOLTERRA, body)


def audit_volterra() -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    _, previous = parent_evidence()
    oracle = qualified_oracle()
    previous.append(oracle["seeds"]["ledger"])
    path = ROOT/VOLTERRA
    result = json.loads((path/"result.json").read_text())
    assert_current(result["source"], result["config"])
    if result["config"]["oracle_sha256"] != sha(ROOT/ORACLE/"result.json"):
        raise ValueError("qualified oracle binding changed")
    prepared, _ = prepare(result["config"])
    reduced = {c: GaussianMixtureShiftSpec(q.means[:, ACTIVE].clone(), q.weights.clone()) for c, (_, q) in prepared.items()}
    planned = jobs("volterra", list(prepared))
    mapping, ledger = seal("volterra", planned, previous)
    protocol = json.loads((path/"protocol.json").read_text())
    check_protocol(result, protocol, mapping, ledger)
    budget, records, replays, maximum = Budget(120000000, 30000, 600), [], [], 0.
    for item, job in zip(result["artifacts"], planned, strict=True):
        if item["path"] != identity(job)+".json" or sha(path/item["path"]) != item["sha256"]:
            raise ValueError("complete schedule artifact inventory changed")
        record = json.loads((path/item["path"]).read_text())
        if record["job"] != job or record["seeds"] != mapping[identity(job)]:
            raise ValueError("schedule identity/stream binding changed")
        chain = hashlib.sha256(b"conditional-risk-target-smc-v1").digest()
        for i, row in enumerate(record["trace"]):
            if row["call"] != i or row["count"] != 128:
                raise ValueError("complete potential trace changed")
            chain = hashlib.sha256(chain+bytes.fromhex(canonical_digest(row))).digest()
        if (chain.hex() != record["trace_sha256"] or sum(t["physical"] for t in record["trace"]) != record["physical"]
                or sum(t["scalar"] for t in record["trace"]) != record["scalar"]):
            raise ValueError("schedule trace/work binding changed")
        cache, q = prepared[job["cell"]][0], reduced[job["cell"]]
        if record["status"] == "complete":
            data = record["result"]
            x = torch.tensor(data["final_x"], dtype=torch.float64)
            w = torch.tensor(data["final_weights"], dtype=torch.float64)
            lineage = torch.tensor(data["initial_lineage"], dtype=torch.int64)
            ds = lineage_summary(lineage, w, x)
            if ds != data["diagnostics"] or island_gate(ds, data["stages"], torch.tensor(data["late_jump_by_particle"], dtype=torch.float64)) != data["gate"]:
                raise ValueError("schedule lineage/mutation gate changed")
            if record["physical"] != 12416 or data["potential_contexts"] != 12416 or len(record["trace"]) != 97:
                raise ValueError("fixed schedule work changed")
            for i, stage in enumerate(data["stages"]):
                if (stage["stage"] != i or stage["beta"] != configuration(job, 128).temperatures[i+1]
                        or stage["resampled"] != ((i+1) % 4 == 0 and i != 31)):
                    raise ValueError("fixed schedule calendar changed")
            fresh = VolterraRiskRecorder(cache, q, budget)(x)
            error = float((fresh.log_m2-torch.tensor(data["final_log_m2"], dtype=torch.float64)).abs().max())
            maximum = max(maximum, error)
            if (error > 1e-8 or not torch.allclose(fresh.log_dq, torch.tensor(data["final_log_dq"], dtype=torch.float64), rtol=0, atol=1e-10)
                    or not torch.allclose(fresh.features, torch.tensor(data["final_features"], dtype=torch.float64), rtol=1e-12, atol=1e-10)):
                raise ValueError("all final conditional target replay failed")
        if job["schedule"] == "linear" and job["replicate"] == 0:
            fresh_record = volterra_island(cache, q, job, mapping[identity(job)], budget)
            if {k: v for k, v in fresh_record.items() if k not in ("wall", "potential_wall")} != {
                    k: v for k, v in record.items() if k not in ("wall", "potential_wall")}:
                raise ValueError("two primary full-island replay failed")
            replays.append(identity(job))
        records.append(record)
    if (summarize_volterra(records, list(prepared)) != result["summary"]
            or sum(r["physical"] for r in records) != result["physical"]
            or sum(r["scalar"] for r in records) != result["scalar"]):
        raise ValueError("whole schedule summary/work changed")
    if json.loads((path/"end_to_end.json").read_text())["result_sha256"] != sha(path/"result.json"):
        raise ValueError("schedule end-to-end result binding changed")
    return {"status": "all_schedule_statistics_final_targets_and_two_primary_islands_replayed",
        "result_sha256": sha(path/"result.json"), "physical": budget.physical, "scalar": budget.scalar,
        "full_island_replays": replays, "maximum_final_log_m2_error": maximum,
        "wall": time.perf_counter()-started, "decision": result["summary"]["decision"], "P2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("oracle", "volterra"))
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        value = audit_oracle() if args.phase == "oracle" else audit_volterra()
        write(ROOT/(ORACLE if args.phase == "oracle" else VOLTERRA)/"audit.json", json_compatible(value))
    else:
        value = run_oracle() if args.phase == "oracle" else run_volterra()
    print(json.dumps(value.get("summary", value)))
