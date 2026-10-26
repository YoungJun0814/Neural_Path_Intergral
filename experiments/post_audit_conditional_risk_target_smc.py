"""Frozen bank-only conditional risk SMC development with failure preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

from experiments.post_audit_analytic_residual_batch import Budget
from experiments.post_audit_conditional_marginal_g import DIRECTORY as PARENT
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.analytic_residual_batch import batched_moments
from src.path_integral.conditional_marginal_g import ACTIVE, json_compatible, moment_error
from src.path_integral.conditional_risk_target_smc import (
    STREAMS,
    Arm,
    RiskEvaluation,
    TargetSMCConfig,
    island_gate,
    lineage_summary,
    mode_diagnostics,
    normalizer_summary,
    run_target_smc,
)
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/conditional_risk_target_smc_v1/session_20261022"
PLAN = "docs/plans/CONDITIONAL_RISK_TARGET_SMC_BANK_2026-10-22_KO.md"
THEORY = "docs/theory/CONDITIONAL_RISK_TARGET_SMC_CONTRACT_2026-10-22_KO.md"
NAMESPACE = "conditional-risk-target-smc-bank-v1"
EXPECTED = "ecd0a93de8dc90a24a3a3e91972c125178986c573c15a8f1ff6e1c33fcf4afbd"


def tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for t in values:
        digest.update(str(tuple(t.shape)).encode())
        digest.update(str(t.dtype).encode())
        digest.update(t.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def one_shot(directory: Path, action: Callable[[Path, dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("one-shot directory; no overwrite or retry")
    directory.mkdir(parents=True, exist_ok=True)
    progress: dict[str, Any] = {"phase": "startup", "physical": 0, "scalar": 0}
    started = time.perf_counter()
    try:
        return action(directory, progress)
    except Exception as error:
        receipt = {"status": "STARTUP_OR_RUN_FAILED", "error": type(error).__name__+": "+str(error),
            "progress": progress, "wall_before_failure_receipt": time.perf_counter()-started,
            "P2_qualified": False, "large_training_authorized": False}
        try:
            write_json_atomic_nonoverwriting(directory/"failure.json", json_compatible(receipt))
        except Exception as storage_error:
            print("Failure receipt could not be persisted: "+type(storage_error).__name__, file=sys.stderr)
        raise


def evidence() -> dict[str, Any]:
    path = ROOT/PARENT
    parent, audit = [json.loads((path/n).read_text()) for n in ("result.json", "audit.json")]
    if (sha(path/"result.json") != EXPECTED or audit["result_sha256"] != EXPECTED
            or parent["summary"]["decision"] != "STOP_MARGINAL_TRAINING_BANK_UNRELIABLE"):
        raise ValueError("audited immutable preceding IID bank STOP required")
    audit_snapshot(ROOT, parent["source"], parent["config"])
    return parent


def jobs(cells: list[str]) -> list[dict[str, Any]]:
    return [{"cell": cell, "island": island, "arm": arm}
        for cell in cells for island in range(4) for arm in ("prior", "component")]


def job_id(job: dict[str, Any]) -> str:
    return f"{job['cell']}-{job['arm']}-{job['island']}"


def seeds_for(cells: list[str], parent: dict[str, Any]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    ledger, mapping = SeedLedger(), {}
    for job in jobs(cells):
        mapping[job_id(job)] = {stream: ledger.allocate(SeedKey(NAMESPACE, job["arm"], "development",
            job["cell"], 32, job["island"], stream)) for stream in STREAMS}
    previous = SeedLedger.from_dict(parent["seeds"]["ledger"])
    if {r.seed for r in previous.records} & {r.seed for r in ledger.records}:
        raise ValueError("new/previous scoped seed collision")
    return mapping, {"ledger": ledger.to_dict(), "sha256": ledger.sha256,
        "previous_seeds_compared": len(previous), "collision_count": 0,
        "scope": "all new external streams and preceding marginal-G 264 sealed streams; not whole history"}


class VolterraRiskRecorder:
    def __init__(self, cache: Any, q: GaussianMixtureShiftSpec, budget: Budget):
        self.cache, self.q, self.budget = cache, q, budget
        self.trace: list[dict[str, Any]] = []
        self.chain = hashlib.sha256(b"conditional-risk-target-smc-v1").digest()
        self.wall = 0.

    def __call__(self, x: torch.Tensor) -> RiskEvaluation:
        started = time.perf_counter()
        before_physical, before_scalar = self.budget.physical, self.budget.scalar
        entry: dict[str, Any] = {"call": len(self.trace), "count": len(x), "input_sha256": tensor_digest(x)}
        try:
            if (x.ndim != 2 or x.shape[1] != 62 or x.dtype != torch.float64 or x.device.type != "cpu"
                    or x.requires_grad or not torch.isfinite(x).all()):
                raise ValueError("finite detached X62 required")
            z = torch.zeros((len(x), 64), dtype=torch.float64)
            z[:, ACTIVE] = x
            profiles = []
            for point in z:
                self.budget.physics(1)
                profiles.append(LastVarianceResidualSlice.create(self.cache, point[None]))
            calculated = batched_moments(profiles, charge=self.budget.charge)
            converted = [moment_error(r) for r in calculated["rows"]]
            if not all(r["relative_numerically_resolved"] for r in calculated["rows"]) or not all(
                    r["structural_numerics_pass"] for r in converted):
                raise FloatingPointError("individual conditional target numerical gate unresolved")
            maximum = max(m["relative_error_proxy_plus_tail"] for r in calculated["rows"] for m in r["moments"])
            dq = torch.logsumexp(x @ self.q.means.T-.5*self.q.means.square().sum(1)+self.q.weights.log(), 1)
            m2 = torch.tensor([r["log_m2"] for r in converted], dtype=torch.float64)
            features = torch.tensor([[math.log(p.preceding_integrated_variance), p.preceding_terminal,
                p.log_initial_last_variance, float(x[j].square().sum()), float(x[j, ACTIVE.index(57)])]
                for j, p in enumerate(profiles)], dtype=torch.float64)
            output = RiskEvaluation(m2, dq, features, maximum)
            output.validate(len(x))
            self.budget.charge(0)
            entry.update({"status": "complete", "output_sha256": tensor_digest(m2, dq, features),
                "maximum_relative_proxy": maximum})
        except Exception as error:
            entry.update({"status": "failed", "error": type(error).__name__+": "+str(error)})
            raise
        finally:
            entry.update({"physical": self.budget.physical-before_physical, "scalar": self.budget.scalar-before_scalar})
            self.chain = hashlib.sha256(self.chain+bytes.fromhex(canonical_digest(entry))).digest()
            self.trace.append(entry)
            self.wall += time.perf_counter()-started
        return output


def smc_config(arm: str) -> TargetSMCConfig:
    if arm not in ("prior", "component"):
        raise ValueError("frozen arm required")
    return TargetSMCConfig(128, tuple((j/32)**4 for j in range(33)), cast(Arm, arm), .35, 4)


def island(cache: Any, q: GaussianMixtureShiftSpec, job: dict[str, Any], streams: dict[str, int],
           budget: Budget) -> dict[str, Any]:
    recorder = VolterraRiskRecorder(cache, q, budget)
    stages: list[dict[str, Any]] = []
    before_p, before_s, started = budget.physical, budget.scalar, time.perf_counter()
    output: dict[str, Any]
    try:
        result = run_target_smc(q, recorder, config=smc_config(job["arm"]), seeds=streams, observer=stages.append)
        output = {"status": "complete", "result": result}
    except Exception as error:
        category = "RESOURCE" if isinstance(error, TimeoutError) else (
            "NUMERICAL" if isinstance(error, (FloatingPointError, ValueError)) else "TECHNICAL")
        output = {"status": "failed", "failure_type": category,
            "error": type(error).__name__+": "+str(error), "completed_stages": stages}
    return json_compatible({"job": job, "seeds": streams, **output, "trace": recorder.trace,
        "trace_sha256": recorder.chain.hex(), "physical": budget.physical-before_p,
        "scalar": budget.scalar-before_s, "wall": time.perf_counter()-started,
        "potential_wall": recorder.wall})


def summarize(records: list[dict[str, Any]], cells: list[str]) -> dict[str, Any]:
    if [r["job"] for r in records] != jobs(cells):
        raise ValueError("all sixteen prespecified jobs in fixed order required")
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        arms: dict[str, Any] = {}
        for arm in ("prior", "component"):
            rows = [r for r in records if r["job"]["cell"] == cell and r["job"]["arm"] == arm]
            if all(r["status"] == "complete" for r in rows):
                normalizer = normalizer_summary([r["result"]["log_normalizer"] for r in rows])
                modes = mode_diagnostics([r["result"] for r in rows]) if arm == "component" else None
                arms[arm] = {"complete": True, "normalizer": normalizer, "modes": modes,
                    "island_gates": [r["result"]["gate"]["pass"] for r in rows]}
            else:
                arms[arm] = {"complete": False, "failed_islands": [r["job"]["island"] for r in rows if r["status"] != "complete"]}
        contrast: dict[str, Any] = {"accuracy_qualified": False, "no_detected_contradiction": None}
        if all(arms[a]["complete"] and arms[a]["normalizer"]["qualified"] for a in arms):
            first, second = [arms[a]["normalizer"] for a in ("prior", "component")]
            scale = max(first["log_mean_normalizer"], second["log_mean_normalizer"])
            x, y = math.exp(first["log_mean_normalizer"]-scale), math.exp(second["log_mean_normalizer"]-scale)
            se = math.hypot(x*first["rse"], y*second["rse"])
            contrast = {"accuracy_qualified": True,
                "no_detected_contradiction": abs(x-y) <= 3.291*se+1e-12*max(x, y),
                "not_equivalence_or_population_accuracy_proof": True}
        summaries.append({"cell": cell, "arms": arms, "contrast": contrast})
    if any(r.get("failure_type") == "RESOURCE" for r in records):
        decision = "STOP_RESOURCE_LIMIT"
    elif any(r.get("failure_type") == "TECHNICAL" for r in records):
        decision = "STOP_TECHNICAL_FAILURE"
    elif any(r["status"] != "complete" for r in records):
        decision = "STOP_NUMERICAL_TARGET_UNRESOLVED"
    elif not all(all(s["arms"]["component"]["island_gates"]) for s in summaries):
        decision = "STOP_GENEALOGY_OR_MUTATION_UNQUALIFIED"
    elif not all(s["arms"]["component"]["normalizer"]["qualified"] and s["arms"]["component"]["modes"]["pass"] for s in summaries):
        decision = "NO_BANK_ADOPTION_ISLAND_OR_MODE_UNRESOLVED"
    elif not all(s["contrast"]["accuracy_qualified"] and s["contrast"]["no_detected_contradiction"] for s in summaries):
        decision = "NO_BANK_ADOPTION_REFERENCE_UNRESOLVED"
    else:
        decision = "BANK_CANDIDATE_FOR_SEPARATE_G_PIPELINE_ONLY"
    return {"cells": summaries, "decision": decision, "actual_G_fits": 0, "event_q_changed": False,
        "P2_qualified": False, "D3_authorized": False, "large_training_authorized": False,
        "not_fixed_precision_or_population_speedup": True}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    end_started = time.perf_counter()

    def body(path: Path, progress: dict[str, Any]) -> dict[str, Any]:
        parent = evidence()
        config = {"namespace": NAMESPACE, "model": parent["config"]["model"], "cells": parent["config"]["cells"],
            "particles": 128, "temperatures": list(smc_config("component").temperatures), "islands": 4,
            "arms": ["prior", "component"], "scale": .35, "resample_every": 4,
            "physical_cap": 200000, "scalar_cap": 800000000, "wall_cap": 2400, "parent_sha256": EXPECTED}
        progress["phase"] = "setup"
        clock = time.perf_counter()
        prepared, setup = prepare(config)
        setup_wall = time.perf_counter()-clock
        reduced = {cell: GaussianMixtureShiftSpec(q.means[:, ACTIVE].clone(), q.weights.clone()) for cell, (_, q) in prepared.items()}
        mapping, ledger = seeds_for(list(prepared), parent)
        progress["phase"] = "source_freeze"
        clock = time.perf_counter()
        source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, PARENT+"/result.json", PARENT+"/audit.json"))
        source_wall = time.perf_counter()-clock
        progress["phase"] = "protocol_io"
        clock = time.perf_counter()
        write_json_atomic_nonoverwriting(path/"protocol.json", json_compatible({"config": config, "source": source, "seeds": ledger, "mapping": mapping}))
        protocol_io = time.perf_counter()-clock
        budget = Budget(800000000, 200000, 2400)
        records, artifacts, io_wall = [], [], 0.
        for job in jobs(list(prepared)):
            progress["phase"] = job_id(job)
            budget.charge(0)
            record = island(prepared[job["cell"]][0], reduced[job["cell"]], job, mapping[job_id(job)], budget)
            progress.update({"physical": budget.physical, "scalar": budget.scalar})
            clock = time.perf_counter()
            name = job_id(job)+".json"
            write_json_atomic_nonoverwriting(path/name, record)
            io_wall += time.perf_counter()-clock
            artifacts.append({"path": name, "sha256": sha(path/name)})
            records.append(record)
            d = record.get("result", {}).get("diagnostics", {})
            print(f"Completed {len(records)}/16 {job_id(job)}: {record['status']}, ancestors={d.get('initial_ancestors')}", flush=True)
        progress["phase"] = "aggregation"
        clock = time.perf_counter()
        summary = summarize(records, list(prepared))
        aggregation_wall = time.perf_counter()-clock
        result = json_compatible({"config": config, "source": source, "seeds": ledger, "artifacts": artifacts,
            "summary": summary, "physical": budget.physical, "scalar": budget.scalar,
            "cost": {"setup": setup, "setup_wall": setup_wall, "source_freeze": source_wall,
                "protocol_io": protocol_io, "island_io": io_wall, "aggregation": aggregation_wall,
                "compute_and_island_io": time.perf_counter()-budget.started}})
        progress["phase"] = "result_io"
        clock = time.perf_counter()
        write_json_atomic_nonoverwriting(path/"result.json", result)
        result_io = time.perf_counter()-clock
        write_json_atomic_nonoverwriting(path/"end_to_end.json", {"result_sha256": sha(path/"result.json"),
            "wall_before_receipt": time.perf_counter()-end_started, "result_io": result_io,
            "audit_and_tests_excluded": True})
        return result

    return one_shot(ROOT/DIRECTORY, body)


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    parent = evidence()
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, digest in result["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("frozen risk-target SMC source changed")
    prepared, _ = prepare(result["config"])
    reduced = {c: GaussianMixtureShiftSpec(q.means[:, ACTIVE].clone(), q.weights.clone()) for c, (_, q) in prepared.items()}
    mapping, ledger = seeds_for(list(prepared), parent)
    protocol = json.loads((path/"protocol.json").read_text())
    if ledger != result["seeds"] or protocol["mapping"] != mapping:
        raise ValueError("sealed seed ledger changed")
    records = []
    budget = Budget(120000000, 30000, 600)
    full_replays = []
    maximum_final_log_error = 0.
    for item, job in zip(result["artifacts"], jobs(list(prepared)), strict=True):
        if item["path"] != job_id(job)+".json" or sha(path/item["path"]) != item["sha256"]:
            raise ValueError("complete island artifact inventory/hash mismatch")
        record = json.loads((path/item["path"]).read_text())
        if record["job"] != job or record["seeds"] != mapping[job_id(job)]:
            raise ValueError("island identity/seed binding changed")
        chain = hashlib.sha256(b"conditional-risk-target-smc-v1").digest()
        for i, row in enumerate(record["trace"]):
            if row["call"] != i or row["count"] != 128:
                raise ValueError("complete potential trace count changed")
            chain = hashlib.sha256(chain+bytes.fromhex(canonical_digest(row))).digest()
        if (chain.hex() != record["trace_sha256"] or sum(t["physical"] for t in record["trace"]) != record["physical"]
                or sum(t["scalar"] for t in record["trace"]) != record["scalar"]):
            raise ValueError("potential trace/accounting changed")
        cache, q = prepared[job["cell"]][0], reduced[job["cell"]]
        if record["status"] == "complete":
            data = record["result"]
            x = torch.tensor(data["final_x"], dtype=torch.float64)
            w = torch.tensor(data["final_weights"], dtype=torch.float64)
            lineage = torch.tensor(data["initial_lineage"], dtype=torch.int64)
            diagnostics = lineage_summary(lineage, w, x)
            if diagnostics != data["diagnostics"] or island_gate(diagnostics, data["stages"],
                    torch.tensor(data["late_jump_by_particle"], dtype=torch.float64)) != data["gate"]:
                raise ValueError("all final lineage/mutation gate summaries changed")
            if record["physical"] != 12416 or data["potential_contexts"] != 12416 or len(record["trace"]) != 97:
                raise ValueError("fixed complete island work changed")
            for i, stage in enumerate(data["stages"]):
                if (stage["stage"] != i or stage["beta"] != smc_config(job["arm"]).temperatures[i+1]
                        or stage["resampled"] != ((i+1) % 4 == 0 and i != 31)):
                    raise ValueError("fixed stage calendar changed")
            fresh = VolterraRiskRecorder(cache, q, budget)(x)
            error = float((fresh.log_m2-torch.tensor(data["final_log_m2"], dtype=torch.float64)).abs().max())
            maximum_final_log_error = max(maximum_final_log_error, error)
            if (error > 1e-8 or not torch.allclose(fresh.log_dq, torch.tensor(data["final_log_dq"], dtype=torch.float64), rtol=0, atol=1e-10)
                    or not torch.allclose(fresh.features, torch.tensor(data["final_features"], dtype=torch.float64), rtol=1e-12, atol=1e-10)):
                raise ValueError("all final full-law potential replays failed")
        if job["arm"] == "component" and job["island"] == 0:
            replay = island(cache, q, job, mapping[job_id(job)], budget)
            if {k: v for k, v in replay.items() if k not in ("wall", "potential_wall")} != {
                    k: v for k, v in record.items() if k not in ("wall", "potential_wall")}:
                raise ValueError("two full primary island trace/stage/failure replays failed")
            full_replays.append(job_id(job))
        records.append(record)
    if summarize(records, list(prepared)) != result["summary"]:
        raise ValueError("all-island statistic/mode/decision recomputation failed")
    if (sum(r["physical"] for r in records) != result["physical"]
            or sum(r["scalar"] for r in records) != result["scalar"]):
        raise ValueError("whole production work changed")
    if json.loads((path/"end_to_end.json").read_text())["result_sha256"] != sha(path/"result.json"):
        raise ValueError("end-to-end result binding changed")
    return {"status": "all_artifacts_statistics_final_targets_and_two_primary_islands_replayed",
        "result_sha256": sha(path/"result.json"), "physical": budget.physical, "scalar": budget.scalar,
        "full_island_replays": full_replays, "maximum_final_log_m2_error": maximum_final_log_error,
        "wall": time.perf_counter()-started, "decision": result["summary"]["decision"], "P2_qualified": False}


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
        print(json.dumps({"decision": value["summary"]["decision"], "physical": value["physical"], "scalar": value["scalar"]}))
