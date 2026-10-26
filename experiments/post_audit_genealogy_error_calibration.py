"""Frozen rare-mode diagnostic calibration; never retroactively qualifies a bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.post_audit_conditional_risk_target_smc import one_shot, tensor_digest
from experiments.post_audit_mixture_oracle_late_bridge import (
    ORACLE,
    VOLTERRA,
    assert_current,
    check_protocol,
    parent_evidence,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.conditional_marginal_g import json_compatible
from src.path_integral.conditional_risk_target_smc import STREAMS, TargetSMCConfig, run_target_smc
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.genealogy_error_calibration import (
    TARGETS,
    KnownMixtureTarget,
    confusion,
    constructed_counterexamples,
    diagnostic_flags,
    independent_reference,
    method_summary,
    rank_correlation,
    truth_errors,
    weighted_moments,
)
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.seed_ledger import SeedKey, SeedLedger

DIRECTORY = "results/post_audit/genealogy_error_calibration_v1/session_20261023"
PLAN = "docs/plans/GENEALOGY_ERROR_CALIBRATION_2026-10-23_KO.md"
NAMESPACE = "genealogy-error-calibration-v1"
EXPECTED = "24b2b05b03e1e706f30c81d5f51c469f818930ab7ff835cc067a0e9719aaa721"
EXPECTED_AUDIT = "574f9cfaff2524201b0eb3cc18e9ece23153ccbb1ba78f3a0a616d38e9d26026"
EXTRA_STREAMS = ("iid-prior", "exact-label", "exact-noise")


def evidence() -> list[dict[str, Any]]:
    _, previous = parent_evidence()
    for directory in (ORACLE, VOLTERRA):
        path = ROOT/directory
        d = json.loads((path/"result.json").read_text())
        previous.append(d["seeds"]["ledger"])
    path = ROOT/VOLTERRA
    if sha(path/"result.json") != EXPECTED or sha(path/"audit.json") != EXPECTED_AUDIT:
        raise ValueError("immutable audited late-bridge STOP required")
    d = json.loads((path/"result.json").read_text())
    assert_current(d["source"], d["config"])
    if d["summary"]["decision"] != "STOP_LATE_BRIDGE_BANK_UNQUALIFIED":
        raise ValueError("preceding STOP must be preserved")
    return previous


def jobs() -> list[dict[str, Any]]:
    return [{"target": t.name, "replicate": r} for t in TARGETS for r in range(32)]


def identity(job: dict[str, Any]) -> str:
    return f"{job['target']}-{job['replicate']}"


def seal(previous: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = SeedLedger()
    mapping = {identity(j): {s: ledger.allocate(SeedKey(NAMESPACE, "diagnostic-calibration", "development",
        j["target"], 1, j["replicate"], s)) for s in STREAMS+EXTRA_STREAMS} for j in jobs()}
    old = {r.seed for payload in previous for r in SeedLedger.from_dict(payload).records}
    if old & {r.seed for r in ledger.records}:
        raise ValueError("scoped stream collision")
    return mapping, {"ledger": ledger.to_dict(), "sha256": ledger.sha256, "previous_seeds_compared": len(old),
        "collision_count": 0, "not_whole_history_or_PRNG_independence_proof": True}


class Budget:
    def __init__(self) -> None:
        self.contexts, self.draws, self.started = 0, 0, time.perf_counter()

    def charge(self, contexts: int = 0, draws: int = 0) -> None:
        if (type(contexts) is not int or type(draws) is not int or contexts < 0 or draws < 0
                or self.contexts+contexts > 2400000 or self.draws+draws > 12288
                or time.perf_counter()-self.started > 300):
            raise TimeoutError("fixed calibration context/draw/wall budget exceeded")
        self.contexts += contexts
        self.draws += draws


def references() -> dict[str, Any]:
    rows = [independent_reference(t) for t in TARGETS]
    examples = {t.name: constructed_counterexamples(t) for t in TARGETS}
    return json_compatible({"references": rows, "counterexamples": examples,
        "pass": all(r["pass"] for r in rows),
        "scalar_reference_evaluations": sum(r["scalar_reference_evaluations"] for r in rows),
        "not_Volterra_or_dimension_transfer_certificate": True})


def replicate(target: KnownMixtureTarget, job: dict[str, Any], streams: dict[str, int], budget: Budget) -> dict[str, Any]:
    before_c, before_d = budget.contexts, budget.draws
    chain = hashlib.sha256(b"genealogy-error-calibration-v1").digest()
    calls = 0

    def evaluate(x: torch.Tensor):
        nonlocal chain, calls
        budget.charge(contexts=len(x))
        v = target.evaluate(x)
        v.validate(len(x))
        chain = hashlib.sha256(chain+bytes.fromhex(tensor_digest(x, v.log_m2, v.log_dq))).digest()
        calls += 1
        return v

    q = GaussianMixtureShiftSpec(torch.zeros((1, 1), dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    config = TargetSMCConfig(128, tuple(j/32 for j in range(33)), "component", .35, 4)
    result = run_target_smc(q, evaluate, config=config, seeds={s: streams[s] for s in STREAMS})
    x, weights = torch.tensor(result["final_x"], dtype=torch.float64), torch.tensor(result["final_weights"], dtype=torch.float64)
    truth = target.truth()
    moments = weighted_moments(x, weights)
    smc = {"log_normalizer": result["log_normalizer"], "moments": moments,
        "errors": truth_errors(moments, result["log_normalizer"], truth),
        "flags": diagnostic_flags(result["diagnostics"], result["gate"]["pass"])}

    iid_x = torch.randn((12416, 1), dtype=torch.float64, generator=torch.Generator().manual_seed(streams["iid-prior"]))
    iid_value = evaluate(iid_x)
    log_total = iid_value.log_m2.logsumexp(0)
    log_z = float(log_total)-torch.tensor(len(iid_x), dtype=torch.float64).log().item()
    normalized = (iid_value.log_m2-log_total).exp()
    iid_moments = weighted_moments(iid_x, normalized)
    iid = {"log_normalizer": log_z, "moments": iid_moments,
        "errors": truth_errors(iid_moments, log_z, truth),
        "nominal_weight_ess": float(1/normalized.square().sum()),
        "maximum_particle_weight": float(normalized.max()),
        "raw_normalizer_not_self_normalized": True, "normalized_moments_are_finite_N_biased_SNIS_diagnostics": True}

    budget.charge(draws=128)
    refreshed = target.sample_exact(128, label_seed=streams["exact-label"], noise_seed=streams["exact-noise"])
    fresh_moments = weighted_moments(refreshed, weights)
    refresh = {"log_normalizer": result["log_normalizer"], "moments": fresh_moments,
        "errors": truth_errors(fresh_moments, result["log_normalizer"], truth),
        "flags": smc["flags"], "weights_lineage_log_normalizer_and_original_flags_unchanged": True,
        "normalizer_not_repaired_by_endpoint_refresh": True,
        "unavailable_exact_target_sampler_in_real_Volterra": True,
        "sample_sha256": tensor_digest(refreshed),
        "conditional_mean_variance": (truth["second_moment"]-truth["mean"]**2)*float(weights.square().sum())}
    return json_compatible({"job": job, "seeds": streams, "smc": smc, "iid": iid, "refresh": refresh,
        "SMC_full_result": result, "potential_calls": calls, "potential_chain_sha256": chain.hex(),
        "contexts": budget.contexts-before_c, "exact_target_draws": budget.draws-before_d,
        "Volterra_physical": 0})


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if [r["job"] for r in records] != jobs():
        raise ValueError("all96 fixed-order calibration jobs required")
    summaries = []
    for target in TARGETS:
        rows = [r for r in records if r["job"]["target"] == target.name]
        methods = {m: method_summary([r[m] for r in rows], target.truth()) for m in ("smc", "iid", "refresh")}
        matrices = {method: {flag: {endpoint: confusion([r[method]["flags"][flag] for r in rows],
            [r[method]["errors"][endpoint] for r in rows]) for endpoint in ("target_accurate", "normalizer_accurate")}
            for flag in ("full", "ancestor_only", "weight_only")} for method in ("smc", "refresh")}
        ancestor = [r["SMC_full_result"]["diagnostics"]["weighted_ancestor_ess"] for r in rows]
        correlations = {endpoint: rank_correlation(ancestor, [r["smc"]["errors"][endpoint] for r in rows])
                        for endpoint in ("target_error_score", "relative_normalizer_error")}
        summaries.append({"target": target.name, "truth": target.truth(), "methods": methods,
            "diagnostic_confusion": matrices, "descriptive_spearman_ancestor_ESS_vs_error": correlations,
            "original_full_gate_pass_count": sum(r["smc"]["flags"]["full"] for r in rows)})
    return {"targets": summaries, "decision": "CALIBRATION_COMPLETE_NO_RETROACTIVE_ADOPTION",
        "event_q_changed": False, "actual_G_fits": 0, "P2_qualified": False,
        "D3_authorized": False, "large_training_authorized": False,
        "production_gate_changed": False, "no_actual_Volterra_reference_qualification": True}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    end_started = time.perf_counter()

    def body(path: Path, progress: dict[str, Any]) -> dict[str, Any]:
        mapping, ledger = seal(evidence())
        config = {"namespace": NAMESPACE, "targets": [asdict(t) for t in TARGETS], "replicates": 32,
            "SMC_particles": 128, "SMC_stages": 32, "SMC_scale": .35, "SMC_resample_every": 4,
            "schedule": "linear", "IID_contexts_per_replicate": 12416,
            "potential_context_cap": 2400000, "exact_target_draw_cap": 12288, "compute_wall_cap": 300,
            "parent_sha256": EXPECTED}
        progress["phase"] = "source_freeze"
        clock = time.perf_counter()
        source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, VOLTERRA+"/result.json", VOLTERRA+"/audit.json"))
        source_wall = time.perf_counter()-clock
        write(path/"protocol.json", json_compatible({"source": source, "config": config, "mapping": mapping, "seeds": ledger}))
        progress["phase"] = "independent_reference"
        clock = time.perf_counter()
        refs = references()
        reference_wall = time.perf_counter()-clock
        write(path/"reference.json", json_compatible(refs))
        budget, records, artifacts = Budget(), [], []
        if refs["pass"]:
            targets = {t.name: t for t in TARGETS}
            for job in jobs():
                progress["phase"] = identity(job)
                try:
                    record = replicate(targets[job["target"]], job, mapping[identity(job)], budget)
                finally:
                    progress.update({"potential_contexts": budget.contexts, "exact_target_draws": budget.draws})
                name = identity(job)+".json"
                write(path/name, record)
                artifacts.append({"path": name, "sha256": sha(path/name)})
                records.append(record)
                if len(records) % 32 == 0:
                    print(f"Completed {len(records)}/96 calibration replicates", flush=True)
            summary = summarize(records)
        else:
            summary = {"decision": "STOP_CALIBRATION_REFERENCE_UNRESOLVED", "P2_qualified": False}
        value = json_compatible({"source": source, "config": config, "seeds": ledger,
            "reference_sha256": sha(path/"reference.json"), "summary": summary, "artifacts": artifacts,
            "potential_contexts": budget.contexts, "exact_target_draws": budget.draws, "Volterra_physical": 0,
            "scalar_reference_evaluations": refs["scalar_reference_evaluations"],
            "cost": {"source_freeze": source_wall, "reference": reference_wall,
                "compute_and_island_io": time.perf_counter()-budget.started}})
        progress["phase"] = "result_io"
        clock = time.perf_counter()
        write(path/"result.json", value)
        write(path/"end_to_end.json", {"result_sha256": sha(path/"result.json"),
            "result_io": time.perf_counter()-clock, "wall_before_receipt": time.perf_counter()-end_started,
            "audit_and_tests_excluded": True})
        return value

    return one_shot(ROOT/DIRECTORY, body)


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    path = ROOT/DIRECTORY
    value = json.loads((path/"result.json").read_text())
    assert_current(value["source"], value["config"])
    mapping, ledger = seal(evidence())
    check_protocol(value, json.loads((path/"protocol.json").read_text()), mapping, ledger)
    refs = json.loads((path/"reference.json").read_text())
    if sha(path/"reference.json") != value["reference_sha256"] or references() != refs:
        raise ValueError("independent reference/counterexample changed")
    targets = {t.name: t for t in TARGETS}
    records, budget = [], Budget()
    if refs["pass"]:
        for job, item in zip(jobs(), value["artifacts"], strict=True):
            if item["path"] != identity(job)+".json" or sha(path/item["path"]) != item["sha256"]:
                raise ValueError("complete calibration artifact inventory changed")
            record = json.loads((path/item["path"]).read_text())
            if replicate(targets[job["target"]], job, mapping[identity(job)], budget) != record:
                raise ValueError("full calibration replicate replay failed")
            records.append(record)
        if summarize(records) != value["summary"]:
            raise ValueError("calibration statistic/label/confusion changed")
    if budget.contexts != value["potential_contexts"] or budget.draws != value["exact_target_draws"]:
        raise ValueError("whole calibration work changed")
    if json.loads((path/"end_to_end.json").read_text())["result_sha256"] != sha(path/"result.json"):
        raise ValueError("calibration end-to-end binding changed")
    return {"status": "all_references_counterexamples_and_96_replicates_fully_replayed" if refs["pass"]
            else "deterministic_reference_stop_replayed_no_random_jobs",
        "result_sha256": sha(path/"result.json"), "potential_contexts": budget.contexts,
        "exact_target_draws": budget.draws, "Volterra_physical": 0, "wall": time.perf_counter()-started,
        "decision": value["summary"]["decision"], "P2_qualified": False, "production_gate_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit()
        write(ROOT/DIRECTORY/"audit.json", result)
    else:
        result = run()
    print(json.dumps({"decision": result.get("decision", result.get("summary", {}).get("decision")),
        "potential_contexts": result["potential_contexts"], "exact_target_draws": result["exact_target_draws"]}))
