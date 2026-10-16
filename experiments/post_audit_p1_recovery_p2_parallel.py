"""Sealed covariance/bridge pilot; P1 production and P2 are separate gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Literal, cast

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.collective_variance_reference import CollectiveVarianceReference
from src.path_integral.conditional_excursion_reference import ConditionalExcursionReference
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.coverage_envelope_reference import CoverageEnvelopeReference
from src.path_integral.proposal_bridge_smc import estimate_proposal_bridge
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import source_tree_digest
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    precision_count,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/P1_RECOVERY_P2_PARALLEL_EXECUTION_2026-10-09_KO.md"
CONFIG = "configs/post_audit/p1_recovery_p2_parallel_v1.yaml"


def seed(identity: str, stream: str) -> int:
    return int(hashlib.sha256(("p1-recovery-p2-parallel-v1/"+identity+"/"+stream).encode()).hexdigest()[:15], 16)


def validate(config: dict[str, Any]) -> None:
    if (config["schema"] != "npi.p1-recovery-p2-parallel.v1" or config["method"] != "parent-as-is"
            or any(isinstance(config[k], bool) or not isinstance(config[k], int) or config[k] != v
                   for k, v in (("replications", 8), ("unit_count", 32768), ("batch_size", 4096)))
            or config["smc"] != {"particles": 1024, "levels": 64, "mutations": 4, "pcn": .35,
                                 "resample_every": 4, "independence_every": 2}
            or config["limits"] != {"B": {"work": 40000000, "wall": 1800.}}):
        raise ValueError("sealed pilot changed")
    if config.get("reference_family", "collective") not in ("collective", "conditional-excursion", "conditional-excursion-directional", "conditional-excursion-arithmetic", "coverage-envelope", "coverage-envelope-f64"):
        raise ValueError("undeclared reference family")
    if config.get("reference_family") == "conditional-excursion-arithmetic":
        anchor = ROOT/config["scale_anchor"]
        if hashlib.sha256(anchor.read_bytes()).hexdigest() != config["scale_anchor_sha256"]:
            raise ValueError("scale anchor changed")
        payload = json.loads(anchor.read_text())
        expected = {cell: {s["target"]: math.exp(s["summary"]["log_mean"]) for s in payload["summaries"]
                          if s["cell"] == cell and s["method"] == "excursion-iid"}
                    for cell in {s["cell"] for s in payload["summaries"]}}
        if config["normalizer_scales"] != expected:
            raise ValueError("scale not bound to pre-existing independent pilot")


def jobs(rows: list[dict[str, Any]], reference_family: str = "collective") -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        cell = row["cell"]["id"]
        physical = "envelope-iid" if reference_family.startswith("coverage-envelope") else ("excursion-iid" if reference_family.startswith("conditional-excursion") else "collective-iid")
        for method in ("guide-iid", physical):
            for rep in range(8):
                for batch in range(8):
                    output.append({"cell": cell, "method": method, "rep": rep, "batch": batch, "target": "paired"})
        for method in ("natural-smc", "proposal-smc"):
            for target in ("mu", "risk"):
                for rep in range(8):
                    output.append({"cell": cell, "method": method, "rep": rep, "batch": 0, "target": target})
    if reference_family.startswith("conditional-excursion") or reference_family.startswith("coverage-envelope"):
        for job in output:
            job["namespace"] = reference_family
    return output


def identity(job: dict[str, Any]) -> str:
    prefix = job["namespace"]+"-" if "namespace" in job else ""
    return prefix+"-".join(str(job[k]) for k in ("cell", "method", "target", "rep", "batch"))


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    physical = next(m for m in ("envelope-iid", "excursion-iid", "collective-iid") if any(r["method"] == m for r in records))
    for cell in sorted({r["cell"] for r in records}):
        for method in ("guide-iid", physical, "natural-smc", "proposal-smc"):
            for target in ("mu", "risk"):
                parts = [r for r in records if r["cell"] == cell and r["method"] == method
                         and r["target"] in (target, "paired")]
                logs = torch.tensor([x for r in parts for x in r["logs"][target]], dtype=torch.float64)
                summary = log_moments(logs)
                rep_logs = [float(torch.logsumexp(torch.tensor([x for r in parts if r["rep"] == rep
                    for x in r["logs"][target]], dtype=torch.float64), 0))
                    - math.log(sum(len(r["logs"][target]) for r in parts if r["rep"] == rep)) for rep in sorted({r["rep"] for r in parts})]
                stability = sensitivity(rep_logs, bootstrap_seed=seed(parts[0].get("namespace", "")+cell+method+target, "bootstrap"))
                wall = sum(r["wall"] for r in parts)
                allocation = precision_count(summary, target_rse=.015, safety_factor=2., minimum=32 if "smc" in method else 65536,
                    maximum=256 if "smc" in method else 4194304, multiple=8 if "smc" in method else 4096)
                required = allocation["required_count"]
                unit_cost = sum(r["actual_work"] for r in parts)/summary["count"]
                output.append({"cell": cell, "method": method, "target": target, "summary": summary,
                    "whole_replication_log_means": rep_logs, "sensitivity": stability,
                    "wall": wall, "allocation": allocation, "forecast_work": math.ceil(required*unit_cost),
                    "forecast_wall": required*wall/summary["count"],
                    "inference_unit": "whole_smc_run" if "smc" in method else "iid_draw",
                    "guide_independence": method != "guide-iid"})
    return output


def decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = []
    physical = next(m for m in ("envelope-iid", "excursion-iid", "collective-iid") if any(r["method"] == m for r in summaries))
    for cell in sorted({r["cell"] for r in summaries}):
        for target in ("mu", "risk"):
            group = {r["method"]: r for r in summaries if r["cell"] == cell and r["target"] == target}
            # Three pairwise comparisons per target and cell. Development only.
            for a, b in ((physical, "natural-smc"), (physical, "proposal-smc"), ("guide-iid", "natural-smc")):
                evidence.append({"cell": cell, "target": target, "methods": [a, b],
                    "equivalence": relative_equivalence(group[a]["summary"], group[b]["summary"], comparisons=12)})
    counts_ok = all(r["allocation"]["status"] == "allocated" for r in summaries)
    return {"pilot_equivalence": evidence, "all_method_allocations_within_cap": counts_ok,
        "production_config": None, "p1_qualified": False, "p2_qualified": False,
        "large_training_authorized": False,
        "scope": "parent0_pilot_only_other_qs_require_fresh_matrix_pilot_and_sealed_production"}


def run(config: dict[str, Any]) -> dict[str, Any]:
    validate(config)
    torch.set_num_threads(1)
    directory = (ROOT/config["run_directory"]).resolve()
    if not directory.is_relative_to((ROOT/"results/post_audit").resolve()):
        raise ValueError("invalid result root")
    directory.mkdir(parents=True, exist_ok=True)
    metadata, rows = inputs(config)
    grid = jobs(rows, config.get("reference_family", "collective"))
    extra = (config.get("protocol_path", PLAN),)+(tuple([config["scale_anchor"]]) if "scale_anchor" in config else ())
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=extra)
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "limits": config["limits"],
        "source_tree": source_tree_digest(ROOT), "expected_units": [identity(j) for j in grid],
        "input_sha256": hashlib.sha256((ROOT/config["proposal_artifact_path"]).read_bytes()).hexdigest(),
        "q_bindings": [{"cell": r["cell"], "q": r["q_digest"]} for r in rows if r["parent_training_rep"] == 0],
        "paired_iid_targets": True, "p2_qualified": False}
    journal = RecoveryJournal(directory, manifest)
    prepared = {}
    def preparation() -> dict[str, Any]:
        calls = 0
        for row in rows:
            if row["parent_training_rep"] != 0:
                continue
            p = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
            q = proposal_from_parameters(row["q_parameters"])
            guide = build_volterra_excursion_guide(p, **config["guide"])
            collective: ConditionalExcursionReference | CollectiveVarianceReference | CoverageEnvelopeReference
            if config.get("reference_family", "").startswith("coverage-envelope"):
                collective = CoverageEnvelopeReference.create(p)
            elif config.get("reference_family", "").startswith("conditional-excursion"):
                collective = ConditionalExcursionReference.create(p)
            else:
                collective = CollectiveVarianceReference.create(p)
            prepared[row["cell"]["id"]] = (CachedVolterraTerminal.create(p), CachedShiftDensity.create(q), q.defensive_mass,
                guide, CachedShiftDensity.create(guide), collective, row["q_digest"])
            calls += 2*p.local_dimension
        return {"actual_work": calls}
    prep = journal.unit("preparation", phase="B", work=1024, maximum_wall=30., operation=preparation)
    records = []
    for job in grid:
        ident = identity(job)
        def operation(j=job, unit=ident):
            evaluator, qdensity, delta, guide, gdensity, collective, digest = prepared[j["cell"]]
            begin = time.perf_counter()
            streams = {s: seed(unit, s) for s in ("path", "label", "whole")}
            if j["target"] == "paired":
                z = (guide.sample(4096, path_seed=streams["path"], label_seed=streams["label"]).samples
                     if j["method"] == "guide-iid" else collective.sample(4096, path_seed=streams["path"], label_seed=streams["label"]))
                density = gdensity.log_q_over_p(z) if j["method"] == "guide-iid" else collective.log_q_over_p(z)
                g = evaluator.log_probability(z)
                log_values = {"mu": (g-density).tolist(), "risk": (2*g-qdensity.log_q_over_p(z)-density).tolist()}
                calls, stages = len(z), None
            else:
                def potential(z):
                    g = evaluator.log_probability(z)
                    return g if j["target"] == "mu" else log_risk_potential(g, qdensity.log_q_over_p(z), defensive_mass=delta)
                arithmetic = config.get("reference_family") == "conditional-excursion-arithmetic"
                temperatures = ((0.,)+tuple(10**(-8+8*i/62) for i in range(63)) if arithmetic
                                else tuple((i/63)**2 for i in range(64)))
                cfg = WeightedSMCConfig(1024, temperatures, 4, .35, 1,
                    streams["whole"], resample_every=4, resampling_scheme="stratified", independence_every=2)
                directional = config.get("reference_family") in ("conditional-excursion-directional", "conditional-excursion-arithmetic", "coverage-envelope", "coverage-envelope-f64")
                scale = (config["normalizer_scales"][j["cell"]][j["target"]]
                         *(delta if j["target"] == "risk" else 1.) if arithmetic else 1.)
                out = estimate_proposal_bridge(potential, proposal=collective, config=cfg,
                    base=cast(Literal["natural", "proposal"], "natural" if j["method"] == "natural-smc" else "proposal"),
                    directional_units=collective.units if directional else None,
                    directional_price_indices=collective.price_indices if directional else None,
                    bridge="arithmetic" if arithmetic else "geometric", normalizer_scale=scale)
                value = out.log_normalizer-(math.log(delta) if j["target"] == "risk" else 0.)
                log_values = {j["target"]: [value]}
                calls, stages = out.potential_evaluations, out.stages
                if calls != 254976:
                    raise ValueError("fixed schedule work mismatch")
            return {**j, "logs": log_values, "actual_work": calls, "wall": time.perf_counter()-begin,
                    "stages": stages, "seeds": streams, "q_digest": digest}
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS cap")
        smc = job["target"] != "paired"
        records.append(journal.unit(ident, phase="B", work=254976 if smc else 4096,
                                     maximum_wall=30. if smc else 5., operation=operation))
        if len(records) % 16 == 0:
            print(f"complete {len(records)}/{len(grid)}", flush=True)
    summaries = summarize(records)
    phase_wall = time.time()-journal._read(directory/"phase-B.clock.json")["started_unix"]
    if phase_wall > 1800:
        raise TimeoutError("summary-inclusive pilot budget exceeded")
    output = {"status": "completed_new_mechanism_pilot", "config": config, "source": source, "manifest": manifest,
        "summaries": summaries, "decision": decision(summaries), "units": len(records),
        "actual_potential_calls_including_preparation": sum(r["actual_work"] for r in records)+prep["actual_work"],
        "phase_wall_including_preparation_io_summary": phase_wall}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-B.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "pilot_wall_failure"})
        raise TimeoutError("final I/O budget exceeded")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    payload = json.loads(path.read_text())
    validate(payload["config"])
    audit_snapshot(ROOT, payload["source"], payload["config"])
    _, rows = inputs(payload["config"])
    grid = jobs(rows, payload["config"].get("reference_family", "collective"))
    manifest = payload["manifest"]
    if manifest["expected_units"] != [identity(j) for j in grid] or payload["units"] != len(grid):
        raise ValueError("changed grid")
    if manifest["input_sha256"] != hashlib.sha256((ROOT/payload["config"]["proposal_artifact_path"]).read_bytes()).hexdigest():
        raise ValueError("input changed")
    journal = RecoveryJournal(path.parent, manifest)
    records = [journal._read(path.parent/(identity(j)+".finish.json"))["result"] for j in grid]
    for job, r in zip(grid, records, strict=True):
        if any(r[k] != v for k, v in job.items()) or r["seeds"] != {s: seed(identity(job), s) for s in ("path", "label", "whole")}:
            raise ValueError("identity/seed changed")
        expected_q = next(x["q_digest"] for x in rows if x["cell"]["id"] == job["cell"] and x["parent_training_rep"] == 0)
        if r["q_digest"] != expected_q:
            raise ValueError("q changed")
    summaries = summarize(records)
    if summaries != payload["summaries"] or decision(summaries) != payload["decision"]:
        raise ValueError("saved summary/decision changed")
    if journal.accounting("B")[0] != payload["actual_potential_calls_including_preparation"]:
        raise ValueError("work changed")
    if (path.parent/"phase-B.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_q_seed_arithmetic_audit_pass_not_precision_certificate", "units": len(grid)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(ROOT/args.audit)))
    else:
        output = run(yaml.safe_load((ROOT/args.config).read_text()))
        print(json.dumps({k: output[k] for k in ("status", "units", "actual_potential_calls_including_preparation")}))


if __name__ == "__main__":
    main()
