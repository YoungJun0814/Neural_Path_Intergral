"""Fixed-count, fresh high-eta q0 validation, not a full P1 pass."""

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

from experiments.post_audit_p1_recovery_p2_parallel import identity, seed, summarize, validate
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.conditional_excursion_reference import ConditionalExcursionReference
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.proposal_bridge_smc import estimate_proposal_bridge
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import relative_equivalence
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/HIGH_ETA_FIXED_PRECISION_VALIDATION_PROTOCOL_2026-10-09_KO.md"
DIRECTORY = "results/post_audit/high_eta_precision_validation_v1/session_20261009"


def grid() -> list[dict[str, Any]]:
    common = {"cell": "high-eta200-k1", "namespace": "high-eta-fixed-precision-v1"}
    jobs = [{**common, "method": method, "target": "paired", "rep": rep, "batch": batch}
            for method in ("guide-iid", "excursion-iid") for rep in range(24) for batch in range(16)]
    jobs += [{**common, "method": method, "target": target, "rep": rep, "batch": 0}
             for method in ("natural-smc", "proposal-smc") for target in ("mu", "risk") for rep in range(64)]
    return jobs


def decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    methods = ("guide-iid", "excursion-iid", "natural-smc", "proposal-smc")
    for target in ("mu", "risk"):
        group = {s["method"]: s for s in summaries if s["target"] == target}
        for method in methods:
            s = group[method]
            passed = (s["summary"]["relative_se"] <= .015
                      and s["summary"]["maximum_fraction"] <= (.1 if "smc" in method else .01)
                      and s["sensitivity"]["maximum_leave_one_out_relative_shift"] <= .05)
            checks.append({"method": method, "target": target, "pass": passed})
        for i, a in enumerate(methods):
            for b in methods[i+1:]:
                agreements.append({"target": target, "methods": [a, b],
                    "equivalence": relative_equivalence(group[a]["summary"], group[b]["summary"], comparisons=30)})
    return {"high_eta_parent0_development_precision_pass": all(x["pass"] for x in checks)
            and all(x["equivalence"]["pass"] for x in agreements),
            "precision_checks": checks, "agreements": agreements,
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
            "scope": "fresh_fixed_count_high_eta_q0_only_not_distribution_free_or_journal_confirmation"}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_arithmetic_bridge_recovery_v1.yaml").read_text())
    validate(input_config)
    metadata, rows = inputs(input_config)
    row = next(r for r in rows if r["cell"]["id"] == "high-eta200-k1" and r["parent_training_rep"] == 0)
    config = {"input": input_config, "iid_blocks": 24, "iid_block_count": 65536, "smc_wholes": 64,
              "scope": "high_eta_parent0_fixed_precision_development", "plan": PLAN}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, input_config["scale_anchor"]))
    jobs = grid()
    streams = [seed(identity(j), s) for j in jobs for s in ("path", "label", "whole")]
    if len(streams) != len(set(streams)):
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "q_digest": row["q_digest"],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"V": {"work": 80000000, "wall": 1800.}}, "expected_units": [identity(j) for j in jobs]}
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    journal = RecoveryJournal(directory, manifest)
    cache: dict[str, Any] = {}
    def prepare():
        p = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
        q = proposal_from_parameters(row["q_parameters"])
        guide = build_volterra_excursion_guide(p, **input_config["guide"])
        cache.update(evaluator=CachedVolterraTerminal.create(p), qdensity=CachedShiftDensity.create(q),
                     delta=q.defensive_mass, guide=guide, gdensity=CachedShiftDensity.create(guide),
                     reference=ConditionalExcursionReference.create(p))
        return {"actual_work": 2*p.local_dimension}
    prep = journal.unit("preparation", phase="V", work=1024, maximum_wall=30., operation=prepare)
    records = []
    for job in jobs:
        ident = identity(job)
        def operation(j=job, unit=ident):
            begin = time.perf_counter()
            streams = {s: seed(unit, s) for s in ("path", "label", "whole")}
            r, evaluator, qdensity, delta = cache["reference"], cache["evaluator"], cache["qdensity"], cache["delta"]
            if j["target"] == "paired":
                z = (cache["guide"].sample(4096, path_seed=streams["path"], label_seed=streams["label"]).samples
                     if j["method"] == "guide-iid" else r.sample(4096, path_seed=streams["path"], label_seed=streams["label"]))
                d = cache["gdensity"].log_q_over_p(z) if j["method"] == "guide-iid" else r.log_q_over_p(z)
                h = evaluator.log_probability(z)
                logs = {"mu": (h-d).tolist(), "risk": (2*h-qdensity.log_q_over_p(z)-d).tolist()}
                calls, stages = 4096, None
            else:
                def potential(z):
                    g = evaluator.log_probability(z)
                    return g if j["target"] == "mu" else log_risk_potential(g, qdensity.log_q_over_p(z), defensive_mass=delta)
                scale = input_config["normalizer_scales"][j["cell"]][j["target"]]*(delta if j["target"] == "risk" else 1.)
                cfg = WeightedSMCConfig(1024, (0.,)+tuple(10**(-8+8*k/62) for k in range(63)), 4, .35, 1,
                                        streams["whole"], resample_every=4, resampling_scheme="stratified", independence_every=2)
                out = estimate_proposal_bridge(potential, proposal=r, config=cfg,
                    base=cast(Literal["natural", "proposal"], "natural" if j["method"] == "natural-smc" else "proposal"),
                    bridge="arithmetic", normalizer_scale=scale, directional_units=r.units, directional_price_indices=r.price_indices)
                logs = {j["target"]: [out.log_normalizer-(math.log(delta) if j["target"] == "risk" else 0.)]}
                calls, stages = out.potential_evaluations, out.stages
                if calls != 254976:
                    raise ValueError("whole-run cost mismatch")
            return {**j, "logs": logs, "actual_work": calls, "wall": time.perf_counter()-begin,
                    "stages": stages, "seeds": streams, "q_digest": row["q_digest"]}
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS cap")
        records.append(journal.unit(ident, phase="V", work=4096 if job["target"] == "paired" else 254976,
                                    maximum_wall=5. if job["target"] == "paired" else 30., operation=operation))
        if len(records) % 64 == 0:
            print(f"fixed validation {len(records)}/{len(jobs)}", flush=True)
    summaries = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "decision": decision(summaries), "units": len(jobs),
              "work": sum(r["actual_work"] for r in records)+prep["actual_work"],
              "phase_wall": time.time()-journal._read(directory/"phase-V.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-V.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "validation_wall_failure"})
        raise TimeoutError("summary and final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    audit_snapshot(ROOT, output["source"], output["config"])
    config, manifest = output["config"]["input"], output["manifest"]
    validate(config)
    _, rows = inputs(config)
    row = next(r for r in rows if r["cell"]["id"] == "high-eta200-k1" and r["parent_training_rep"] == 0)
    if (manifest["q_digest"] != row["q_digest"] or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["proposal_artifact_path"]).read_bytes()).hexdigest()
            or manifest["expected_units"] != [identity(j) for j in grid()] or output["units"] != len(grid())):
        raise ValueError("source/input/q/grid binding mismatch")
    journal = RecoveryJournal(path.parent, manifest)
    records = [journal._read(path.parent/(identity(j)+".finish.json"))["result"] for j in grid()]
    for j, r in zip(grid(), records, strict=True):
        if (any(r[k] != v for k, v in j.items()) or r["q_digest"] != row["q_digest"]
                or r["seeds"] != {s: seed(identity(j), s) for s in ("path", "label", "whole")}
                or r["actual_work"] != (4096 if j["target"] == "paired" else 254976)):
            raise ValueError("changed unit")
    summaries = summarize(records)
    if (summaries != output["summaries"] or decision(summaries) != output["decision"]
            or journal.accounting("V")[0] != output["work"]):
        raise ValueError("changed arithmetic/accounting")
    if (path.parent/"phase-V.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_q_seed_arithmetic_audit_pass_not_coverage_certificate", "units": len(records),
            "decision": output["decision"]["high_eta_parent0_development_precision_pass"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "units", "phase_wall", "decision")}))
