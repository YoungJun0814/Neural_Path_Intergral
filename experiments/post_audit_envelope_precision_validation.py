"""Bounded-memory fixed precision validation of the corrected coverage envelope."""

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

from experiments.post_audit_p1_recovery_p2_parallel import identity, seed, validate
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.coverage_envelope_reference import CoverageEnvelopeReference
from src.path_integral.proposal_bridge_smc import estimate_proposal_bridge
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    merge_moments,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/ENVELOPE_FIXED_PRECISION_VALIDATION_2026-10-10_KO.md"
ANCHOR = "results/post_audit/p1_coverage_envelope_recovery_v2/session_20261010/result.json"
ANCHOR_HASH = "b7074b2a135415bdac0cf62ec42494eab561f027e44a5b046b114f3a1c0d4f44"
DIRECTORY = "results/post_audit/envelope_precision_validation_v1/session_20261010"


def grid() -> list[dict[str, Any]]:
    common = {"cell": "high-eta200-k1", "namespace": "envelope-fixed-precision-v1"}
    jobs = [{**common, "method": method, "target": "paired", "rep": rep, "batch": b}
            for method in ("guide-iid", "envelope-iid") for rep in range(96) for b in range(8)]
    jobs += [{**common, "method": method, "target": target, "rep": rep, "batch": 0}
             for method, count in (("natural-smc", 176), ("proposal-smc", 64))
             for target in ("mu", "risk") for rep in range(count)]
    return jobs


def prepare(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    validate(config)
    metadata, rows = inputs(config)
    row = next(r for r in rows if r["cell"]["id"] == "high-eta200-k1" and r["parent_training_rep"] == 0)
    p = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
    q = proposal_from_parameters(row["q_parameters"])
    guide = build_volterra_excursion_guide(p, **config["guide"])
    return row, {"evaluator": CachedVolterraTerminal.create(p), "qdensity": CachedShiftDensity.create(q),
                 "delta": q.defensive_mass, "guide": guide, "gdensity": CachedShiftDensity.create(guide),
                 "reference": CoverageEnvelopeReference.create(p), "preparation_work": 2*p.local_dimension}


def unit_values(job: dict[str, Any], cache: dict[str, Any], q_digest: str) -> dict[str, Any]:
    start = time.perf_counter()
    streams = {s: seed(identity(job), s) for s in ("path", "label", "whole")}
    evaluator, r, delta = cache["evaluator"], cache["reference"], cache["delta"]
    if job["target"] == "paired":
        z = (cache["guide"].sample(4096, path_seed=streams["path"], label_seed=streams["label"]).samples
             if job["method"] == "guide-iid" else r.sample(4096, path_seed=streams["path"], label_seed=streams["label"]))
        d = cache["gdensity"].log_q_over_p(z) if job["method"] == "guide-iid" else r.log_q_over_p(z)
        g = evaluator.log_probability(z)
        values = {"mu": g-d, "risk": 2*g-cache["qdensity"].log_q_over_p(z)-d}
        payload: dict[str, Any] = {"moments": {t: log_moments(v) for t, v in values.items()}}
        calls, stages = 4096, None
    else:
        def potential(z):
            g = evaluator.log_probability(z)
            return g if job["target"] == "mu" else log_risk_potential(g, cache["qdensity"].log_q_over_p(z), defensive_mass=delta)
        cfg = WeightedSMCConfig(1024, tuple((k/63)**2 for k in range(64)), 4, .35, 1, streams["whole"],
                               resample_every=4, resampling_scheme="stratified", independence_every=2)
        out = estimate_proposal_bridge(potential, proposal=r, config=cfg,
            base=cast(Literal["natural", "proposal"], "natural" if job["method"] == "natural-smc" else "proposal"),
            directional_units=r.units, directional_price_indices=r.price_indices)
        payload = {"log_normalizer": out.log_normalizer-(math.log(delta) if job["target"] == "risk" else 0.)}
        calls, stages = out.potential_evaluations, out.stages
        if calls != 254976:
            raise ValueError("whole-run work mismatch")
    return {**job, **payload, "actual_work": calls, "wall": time.perf_counter()-start,
            "stages": stages, "seeds": streams, "q_digest": q_digest}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in ("guide-iid", "envelope-iid", "natural-smc", "proposal-smc"):
        for target in ("mu", "risk"):
            parts = [r for r in records if r["method"] == method and r["target"] in ("paired", target)]
            if "iid" in method:
                summary = merge_moments([r["moments"][target] for r in parts])
                reps = sorted({r["rep"] for r in parts})
                rep_logs = [merge_moments([r["moments"][target] for r in parts if r["rep"] == rep])["log_mean"] for rep in reps]
            else:
                rep_logs = [r["log_normalizer"] for r in parts]
                summary = log_moments(torch.tensor(rep_logs, dtype=torch.float64))
            result.append({"cell": "high-eta200-k1", "method": method, "target": target, "summary": summary,
                           "whole_replication_log_means": rep_logs,
                           "sensitivity": sensitivity(rep_logs, bootstrap_seed=seed("envelope-validation-"+method+target, "bootstrap")),
                           "wall": sum(r["wall"] for r in parts),
                           "inference_unit": "iid_draw_with_equal_size_block_sensitivity" if "iid" in method else "whole_smc_run"})
    return result


def decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    methods = ("guide-iid", "envelope-iid", "natural-smc", "proposal-smc")
    for target in ("mu", "risk"):
        group = {s["method"]: s for s in summaries if s["target"] == target}
        for method in methods:
            s = group[method]
            passed = (s["summary"]["relative_se"] <= .015 and s["summary"]["maximum_fraction"] <= (.01 if "iid" in method else .1)
                      and s["sensitivity"]["maximum_leave_one_out_relative_shift"] <= .05)
            checks.append({"method": method, "target": target, "pass": passed})
        for i, a in enumerate(methods):
            for b in methods[i+1:]:
                agreements.append({"methods": [a, b], "target": target,
                    "equivalence": relative_equivalence(group[a]["summary"], group[b]["summary"], comparisons=30)})
    return {"high_eta_parent0_development_precision_pass": all(x["pass"] for x in checks)
            and all(x["equivalence"]["pass"] for x in agreements), "precision_checks": checks, "agreements": agreements,
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
            "scope": "high_eta_q0_fixed_count_development_not_distribution_free_confirmation"}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    if hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != ANCHOR_HASH:
        raise ValueError("changed pre-existing pilot anchor")
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    config = {"input": input_config, "iid_blocks": 96, "iid_block_count": 32768,
              "smc_wholes": {"natural-smc": 176, "proposal-smc": 64}, "anchor": ANCHOR,
              "anchor_sha256": ANCHOR_HASH, "scope": "scoped_high_eta_q0_precision_validation"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    metadata, rows = inputs(input_config)
    row = next(r for r in rows if r["cell"]["id"] == "high-eta200-k1" and r["parent_training_rep"] == 0)
    jobs = grid()
    seeds = [seed(identity(j), s) for j in jobs for s in ("path", "label", "whole")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "q_digest": row["q_digest"],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "expected_units": [identity(j) for j in jobs], "limits": {"V": {"work": 160000000, "wall": 3600.}}}
    journal = RecoveryJournal(directory, manifest)
    saved: dict[str, Any] = {}
    def preparation():
        _, cache = prepare(input_config)
        saved.update(cache)
        return {"actual_work": cache["preparation_work"]}
    prep = journal.unit("preparation", phase="V", work=1024, maximum_wall=30., operation=preparation)
    records = []
    for j in jobs:
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS cap")
        def operation(job=j):
            return unit_values(job, saved, row["q_digest"])
        records.append(journal.unit(identity(j), phase="V", work=4096 if j["target"] == "paired" else 254976,
                                    maximum_wall=5. if j["target"] == "paired" else 30., operation=operation))
        if len(records) % 128 == 0:
            print(f"envelope fixed validation {len(records)}/{len(jobs)}", flush=True)
    summaries = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "decision": decision(summaries), "units": len(jobs), "work": sum(r["actual_work"] for r in records)+prep["actual_work"],
              "phase_wall": time.time()-journal._read(directory/"phase-V.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-V.clock.json")["started_unix"] > 3600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    audit_snapshot(ROOT, output["source"], output["config"])
    manifest, config = output["manifest"], output["config"]
    if (config["anchor"] != ANCHOR or config["anchor_sha256"] != ANCHOR_HASH
            or hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != ANCHOR_HASH
            or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["expected_units"] != [identity(j) for j in grid()] or output["units"] != len(grid())
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()):
        raise ValueError("source/config/grid/input mismatch")
    row, cache = prepare(config["input"])
    if manifest["q_digest"] != row["q_digest"]:
        raise ValueError("q binding mismatch")
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
        raise ValueError("summary/accounting mismatch")
    replayed = 0
    for method in ("guide-iid", "envelope-iid"):
        for rep in (0, 95):
            job = next(j for j in grid() if j["method"] == method and j["rep"] == rep and j["batch"] == 0)
            original = next(r for r in records if identity(r) == identity(job))
            replay = unit_values(job, cache, row["q_digest"])
            if replay["moments"] != original["moments"]:
                raise ValueError("IID moment replay mismatch")
            replayed += 4096
    if (path.parent/"phase-V.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_q_seed_moment_arithmetic_replay_pass_not_coverage_certificate", "units": len(records),
            "replayed_paths": replayed, "decision": output["decision"]["high_eta_parent0_development_precision_pass"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "phase_wall", "units", "decision")}))
