"""Fixed known-component stratification with ordinary full-mixture weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.fixed_stratum_reference import proportional_guide_counts, stratified_summary
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    merge_moments,
    precision_count,
    sensitivity,
)
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

ROOT = Path(__file__).resolve().parents[1]
CELL = "canonical-h005-k1"
PLAN = "docs/plans/CANONICAL_STATIC_GUIDE_STRATIFICATION_PILOT_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/canonical_static_stratification_v1/session_20261010"
TARGETS = ["mu"]+[f"risk-q{j}" for j in range(5)]
BLOCK_SIZE = 29760


def jobs() -> list[dict[str, Any]]:
    return [{"method": method, "rep": rep} for method in ("iid", "stratified") for rep in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{CELL}-{j['method']}-{j['rep']}"


def seed(j: dict[str, Any], stream: str) -> int:
    return int(hashlib.sha256(("canonical-static-strata-v1/"+identity(j)+"/"+stream).encode()).hexdigest()[:15], 16)


def prepare(metadata: dict[str, Any], rows: list[dict[str, Any]], input_config: dict[str, Any]) -> dict[str, Any]:
    family = sorted([r for r in rows if r["cell"]["id"] == CELL], key=lambda r: r["parent_training_rep"])
    if [r["parent_training_rep"] for r in family] != list(range(5)):
        raise ValueError("incomplete frozen q family")
    p = _problem(metadata["config"]["model"], family[0]["cell"], 32)
    guide = build_volterra_excursion_guide(p, **input_config["guide"])
    qs = [proposal_from_parameters(r["q_parameters"]) for r in family]
    counts = proportional_guide_counts(guide.weights, multiplier=8)
    density = CachedShiftDensity.create(guide)  # Enforces identity covariance.
    if len(counts) != 373 or sum(counts) != BLOCK_SIZE:
        raise ValueError("changed guide design")
    return {"evaluator": CachedVolterraTerminal.create(p), "guide": guide, "density": density,
            "qdensities": [CachedShiftDensity.create(q) for q in qs], "q_digests": [r["q_digest"] for r in family],
            "stratum_counts": counts}


def unit_values(j: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    streams = {s: seed(j, s) for s in (("path", "label") if j["method"] == "iid" else ("path",))}
    if j["method"] == "iid":
        z = c["guide"].sample(BLOCK_SIZE, path_seed=streams["path"], label_seed=streams["label"]).samples
    else:
        labels = torch.repeat_interleave(torch.arange(len(c["stratum_counts"])), torch.tensor(c["stratum_counts"]))
        noise = torch.randn((BLOCK_SIZE, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(streams["path"]))
        z = noise+c["density"].means[labels]
    parts = []
    for batch in z.split(4096):
        g, r = c["evaluator"].log_probability(batch), c["density"].log_q_over_p(batch)
        parts.append(torch.stack([g-r]+[2*g-q.log_q_over_p(batch)-r for q in c["qdensities"]], 1))
    values = torch.cat(parts)
    if j["method"] == "iid":
        moments = [log_moments(values[:, k]) for k in range(6)]
        strata = None
    else:
        strata = []
        offset = 0
        for count in c["stratum_counts"]:
            strata.append([log_moments(values[offset:offset+count, k]) for k in range(6)])
            offset += count
        moments = [stratified_summary([s[k] for s in strata], c["guide"].weights.log()) for k in range(6)]
    return {**j, "seeds": streams, "actual_work": BLOCK_SIZE, "moments": moments, "strata": strata,
            "stratum_counts": list(c["stratum_counts"]) if j["method"] == "stratified" else None,
            "q_digests": c["q_digests"], "wall": time.perf_counter()-begin, "guide_independent_kernel": False}


def summarize(records: list[dict[str, Any]], c: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for method in ("iid", "stratified"):
        group = [r for r in records if r["method"] == method]
        if len(group) != 8 or {r["rep"] for r in group} != set(range(8)):
            raise ValueError("incomplete block grid")
        components: list[dict[str, Any]] = []
        for k, target in enumerate(TARGETS):
            if method == "iid":
                stats = merge_moments([r["moments"][k] for r in group])
            else:
                strata = [merge_moments([r["strata"][s][k] for r in group]) for s in range(len(c["stratum_counts"]))]
                stats = stratified_summary(strata, c["guide"].weights.log())
            logs = [r["moments"][k]["log_mean"] for r in group]
            components.append({"target": target, "summary": stats,
                "sensitivity": sensitivity(logs, bootstrap_seed=seed({"method": method, "rep": 200}, target)),
                "allocation": precision_count(stats, target_rse=.015, safety_factor=2,
                    minimum=32*BLOCK_SIZE, maximum=8388608, multiple=BLOCK_SIZE)})
        count = max(s["allocation"]["required_count"] for s in components)
        mean_block_wall = sum(r["wall"] for r in group)/8
        out.append({"method": method, "components": components, "shared_draw_required": count,
                    "allocation_feasible": count <= 8388608, "forecast_wall": count/BLOCK_SIZE*mean_block_wall,
                    "selection_cost_proxy": max(s["summary"]["relative_se"]**2*s["summary"]["count"] for s in components)*mean_block_wall/BLOCK_SIZE,
                    "scope": "known_stratum_development_not_independent_reference_qualification"})
    return out


def selected_method(summaries: list[dict[str, Any]]) -> str | None:
    feasible = [s for s in summaries if s["allocation_feasible"]]
    return min(feasible, key=lambda s: (s["selection_cost_proxy"], s["method"]))["method"] if feasible else None


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"input": input_config, "blocks": 8, "multiplier": 8, "block_size": BLOCK_SIZE,
              "scope": "canonical_static_known_stratum_pilot"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in jobs()],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"T": {"work": 600000, "wall": 600.}}}
    journal = RecoveryJournal(directory, manifest)
    c: dict[str, Any] = {}
    def preparation():
        c.update(prepare(metadata, rows, input_config))
        return {"actual_work": 64}
    journal.unit("preparation", phase="T", work=64, maximum_wall=30., operation=preparation)
    records = []
    for j in jobs():
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            return unit_values(job, c)
        records.append(journal.unit(identity(j), phase="T", work=BLOCK_SIZE, maximum_wall=30.,
                                    operation=operation))
    summaries = summarize(records, c)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "selected_method": selected_method(summaries), "units": len(records), "work": journal.accounting("T")[0],
              "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-T.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-T.clock.json")["started_unix"] > 600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()
            or manifest["expected_units"] != [identity(j) for j in jobs()]):
        raise ValueError("changed source/input/grid")
    metadata, rows = inputs(config["input"])
    c = prepare(metadata, rows, config["input"])
    journal = RecoveryJournal(path.parent, manifest)
    records = []
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed block")
        r = finish["result"]
        streams = {s: seed(j, s) for s in (("path", "label") if j["method"] == "iid" else ("path",))}
        if (any(r[k] != v for k, v in j.items()) or r["seeds"] != streams or r["actual_work"] != BLOCK_SIZE
                or r["q_digests"] != c["q_digests"] or r["guide_independent_kernel"]
                or r["stratum_counts"] != (list(c["stratum_counts"]) if j["method"] == "stratified" else None)):
            raise ValueError("changed unit binding")
        if j["rep"] == 0:
            replay = unit_values(j, c)
            if replay["moments"] != r["moments"] or replay["strata"] != r["strata"]:
                raise ValueError("block replay mismatch")
        records.append(r)
    summaries = summarize(records, c)
    if (summaries != output["summaries"] or selected_method(summaries) != output["selected_method"]
            or output["units"] != 16 or journal.accounting("T")[0] != output["work"] or output["work"] != 476224
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("changed summary/cost/gates")
    if (path.parent/"phase-T.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_q_seed_stratum_variance_accounting_replay_pass_not_precision_certificate",
            "replayed_draws": 2*BLOCK_SIZE, "audit_forward_evaluations": 64+2*BLOCK_SIZE,
            "selected_method": output["selected_method"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "units", "phase_wall", "selected_method")}))
