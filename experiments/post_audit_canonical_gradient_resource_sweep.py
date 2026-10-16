"""Bounded fresh resource-design pilot for guide-free gradient family SMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_canonical_gradient_recovery import CELL, family_weights
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_umbrella_reference_pilot import prepare
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, precision_count, sensitivity
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/CANONICAL_GRADIENT_RESOURCE_SWEEP_2026-10-10_KO.md"
ANCHOR = "results/post_audit/canonical_gradient_recovery_v1/session_20261010/result.json"
DIRECTORY = "results/post_audit/canonical_gradient_resource_v1/session_20261010"
DESIGNS = {"more-moves": (1024, 64, 8, .35), "more-particles": (2048, 64, 4, .35),
           "more-temperatures": (1024, 128, 4, .35), "larger-move": (1024, 64, 4, .55)}


def jobs() -> list[dict[str, Any]]:
    return [{"design": d, "rep": rep} for d in DESIGNS for rep in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{CELL}-{j['design']}-{j['rep']}"


def seed(j: dict[str, Any]) -> int:
    return int(hashlib.sha256(("canonical-gradient-resource-v1/"+identity(j)).encode()).hexdigest()[:15], 16)


def expected_work(j: dict[str, Any]) -> int:
    particles, points, moves, _ = DESIGNS[j["design"]]
    return particles*(1+(points-2)*moves)


def unit_values(j: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    particles, points, moves, scale = DESIGNS[j["design"]]
    weights = family_weights("risks-pcnl", c["log_weights"].tolist())
    def components(z):
        g = c["evaluator"].log_probability(z)
        return torch.stack([log_risk_potential(g, q.log_q_over_p(z), defensive_mass=d)
                            for q, d in zip(c["qdensities"], c["deltas"], strict=True)], 1)
    cfg = WeightedSMCConfig(particles, tuple((k/(points-1))**2 for k in range(points)), moves, scale,
                            1, seed(j), resample_every=4, resampling_scheme="stratified")
    r = estimate_gradient_umbrella_normalizers(components, dimension=c["dimension"],
        log_mixture_weights=weights, config=cfg, maximum_drift_norm=20.)
    correction = torch.tensor([math.log(d) for d in c["deltas"]], dtype=torch.float64)
    return {**j, "seed": seed(j), "design_parameters": list(DESIGNS[j["design"]]),
            "actual_work": r.potential_evaluations, "backward_evaluations": r.gradient_evaluations,
            "q_digests": c["q_digests"], "log_mixture_weights": weights.tolist(),
            "log_normalizer": r.log_normalizer, "log_scaled_component_normalizers": r.log_component_normalizers.tolist(),
            "log_estimand_values": (r.log_component_normalizers-correction).tolist(),
            "stages": list(r.stages), "wall": time.perf_counter()-begin, "guide_independent_kernel": True}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for design in DESIGNS:
        group = [r for r in records if r["design"] == design]
        if len(group) != 8 or {r["rep"] for r in group} != set(range(8)):
            raise ValueError("incomplete design grid")
        components: list[dict[str, Any]] = []
        for j in range(5):
            logs = [r["log_estimand_values"][j] for r in group]
            stats = log_moments(torch.tensor(logs, dtype=torch.float64))
            components.append({"target": f"risk-q{j}", "summary": stats, "whole_log_means": logs,
                "sensitivity": sensitivity(logs, bootstrap_seed=seed({"design": design+f"q{j}", "rep": 200})),
                "allocation": precision_count(stats, target_rse=.015, safety_factor=2, minimum=32, maximum=256, multiple=1)})
        count = max(c["allocation"]["required_count"] for c in components)
        wall = sum(r["wall"] for r in group)/8
        work = count*expected_work(group[0])
        feasible = count <= 256 and work <= 160000000 and count*wall <= 1800
        stage_ess = [r["stages"][-1]["ess_before_resampling"] for r in group]
        ancestors = [r["stages"][-1]["surviving_initial_ancestors"] for r in group]
        out.append({"design": design, "components": components, "shared_whole_required": count,
                    "normalizer_summary": log_moments(torch.tensor([r["log_normalizer"] for r in group], dtype=torch.float64)),
                    "forecast_forward_evaluations": work, "forecast_backward_evaluations": work,
                    "forecast_wall": count*wall, "allocation_feasible": feasible,
                    "selection_cost_proxy": max(c["summary"]["relative_se"]**2*8 for c in components)*wall,
                    "endpoint_ess": stage_ess, "initial_ancestors": ancestors,
                    "scope": "risk_family_development_design_not_full_reference_qualification"})
    return out


def selected_design(summaries: list[dict[str, Any]]) -> str | None:
    feasible = [s for s in summaries if s["allocation_feasible"]]
    return min(feasible, key=lambda s: (s["selection_cost_proxy"], s["design"]))["design"] if feasible else None


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    weights = anchor["config"]["log_mixture_weights"]
    config = {"input": input_config, "anchor": ANCHOR,
              "anchor_sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest(),
              "log_mixture_weights": weights, "designs": {d: list(v) for d, v in DESIGNS.items()},
              "wholes_per_design": 8, "backward_cap": 18000000}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    if len(grid) != 32 or len({seed(j) for j in grid}) != 32:
        raise ValueError("unexpected seed/grid")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"S": {"work": 18000000, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    cells: dict[str, Any] = {}
    def preparation():
        cells.update(prepare(metadata, rows, weights))
        return {"actual_work": 128}
    journal.unit("preparation", phase="S", work=1024, maximum_wall=30., operation=preparation)
    records: list[dict[str, Any]] = []
    backward = 0
    for j in grid:
        work = expected_work(j)
        if backward+work > config["backward_cap"]:
            raise RuntimeError("backward reservation cap")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            out = unit_values(job, cells[CELL])
            if (out["actual_work"], out["backward_evaluations"]) != (expected_work(job), expected_work(job)):
                raise ValueError("forward/backward accounting")
            return out
        records.append(journal.unit(identity(j), phase="S", work=work, maximum_wall=60., operation=operation))
        backward += work
        if len(records) % 8 == 0:
            print(f"canonical gradient resources {len(records)}/32", flush=True)
    summaries = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "selected_design": selected_design(summaries), "units": len(records), "work": journal.accounting("S")[0],
              "backward_evaluations": backward, "p1_qualified": False, "p2_qualified": False,
              "large_training_authorized": False, "phase_wall": time.time()-journal._read(directory/"phase-S.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-S.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["anchor"] != ANCHOR or hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != config["anchor_sha256"]
            or config["designs"] != {d: list(v) for d, v in DESIGNS.items()}
            or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()
            or manifest["expected_units"] != [identity(j) for j in jobs()]):
        raise ValueError("changed source/anchor/input/design/grid")
    anchor = json.loads((ROOT/ANCHOR).read_text())
    weights = anchor["config"]["log_mixture_weights"]
    if weights != config["log_mixture_weights"]:
        raise ValueError("changed frozen weights")
    metadata, rows = inputs(config["input"])
    c = prepare(metadata, rows, weights)[CELL]
    w = family_weights("risks-pcnl", weights[CELL])
    correction = torch.tensor([math.log(d) for d in c["deltas"]], dtype=torch.float64)
    journal = RecoveryJournal(path.parent, manifest)
    records, audit_forward, audit_backward, replayed = [], 128, 0, 0
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed whole")
        r = finish["result"]
        if (any(r[k] != v for k, v in j.items()) or r["seed"] != seed(j)
                or r["design_parameters"] != list(DESIGNS[j["design"]])
                or (r["actual_work"], r["backward_evaluations"]) != (expected_work(j), expected_work(j))
                or r["q_digests"] != c["q_digests"] or not r["guide_independent_kernel"]
                or r["log_mixture_weights"] != w.tolist()):
            raise ValueError("changed unit binding")
        vector = torch.tensor(r["log_scaled_component_normalizers"], dtype=torch.float64)
        if abs(float(torch.logsumexp(vector+w, 0))-r["log_normalizer"]) > 1e-10 or (vector-correction).tolist() != r["log_estimand_values"]:
            raise ValueError("unnormalized identity/risk correction")
        if j["rep"] == 0:
            replay = unit_values(j, c)
            if any(replay[k] != r[k] for k in ("log_normalizer", "log_scaled_component_normalizers", "stages")):
                raise ValueError("whole replay mismatch")
            audit_forward += replay["actual_work"]
            audit_backward += replay["backward_evaluations"]
            replayed += 1
        records.append(r)
    summaries = summarize(records)
    if (summaries != output["summaries"] or selected_design(summaries) != output["selected_design"]
            or output["units"] != 32 or journal.accounting("S")[0] != output["work"] or output["work"] != 14327936
            or sum(r["backward_evaluations"] for r in records) != output["backward_evaluations"]
            or output["backward_evaluations"] != 14327808
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("changed summary/accounting/gates")
    if (path.parent/"phase-S.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_anchor_q_seed_identity_cost_whole_replay_pass_not_precision_certificate",
            "replayed_wholes": replayed, "audit_forward_evaluations": audit_forward,
            "audit_backward_evaluations": audit_backward, "selected_design": output["selected_design"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "backward_evaluations", "units", "phase_wall", "selected_design")}))
