"""Frozen-family endpoint reference pilot with correct unnormalized SMC measure."""

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

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.coverage_envelope_reference import CoverageEnvelopeReference
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, precision_count, sensitivity
from src.path_integral.umbrella_normalizer_smc import estimate_umbrella_normalizers
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/FROZEN_FAMILY_UMBRELLA_REFERENCE_PILOT_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/umbrella_reference_pilot_v1/session_20261010"
ANCHOR = "results/post_audit/full_q_reference_allocation_v1/session_20261010/result.json"


def seed(identity: str) -> int:
    return int(hashlib.sha256(("umbrella-reference-v1/"+identity).encode()).hexdigest()[:15], 16)


def jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"cell": r["cell"]["id"], "method": method, "rep": rep}
            for r in rows if r["parent_training_rep"] == 0
            for method in ("free-full", "free-block", "envelope") for rep in range(8)]


def identity(j: dict[str, Any]) -> str:
    return j["cell"]+"-"+j["method"]+f"-{j['rep']}"


def expected_work(j: dict[str, Any]) -> int:
    return 254976 if j["method"] == "envelope" else 508928


def weights_from_anchor(anchor: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    result = {}
    for cell in sorted({r["cell"]["id"] for r in rows}):
        normalizers = []
        for component in range(6):
            target, rep = ("mu", 0) if component == 0 else ("risk", component-1)
            matches = [s for s in anchor["summaries"] if s["cell"] == cell and s["method"] == "guide-iid"
                       and s["target"] == target and s["parent_training_rep"] == rep]
            if len(matches) != 1:
                raise ValueError("missing static anchor component")
            delta = (1. if component == 0 else proposal_from_parameters(next(
                r["q_parameters"] for r in rows if r["cell"]["id"] == cell and r["parent_training_rep"] == rep)).defensive_mass)
            value = matches[0]["summary"]["log_mean"]+math.log(delta)
            if not math.isfinite(value):
                raise ValueError("nonpositive anchor mean")
            normalizers.append(value)
        inverse = -torch.tensor(normalizers, dtype=torch.float64)
        result[cell] = (inverse-torch.logsumexp(inverse, 0)).tolist()
    return result


def prepare(metadata: dict[str, Any], rows: list[dict[str, Any]], log_weights: dict[str, list[float]]) -> dict[str, Any]:
    result = {}
    for cell in sorted(log_weights):
        family = sorted([r for r in rows if r["cell"]["id"] == cell], key=lambda r: r["parent_training_rep"])
        if [r["parent_training_rep"] for r in family] != list(range(5)):
            raise ValueError("requires whole frozen five-q family")
        p = _problem(metadata["config"]["model"], family[0]["cell"], 32)
        qs = [proposal_from_parameters(r["q_parameters"]) for r in family]
        result[cell] = {"evaluator": CachedVolterraTerminal.create(p), "dimension": p.local_dimension,
                        "qdensities": [CachedShiftDensity.create(q) for q in qs],
                        "deltas": [q.defensive_mass for q in qs], "q_digests": [r["q_digest"] for r in family],
                        "reference": CoverageEnvelopeReference.create(p),
                        "log_weights": torch.tensor(log_weights[cell], dtype=torch.float64)}
    return result


def unit_values(j: dict[str, Any], cells: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    c = cells[j["cell"]]
    def components(z):
        g = c["evaluator"].log_probability(z)
        return torch.stack([g]+[log_risk_potential(g, q.log_q_over_p(z), defensive_mass=delta)
                                for q, delta in zip(c["qdensities"], c["deltas"], strict=True)], 1)
    guided = j["method"] == "envelope"
    whole_seed = seed(j.get("namespace", "")+identity(j))
    cfg = WeightedSMCConfig(1024, tuple((k/63)**2 for k in range(64)), 4 if guided else 8, .35, 1,
                           whole_seed, resample_every=4, resampling_scheme="stratified",
                           independence_every=2 if guided else 0)
    r = c["reference"]
    result = estimate_umbrella_normalizers(components, dimension=c["dimension"],
        log_mixture_weights=c["log_weights"], config=cfg, use_blocks=j["method"] == "free-block",
        proposal=r if guided else None, directional_units=r.units if guided else None,
        directional_price_indices=r.price_indices if guided else None)
    correction = torch.tensor([0.]+[math.log(d) for d in c["deltas"]], dtype=torch.float64)
    return {**j, "seed": whole_seed, "actual_work": result.potential_evaluations,
            "log_normalizer": result.log_normalizer,
            "log_scaled_component_normalizers": result.log_component_normalizers.tolist(),
            "log_estimand_values": (result.log_component_normalizers-correction).tolist(),
            "component_targets": ["mu"]+[f"risk-q{rep}" for rep in range(5)],
            "q_digests": c["q_digests"], "log_mixture_weights": c["log_weights"].tolist(),
            "stages": list(result.stages), "wall": time.perf_counter()-begin,
            "guide_independent_kernel": not guided}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cell in sorted({r["cell"] for r in records}):
        for method in ("free-full", "free-block", "envelope"):
            group = [r for r in records if r["cell"] == cell and r["method"] == method]
            if len(group) != 8 or {r["rep"] for r in group} != set(range(8)):
                raise ValueError("incomplete whole grid")
            components: list[dict[str, Any]] = []
            for j, target in enumerate(["mu"]+[f"risk-q{rep}" for rep in range(5)]):
                logs = [r["log_estimand_values"][j] for r in group]
                stats = log_moments(torch.tensor(logs, dtype=torch.float64))
                components.append({"target": target, "summary": stats,
                    "whole_log_means": logs, "sensitivity": sensitivity(logs, bootstrap_seed=seed(cell+method+target+"bootstrap")),
                    "allocation": precision_count(stats, target_rse=.015, safety_factor=2,
                                                   minimum=32, maximum=256, multiple=1)})
            required = max(s["allocation"]["required_count"] for s in components)
            wall = sum(r["wall"] for r in group)
            result.append({"cell": cell, "method": method, "components": components,
                "shared_whole_required": required, "shared_whole_cap": 256,
                "shared_allocation_feasible": required <= 256,
                "forecast_work": required*expected_work(group[0]), "forecast_wall": required*wall/len(group),
                "wall": wall, "guide_independent_kernel": method != "envelope",
                "scope": "correlated_component_outputs_independent_whole_development_only"})
    return result


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    log_weights = weights_from_anchor(anchor, rows)
    config = {"input": input_config, "anchor": ANCHOR,
              "anchor_sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest(),
              "log_mixture_weights": log_weights, "whole_repetitions": 8, "steps": 32,
              "scope": "frozen_family_umbrella_reference_development"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR,
        "docs/theory/FROZEN_FAMILY_UMBRELLA_UNNORMALIZED_MEASURE_2026-10-10_KO.md"))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs(rows)
    if len(grid) != 48 or len({seed(identity(j)) for j in grid}) != len(grid):
        raise ValueError("unexpected grid/seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"U": {"work": 25000000, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    cells: dict[str, Any] = {}
    def preparation():
        cells.update(prepare(metadata, rows, log_weights))
        return {"actual_work": 128}
    journal.unit("preparation", phase="U", work=1024, maximum_wall=30., operation=preparation)
    records = []
    for j in grid:
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            out = unit_values(job, cells)
            if out["actual_work"] != expected_work(job):
                raise ValueError("unexpected actual work")
            return out
        records.append(journal.unit(identity(j), phase="U", work=expected_work(j), maximum_wall=30., operation=operation))
        if len(records) % 8 == 0:
            print(f"umbrella reference pilot {len(records)}/{len(grid)}", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records),
              "work": journal.accounting("U")[0], "units": len(grid), "p1_qualified": False,
              "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-U.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-U.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary and final I/O wall cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["anchor"] != ANCHOR
            or hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != config["anchor_sha256"]
            or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()):
        raise ValueError("changed source/anchor/input")
    metadata, rows = inputs(config["input"])
    log_weights = weights_from_anchor(json.loads((ROOT/ANCHOR).read_text()), rows)
    if log_weights != config["log_mixture_weights"]:
        raise ValueError("changed fixed umbrella weights")
    grid = jobs(rows)
    if manifest["expected_units"] != [identity(j) for j in grid]:
        raise ValueError("changed completion grid")
    cells = prepare(metadata, rows, log_weights)
    journal = RecoveryJournal(path.parent, manifest)
    records, replayed = [], 0
    for j in grid:
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed unit")
        record = finish["result"]
        if (any(record[k] != v for k, v in j.items()) or record["seed"] != seed(identity(j))
                or record["actual_work"] != expected_work(j)
                or record["q_digests"] != cells[j["cell"]]["q_digests"]
                or record["log_mixture_weights"] != log_weights[j["cell"]]
                or record["guide_independent_kernel"] != (j["method"] != "envelope")):
            raise ValueError("changed unit binding")
        vector = torch.tensor(record["log_scaled_component_normalizers"], dtype=torch.float64)
        w = torch.tensor(log_weights[j["cell"]], dtype=torch.float64)
        if abs(float(torch.logsumexp(vector+w, 0))-record["log_normalizer"]) > 1e-10:
            raise ValueError("unnormalized family measure identity failed")
        correction = torch.tensor([0.]+[math.log(d) for d in cells[j["cell"]]["deltas"]], dtype=torch.float64)
        if (vector-correction).tolist() != record["log_estimand_values"]:
            raise ValueError("incorrect scaled-risk correction")
        records.append(record)
        if j["rep"] == 0:
            replay = unit_values(j, cells)
            for k in ("log_normalizer", "log_scaled_component_normalizers", "log_estimand_values", "stages"):
                if replay[k] != record[k]:
                    raise ValueError("whole-run replay mismatch")
            replayed += 1
    if (summarize(records) != output["summaries"] or journal.accounting("U")[0] != output["work"]
            or output["work"] != 128+sum(expected_work(j) for j in grid) or output["units"] != len(grid)
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("changed summary/accounting/flags")
    if (path.parent/"phase-U.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_anchor_q_seed_identity_arithmetic_whole_replay_pass_not_precision_certificate",
            "units": len(grid), "replayed_wholes": replayed,
            "audit_potential_calls": 128+sum(expected_work(j) for j in grid if j["rep"] == 0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "phase_wall", "p1_qualified", "units")}))
