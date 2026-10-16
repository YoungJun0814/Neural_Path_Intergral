"""Bounded risk schedule controls and independently calibrated proposal geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_bank_risk_diagnostics import smc_config, smc_work
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from src.path_integral.conditional_second_moment import (
    log_risk_potential,
    summarize_risk_replicates,
)
from src.path_integral.path_geometry_diagnostics import calibrate_path_geometry
from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import estimate_weighted_tempered_normalizer

ROOT = Path(__file__).resolve().parents[1]


def run(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(config["torch_threads"])
    content = (ROOT/config["proposal_artifact_path"]).read_bytes()
    original = json.loads(content)
    ledger = SeedLedger()
    protocol = "r2-risk-geometry-"+canonical_digest(config)[:16]
    started, work = time.perf_counter(), 0
    records = []
    candidates = config["candidates"]
    if len(candidates) != 3 or len({c["id"] for c in candidates}) != 3:
        raise ValueError("three distinct preregistered risk controls required")
    jobs = original["records"]
    for old in jobs:
        base_start, base_work = time.perf_counter(), work
        cell, parent_rep = old["cell"], old["parent_training_rep"]
        item = next(x for x in old["candidates"] if x["method"] == config["method"])
        if canonical_digest(item["proposal_parameters"]) != item["proposal_digest"]:
            raise ValueError("proposal binding mismatch")
        proposal = proposal_from_parameters(item["proposal_parameters"])
        problem = _problem(original["config"]["model"], cell, original["config"]["smc"]["steps"])
        row: dict[str, Any] = {"cell": cell, "parent_training_rep": parent_rep,
            "proposal_digest": item["proposal_digest"], "candidates": [], "failure_reasons": []}
        records.append(row)

        def seed(role: str, regime: str, level: int, stream: str,
                 task: str = problem.task_id, rep: int = parent_rep) -> int:
            return ledger.allocate(SeedKey(protocol, role, regime, task, level, rep, stream))

        def check(n: int) -> None:
            if work+n > config["budget"]["max_potential_evaluations"]:
                raise TimeoutError("unresolved_potential_budget")
            if time.perf_counter()-started >= config["budget"]["max_wall_seconds"]:
                raise TimeoutError("unresolved_wall_budget")

        def payoff(x: torch.Tensor, fixed_problem: Any = problem) -> torch.Tensor:
            nonlocal work
            check(len(x))
            work += len(x)
            return _log_potential(fixed_problem, x)

        try:
            geom_spec = config["geometry"]
            training = proposal.sample(geom_spec["training_count"],
                path_seed=seed("geometry-training", "shared", 0, "path"),
                label_seed=seed("geometry-training", "shared", 0, "label")).samples
            calibration = proposal.sample(geom_spec["calibration_count"],
                path_seed=seed("geometry-calibration", "shared", 0, "path"),
                label_seed=seed("geometry-calibration", "shared", 0, "label")).samples
            geometry = calibrate_path_geometry(training, calibration,
                torch.stack([c.mean for c in proposal.components]), rank=geom_spec["rank"],
                outside_probability=geom_spec["outside_probability"])
            row["geometry"] = {k: v.tolist() if isinstance(v, torch.Tensor) else v
                                for k, v in asdict(geometry).items()}
            row["geometry_role"] = "surrogate_geometry_from_frozen_proposal_not_original_smc_bank"
            for candidate in candidates:
                stage_start, stage_work = time.perf_counter(), work
                result: dict[str, Any] = {"id": candidate["id"], "risk_replicates": [], "failure_reasons": []}
                row["candidates"].append(result)
                logs, flags = [], []
                try:
                    direct = config["direct"]
                    for batch in range(math.ceil(direct["count"]/direct["batch_size"])):
                        n = min(direct["batch_size"], direct["count"]-batch*direct["batch_size"])
                        check(n)
                        draw = proposal.sample(n, path_seed=seed("direct-final", candidate["id"], batch, "path"),
                            label_seed=seed("direct-final", candidate["id"], batch, "label"))
                        logs.append(payoff(draw.samples)+draw.log_p_over_q)
                        flags.append(geometry.outside(draw.samples))
                    values, outside = torch.cat(logs), torch.cat(flags)
                    result["direct"] = asdict(summarize_log_contributions(values))
                    result["direct_m2"] = asdict(summarize_log_contributions(2*values))
                    result["outside_direct"] = {"count": int(outside.sum()), "fraction": float(outside.double().mean()),
                        "contribution_share": float(torch.exp(torch.logsumexp(values[outside], 0)-torch.logsumexp(values, 0))),
                        "second_moment_share": float(torch.exp(torch.logsumexp(2*values[outside], 0)-torch.logsumexp(2*values, 0)))}
                    spec = {**config["risk"], **candidate}

                    def risk(x: torch.Tensor, fixed_q: Any = proposal) -> torch.Tensor:
                        return log_risk_potential(payoff(x), fixed_q.log_q_over_p(x), defensive_mass=fixed_q.defensive_mass)

                    for rep in range(spec["replicates"]):
                        check(smc_work(spec, spec["particles"]))
                        smc = estimate_weighted_tempered_normalizer(risk, dimension=problem.local_dimension,
                            config=smc_config(spec, seed=seed("independent-risk", candidate["id"], rep, "whole"), retain=True))
                        if smc.final_particles is None or smc.final_weights is None:
                            raise RuntimeError("missing terminal risk geometry")
                        selected = geometry.outside(smc.final_particles)
                        fraction = float(smc.final_weights[selected].sum())
                        log_z = float(smc.log_replicate_estimates[0])
                        result["risk_replicates"].append({"replicate": rep, "log_normalizer": log_z,
                            "outside_weight_fraction": fraction,
                            "outside_log_m2": log_z-math.log(proposal.defensive_mass)+math.log(fraction) if fraction > 0 else None,
                            "potential_evaluations": smc.potential_evaluations,
                            "unique_initial_ancestors": smc.replicate_diagnostics[0]["final_unique_initial_ancestors"]})
                    result["status"] = "completed_development"
                except (ValueError, FloatingPointError, RuntimeError, TimeoutError) as error:
                    result["status"] = "unresolved"
                    result["failure_reasons"].append(f"{type(error).__name__}: {error}")
                z = torch.tensor([r["log_normalizer"] for r in result["risk_replicates"]], dtype=torch.float64)
                result["risk"] = summarize_risk_replicates(z, defensive_mass=proposal.defensive_mass,
                    expected_replicates=config["risk"]["replicates"], maximum_relative_se=config["risk"]["maximum_relative_se"])
                if len(z) >= 2:
                    out = torch.tensor([r["outside_log_m2"] if r["outside_log_m2"] is not None else -math.inf
                                        for r in result["risk_replicates"]], dtype=torch.float64)
                    outside_summary = summarize_log_contributions(out)
                    result["outside_risk_m2"] = asdict(outside_summary)
                    result["outside_risk_m2_share"] = math.exp(outside_summary.log_mean-result["risk"]["log_mean"]) if outside_summary.log_mean is not None else 0.
                if "direct_m2" in result and result["risk"]["mean"] is not None:
                    result["risk_over_direct_m2"] = math.exp(result["risk"]["log_mean"]-result["direct_m2"]["log_mean"])
                result["potential_evaluations"] = work-stage_work
                result["wall_seconds_including_failure"] = time.perf_counter()-stage_start
                print(json.dumps({"finished": [cell["id"], parent_rep, candidate["id"]],
                                  "risk_status": result["risk"]["status"]}), flush=True)
            row["status"] = "completed_development"
        except (ValueError, FloatingPointError, RuntimeError, TimeoutError) as error:
            row["status"] = "unresolved"
            row["failure_reasons"].append(f"{type(error).__name__}: {error}")
        row["potential_evaluations"] = work-base_work
        row["wall_seconds_including_failure"] = time.perf_counter()-base_start
    if source_tree_digest(ROOT) != source["source_tree_digest"]:
        raise RuntimeError("source changed during experiment")
    decisions = []
    for cell_id in sorted({r["cell"]["id"] for r in records}):
        cell_rows = [r for r in records if r["cell"]["id"] == cell_id]
        passes = {c["id"]: sum(any(x["id"] == c["id"] and x["risk"]["status"] == "development_precision_pass_not_oracle"
                                 for x in r["candidates"]) for r in cell_rows) for c in candidates}
        qualified_designs = [k for k, v in passes.items() if v >= config["decision"]["required_precision_passes_per_cell"]]
        pair_consistent = len(qualified_designs) >= config["decision"]["minimum_precision_designs_for_crosscheck"]
        if pair_consistent:
            for r in cell_rows:
                means = [x["risk"]["mean"] for x in r["candidates"] if x["id"] in qualified_designs]
                if max(means)/min(means)-1 > config["decision"]["maximum_relative_normalizer_difference"]:
                    pair_consistent = False
        decisions.append({"cell": cell_id, "precision_passes": passes, "cross_design_stability": pair_consistent,
                          "correction_experiment_ready": pair_consistent})
    return {"schema": "npi.post-audit.r2-risk-geometry.v1", "source": source, "config": config,
        "proposal_artifact_sha256": hashlib.sha256(content).hexdigest(), "seed_ledger": ledger.to_dict(),
        "records": records, "decisions": decisions, "potential_evaluations": work,
        "physical_wall_seconds": time.perf_counter()-started, "performance_claim_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/post_audit/r2_risk_geometry_stability_v1.yaml")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(json.loads(args.audit.read_text(encoding="utf-8"))), allow_nan=False))
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT/config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    source = freeze_source(output, config)
    payload = run(config, source)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "decisions": payload["decisions"], "wall_seconds": payload["physical_wall_seconds"]}), flush=True)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    source, config = payload["source"], payload["config"]
    torch.set_num_threads(config["torch_threads"])
    if payload["schema"] != "npi.post-audit.r2-risk-geometry.v1" or canonical_digest(config) != source["config_digest"]:
        raise ValueError("schema/config binding mismatch")
    archive_path = ROOT/source["snapshot_path"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["snapshot_sha256"]:
        raise ValueError("snapshot mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        if json.loads(archive.read("RUN_CONFIG.json")) != config:
            raise ValueError("archive config mismatch")
        for name, sha in source["snapshot_file_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("source entry mismatch")
    original_bytes = (ROOT/config["proposal_artifact_path"]).read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != payload["proposal_artifact_sha256"]:
        raise ValueError("proposal artifact changed")
    original = json.loads(original_bytes)
    expected = {(r["cell"]["id"], r["parent_training_rep"]): r for r in original["records"]}
    if len(payload["records"]) != len(expected):
        raise ValueError("missing parent diagnostic")
    identities = set()
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    for row in payload["records"]:
        key = (row["cell"]["id"], row["parent_training_rep"])
        if key not in expected or key in identities:
            raise ValueError("undeclared/duplicate parent")
        identities.add(key)
        item = next(x for x in expected[key]["candidates"] if x["method"] == config["method"])
        if item["proposal_digest"] != row["proposal_digest"]:
            raise ValueError("frozen proposal mismatch")
        q = proposal_from_parameters(item["proposal_parameters"])
        if "geometry" in row:
            problem = _problem(original["config"]["model"], row["cell"], original["config"]["smc"]["steps"])
            protocol = "r2-risk-geometry-"+canonical_digest(config)[:16]

            def geometry_seed(role: str, stream: str, task: str = problem.task_id,
                              rep: int = row["parent_training_rep"], namespace: str = protocol) -> int:
                return ledger.lookup(SeedKey(namespace, role, "shared", task, 0, rep, stream))

            gs = config["geometry"]
            train = q.sample(gs["training_count"], path_seed=geometry_seed("geometry-training", "path"),
                             label_seed=geometry_seed("geometry-training", "label")).samples
            cal = q.sample(gs["calibration_count"], path_seed=geometry_seed("geometry-calibration", "path"),
                           label_seed=geometry_seed("geometry-calibration", "label")).samples
            rebuilt = calibrate_path_geometry(train, cal, torch.stack([c.mean for c in q.components]),
                rank=gs["rank"], outside_probability=gs["outside_probability"])
            for name, value in asdict(rebuilt).items():
                saved = row["geometry"][name]
                if isinstance(value, torch.Tensor):
                    if not torch.allclose(value, torch.tensor(saved, dtype=torch.float64), rtol=1e-8, atol=1e-8):
                        raise ValueError("training/calibration geometry replay mismatch")
                elif not math.isclose(value, saved, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("calibration threshold replay mismatch")
        if row["status"] == "completed_development" and {c["id"] for c in row["candidates"]} != {c["id"] for c in config["candidates"]}:
            raise ValueError("missing schedule")
        for c in row["candidates"]:
            z = torch.tensor([r["log_normalizer"] for r in c["risk_replicates"]], dtype=torch.float64)
            recomputed = summarize_risk_replicates(z, defensive_mass=q.defensive_mass,
                expected_replicates=config["risk"]["replicates"], maximum_relative_se=config["risk"]["maximum_relative_se"])
            if canonical_digest(recomputed) != canonical_digest(c["risk"]):
                raise ValueError("risk arithmetic mismatch")
            if len(z) >= 2:
                logs = []
                for r in c["risk_replicates"]:
                    fraction = r["outside_weight_fraction"]
                    if not 0 <= fraction <= 1+1e-12:
                        raise ValueError("invalid outside weight fraction")
                    value = r["log_normalizer"]-math.log(q.defensive_mass)+math.log(fraction) if fraction else None
                    if value != r["outside_log_m2"]:
                        raise ValueError("restricted normalizer arithmetic mismatch")
                    logs.append(value if value is not None else -math.inf)
                outside = summarize_log_contributions(torch.tensor(logs, dtype=torch.float64))
                if canonical_digest(asdict(outside)) != canonical_digest(c["outside_risk_m2"]):
                    raise ValueError("outside replicate summary mismatch")
                share = math.exp(outside.log_mean-recomputed["log_mean"]) if outside.log_mean is not None else 0.
                if not math.isclose(share, c["outside_risk_m2_share"], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("outside ratio mismatch")
            if c["status"] == "completed_development":
                planned = config["direct"]["count"]+config["risk"]["replicates"]*smc_work(config["risk"], config["risk"]["particles"])
                if c["potential_evaluations"] != planned:
                    raise ValueError("stage work mismatch")
        if sum(c["potential_evaluations"] for c in row["candidates"]) != row["potential_evaluations"]:
            raise ValueError("parent work mismatch")
    if sum(r["potential_evaluations"] for r in payload["records"]) != payload["potential_evaluations"]:
        raise ValueError("work total mismatch")
    if payload["potential_evaluations"] > config["budget"]["max_potential_evaluations"] or payload["performance_claim_authorized"]:
        raise ValueError("budget/performance violation")
    # Recompute the conservative decision independently from the stored decision.
    if ({d["cell"] for d in payload["decisions"]} != {r["cell"]["id"] for r in payload["records"]}
            or len(payload["decisions"]) != len({d["cell"] for d in payload["decisions"]})):
        raise ValueError("missing/duplicate cell decision")
    for decision in payload["decisions"]:
        rows = [r for r in payload["records"] if r["cell"]["id"] == decision["cell"]]
        passes = {d["id"]: sum(any(c["id"] == d["id"] and c["risk"]["status"] == "development_precision_pass_not_oracle"
                    for c in r["candidates"]) for r in rows) for d in config["candidates"]}
        stable = [k for k, v in passes.items() if v >= config["decision"]["required_precision_passes_per_cell"]]
        ready = len(stable) >= config["decision"]["minimum_precision_designs_for_crosscheck"]
        if ready:
            for r in rows:
                means = [c["risk"]["mean"] for c in r["candidates"] if c["id"] in stable]
                if max(means)/min(means)-1 > config["decision"]["maximum_relative_normalizer_difference"]:
                    ready = False
        if passes != decision["precision_passes"] or ready != decision["correction_experiment_ready"] or ready != decision["cross_design_stability"]:
            raise ValueError("stale correction readiness")
    return {"status": "source_and_arithmetic_pass_not_statistical_certificate", "parents": len(identities),
            "seed_streams": len(ledger.records)}


if __name__ == "__main__":
    main()
