"""Plan S2: three matched-work banks and independent frozen-proposal risk SMC.

Risk normalizers are not trained proposals or reference probabilities. Terminal
particles are never used as independent units for a second-moment SE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zipfile
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from experiments.post_audit_r15_reference_design import temperatures
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.conditional_second_moment import (
    log_risk_potential,
    summarize_risk_replicates,
)
from src.path_integral.finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture
from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_bank_mixture import (
    fit_weighted_bank_mixture,
    proposal_from_parameters,
)
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def smc_config(spec: dict[str, Any], *, seed: int, particles: int | None = None,
               retain: bool = False) -> WeightedSMCConfig:
    return WeightedSMCConfig(
        particles=int(spec["particles"] if particles is None else particles),
        temperatures=temperatures(int(spec["levels"]), int(spec["bridge_power"])),
        mutation_steps=int(spec["mutation_steps"]), pcn_scale=float(spec["pcn_scale"]),
        replicates=1, seed=seed, resample_every=int(spec["resample_every"]),
        resampling_scheme=cast(Literal["multinomial", "stratified"], spec["resampling_scheme"]),
        retain_final_particles=retain, independence_every=int(spec.get("independence_every", 0)))


def smc_work(spec: dict[str, Any], particles: int) -> int:
    return particles * (1 + (int(spec["levels"]) - 1) * int(spec["mutation_steps"]))


def accuracy(summary: dict[str, Any], ref: dict[str, Any], spec: dict[str, Any],
             *, complete: bool) -> dict[str, Any]:
    reasons = []
    if not complete or summary.get("log_mean") is None or summary.get("relative_se") is None:
        return {"qualified": False, "failure_reasons": ["incomplete_or_zero_direct"]}
    mean = math.exp(summary["log_mean"])
    se = mean * summary["relative_se"]
    upper = abs(mean-ref["mean"]) + spec["confidence_z"] * math.hypot(se, ref["standard_error"])
    if summary["relative_se"] > spec["target_relative_se"]:
        reasons.append("insufficient_direct_precision")
    if ref["standard_error"] > spec["reference_se_fraction_of_method_se"] * se:
        reasons.append("unresolved_reference_precision")
    if upper > spec["relative_equivalence_margin"] * ref["mean"]:
        reasons.append("equivalence_not_established")
    return {"mean": mean, "standard_error": se, "equivalence_upper_difference": upper,
            "qualified": not reasons, "failure_reasons": reasons}


def run(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    ledger = SeedLedger()
    protocol = "post-audit-r2-bank-risk-" + canonical_digest(config)[:16]
    started = time.perf_counter()
    evaluations = 0
    records: list[dict[str, Any]] = []
    reference_bytes = (ROOT / config["reference_path"]).read_bytes()
    references = json.loads(reference_bytes)
    jobs: list[dict[str, Any]] = []
    artifact_sha: str | None = None
    if config["mode"] == "bank_islands":
        candidates = config["bank_candidates"]
        if len(candidates) != 3 or len({x["id"] for x in candidates}) != 3:
            raise ValueError("exactly three distinct preregistered island designs required")
        work = {smc_work(config["smc"], int(c["particles"])) * c["islands"] for c in candidates}
        if len(work) != 1:
            raise ValueError("bank potential budgets are not equal")
        for cell in config["cells"]:
            for rep in range(config["parent_training_replicates"]):
                for design in candidates:
                    jobs.append({"cell": cell, "rep": rep, "method": design["id"], "design": design,
                                 "model": config["model"], "steps": config["smc"]["steps"]})
    elif config["mode"] == "frozen_proposals":
        artifact_bytes = (ROOT / config["proposal_artifact_path"]).read_bytes()
        artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
        artifact = json.loads(artifact_bytes)
        if len(config["methods"]) != len(set(config["methods"])):
            raise ValueError("duplicate frozen method")
        for old in artifact["records"]:
            for method in config["methods"]:
                item = next(x for x in old["candidates"] if x["method"] == method)
                if canonical_digest(item["proposal_parameters"]) != item["proposal_digest"]:
                    raise ValueError("frozen proposal binding mismatch")
                jobs.append({"cell": old["cell"], "rep": old["parent_training_rep"], "method": method,
                             "parameters": item["proposal_parameters"], "model": artifact["config"]["model"],
                             "steps": artifact["config"]["smc"]["steps"]})
    else:
        raise ValueError("unsupported diagnostic mode")

    def check_budget(n: int) -> None:
        if evaluations+n > config["budget"]["max_potential_evaluations"]:
            raise TimeoutError("unresolved_potential_budget")
        if time.perf_counter()-started >= config["budget"]["max_wall_seconds"]:
            raise TimeoutError("unresolved_wall_budget")

    def counted_payoff(problem: Any, samples: torch.Tensor) -> torch.Tensor:
        # Account even for evaluations that fail. Checking inside each SMC call
        # also permits an honest wall-budget interruption of a whole replicate.
        nonlocal evaluations
        check_budget(samples.shape[0])
        evaluations += samples.shape[0]
        return _log_potential(problem, samples)

    for job in jobs:
        row_start, row_work = time.perf_counter(), evaluations
        problem = _problem(job["model"], job["cell"], job["steps"])
        ref = next(x["new_reference"] for x in references["cells"] if x["cell"]["id"] == job["cell"]["id"])
        row: dict[str, Any] = {"cell": job["cell"], "parent_training_rep": job["rep"],
                               "method": job["method"], "task_id": problem.task_id,
                               "role": config["role"], "failure_reasons": [], "costs": {},
                               "bank_diagnostics": [], "direct_batches": [], "risk_replicates": [],
                               "model": job["model"], "steps": job["steps"],
                               "reference": ref, "qualified": False}
        records.append(row)

        def seed(role: str, stream: str, level: int = 0,
                 task: str = problem.task_id, rep: int = job["rep"], method: str = job["method"]) -> int:
            return ledger.allocate(SeedKey(protocol, role, method, task, level, rep, stream))

        active_stage, stage_start = "offline", time.perf_counter()
        try:
            if config["mode"] == "bank_islands":
                design = job["design"]
                samples, weights = [], []
                for island in range(design["islands"]):
                    count = int(design["particles"])
                    check_budget(smc_work(config["smc"], count))
                    bank = estimate_weighted_tempered_normalizer(
                        partial(counted_payoff, problem), dimension=problem.local_dimension,
                        config=smc_config(config["smc"], seed=seed("parent-training", "island", island),
                                          particles=count, retain=True))
                    if bank.final_particles is None or bank.final_weights is None:
                        raise RuntimeError("missing bank")
                    samples.append(bank.final_particles)
                    weights.append(bank.final_weights / design["islands"])
                    row["bank_diagnostics"].append({"island": island,
                                                     "potential_evaluations": bank.potential_evaluations,
                                                     **bank.replicate_diagnostics[0]})
                fit = fit_weighted_bank_mixture(torch.cat(samples), torch.cat(weights), **config["mixture"])
                proposal = fit.proposal
                row["bank_summary"] = {
                    "particle_count": fit.bank_count, "weighted_particle_ess": fit.weighted_bank_ess,
                    "unique_initial_ancestors_total": sum(int(x["final_unique_initial_ancestors"])
                                                          for x in row["bank_diagnostics"]),
                    "training_potential_evaluations": sum(x["potential_evaluations"] for x in row["bank_diagnostics"]),
                    "learned_components": len(proposal.components)-1,
                    "cluster_masses": fit.cluster_masses}
            else:
                proposal = proposal_from_parameters(job["parameters"])
            row["proposal_parameters"] = proposal_parameters(proposal)
            row["proposal_digest"] = canonical_digest(row["proposal_parameters"])
            row["costs"][active_stage] = time.perf_counter()-stage_start
            active_stage, stage_start = "bank_diagnostic", time.perf_counter()
            n = int(config["evaluation"]["bank_diagnostic_count"])
            if n:
                check_budget(n)
                draw = proposal.sample(n, path_seed=seed("iid-bank-diagnostic", "path"),
                                       label_seed=seed("iid-bank-diagnostic", "label"))
                log_g = counted_payoff(problem, draw.samples)
                normalized = torch.softmax(log_g+draw.log_p_over_q, 0)
                row["iid_bank_diagnostic"] = {"count": n, "target_weight_ess": float(1/normalized.square().sum()),
                                               "maximum_target_weight": float(normalized.max()), "used_for_fit": False}
            row["costs"][active_stage] = time.perf_counter()-stage_start
            active_stage, stage_start = "direct_inference", time.perf_counter()
            values = []
            spec = config["evaluation"]
            for batch in range(math.ceil(spec["direct_count"]/spec["batch_size"])):
                n = min(spec["batch_size"], spec["direct_count"]-batch*spec["batch_size"])
                check_budget(n)
                draw = proposal.sample(n, path_seed=seed("direct-final", "path", batch),
                                       label_seed=seed("direct-final", "label", batch))
                logs = counted_payoff(problem, draw.samples)+draw.log_p_over_q
                values.append(logs)
                row["direct_batches"].append({"batch": batch, "contribution": asdict(summarize_log_contributions(logs)),
                                                "second_moment": asdict(summarize_log_contributions(2*logs))})
            all_logs = torch.cat(values)
            row["direct"] = asdict(summarize_log_contributions(all_logs))
            row["direct_second_moment"] = asdict(summarize_log_contributions(2*all_logs))
            ordered = torch.sort(all_logs, descending=True).values
            n_top = max(1, math.ceil(.01*ordered.numel()))
            row["direct_tail"] = {"top_one_percent_contribution_share": float(torch.exp(
                torch.logsumexp(ordered[:n_top], 0)-torch.logsumexp(ordered, 0))),
                "top_one_percent_second_moment_share": float(torch.exp(
                torch.logsumexp(2*ordered[:n_top], 0)-torch.logsumexp(2*ordered, 0)))}
            row["accuracy"] = accuracy(row["direct"], ref, config["qualification"], complete=True)
            row["qualified"] = row["accuracy"]["qualified"]
            row["costs"][active_stage] = time.perf_counter()-stage_start
            active_stage, stage_start = "independent_risk", time.perf_counter()

            def risk_potential(x: torch.Tensor, frozen_q: DefensiveFiniteRankGaussianMixture = proposal,
                               frozen_problem: Any = problem) -> torch.Tensor:
                return log_risk_potential(counted_payoff(frozen_problem, x), frozen_q.log_q_over_p(x),
                                          defensive_mass=frozen_q.defensive_mass)

            risk_spec = config["risk"]
            for rep in range(risk_spec["replicates"]):
                check_budget(smc_work(risk_spec, int(risk_spec["particles"])))
                result = estimate_weighted_tempered_normalizer(
                    risk_potential, dimension=problem.local_dimension,
                    config=smc_config(risk_spec, seed=seed("independent-risk", "whole-smc", rep)))
                row["risk_replicates"].append({"replicate": rep,
                    "log_normalizer": float(result.log_replicate_estimates[0]),
                    "potential_evaluations": result.potential_evaluations,
                    "diagnostic": result.replicate_diagnostics[0]})
            row["status"] = "completed_development_diagnostic"
        except (ValueError, FloatingPointError, RuntimeError, TimeoutError) as error:
            row["status"] = "unresolved"
            row["failure_reasons"].append(f"{type(error).__name__}: {error}")
            row["qualified"] = False
        finally:
            row["costs"][active_stage] = time.perf_counter()-stage_start
        if "proposal_parameters" in row:
            proposal = proposal_from_parameters(row["proposal_parameters"])
            log_z = torch.tensor([x["log_normalizer"] for x in row["risk_replicates"]], dtype=torch.float64)
            row["risk"] = summarize_risk_replicates(log_z, defensive_mass=proposal.defensive_mass,
                expected_replicates=config["risk"]["replicates"], maximum_relative_se=config["risk"]["maximum_relative_se"])
            if row["risk"]["status"] != "development_precision_pass_not_oracle":
                row["failure_reasons"].append(row["risk"]["status"])
            if row["risk"]["mean"] is not None and "direct_second_moment" in row:
                direct_m2 = row["direct_second_moment"]
                mean = math.exp(direct_m2["log_mean"])
                se = mean * direct_m2["relative_se"]
                denominator = math.hypot(se, row["risk"]["standard_error"])
                row["risk_crosscheck"] = {"direct_m2": mean, "direct_m2_standard_error": se,
                    "risk_m2_over_direct_m2": row["risk"]["mean"]/mean,
                    "difference_over_combined_se": (row["risk"]["mean"]-mean)/denominator if denominator else None,
                    "role": "exploratory_independent_mechanisms_not_equivalence_test",
                    "risk_m2_over_reference_mean_squared": row["risk"]["mean"]/ref["mean"]**2}
        row["potential_evaluations"] = evaluations-row_work
        row["physical_wall_seconds"] = time.perf_counter()-row_start
        row["recorded_stage_wall_seconds"] = sum(row["costs"].values())
        print(json.dumps({"finished": [job["cell"]["id"], job["rep"], job["method"]],
                          "status": row["status"], "risk_status": row.get("risk", {}).get("status")}), flush=True)
    if source_tree_digest(ROOT) != source["source_tree_digest"]:
        raise RuntimeError("runtime source changed during execution")
    return {"schema": "npi.post-audit.r2-bank-risk.v1", "source": source, "config": config,
            "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(), "proposal_artifact_sha256": artifact_sha,
            "records": records, "seed_ledger": ledger.to_dict(), "expected_jobs": len(jobs),
            "potential_evaluations": evaluations, "physical_wall_seconds": time.perf_counter()-started,
            "performance_claim_authorized": False,
            "limitations": ["Five independent fits are engineering diagnostics, not confirmation.",
                "Risk-SMC RSE is based on eight whole replicates, not a certified tail bound.",
                "Risk/reference ratio uses a noisy historical reference and is descriptive.",
                "Fixed direct sample count is not matched-precision total cost.",
                "Source snapshot, import and artifact I/O are excluded from measured stages."]}


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["schema"] != "npi.post-audit.r2-bank-risk.v1":
        raise ValueError("unsupported artifact")
    config, source = payload["config"], payload["source"]
    if canonical_digest(config) != source["config_digest"]:
        raise ValueError("config binding mismatch")
    archive_path = ROOT / source["snapshot_path"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["snapshot_sha256"]:
        raise ValueError("snapshot mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        for name, sha in source["snapshot_file_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("snapshot entry mismatch")
        if json.loads(archive.read("RUN_CONFIG.json")) != config:
            raise ValueError("archived config mismatch")
    reference_bytes = (ROOT/config["reference_path"]).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != payload["reference_sha256"]:
        raise ValueError("reference changed")
    references = json.loads(reference_bytes)
    expected_proposals = {}
    if config["mode"] == "frozen_proposals":
        original_bytes = (ROOT/config["proposal_artifact_path"]).read_bytes()
        if hashlib.sha256(original_bytes).hexdigest() != payload["proposal_artifact_sha256"]:
            raise ValueError("proposal source artifact changed")
        original = json.loads(original_bytes)
        expected_keys = set()
        for old in original["records"]:
            for method in config["methods"]:
                key = (old["cell"]["id"], old["parent_training_rep"], method)
                expected_keys.add(key)
                expected_proposals[key] = next(x["proposal_digest"] for x in old["candidates"] if x["method"] == method)
    else:
        expected_keys = {(cell["id"], rep, design["id"]) for cell in config["cells"]
                         for rep in range(config["parent_training_replicates"])
                         for design in config["bank_candidates"]}
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    if len(payload["records"]) != len(expected_keys) or payload["expected_jobs"] != len(expected_keys):
        raise ValueError("missing failed job")
    identities = set()
    for row in payload["records"]:
        key = (row["cell"]["id"], row["parent_training_rep"], row["method"])
        if key in identities:
            raise ValueError("duplicate job")
        if key not in expected_keys:
            raise ValueError("undeclared job")
        identities.add(key)
        ref = next(x["new_reference"] for x in references["cells"] if x["cell"]["id"] == row["cell"]["id"])
        if canonical_digest(ref) != canonical_digest(row["reference"]):
            raise ValueError("row reference binding mismatch")
        if "proposal_parameters" not in row:
            if row["status"] != "unresolved" or not row["failure_reasons"]:
                raise ValueError("missing proposal without failed run")
            continue
        if canonical_digest(row["proposal_parameters"]) != row["proposal_digest"]:
            raise ValueError("proposal binding mismatch")
        if expected_proposals and expected_proposals[key] != row["proposal_digest"]:
            raise ValueError("frozen proposal changed")
        proposal = proposal_from_parameters(row["proposal_parameters"])
        if proposal.defensive_mass < .1-1e-12:
            raise ValueError("lost defensive protection")
        z = torch.tensor([x["log_normalizer"] for x in row["risk_replicates"]], dtype=torch.float64)
        recomputed = summarize_risk_replicates(z, defensive_mass=proposal.defensive_mass,
            expected_replicates=config["risk"]["replicates"], maximum_relative_se=config["risk"]["maximum_relative_se"])
        if canonical_digest(recomputed) != canonical_digest(row["risk"]):
            raise ValueError("whole-replicate risk arithmetic mismatch")
        for field, batch_field in [("direct", "contribution"), ("direct_second_moment", "second_moment")]:
            if field not in row:
                continue
            parts = [x[batch_field] for x in row["direct_batches"]]
            n = sum(x["count"] for x in parts)
            log_mean = float(torch.logsumexp(torch.tensor([x["log_mean"]+math.log(x["count"]) for x in parts], dtype=torch.float64), 0)-math.log(n))
            log_second = float(torch.logsumexp(torch.tensor([x["log_second_moment"]+math.log(x["count"]) for x in parts], dtype=torch.float64), 0)-math.log(n))
            rse = math.sqrt(max(0., math.expm1(log_second-2*log_mean))/(n-1))
            if (row[field]["count"] != n or not math.isclose(row[field]["log_mean"], log_mean, abs_tol=1e-12)
                    or not math.isclose(row[field]["relative_se"], rse, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("direct moment mismatch")
        if "accuracy" in row:
            expected = accuracy(row["direct"], row["reference"], config["qualification"],
                                complete=row["direct"]["count"] == config["evaluation"]["direct_count"])
            if canonical_digest(expected) != canonical_digest(row["accuracy"]):
                raise ValueError("accuracy mismatch")
        if row["qualified"] and (row["status"] != "completed_development_diagnostic" or not row["accuracy"]["qualified"]):
            raise ValueError("stale qualification")
        if not math.isclose(sum(row["costs"].values()), row["recorded_stage_wall_seconds"], abs_tol=1e-12):
            raise ValueError("failed cost missing")
        if row["status"] == "completed_development_diagnostic":
            expected_work = sum(x["potential_evaluations"] for x in row["bank_diagnostics"])
            expected_work += row.get("iid_bank_diagnostic", {}).get("count", 0)
            expected_work += sum(x["contribution"]["count"] for x in row["direct_batches"])
            expected_work += sum(x["potential_evaluations"] for x in row["risk_replicates"])
            if expected_work != row["potential_evaluations"]:
                raise ValueError("stage work mismatch")
    if sum(x["potential_evaluations"] for x in payload["records"]) != payload["potential_evaluations"]:
        raise ValueError("work mismatch")
    if payload["potential_evaluations"] > config["budget"]["max_potential_evaluations"] or payload["performance_claim_authorized"]:
        raise ValueError("budget or performance gate violation")
    return {"status": "integrity_arithmetic_pass_not_statistical_validation", "jobs": len(identities),
            "seed_streams": len(ledger.records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/post_audit/r2_bank_islands_risk_v1.yaml")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(json.loads(args.audit.read_text(encoding="utf-8"))), allow_nan=False))
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT/config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = freeze_source(output, config)
    payload = run(config, source)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "potential_evaluations": payload["potential_evaluations"],
                      "wall_seconds": payload["physical_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
