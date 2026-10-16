"""Bounded conditional/nested and guide-free kernel references; no model fitting.

Modes are micro, kernel, block and sealed production. Pilot data are never
pooled into production; any failed whole run invalidates the experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import audit as legacy_audit
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.gaussian_mixture_marginal import ShiftMixtureMarginal, require_shift_mixture
from src.path_integral.nested_reference_statistics import nested_variance_diagnostic
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.structural_v2_conditional_reference import (
    block_log_inner_risk,
    last_pair_cache,
    nested_log_means,
)
from src.path_integral.structural_v2_contract import MeasurementContract, audit_sample_uses
from src.path_integral.structural_v2_inventory import audit_snapshot, file_sha256
from src.path_integral.structural_v2_reference import (
    log_moments,
    merge_moments,
    precision_count,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.structural-v2.reference-redesign.v1"
PLAN = "docs/plans/STRUCTURAL_V2_REFERENCE_METHOD_REDESIGN_2026-10-08_KO.md"


def validate_config(config: dict[str, Any]) -> None:
    if (config["schema"] != "npi.structural-v2.reference-redesign-config.v1"
            or config["mode"] not in ("micro", "kernel", "block", "production")
            or config["torch_threads"] != 1 or isinstance(config["torch_threads"], bool)
            or config["method"] != "parent-as-is" or config["pilot_parent_rep"] != 0):
        raise ValueError("invalid redesign mode/target")
    expected_outer_batch = 256 if config["mode"] == "block" else 512
    if config["inner_counts"] != [1, 4, 16, 64] or config["outer_batches"] != 4 or config["outer_batch_size"] != expected_outer_batch:
        raise ValueError("microstudy candidate grid changed")
    if config["whole_replicates"] != 8 or config["kernels"] != ["pcn", "elliptical_slice"]:
        raise ValueError("guide-free pilot contract changed")
    if config["fixed_block_fractions"] != [.25, .5, .75]:
        raise ValueError("fixed block positions changed")
    for name in ("max_potential_evaluations", "max_process_rss_bytes", "minimum_free_disk_bytes"):
        value = config["budget"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("invalid integer budget")
    if not math.isfinite(config["budget"]["max_wall_seconds"]) or config["budget"]["max_wall_seconds"] <= 0:
        raise ValueError("invalid wall budget")
    if config["mode"] == "production" and not config.get("production_jobs"):
        raise ValueError("production needs a sealed allocation")


def jobs_for(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if config["mode"] == "production":
        return config["production_jobs"]
    jobs = []
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        common = {"cell_id": row["cell"]["id"], "parent_training_rep": 0}
        if config["mode"] == "kernel":
            for kind in ("risk", "probability"):
                for kernel in config["kernels"]:
                    jobs.append({**common, "estimand": kind, "method": kernel,
                                 "count": config["whole_replicates"], "inner": None, "block": None})
        else:
            blocks = [None] if config["mode"] == "micro" else config["fixed_block_fractions"]
            for block in blocks:
                jobs.append({**common, "estimand": "risk", "method": "nested",
                             "count": config["outer_batches"] * config["outer_batch_size"],
                             "inner": None, "block": block})
            if config["mode"] == "micro":
                jobs.append({**common, "estimand": "probability", "method": "conditional",
                             "count": config["outer_batches"] * config["outer_batch_size"],
                             "inner": 1, "block": None})
    return jobs


def experiment_identity(job: dict[str, Any]) -> str:
    return f"{job['cell_id']}/{job['parent_training_rep']}/{job['estimand']}/{job['method']}/{job['block']}"


def run(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    verify_production(config)
    torch.set_num_threads(1)
    metadata, rows = inputs(config)
    for row in rows:
        require_shift_mixture(proposal_from_parameters(row["q_parameters"]))
    started = time.perf_counter()
    process, ledger = psutil.Process(), SeedLedger()
    uses: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    counters = {"potential_equivalent_evaluations": 0, "fft_path_evaluations": 0,
                "cdf_evaluations": 0, "density_component_evaluations": 0,
                "guide_probe_path_steps": 0}
    peak = process.memory_info().rss
    budget = config["budget"]
    if shutil.disk_usage(ROOT).free < budget["minimum_free_disk_bytes"]:
        raise RuntimeError("insufficient disk")

    def charge(*, paths: int = 0, cdfs: int = 0, densities: int = 0) -> None:
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if (counters["potential_equivalent_evaluations"] + paths + cdfs > budget["max_potential_evaluations"]
                or time.perf_counter() - started > budget["max_wall_seconds"]
                or peak > min(budget["max_process_rss_bytes"], psutil.virtual_memory().total // 4)):
            raise TimeoutError("redesign budget exhausted; no selective run replacement")
        counters["potential_equivalent_evaluations"] += paths + cdfs
        counters["fft_path_evaluations"] += paths
        counters["cdf_evaluations"] += cdfs
        counters["density_component_evaluations"] += densities

    aborted = False
    for job in jobs_for(config, rows):
        before, timer = counters.copy(), time.perf_counter()
        identity = experiment_identity(job)
        row = next(r for r in rows if r["cell"]["id"] == job["cell_id"]
                   and r["parent_training_rep"] == job["parent_training_rep"])
        problem = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
        q = proposal_from_parameters(row["q_parameters"])
        risk = job["estimand"] == "risk"
        smc_job = job["method"] in ("pcn", "elliptical_slice")
        static_job = job["method"] == "static-iid"
        target_digest = canonical_digest({"model": metadata["config"]["model"], "cell": row["cell"], "steps": problem.steps})
        result: dict[str, Any] = {**job, "identity": identity, "q_digest": row["q_digest"] if risk else None,
                                  "target_digest": target_digest, "status": "not_run_after_protocol_failure" if aborted else "running",
                                  "failure_reasons": [], "whole_runs": [], "candidates": []}
        records.append(result)
        if aborted:
            result.update(counters={key: 0 for key in counters}, wall_seconds=0.)
            continue
        def seed(stream: str, level: int, role: str | None = None,
                 fixed_job: dict[str, Any] = job, fixed_problem: Any = problem,
                 consumer: str = identity, fixed_smc_job: bool = smc_job) -> int:
            use_role = role or ("allocation-pilot" if config["mode"] != "production" else
                                "reference-whole-smc" if fixed_smc_job else "reference-iid")
            key = SeedKey("reference-redesign-" + canonical_digest(config)[:16], use_role,
                          fixed_job["estimand"] + "-" + fixed_job["method"] + "-" + str(fixed_job["block"]),
                          fixed_problem.task_id, level, fixed_job["parent_training_rep"], stream)
            value = ledger.allocate(key)
            uses.append({"use_id": f"{consumer}/{level}/{stream}", "consumer_id": consumer,
                         "role": use_role, "seed_key": asdict(key), "pair_group": None})
            return value

        try:
            guide = None
            if not smc_job:
                charge(paths=problem.local_dimension)
                counters["guide_probe_path_steps"] += problem.local_dimension * problem.steps
                guide = build_volterra_excursion_guide(problem, **config["guide"])
                require_shift_mixture(guide)
            guide_params = proposal_parameters(guide) if guide is not None else None
            guide_digest = canonical_digest(guide_params) if guide_params is not None else None
            result.update(guide_parameters=guide_params, guide_digest=guide_digest)
            estimator = ("smc_normalizer" if smc_job else "auxiliary_is" if static_job and risk else
                         "ordinary_is" if static_job else "nested_auxiliary_is" if risk else "conditional_mean_is")
            contract = MeasurementContract(identity, target_digest, problem.steps,
                "raw_event_second_moment" if risk else "event_probability", estimator,
                "reference-whole-smc" if smc_job else "reference-iid",
                "whole_smc_run" if smc_job else "iid_draw" if static_job else "iid_outer_draw",
                row["q_digest"] if risk else guide_digest,
                guide_digest if risk and not smc_job else None, parent_training_rep=job["parent_training_rep"], reference_rep=0)
            result["measurement_contract"] = contract.to_dict()
            if smc_job:
                def potential(z: torch.Tensor, fixed_q: Any = q, fixed_problem: Any = problem,
                              fixed_risk: bool = risk) -> torch.Tensor:
                    charge(paths=len(z), cdfs=len(z), densities=len(z) * len(fixed_q.components) if fixed_risk else 0)
                    logg = evaluate_rbergomi_conditional_terminal(fixed_problem, z).payoffs.log_left_probability
                    return log_risk_potential(logg, fixed_q.log_q_over_p(z), defensive_mass=fixed_q.defensive_mass) if fixed_risk else logg
                spec = config["smc"]
                for rep in range(job["count"]):
                    substart, calls_before = time.perf_counter(), counters["fft_path_evaluations"]
                    cfg = WeightedSMCConfig(spec["particles"],
                        tuple((i / (spec["levels"] - 1))**spec["bridge_power"] for i in range(spec["levels"])),
                        spec["mutation_steps"], spec["pcn_scale"], 1, seed("whole", rep),
                        resample_every=spec["resample_every"], resampling_scheme="stratified",
                        mutation_kernel=job["method"], slice_maximum_attempts=spec["slice_maximum_attempts"])
                    output = estimate_weighted_tempered_normalizer(potential, dimension=problem.local_dimension, config=cfg)
                    actual_calls = counters["fft_path_evaluations"] - calls_before
                    if output.potential_evaluations != actual_calls:
                        raise ValueError("SMC callback work mismatch")
                    logvalue = float(output.log_replicate_estimates[0]) - (math.log(q.defensive_mass) if risk else 0)
                    result["whole_runs"].append({"log_estimand_estimate": logvalue,
                        "potential_evaluations": actual_calls, "diagnostics": output.replicate_diagnostics[0],
                        "wall_seconds": time.perf_counter() - substart})
                logs = [r["log_estimand_estimate"] for r in result["whole_runs"]]
                result["summary"] = log_moments(torch.tensor(logs, dtype=torch.float64))
                result["log_unit_contributions"] = logs
                result["block_log_means"] = logs
            else:
                assert guide is not None
                block = problem.steps - 1 if job["block"] is None else min(problem.steps - 1, int(problem.steps * job["block"]))
                indices = tuple(i for i in range(problem.local_dimension) if i not in (2 * block, 2 * block + 1))
                marginal = ShiftMixtureMarginal.from_full(guide, indices)
                inner_counts = [job["inner"]] if config["mode"] == "production" else config["inner_counts"] if risk else [1]
                accum: dict[int, dict[str, Any]] = {inner_count: {"first": [], "second": [], "moments": [], "block_logs": [], "timing": 0.}
                                                   for inner_count in inner_counts}
                batch_count = config["outer_batch_size"] if config["mode"] != "production" else config["production_batch_size"]
                if job["count"] % batch_count:
                    raise ValueError("outer counts need complete equal batches")
                for batch in range(job["count"] // batch_count):
                    cache_start = time.perf_counter()
                    if static_job:
                        draw = guide.sample(batch_count, path_seed=seed("outer-path", batch), label_seed=seed("outer-label", batch))
                        charge(paths=batch_count, cdfs=batch_count,
                               densities=batch_count * (len(guide.components) + (len(q.components) if risk else 0)))
                        logg = evaluate_rbergomi_conditional_terminal(problem, draw.samples).payoffs.log_left_probability
                        static_values = 2 * logg - q.log_q_over_p(draw.samples) - draw.log_q_over_p if risk else logg - draw.log_q_over_p
                        accum[inner_counts[0]]["moments"].append(log_moments(static_values))
                        accum[inner_counts[0]]["block_logs"].append(float(torch.logsumexp(static_values, 0)) - math.log(batch_count))
                        accum[inner_counts[0]]["timing"] += time.perf_counter() - cache_start
                        continue
                    outer = marginal.sample(batch_count, path_seed=seed("outer-path", batch), label_seed=seed("outer-label", batch))
                    charge(densities=batch_count * len(guide.components))
                    log_outer_ratio = marginal.log_q_over_p(outer)
                    if job["block"] is None:
                        charge(paths=batch_count, densities=batch_count * len(q.components))
                        cached = last_pair_cache(problem, outer, q)
                    cache_wall = time.perf_counter() - cache_start
                    for inner_count in inner_counts:
                        for repeat in range(1 if config["mode"] == "production" or not risk else 2):
                            inner_start = time.perf_counter()
                            if not risk:
                                charge(cdfs=batch_count)
                                values = cached.log_mu_mean() - log_outer_ratio
                            else:
                                pair = torch.randn((batch_count, inner_count, 2), dtype=torch.float64,
                                    generator=torch.Generator().manual_seed(seed(f"inner-{inner_count}-{repeat}", batch)))
                                charge(paths=batch_count * inner_count if job["block"] is not None else 0,
                                       cdfs=batch_count * inner_count, densities=batch_count * inner_count * len(q.components))
                                log_inner = (cached.log_inner_risk(pair) if job["block"] is None else
                                             block_log_inner_risk(problem, outer, pair, q, block))
                                values = nested_log_means(log_inner, log_outer_ratio)
                            if config["mode"] == "production":
                                accum[inner_count]["moments"].append(log_moments(values))
                            else:
                                accum[inner_count]["first" if repeat == 0 else "second"].append(values)
                            if repeat == 0:
                                accum[inner_count]["timing"] += cache_wall + time.perf_counter() - inner_start
                                accum[inner_count]["block_logs"].append(float(torch.logsumexp(values, 0)) - math.log(batch_count))
                for inner_count, data in accum.items():
                    all_logs = torch.cat(data["first"]) if data["first"] else None
                    summary = log_moments(all_logs) if all_logs is not None else merge_moments(data["moments"])
                    candidate = {"inner": inner_count, "summary": summary,
                        "block_log_means": data["block_logs"],
                        "single_pass_wall_seconds": data["timing"], "se_unit": "iid_draw" if static_job else "iid_outer_draw"}
                    if all_logs is not None:
                        candidate["log_unit_contributions"] = all_logs.tolist()
                    else:
                        candidate["batch_moments"] = data["moments"]
                    candidate["cv2_wall_per_outer"] = summary["relative_se"]**2 * data["timing"]
                    if data["second"]:
                        assert all_logs is not None
                        candidate["nested_variance_diagnostic"] = nested_variance_diagnostic(all_logs, torch.cat(data["second"]))
                        candidate["second_log_unit_contributions"] = torch.cat(data["second"]).tolist()
                    result["candidates"].append(candidate)
                if len(result["candidates"]) == 1:
                    single = result["candidates"][0]
                    result.update({key: single[key] for key in
                                   ("summary", "log_unit_contributions", "block_log_means", "batch_moments") if key in single})
            if "summary" in result:
                result["sensitivity"] = sensitivity(result["block_log_means"], bootstrap_seed=seed("bootstrap", 0, "audit"))
            result["status"] = "completed"
        except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as error:
            result["status"] = "protocol_failure"
            result["failure_reasons"].append(f"{type(error).__name__}: {error}")
            aborted = True
        result.update(counters={key: counters[key] - before[key] for key in counters},
                      wall_seconds=time.perf_counter() - timer)
        print(json.dumps({"finished": identity, "status": result["status"], "wall": result["wall_seconds"],
                          "work": counters["potential_equivalent_evaluations"],
                          "rse": result.get("summary", {}).get("relative_se")}), flush=True)
    if source_tree_digest(ROOT) != source["source_tree_digest"]:
        raise RuntimeError("runtime source changed during experiment")
    peak = max(peak, process.memory_info().rss)
    payload = {"schema": SCHEMA, "config": config, "source": source,
        "proposal_artifact_sha256": file_sha256(ROOT / config["proposal_artifact_path"])[0],
        "frozen_qs": rows, "records": records, "seed_ledger": ledger.to_dict(), "sample_uses": uses,
        "counters": counters, "peak_process_rss_bytes": peak,
        "physical_wall_seconds": time.perf_counter() - started, "model_q_changed": False,
        "p2_authorized": False, "performance_claim_authorized": False,
        "status": "protocol_failure" if aborted else "completed_design_pilot" if config["mode"] != "production" else "completed_production"}
    payload["selection"] = select_nested(payload) if config["mode"] in ("micro", "block") else None
    payload["qualification"] = qualify(payload)
    return payload


def select_nested(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in payload["records"] if r["estimand"] == "risk"]
    if not rows or any(r["status"] != "completed" for r in rows):
        return {"status": "unresolved_incomplete", "candidates": []}
    positions = {r["block"] for r in rows}
    candidates = []
    for block in sorted(positions, key=str):
        matching = [r for r in rows if r["block"] == block]
        for inner_count in payload["config"]["inner_counts"]:
            scores, ratios = [], []
            for r in matching:
                c = next(c for c in r["candidates"] if c["inner"] == inner_count)
                baseline = r["candidates"][0]["cv2_wall_per_outer"]
                score = c["cv2_wall_per_outer"]
                scores.append(score)
                ratios.append(score / baseline if baseline > 0 else math.inf)
            candidates.append({"block": block, "inner": inner_count, "worst_cv2_wall_per_outer": max(scores),
                               "worst_ratio_to_L1": max(ratios), "eligible": max(ratios) <= .8})
    eligible = [c for c in candidates if c["eligible"]]
    best = min(eligible, key=lambda c: c["worst_cv2_wall_per_outer"]) if eligible else None
    return {"status": "eligible_development_candidate" if best else "unresolved_no_two_cell_cost_gain",
            "selected": best, "candidates": candidates, "claim": "point_estimate_screen_not_significance"}


def qualify(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["config"]["mode"] != "production":
        return {"status": "not_run_design_pilot", "p2_authorized": False}
    if payload["status"] != "completed_production":
        return {"status": "unresolved_protocol_failure", "p2_authorized": False}
    findings, passed = [], True
    for row in payload["frozen_qs"]:
        for kind in ("risk", "probability"):
            if kind == "probability" and row["parent_training_rep"] != 0:
                continue
            matches = [r for r in payload["records"] if r["cell_id"] == row["cell"]["id"]
                       and r["parent_training_rep"] == row["parent_training_rep"] and r["estimand"] == kind]
            okay = len(matches) == (3 if kind == "risk" else 2)
            checks = []
            for r in matches:
                s = r["sensitivity"]
                good = (r["summary"]["relative_se"] <= .025 and s["maximum_leave_one_out_relative_shift"] <= .05
                        and s["maximum_unit_contribution_fraction"] <= .15)
                if r["method"] not in ("pcn", "elliptical_slice"):
                    good = good and s["between_unit_relative_se"] <= 2 * r["summary"]["relative_se"]
                checks.append({"method": r["method"], "pass": good})
                okay = okay and good
            pairs = []
            for i, a in enumerate(matches):
                for b in matches[i + 1:]:
                    comparison = relative_equivalence(a["summary"], b["summary"], comparisons=30 if kind == "risk" else 2)
                    pairs.append({"methods": [a["method"], b["method"]], **comparison})
                    okay = okay and comparison["pass"]
            passed = passed and okay
            findings.append({"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
                             "estimand": kind, "pass": okay, "precision_sensitivity": checks, "pairwise": pairs})
    return {"status": "pass_finite_grid_development" if passed else "unresolved_reference", "p2_authorized": passed,
            "findings": findings, "coverage_claim": "empirical_corroboration_not_tail_oracle"}


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    validate_config(config)
    # Norm reductions can differ by a few ulps across CPU thread counts.
    # Reconstruct the frozen guide under the measured runtime convention.
    torch.set_num_threads(config["torch_threads"])
    verify_production(config)
    if payload["schema"] != SCHEMA or payload["model_q_changed"] is not False or payload["p2_authorized"] is not False:
        raise ValueError("invalid schema/claim")
    audit_snapshot(ROOT, payload["source"], config)
    if file_sha256(ROOT / config["proposal_artifact_path"])[0] != payload["proposal_artifact_sha256"]:
        raise ValueError("q artifact changed")
    metadata, rows = inputs(config)
    if rows != payload["frozen_qs"]:
        raise ValueError("q grid changed")
    jobs = jobs_for(config, rows)
    if len(jobs) != len(payload["records"]):
        raise ValueError("missing job")
    contracts = []
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    for job, record in zip(jobs, payload["records"], strict=True):
        if any(record[key] != value for key, value in job.items()) or record["identity"] != experiment_identity(job):
            raise ValueError("job changed")
        if "measurement_contract" in record:
            contract = MeasurementContract.from_dict(record["measurement_contract"])
            if contract.consumer_id != record["identity"] or contract.target_digest != record["target_digest"]:
                raise ValueError("contract binding changed")
            contracts.append(contract)
        if record["status"] != "completed":
            if record["status"] == "protocol_failure" and not record["failure_reasons"]:
                raise ValueError("unexplained protocol failure")
            continue
        fixed = next(r for r in rows if r["cell"]["id"] == job["cell_id"]
                     and r["parent_training_rep"] == job["parent_training_rep"])
        problem = _problem(metadata["config"]["model"], fixed["cell"], metadata["config"]["smc"]["steps"])
        target_digest = canonical_digest({"model": metadata["config"]["model"], "cell": fixed["cell"], "steps": problem.steps})
        risk, smc_job = job["estimand"] == "risk", job["method"] in ("pcn", "elliptical_slice")
        expected_guide = None if smc_job else proposal_parameters(build_volterra_excursion_guide(problem, **config["guide"]))
        guide_digest = canonical_digest(expected_guide) if expected_guide is not None else None
        if (record["target_digest"] != target_digest or record["q_digest"] != (fixed["q_digest"] if risk else None)
                or record["guide_parameters"] != expected_guide or record["guide_digest"] != guide_digest
                or contract.q_digest != (fixed["q_digest"] if risk else guide_digest)
                or contract.r_digest != (guide_digest if risk and not smc_job else None)
                or contract.grid_steps != problem.steps or contract.parent_training_rep != job["parent_training_rep"]
                or contract.estimand_kind != ("raw_event_second_moment" if risk else "event_probability")):
            raise ValueError("target/q/marginal/estimand binding changed")
        for use in [u for u in payload["sample_uses"] if u["consumer_id"] == record["identity"]]:
            level, stream = use["use_id"].rsplit("/", 2)[-2:]
            key = SeedKey("reference-redesign-" + canonical_digest(config)[:16],
                "audit" if stream == "bootstrap" else "allocation-pilot" if config["mode"] != "production" else
                "reference-whole-smc" if smc_job else "reference-iid",
                job["estimand"] + "-" + job["method"] + "-" + str(job["block"]),
                problem.task_id, int(level), job["parent_training_rep"], stream)
            if use["seed_key"] != asdict(key):
                raise ValueError("seed namespace/target changed")
        if "summary" in record:
            regenerated = (merge_moments(record["batch_moments"]) if "batch_moments" in record else
                           log_moments(torch.tensor(record["log_unit_contributions"], dtype=torch.float64)))
            if regenerated != record["summary"] or regenerated["count"] != job["count"]:
                raise ValueError("summary/count changed")
            if "batch_moments" in record:
                expected_block_logs = [b["log_mean"] for b in record["batch_moments"]]
            elif smc_job:
                expected_block_logs = [w["log_estimand_estimate"] for w in record["whole_runs"]]
                if len(record["whole_runs"]) != job["count"]:
                    raise ValueError("whole-run count changed")
            else:
                values = torch.tensor(record["log_unit_contributions"], dtype=torch.float64)
                expected_block_logs = [float(torch.logsumexp(v, 0)) - math.log(len(v))
                                       for v in values.split(config["outer_batch_size"])]
            if expected_block_logs != record["block_log_means"]:
                raise ValueError("block/whole means changed")
            bootstrap_uses = [u for u in payload["sample_uses"] if u["use_id"] == f"{record['identity']}/0/bootstrap"]
            if len(bootstrap_uses) != 1:
                raise ValueError("missing or duplicate bootstrap stream")
            key = SeedKey(**bootstrap_uses[0]["seed_key"])
            if record["sensitivity"] != sensitivity(expected_block_logs, bootstrap_seed=ledger.lookup(key)):
                raise ValueError("sensitivity changed")
        for candidate in record["candidates"]:
            regenerated_candidate = (merge_moments(candidate["batch_moments"]) if "batch_moments" in candidate else
                                     log_moments(torch.tensor(candidate["log_unit_contributions"], dtype=torch.float64)))
            if regenerated_candidate != candidate["summary"]:
                raise ValueError("nested summary changed")
            if "second_log_unit_contributions" in candidate:
                diagnostic = nested_variance_diagnostic(torch.tensor(candidate["log_unit_contributions"], dtype=torch.float64),
                    torch.tensor(candidate["second_log_unit_contributions"], dtype=torch.float64))
                if diagnostic != candidate["nested_variance_diagnostic"]:
                    raise ValueError("nested variance diagnostic changed")
    for key, value in payload["counters"].items():
        if sum(r["counters"][key] for r in payload["records"]) != value:
            raise ValueError("work total changed")
    if payload["selection"] != (select_nested(payload) if config["mode"] in ("micro", "block") else None):
        raise ValueError("selection changed")
    if payload["qualification"] != qualify(payload):
        raise ValueError("qualification changed")
    expected_ids = set()
    for record in payload["records"]:
        if record["status"] != "completed":
            expected_ids.update(u["use_id"] for u in payload["sample_uses"] if u["consumer_id"] == record["identity"])
            continue
        identity = record["identity"]
        if record["method"] in ("pcn", "elliptical_slice"):
            expected_ids.update(f"{identity}/{rep}/whole" for rep in range(record["count"]))
        else:
            batch_size = config["production_batch_size"] if config["mode"] == "production" else config["outer_batch_size"]
            for batch in range(record["count"] // batch_size):
                expected_ids.update(f"{identity}/{batch}/{stream}" for stream in ("outer-path", "outer-label"))
                if record["method"] == "nested":
                    counts = [record["inner"]] if config["mode"] == "production" else config["inner_counts"]
                    for inner_count in counts:
                        for repeat in range(1 if config["mode"] == "production" else 2):
                            expected_ids.add(f"{identity}/{batch}/inner-{inner_count}-{repeat}")
        if "summary" in record:
            expected_ids.add(f"{identity}/0/bootstrap")
    if any(r.key.protocol != "reference-redesign-" + canonical_digest(config)[:16] for r in ledger.records):
        raise ValueError("seed protocol binding changed")
    role_audit = audit_sample_uses(payload["sample_uses"], ledger, contracts,
                                  expected_use_ids=expected_ids)
    return {"status": "binding_arithmetic_audit_not_tail_certificate", "records": len(jobs), "roles": role_audit,
            "p2_authorized": payload["qualification"]["p2_authorized"]}


def verify_production(config: dict[str, Any]) -> None:
    if config["mode"] != "production":
        return
    for binding in config["pilot_bindings"]:
        if file_sha256(ROOT / binding["path"])[0] != binding["sha256"]:
            raise ValueError("production pilot changed")
    if canonical_digest(config["production_jobs"]) != config["production_jobs_digest"]:
        raise ValueError("sealed production allocation changed")
    allocation_path = ROOT / config["allocation_path"]
    if file_sha256(allocation_path)[0] != config["allocation_sha256"]:
        raise ValueError("sealed allocation report changed")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    expected = allocation["production_config"]
    if expected is None:
        raise ValueError("production locked by allocation failure")
    expected = {**expected, "allocation_path": config["allocation_path"], "allocation_sha256": config["allocation_sha256"]}
    if config != expected:
        raise ValueError("production config changed after allocation")


def production_allocation(micro: dict[str, Any], kernel: dict[str, Any], baseline: dict[str, Any],
                          *, bindings: list[dict[str, str]], block: dict[str, Any] | None = None) -> dict[str, Any]:
    """Allocate once from completed pilots; no count extension on final results."""
    reasons = []
    if any(p["status"] != "completed_design_pilot" for p in (micro, kernel)):
        reasons.append("incomplete_primary_pilot")
    if block is not None and block["status"] != "completed_design_pilot":
        reasons.append("incomplete_block_pilot")
    if any(p["proposal_artifact_sha256"] != micro["proposal_artifact_sha256"] for p in (kernel, baseline)):
        raise ValueError("allocation target artifact mismatch")
    nested_source = micro
    selected = micro["selection"].get("selected")
    if block is not None and block["selection"].get("selected"):
        candidate = block["selection"]["selected"]
        micro_best_score = min(c["worst_cv2_wall_per_outer"] for c in micro["selection"]["candidates"])
        if candidate["worst_cv2_wall_per_outer"] <= .8 * micro_best_score:
            selected, nested_source = candidate, block
    if selected is None:
        reasons.append("no_two_cell_20_percent_nested_cost_gain")
        # Diagnostic forecast only; this fallback never authorizes production.
        candidates = micro["selection"].get("candidates", [])
        if not candidates:
            return {"status": "unresolved_incomplete_pilot", "failure_reasons": reasons,
                    "production_config": None, "production_jobs": [], "p2_authorized": False}
        selected = min(candidates, key=lambda c: c["worst_cv2_wall_per_outer"])
    jobs = []
    for row in micro["frozen_qs"]:
        for kind in ("risk", "probability"):
            if kind == "probability" and row["parent_training_rep"] != 0:
                continue
            methods = ("nested", "elliptical_slice", "static-iid") if kind == "risk" else ("conditional", "elliptical_slice")
            for method in methods:
                pilot_source = baseline if method == "static-iid" else kernel if method == "elliptical_slice" else micro if kind == "probability" else nested_source
                pilot_method = method
                matches = [r for r in pilot_source["records"] if r["cell_id"] == row["cell"]["id"]
                    and r["estimand"] == kind and r["method"] == pilot_method
                    and r["parent_training_rep"] == (row["parent_training_rep"] if method == "static-iid" else 0)
                    and (method != "nested" or r["block"] == selected["block"])]
                if len(matches) != 1 or matches[0]["status"] != "completed":
                    reasons.append(f"missing_completed_pilot:{row['cell']['id']}/{kind}/{method}")
                    continue
                pilot = matches[0]
                if method == "nested":
                    candidate = next(c for c in pilot["candidates"] if c["inner"] == selected["inner"])
                    summary, wall = candidate["summary"], candidate["single_pass_wall_seconds"]
                else:
                    summary, wall = pilot["summary"], pilot["wall_seconds"]
                is_smc = method == "elliptical_slice"
                planned = precision_count(summary, target_rse=.015, safety_factor=3.,
                    minimum=32 if is_smc else 1048576, maximum=256 if is_smc else 8388608,
                    multiple=1 if is_smc else 65536)
                count = planned["required_count"]
                if planned["status"] != "allocated":
                    reasons.append(f"sample_cap:{row['cell']['id']}/{row['parent_training_rep']}/{kind}/{method}")
                inner_count = selected["inner"] if method == "nested" else 1
                block_position = selected["block"] if method == "nested" else None
                work_per = (pilot["counters"]["potential_equivalent_evaluations"] / summary["count"] if is_smc else
                            2 if method in ("conditional", "static-iid") else
                            inner_count + 1 if block_position is None else 2 * inner_count)
                job = {"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
                       "estimand": kind, "method": method, "inner": inner_count, "block": block_position,
                       "count": count, "allocation": planned,
                       "forecast_work": math.ceil(count * work_per * (1.5 if is_smc else 1.)),
                       "forecast_wall_seconds": wall / summary["count"] * count * 1.5}
                jobs.append(job)
    work, wall = sum(j["forecast_work"] for j in jobs), sum(j["forecast_wall_seconds"] for j in jobs)
    if work > 160000000:
        reasons.append("production_work_cap")
    if wall > 7200:
        reasons.append("production_wall_cap")
    if len(jobs) != 34:
        reasons.append("incomplete_production_grid")
    config = copy.deepcopy(micro["config"])
    config.update(mode="production", output_path="results/post_audit/reference_redesign_production_v1.json",
                  production_batch_size=8192, production_jobs=jobs,
                  production_jobs_digest=canonical_digest(jobs), pilot_bindings=bindings,
                  scope="fresh_finite_grid_development_reference_not_confirmation")
    config["budget"].update(max_potential_evaluations=160000000, max_wall_seconds=7200.)
    return {"status": "allocated" if not reasons else "unresolved_production_budget_or_gain",
            "failure_reasons": reasons, "selected_nested_candidate": selected,
            "selected_nested_mode": nested_source["config"]["mode"], "production_jobs": jobs,
            "forecast_potential_equivalent_evaluations": work, "forecast_wall_seconds": wall,
            "production_config": config if not reasons else None, "pilot_bindings": bindings,
            "p2_authorized": False,
            "scope": "rep0_nested_and_ellipse_variability_forecast_not_uniform_q_precision_theorem"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/post_audit/reference_redesign_micro_v1.yaml")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--allocate", action="store_true")
    args = parser.parse_args()
    if args.allocate:
        paths = ["results/post_audit/reference_redesign_micro_v1.json",
                 "results/post_audit/reference_redesign_kernel_v1.json",
                 "results/post_audit/structural_v2_p1_repair_v1.json"]
        values = [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in paths]
        audit(values[0])
        audit(values[1])
        legacy_audit(values[2])
        block_path = "results/post_audit/reference_redesign_block_v1.json"
        block = None
        if (ROOT / block_path).exists():
            paths.append(block_path)
            block = json.loads((ROOT / block_path).read_text(encoding="utf-8"))
            audit(block)
        bindings = [{"path": path, "sha256": file_sha256(ROOT / path)[0]} for path in paths]
        allocated = production_allocation(*values, bindings=bindings, block=block)
        allocation_path = ROOT / "results/post_audit/reference_redesign_allocation_v1.json"
        with allocation_path.open("x", encoding="utf-8") as handle:
            json.dump(allocated, handle, indent=2, allow_nan=False)
        if allocated["production_config"] is not None:
            config = {**allocated["production_config"], "allocation_path": allocation_path.relative_to(ROOT).as_posix(),
                      "allocation_sha256": file_sha256(allocation_path)[0]}
            config_path = ROOT / "configs/post_audit/reference_redesign_production_v1.yaml"
            with config_path.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        print(json.dumps({key: value for key, value in allocated.items() if key not in ("production_config", "production_jobs")}, allow_nan=False))
        return
    if args.audit:
        print(json.dumps(audit(json.loads(args.audit.read_text(encoding="utf-8"))), allow_nan=False))
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    source = freeze_source(output, config, extra_snapshot_paths=(PLAN,))
    payload = run(config, source)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "status": payload["status"], "selection": payload["selection"],
                      "qualification": payload["qualification"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
