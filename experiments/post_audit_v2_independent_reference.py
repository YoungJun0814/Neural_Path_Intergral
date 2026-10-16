"""P1 disjoint pilot -> frozen allocation -> independent raw references.

No estimator fitting, model selection, or confirmation occurs here. Failed
allocation keeps P2 locked; pilot observations are never production references.
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
from typing import Any, Literal, cast

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.conditional_second_moment import (
    log_auxiliary_second_moment,
    log_risk_potential,
)
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.structural_v2_contract import MeasurementContract, audit_sample_uses
from src.path_integral.structural_v2_inventory import audit_snapshot, file_sha256, stream_evidence
from src.path_integral.structural_v2_reference import (
    exact_smc_work,
    log_moments,
    merge_moments,
    precision_count,
    relative_equivalence,
    sensitivity,
)
from src.path_integral.structural_v2_terminal_geometry import (
    PARTITION,
    mode_contributions,
    terminal_geometry,
)
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.structural-v2.raw-reference.v1"


def fixed_smc_config(spec: dict[str, Any], *, seed: int) -> WeightedSMCConfig:
    """Here levels counts temperature POINTS, unlike legacy interval-count helper."""
    exact_smc_work(spec)
    power = spec["bridge_power"]
    if isinstance(power, bool) or not isinstance(power, int) or power < 1:
        raise ValueError("invalid bridge power")
    return WeightedSMCConfig(particles=spec["particles"],
        temperatures=tuple((i / (spec["levels"] - 1))**power for i in range(spec["levels"])),
        mutation_steps=spec["mutation_steps"], pcn_scale=spec["pcn_scale"], replicates=1, seed=seed,
        resample_every=spec["resample_every"],
        resampling_scheme=cast(Literal["multinomial", "stratified"], spec["resampling_scheme"]),
        independence_every=spec.get("independence_every", 0),
        retain_final_particles=spec.get("retain_final_particles", False))


def validate_config(config: dict[str, Any]) -> None:
    if config["schema"] != "npi.structural-v2.reference-config.v1" or config["mode"] not in {"pilot", "production"}:
        raise ValueError("invalid P1 config schema/mode")
    if config["torch_threads"] != 1 or isinstance(config["torch_threads"], bool):
        raise ValueError("sequential one-thread scientific timing required")
    if config["pilot_parent_rep"] != 0:
        raise ValueError("pilot parent must be fixed before result inspection")
    if [s["id"] for s in config["schedules"]] != ["local-A", "local-B", "global-C"]:
        raise ValueError("requires two guide-free and one static-global schedule")
    for s in config["schedules"]:
        exact_smc_work(s)
        fixed_smc_config(s, seed=1)
        if bool(s.get("independence_every", 0)) != (s["id"] == "global-C"):
            raise ValueError("local schedule must be guide-free")
    for section, keys in (("pilot", ("whole_smc_replicates", "iid_count", "iid_blocks", "batch_size")),
                          ("budget", ("max_potential_evaluations", "max_process_rss_bytes", "minimum_free_disk_bytes"))):
        for k in keys:
            v = config[section][k]
            if isinstance(v, bool) or not isinstance(v, int) or v < (0 if k == "minimum_free_disk_bytes" else 2):
                raise ValueError("invalid P1 count/budget")
    if config["pilot"]["iid_count"] % (config["pilot"]["iid_blocks"] * config["pilot"]["batch_size"]):
        raise ValueError("IID pilot blocks must have equal complete batches")
    if config["qualification"]["risk_comparisons"] != 30 or config["qualification"]["probability_comparisons"] != 2:
        raise ValueError("comparison family cannot be silently reduced")
    if config["guide"]["defensive_mass"] != .1:
        raise ValueError("static guide floor must remain .1")
    if not isinstance(config.get("probability_local_A_pilot", False), bool):
        raise ValueError("probability local pilot switch must be boolean")


def inputs(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    rows = []
    for name, value in stream_evidence(ROOT / config["proposal_artifact_path"]):
        if name != "record":
            metadata[name] = value
            continue
        selected = [x for x in value["candidates"] if x["method"] == config["method"]]
        if len(selected) != 1 or canonical_digest(selected[0]["proposal_parameters"]) != selected[0]["proposal_digest"]:
            raise ValueError("frozen q binding mismatch")
        q = proposal_from_parameters(selected[0]["proposal_parameters"])
        if q.defensive_mass < .1 - 1e-12:
            raise ValueError("q natural floor below contract")
        rows.append({"cell": value["cell"], "parent_training_rep": value["parent_training_rep"],
                     "q_parameters": selected[0]["proposal_parameters"], "q_digest": selected[0]["proposal_digest"]})
    identities = {(r["cell"]["id"], r["parent_training_rep"]) for r in rows}
    cells = {r["cell"]["id"] for r in rows}
    if len(rows) != 10 or len(identities) != 10 or len(cells) != 2 or any(
        {r["parent_training_rep"] for r in rows if r["cell"]["id"] == cell} != set(range(5)) for cell in cells
    ):
        raise ValueError("requires fixed two-cell five-q development grid")
    return metadata, rows


def pilot_jobs(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for row in rows:
        jobs.append({"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
                     "estimand": "risk", "method": "static-iid", "count": config["pilot"]["iid_count"]})
        if row["parent_training_rep"] != config["pilot_parent_rep"]:
            continue
        for kind in ("risk", "probability"):
            if kind == "probability":
                jobs.append({"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
                             "estimand": kind, "method": "static-iid", "count": config["pilot"]["iid_count"]})
            for schedule in config["schedules"]:
                if kind == "probability" and schedule["id"] == "local-A" and not config.get("probability_local_A_pilot", False):
                    continue
                jobs.append({"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
                             "estimand": kind, "method": schedule["id"],
                             "count": config["pilot"]["whole_smc_replicates"]})
    return jobs


def allocate(payload: dict[str, Any]) -> dict[str, Any]:
    """No production mean is inspected; no count is extended after production."""
    config, records, spec = payload["config"], payload["records"], payload["config"]["allocation"]
    if config["mode"] != "pilot" or any(r["status"] != "completed" for r in records):
        return {"status": "unresolved_incomplete_pilot", "production_jobs": [], "p2_authorized": False}
    locals_ = [s["id"] for s in config["schedules"] if s.get("independence_every", 0) == 0]
    scores = {method: max(r["summary"]["count"] * r["summary"]["relative_se"]**2 *
                         r["wall_seconds"] / r["count"] for r in records
                         if r["estimand"] == "risk" and r["method"] == method) for method in locals_}
    chosen = min(locals_, key=lambda method: (scores[method], method))
    probability_scores = {method: max(r["summary"]["count"] * r["summary"]["relative_se"]**2 *
        r["wall_seconds"] / r["count"] for r in records if r["estimand"] == "probability" and r["method"] == method)
        for method in locals_} if config.get("probability_local_A_pilot", False) else None
    chosen_probability = min(locals_, key=lambda method: (probability_scores[method], method)) if probability_scores is not None else "local-B"
    jobs, reasons = [], []
    for row in payload["frozen_qs"]:
        for kind in ("risk", "probability"):
            if kind == "probability" and row["parent_training_rep"] != config["pilot_parent_rep"]:
                continue
            methods = ("static-iid", chosen, "global-C") if kind == "risk" else ("static-iid", chosen_probability)
            for method in methods:
                pilots = [r for r in records if r["cell_id"] == row["cell"]["id"] and r["estimand"] == kind
                          and r["method"] == method and (method != "static-iid" or
                          r["parent_training_rep"] == row["parent_training_rep"])]
                if len(pilots) != 1:
                    raise ValueError("missing allocation pilot")
                pilot = pilots[0]
                iid = method == "static-iid"
                allocation = precision_count(pilot["summary"], target_rse=spec["target_relative_se"],
                    safety_factor=spec["safety_factor"], minimum=spec["minimum_iid_count"] if iid else spec["minimum_whole_smc_replicates"],
                    maximum=spec["maximum_iid_count"] if iid else spec["maximum_whole_smc_replicates"],
                    multiple=config["pilot"]["batch_size"] if iid else 1)
                count = allocation["required_count"]
                schedule = next((s for s in config["schedules"] if s["id"] == method), None)
                job = {"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
                       "estimand": kind, "method": method, "count": count, "allocation": allocation,
                       "pilot_parent_rep": pilot["parent_training_rep"],
                       "forecast_potential_evaluations": count if iid else count * (exact_smc_work(schedule) +
                           (schedule["particles"] if schedule.get("retain_final_particles", False) else 0)),
                       "forecast_wall_seconds": pilot["wall_seconds"] / pilot["count"] * count * spec["wall_safety_factor"]}
                jobs.append(job)
                if allocation["status"] != "allocated":
                    reasons.append(f"{row['cell']['id']}/{row['parent_training_rep']}/{kind}/{method}: sample_cap")
    work = sum(j["forecast_potential_evaluations"] for j in jobs)
    wall = sum(j["forecast_wall_seconds"] for j in jobs)
    if work > spec["production_max_potential_evaluations"]:
        reasons.append("production_potential_cap")
    if wall > spec["production_max_wall_seconds"]:
        reasons.append("production_wall_cap")
    result = {"status": "allocated" if not reasons else "unresolved_production_budget", "selected_local_schedule": chosen,
            "selection_scores_worst_cell_cv2_wall_per_run": scores, "failure_reasons": reasons,
            "production_jobs": jobs, "forecast_potential_evaluations": work, "forecast_wall_seconds": wall,
            "scope": "rep0_smc_variability_forecast_for_other_fixed_qs_not_uniform_precision_guarantee",
            "p2_authorized": False}
    if probability_scores is not None:
        result.update(selected_probability_local_schedule=chosen_probability,
                      probability_selection_scores_worst_cell_cv2_wall_per_run=probability_scores)
    return result


def qualification(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["config"]["mode"] != "production":
        return {"status": "not_run_pilot_only", "p2_authorized": False}
    spec, records = payload["config"]["qualification"], payload["records"]
    findings, passed = [], True
    for row in payload["frozen_qs"]:
        for kind in ("risk", "probability"):
            if kind == "probability" and row["parent_training_rep"] != payload["config"]["pilot_parent_rep"]:
                continue
            matches = [r for r in records if r["cell_id"] == row["cell"]["id"]
                       and r["parent_training_rep"] == row["parent_training_rep"] and r["estimand"] == kind]
            okay = len(matches) == (3 if kind == "risk" else 2)
            checks = []
            for r in matches:
                s = r.get("sensitivity", {})
                good = r["status"] == "completed" and r["summary"]["relative_se"] <= spec["maximum_relative_se"]
                good = good and s.get("maximum_leave_one_out_relative_shift", 1.) <= spec["maximum_leave_one_out_relative_shift"]
                good = good and s.get("maximum_unit_contribution_fraction", 1.) <= spec["maximum_unit_contribution_fraction"]
                if r["method"] == "static-iid" and good:
                    good = s["between_unit_relative_se"] <= spec["maximum_block_between_to_iid_rse_ratio"] * r["summary"]["relative_se"]
                checks.append({"method": r["method"], "pass": good})
                okay = okay and good
            pairs = []
            if all(r["status"] == "completed" for r in matches):
                for i, a in enumerate(matches):
                    for b in matches[i + 1:]:
                        pair = relative_equivalence(a["summary"], b["summary"],
                            comparisons=spec["risk_comparisons"] if kind == "risk" else spec["probability_comparisons"],
                            alpha=spec["family_alpha"], margin=spec["relative_margin"])
                        pairs.append({"methods": [a["method"], b["method"]], **pair})
                        okay = okay and pair["pass"]
            findings.append({"cell_id": row["cell"]["id"], "parent_training_rep": row["parent_training_rep"],
                             "estimand": kind, "pass": okay, "precision_sensitivity": checks, "pairwise": pairs})
            passed = passed and okay
    return {"status": "pass_finite_grid_development" if passed else "unresolved_reference", "p2_authorized": passed,
            "findings": findings, "method_se_fraction_check": "deferred_until_a_specific_comparison_method_exists",
            "coverage_claim": "empirical_corroboration_not_distribution_free_oracle"}


def production_config(pilot: dict[str, Any], pilot_path: Path) -> dict[str, Any]:
    """Seal equal-block counts before production; never round after seeing finals."""
    if pilot["allocation"]["status"] != "allocated":
        raise ValueError("production locked: pilot allocation exceeds preregistered budget")
    config = copy.deepcopy(pilot["config"])
    jobs = copy.deepcopy(pilot["allocation"]["production_jobs"])
    multiple = config["pilot"]["batch_size"] * config["pilot"]["iid_blocks"]
    for job in jobs:
        original_count = job["count"]
        if job["method"] == "static-iid":
            job["count"] = math.ceil(original_count / multiple) * multiple
            if job["count"] > config["allocation"]["maximum_iid_count"]:
                raise ValueError("equal-block production rounding exceeds sample cap")
            job["forecast_potential_evaluations"] = job["count"]
            job["forecast_wall_seconds"] *= job["count"] / original_count
        job["production_rounding_from_pilot_forecast_count"] = original_count
    if (sum(j["forecast_potential_evaluations"] for j in jobs) > config["allocation"]["production_max_potential_evaluations"]
            or sum(j["forecast_wall_seconds"] for j in jobs) > config["allocation"]["production_max_wall_seconds"]):
        raise ValueError("equal-block sealed production exceeds total budget")
    config.update(mode="production", output_path="results/post_audit/structural_v2_p1_production_v1.json",
        production_jobs=jobs, pilot_artifact_sha256=file_sha256(pilot_path)[0],
        pilot_artifact_path=pilot_path.resolve().relative_to(ROOT).as_posix())
    config["budget"].update(max_potential_evaluations=config["allocation"]["production_max_potential_evaluations"],
                             max_wall_seconds=config["allocation"]["production_max_wall_seconds"])
    return config


def verify_production_binding(config: dict[str, Any]) -> None:
    if config["mode"] != "production":
        return
    pilot_path = ROOT / config["pilot_artifact_path"]
    if file_sha256(pilot_path)[0] != config["pilot_artifact_sha256"]:
        raise ValueError("allocation pilot binding changed")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    audit(pilot)
    if config != production_config(pilot, pilot_path):
        raise ValueError("production config, endpoint or allocation changed after pilot")


def run(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    verify_production_binding(config)
    if config.get("repair_baseline_path"):
        baseline_path = ROOT / config["repair_baseline_path"]
        if file_sha256(baseline_path)[0] != config["repair_baseline_sha256"]:
            raise ValueError("repair baseline binding changed")
        audit(json.loads(baseline_path.read_text(encoding="utf-8")))
    torch.set_num_threads(config["torch_threads"])
    old, rows = inputs(config)
    sha, _ = file_sha256(ROOT / config["proposal_artifact_path"])
    jobs = pilot_jobs(config, rows) if config["mode"] == "pilot" else config["production_jobs"]
    start, work, density = time.perf_counter(), 0, 0
    peak, process, ledger, uses, contracts, records = 0, psutil.Process(), SeedLedger(), [], [], []
    budget = config["budget"]
    if shutil.disk_usage(ROOT).free < budget["minimum_free_disk_bytes"]:
        raise RuntimeError("insufficient free disk")

    def check(n: int = 0) -> None:
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if work + n > budget["max_potential_evaluations"] or time.perf_counter() - start > budget["max_wall_seconds"]:
            raise TimeoutError("unresolved_potential_or_wall_budget")
        if peak > min(budget["max_process_rss_bytes"], psutil.virtual_memory().total // 4):
            raise TimeoutError("unresolved_process_memory_budget")

    for job in jobs:
        stage, before, before_density = time.perf_counter(), work, density
        row = next(r for r in rows if r["cell"]["id"] == job["cell_id"] and r["parent_training_rep"] == job["parent_training_rep"])
        problem = _problem(old["config"]["model"], row["cell"], old["config"]["smc"]["steps"])
        q = proposal_from_parameters(row["q_parameters"])
        iid, risk = job["method"] == "static-iid", job["estimand"] == "risk"
        identity = f"{job['cell_id']}/{job['parent_training_rep']}/{job['estimand']}/{job['method']}"
        contract = MeasurementContract(consumer_id=identity,
            target_digest=canonical_digest({"model": old["config"]["model"], "cell": row["cell"], "steps": problem.steps}),
            grid_steps=problem.steps, estimand_kind="raw_event_second_moment" if risk else "event_probability",
            estimator_kind="auxiliary_is" if iid and risk else "ordinary_is" if iid else "smc_normalizer",
            sample_role="reference-iid" if iid else "reference-whole-smc", se_unit="iid_draw" if iid else "whole_smc_run",
            q_digest=row["q_digest"] if risk else None, r_digest=None,
            parent_training_rep=row["parent_training_rep"], reference_rep=0) if not iid else None
        result: dict[str, Any] = {**job, "identity": identity, "q_digest": row["q_digest"] if risk else None,
                                  "blocks": [], "whole_runs": [], "status": "unresolved", "failure_reasons": []}
        records.append(result)

        def seed(stream: str, level: int, role: str | None = None, *, is_iid: bool = iid,
                 fixed_job: dict[str, Any] = job, fixed_row: dict[str, Any] = row,
                 task_id: str = problem.task_id, consumer_id: str = identity) -> int:
            sample_role = role or ("allocation-pilot" if config["mode"] == "pilot" else
                                  "reference-iid" if is_iid else "reference-whole-smc")
            key = SeedKey("structural-v2-reference-" + canonical_digest(config)[:16], sample_role,
                          fixed_job["estimand"] + "-" + fixed_job["method"], task_id, level,
                          fixed_row["parent_training_rep"], stream)
            value = ledger.allocate(key)
            uses.append({"use_id": f"{consumer_id}/{level}/{stream}", "consumer_id": consumer_id,
                         "role": sample_role, "seed_key": asdict(key), "pair_group": None})
            return value

        def payoff(x: torch.Tensor, fixed_problem: Any = problem) -> torch.Tensor:
            nonlocal work
            check(len(x))
            work += len(x)
            return _log_potential(fixed_problem, x)

        try:
            check()
            guide = build_volterra_excursion_guide(problem, **config["guide"]) if iid or job["method"] == "global-C" else None
            result["guide_parameters"] = proposal_parameters(guide) if guide is not None else None
            result["guide_digest"] = canonical_digest(result["guide_parameters"]) if guide is not None else None
            result["guide_probe_path_steps"] = problem.local_dimension * problem.steps if guide is not None else 0
            if iid:
                assert guide is not None
                contract = MeasurementContract(consumer_id=identity,
                    target_digest=canonical_digest({"model": old["config"]["model"], "cell": row["cell"], "steps": problem.steps}),
                    grid_steps=problem.steps, estimand_kind="raw_event_second_moment" if risk else "event_probability",
                    estimator_kind="auxiliary_is" if risk else "ordinary_is", sample_role="reference-iid", se_unit="iid_draw",
                    q_digest=row["q_digest"] if risk else result["guide_digest"], r_digest=result["guide_digest"] if risk else None,
                    parent_training_rep=row["parent_training_rep"], reference_rep=0)
            assert contract is not None
            contracts.append(contract)
            result["measurement_contract"] = contract.to_dict()
            if iid:
                count, blocks, batch_size = job["count"], config["pilot"]["iid_blocks"], config["pilot"]["batch_size"]
                if count % blocks or (count // blocks) % batch_size:
                    raise ValueError("equal IID blocks must contain complete batches")
                assert guide is not None
                for block in range(blocks):
                    logs = []
                    for batch in range(count // blocks // batch_size):
                        level = block * (count // blocks // batch_size) + batch
                        check(batch_size)
                        draw = guide.sample(batch_size, path_seed=seed("path", level), label_seed=seed("label", level))
                        density += batch_size * len(guide.components)
                        logg = payoff(draw.samples)
                        if risk:
                            density += batch_size * len(q.components)
                            values = log_auxiliary_second_moment(logg, q.log_q_over_p(draw.samples), draw.log_q_over_p)
                        else:
                            values = logg + draw.log_p_over_q
                        logs.append(values)
                    result["blocks"].append(log_moments(torch.cat(logs)))
                result["summary"] = merge_moments(result["blocks"])
                unit_means = [b["log_mean"] for b in result["blocks"]]
            else:
                schedule = next(s for s in config["schedules"] if s["id"] == job["method"])

                def potential(x: torch.Tensor, is_risk: bool = risk, fixed_q: Any = q,
                              fixed_payoff: Any = payoff) -> torch.Tensor:
                    nonlocal density
                    logg = fixed_payoff(x)
                    if is_risk:
                        density += len(x) * len(fixed_q.components)
                        return log_risk_potential(logg, fixed_q.log_q_over_p(x), defensive_mass=fixed_q.defensive_mass)
                    return logg

                for rep in range(job["count"]):
                    expected = exact_smc_work(schedule)
                    diagnostic_count = schedule["particles"] if schedule.get("retain_final_particles", False) else 0
                    check(expected + diagnostic_count)
                    run_before = work
                    smc = estimate_weighted_tempered_normalizer(potential, dimension=problem.local_dimension,
                        config=fixed_smc_config(schedule, seed=seed("whole", rep)), independence_proposal=guide)
                    if smc.potential_evaluations != expected or work - run_before != expected:
                        raise ValueError("SMC work formula disagrees with actual calls")
                    global_moves = sum(int(s["global_mutation_proposals"]) for s in smc.replicate_diagnostics[0]["stages"])
                    if guide is not None:
                        density += global_moves * len(guide.components) * 2
                    logz = float(smc.log_replicate_estimates[0])
                    result["whole_runs"].append({"replicate": rep, "log_normalizer": logz,
                        "log_estimand_estimate": logz - math.log(q.defensive_mass) if risk else logz,
                        "potential_evaluations": expected, "diagnostics": smc.replicate_diagnostics[0]})
                    if diagnostic_count:
                        if smc.final_particles is None or smc.final_weights is None:
                            raise RuntimeError("missing declared terminal diagnostic particles")
                        check(diagnostic_count)
                        work += diagnostic_count
                        result["whole_runs"][-1]["terminal_geometry"] = terminal_geometry(problem, smc.final_particles, smc.final_weights)
                        result["whole_runs"][-1]["diagnostic_path_evaluations"] = diagnostic_count
                unit_means = [r["log_estimand_estimate"] for r in result["whole_runs"]]
                result["summary"] = log_moments(torch.tensor(unit_means, dtype=torch.float64))
                if schedule.get("retain_final_particles", False):
                    result["mode_contributions"] = mode_contributions(result["whole_runs"])
            result["sensitivity"] = sensitivity(unit_means, bootstrap_seed=seed("bootstrap", 0, "audit"),
                                                bootstrap_replicates=config["qualification"]["bootstrap_replicates"])
            result["status"] = "completed"
        except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as error:
            result["failure_reasons"].append(f"{type(error).__name__}: {error}")
        result.update(potential_evaluations=work - before, density_component_evaluations=density - before_density,
                      path_steps=(work - before) * problem.steps, wall_seconds=time.perf_counter() - stage)
        print(json.dumps({"finished": identity, "status": result["status"],
                          "rse": result.get("summary", {}).get("relative_se"), "work": work}), flush=True)
    if source_tree_digest(ROOT) != source["source_tree_digest"]:
        raise RuntimeError("runtime source changed during experiment")
    payload = {"schema": SCHEMA, "config": config, "source": source, "proposal_artifact_sha256": sha,
        "frozen_qs": rows, "records": records, "seed_ledger": ledger.to_dict(), "sample_uses": uses,
        "potential_evaluations": work, "density_component_evaluations": density,
        "path_steps": sum(r["path_steps"] for r in records), "peak_process_rss_bytes": peak,
        "physical_wall_seconds": time.perf_counter() - start, "model_q_changed": False,
        "performance_claim_authorized": False, "low_dimensional_oracle": "tests/test_structural_v2_reference.py"}
    if any(s.get("retain_final_particles", False) for s in config["schedules"]):
        payload["diagnostic_path_evaluations"] = sum(r.get("diagnostic_path_evaluations", 0)
            for record in records for r in record["whole_runs"])
    payload["allocation"] = allocate(payload) if config["mode"] == "pilot" else None
    payload["qualification"] = qualification(payload)
    return payload


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    validate_config(config)
    torch.set_num_threads(config["torch_threads"])
    verify_production_binding(config)
    if payload["schema"] != SCHEMA or payload["model_q_changed"] is not False or payload["performance_claim_authorized"] is not False:
        raise ValueError("schema or unauthorized scientific claim")
    audit_snapshot(ROOT, payload["source"], config)
    if config.get("repair_baseline_path") and file_sha256(ROOT / config["repair_baseline_path"])[0] != config["repair_baseline_sha256"]:
        raise ValueError("repair baseline changed")
    if file_sha256(ROOT / config["proposal_artifact_path"])[0] != payload["proposal_artifact_sha256"]:
        raise ValueError("historical q artifact changed")
    old, rows = inputs(config)
    if rows != payload["frozen_qs"]:
        raise ValueError("frozen q completion grid changed")
    jobs = pilot_jobs(config, rows) if config["mode"] == "pilot" else config["production_jobs"]
    records = payload["records"]
    if len(records) != len(jobs):
        raise ValueError("missing reference job")
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    contracts = []
    expected_use_ids = set()
    for job, row in zip(jobs, records, strict=True):
        if any(row[k] != v for k, v in job.items()):
            raise ValueError("reference job or allocation changed")
        if row["status"] != "completed":
            if not row["failure_reasons"]:
                raise ValueError("unexplained reference failure")
            if "measurement_contract" in row:
                contracts.append(MeasurementContract.from_dict(row["measurement_contract"]))
            continue
        contracts.append(MeasurementContract.from_dict(row["measurement_contract"]))
        fixed = next(r for r in rows if r["cell"]["id"] == job["cell_id"] and r["parent_training_rep"] == job["parent_training_rep"])
        problem = _problem(old["config"]["model"], fixed["cell"], old["config"]["smc"]["steps"])
        guide = build_volterra_excursion_guide(problem, **config["guide"]) if job["method"] in {"static-iid", "global-C"} else None
        params = proposal_parameters(guide) if guide is not None else None
        if row["guide_parameters"] != params or row["guide_digest"] != (canonical_digest(params) if params is not None else None):
            raise ValueError("frozen static guide changed")
        if row["q_digest"] != (fixed["q_digest"] if job["estimand"] == "risk" else None):
            raise ValueError("risk q or probability estimand binding changed")
        c = contracts[-1]
        if c.target_digest != canonical_digest({"model": old["config"]["model"], "cell": fixed["cell"], "steps": problem.steps}) or c.grid_steps != problem.steps:
            raise ValueError("finite-grid target changed")
        if c.estimand_kind != ("raw_event_second_moment" if job["estimand"] == "risk" else "event_probability"):
            raise ValueError("raw M2/probability contract mismatch")
        if c.q_digest != (fixed["q_digest"] if job["estimand"] == "risk" else row["guide_digest"] if job["method"] == "static-iid" else None):
            raise ValueError("measurement q changed")
        if c.r_digest != (row["guide_digest"] if job["method"] == "static-iid" and job["estimand"] == "risk" else None):
            raise ValueError("measurement auxiliary changed")
        if c.parent_training_rep != job["parent_training_rep"] or c.reference_rep != 0:
            raise ValueError("measurement repetition identity changed")
        if c.estimator_kind != ("auxiliary_is" if job["method"] == "static-iid" and job["estimand"] == "risk"
                                else "ordinary_is" if job["method"] == "static-iid" else "smc_normalizer"):
            raise ValueError("measurement estimator changed")
        if row["identity"] != f"{job['cell_id']}/{job['parent_training_rep']}/{job['estimand']}/{job['method']}":
            raise ValueError("consumer identity changed")
        for use in (u for u in payload["sample_uses"] if u["consumer_id"] == row["identity"]):
            key = SeedKey(**use["seed_key"])
            role = "audit" if key.stream == "bootstrap" else "allocation-pilot" if config["mode"] == "pilot" else (
                "reference-iid" if job["method"] == "static-iid" else "reference-whole-smc")
            expected_key = SeedKey("structural-v2-reference-" + canonical_digest(config)[:16], role,
                job["estimand"] + "-" + job["method"], problem.task_id, key.level, job["parent_training_rep"], key.stream)
            if key != expected_key or use["use_id"] != f"{row['identity']}/{key.level}/{key.stream}":
                raise ValueError("seed namespace/role binding changed")
        expected_use_ids.add(f"{row['identity']}/0/bootstrap")
        if row["method"] == "static-iid":
            expected = merge_moments(row["blocks"])
            logs = [b["log_mean"] for b in row["blocks"]]
            count = sum(b["count"] for b in row["blocks"])
            blocks = config["pilot"]["iid_blocks"]
            if count != job["count"] or row["whole_runs"] or len(row["blocks"]) != blocks or any(
                b["count"] != job["count"] // blocks for b in row["blocks"]
            ):
                raise ValueError("IID completion/SE mismatch")
            expected_work = count
            for batch in range(count // config["pilot"]["batch_size"]):
                expected_use_ids.update(f"{row['identity']}/{batch}/{stream}" for stream in ("path", "label"))
        else:
            logs = [r["log_estimand_estimate"] for r in row["whole_runs"]]
            if len(logs) != job["count"] or row["blocks"]:
                raise ValueError("whole SMC completion/SE mismatch")
            delta = proposal_from_parameters(fixed["q_parameters"]).defensive_mass
            for rep, r in enumerate(row["whole_runs"]):
                if r["replicate"] != rep or not math.isclose(r["log_estimand_estimate"],
                    r["log_normalizer"] - (math.log(delta) if job["estimand"] == "risk" else 0), abs_tol=1e-12):
                    raise ValueError("risk normalizer conversion mismatch")
                schedule = next(s for s in config["schedules"] if s["id"] == job["method"])
                stages = r["diagnostics"]["stages"]
                if r["potential_evaluations"] != exact_smc_work(schedule) or len(stages) != schedule["levels"] - 1 or stages[-1]["beta"] != 1.:
                    raise ValueError("whole SMC schedule/work changed")
                if stages[-1]["log_normalizer"] != r["log_normalizer"]:
                    raise ValueError("SMC terminal normalizer binding changed")
                expected_diagnostic = schedule["particles"] if schedule.get("retain_final_particles", False) else 0
                if r.get("diagnostic_path_evaluations", 0) != expected_diagnostic:
                    raise ValueError("terminal diagnostic work changed")
                if expected_diagnostic:
                    geometry = r["terminal_geometry"]
                    counts = geometry["particle_mode_counts"]
                    if (geometry["partition"] != PARTITION or geometry["particles"] != expected_diagnostic
                            or len(counts) != 16 or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts)
                            or sum(counts) != expected_diagnostic):
                        raise ValueError("terminal mode particle completion mismatch")
                    for metric in ("weighted_integrated_variance", "weighted_log_conditional_probability",
                                   "weighted_largest_left_variance_share", "weighted_peak_time_fraction", "terminal_weight_ess"):
                        if not math.isfinite(geometry[metric]):
                            raise ValueError("nonfinite terminal geometry")
                    if (geometry["weighted_integrated_variance"] <= 0 or geometry["weighted_log_conditional_probability"] > 0
                            or not 0 <= geometry["weighted_largest_left_variance_share"] <= 1
                            or not 0 <= geometry["weighted_peak_time_fraction"] < 1
                            or not 1 - 1e-10 <= geometry["terminal_weight_ess"] <= expected_diagnostic + 1e-8):
                        raise ValueError("terminal geometry bounds violated")
            expected = log_moments(torch.tensor(logs, dtype=torch.float64))
            expected_work = job["count"] * (exact_smc_work(schedule) + expected_diagnostic)
            if expected_diagnostic and row["mode_contributions"] != mode_contributions(row["whole_runs"]):
                raise ValueError("whole-run mode contribution arithmetic changed")
            expected_use_ids.update(f"{row['identity']}/{rep}/whole" for rep in range(job["count"]))
        if row["summary"] != expected or row["potential_evaluations"] != expected_work:
            raise ValueError("saved reference moments/work changed")
        use = next(u for u in payload["sample_uses"] if u["consumer_id"] == row["identity"] and u["seed_key"]["stream"] == "bootstrap")
        regenerated = sensitivity(logs, bootstrap_seed=ledger.lookup(SeedKey(**use["seed_key"])),
                                  bootstrap_replicates=config["qualification"]["bootstrap_replicates"])
        if row["sensitivity"] != regenerated:
            raise ValueError("whole-run/block sensitivity changed")
    # A failed job retains exactly its observed partial seed uses, never a pass.
    failed_ids = {r["identity"] for r in records if r["status"] != "completed"}
    expected_use_ids.update(u["use_id"] for u in payload["sample_uses"] if u["consumer_id"] in failed_ids)
    role_audit = audit_sample_uses(payload["sample_uses"], ledger, contracts, expected_use_ids=expected_use_ids)
    if payload["potential_evaluations"] != sum(r["potential_evaluations"] for r in records):
        raise ValueError("total reference work mismatch")
    if "diagnostic_path_evaluations" in payload and payload["diagnostic_path_evaluations"] != sum(
        r.get("diagnostic_path_evaluations", 0) for record in records for r in record["whole_runs"]
    ):
        raise ValueError("total diagnostic work mismatch")
    if payload["allocation"] != (allocate(payload) if config["mode"] == "pilot" else None) or payload["qualification"] != qualification(payload):
        raise ValueError("stale allocation/qualification")
    return {"status": "reference_binding_arithmetic_audit_pass_not_tail_certificate", "records": len(records),
            "roles": role_audit, "p2_authorized": payload["qualification"]["p2_authorized"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/post_audit/structural_v2_p1_pilot_v1.yaml")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--production-from", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(json.loads(args.audit.read_text(encoding="utf-8"))), allow_nan=False))
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.production_from:
        pilot = json.loads(args.production_from.read_text(encoding="utf-8"))
        audit(pilot)
        config = production_config(pilot, args.production_from)
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    extra = ("docs/plans/MODEL_STRUCTURAL_IMPROVEMENT_PLAN_V2_2026-10-08_KO.md",)
    if config.get("repair_protocol_path"):
        extra += (config["repair_protocol_path"],)
    source = freeze_source(output, config, extra_snapshot_paths=extra)
    payload = run(config, source)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "allocation": payload["allocation"],
                      "qualification": payload["qualification"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
