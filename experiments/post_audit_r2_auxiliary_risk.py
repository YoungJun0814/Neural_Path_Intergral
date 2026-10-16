"""Independent ordinary IS risk estimator with matched event/risk training banks."""

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
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.conditional_second_moment import (
    log_auxiliary_second_moment,
    log_risk_potential,
)
from src.path_integral.finite_rank_gaussian_transport import combine_defensive_gaussian_mixtures
from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import (
    fit_weighted_bank_mixture,
    proposal_from_parameters,
)
from src.path_integral.weighted_tempered_smc import estimate_weighted_tempered_normalizer

ROOT = Path(__file__).resolve().parents[1]


def run(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(config["torch_threads"])
    raw = (ROOT/config["proposal_artifact_path"]).read_bytes()
    original = json.loads(raw)
    guide_raw = (ROOT/config["guide_artifact_path"]).read_bytes() if config.get("guide_artifact_path") else None
    guide_payload = json.loads(guide_raw) if guide_raw else None
    protocol, ledger = "r2-aux-risk-"+canonical_digest(config)[:16], SeedLedger()
    start, work = time.perf_counter(), 0
    records = []
    repetitions = config.get("training_repetitions", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("invalid whole-training repetitions")
    jobs = [(old, fit_rep) for old in original["records"] for fit_rep in range(repetitions)]
    for old, fit_rep in jobs:
        row_start, row_work = time.perf_counter(), work
        item = next(x for x in old["candidates"] if x["method"] == config["method"])
        if canonical_digest(item["proposal_parameters"]) != item["proposal_digest"]:
            raise ValueError("q binding mismatch")
        q = proposal_from_parameters(item["proposal_parameters"])
        problem = _problem(original["config"]["model"], old["cell"], original["config"]["smc"]["steps"])
        row: dict[str, Any] = {"cell": old["cell"], "parent_training_rep": old["parent_training_rep"],
            "q_digest": item["proposal_digest"], "auxiliary_training_rep": fit_rep,
            "estimators": [], "training": [], "failure_reasons": []}
        records.append(row)

        def seed(role: str, regime: str, level: int, stream: str,
                 task: str = problem.task_id, rep: int = old["parent_training_rep"],
                 fitting_rep: int = fit_rep) -> int:
            return ledger.allocate(SeedKey(protocol, role, regime, task, level, rep,
                                           f"fit-{fitting_rep}-{stream}"))

        def check(n: int) -> None:
            if work+n > config["budget"]["max_potential_evaluations"]:
                raise TimeoutError("unresolved_potential_budget")
            if time.perf_counter()-start >= config["budget"]["max_wall_seconds"]:
                raise TimeoutError("unresolved_wall_budget")

        def payoff(x: torch.Tensor, fixed_problem: Any = problem) -> torch.Tensor:
            nonlocal work
            check(len(x))
            work += len(x)
            return _log_potential(fixed_problem, x)

        try:
            auxiliaries = [("direct-q", q)]
            static_guide = None
            if config.get("volterra_guide"):
                stage = time.perf_counter()
                static_guide = build_volterra_excursion_guide(problem, **config["volterra_guide"])
                row["closed_form_guide_parameters"] = proposal_parameters(static_guide)
                row["driver_basis_probe_paths"] = problem.local_dimension
                row["guide_construction_wall_seconds"] = time.perf_counter()-stage
                row["replication_unit"] = "fixed_r_iid_evaluation" if config.get("static_only") else "whole_auxiliary_training_and_final"
            kinds = () if config.get("static_only") else ("event", "risk")
            for kind in kinds:
                stage, stage_work = time.perf_counter(), work
                samples, weights, diagnostics = [], [], []
                spec = config["training"]
                guide = static_guide if spec.get("independence_every", 0) else None
                if guide_payload is not None:
                    guide_rows = [x for x in guide_payload["records"] if x["q_digest"] == row["q_digest"]]
                    members = [proposal_from_parameters(e["r_parameters"]) for x in guide_rows
                               for e in x["estimators"] if e["id"] == kind+"-auxiliary"]
                    if len(members) != guide_payload["config"].get("training_repetitions", 1):
                        raise ValueError("incomplete frozen guide ensemble")
                    guide = combine_defensive_gaussian_mixtures(tuple(members), tuple([1/len(members)]*len(members)))

                def potential(x: torch.Tensor, mode: str = kind, fixed_q: Any = q) -> torch.Tensor:
                    logg = payoff(x)
                    return logg if mode == "event" else log_risk_potential(logg, fixed_q.log_q_over_p(x),
                                                                         defensive_mass=fixed_q.defensive_mass)

                for island in range(spec["islands"]):
                    check(smc_work(spec, spec["particles"]))
                    bank = estimate_weighted_tempered_normalizer(potential, dimension=problem.local_dimension,
                        config=smc_config(spec, seed=seed("auxiliary-training", kind, island, "whole"), retain=True),
                        independence_proposal=guide)
                    if bank.final_particles is None or bank.final_weights is None:
                        raise RuntimeError("missing auxiliary training bank")
                    samples.append(bank.final_particles)
                    weights.append(bank.final_weights/spec["islands"])
                    diagnostics.extend(bank.replicate_diagnostics)
                fit = fit_weighted_bank_mixture(torch.cat(samples), torch.cat(weights), **config["mixture"])
                members = [fit.proposal]
                if config.get("island_ensemble", False):
                    members.extend(fit_weighted_bank_mixture(x, w, **config["mixture"]).proposal
                                   for x, w in zip(samples, weights, strict=True))
                    family_weights = (.5, *([.5/spec["islands"]]*spec["islands"]))
                    auxiliary = combine_defensive_gaussian_mixtures(tuple(members), family_weights)
                else:
                    family_weights, auxiliary = (1.,), fit.proposal
                if static_guide is not None and config.get("guide_output_mass", 0):
                    mass = config["guide_output_mass"]
                    auxiliary = combine_defensive_gaussian_mixtures((auxiliary, static_guide), (1-mass, mass))
                auxiliaries.append((kind+"-auxiliary", auxiliary))
                row["training"].append({"kind": kind, "potential_evaluations": work-stage_work,
                    "wall_seconds": time.perf_counter()-stage, "weighted_particle_ess": fit.weighted_bank_ess,
                    "whole_island_diagnostics": diagnostics,
                    "guide_parameters": proposal_parameters(guide) if guide is not None else None,
                    "fit_member_parameters": [proposal_parameters(m) for m in members],
                    "fit_member_weights": list(family_weights),
                    "cluster_masses": list(fit.cluster_masses)})
            if static_guide is not None and config.get("static_control", False):
                auxiliaries.append(("volterra-only", static_guide))
            # r candidates all frozen before any final risk observations.
            for name, r in auxiliaries:
                row["estimators"].append({"id": name, "r_parameters": proposal_parameters(r),
                    "r_digest": canonical_digest(proposal_parameters(r)), "batches": [], "failure_reasons": []})
            for (name, r), result in zip(auxiliaries, row["estimators"], strict=True):
                stage, stage_work = time.perf_counter(), work
                logs = []
                try:
                    spec = config["evaluation"]
                    count = spec.get("counts", {}).get(name, spec["count"])
                    for batch in range(math.ceil(count/spec["batch_size"])):
                        n = min(spec["batch_size"], count-batch*spec["batch_size"])
                        check(n)
                        draw = r.sample(n, path_seed=seed("auxiliary-final", name, batch, "path"),
                                        label_seed=seed("auxiliary-final", name, batch, "label"))
                        values = log_auxiliary_second_moment(payoff(draw.samples), q.log_q_over_p(draw.samples),
                                                             draw.log_q_over_p)
                        logs.append(values)
                        result["batches"].append(asdict(summarize_log_contributions(values)))
                    all_logs = torch.cat(logs)
                    result["summary"] = asdict(summarize_log_contributions(all_logs))
                    if spec.get("fixed_r_block_count"):
                        if count % spec["fixed_r_block_count"]:
                            raise ValueError("fixed-r independent blocks require equal predetermined sizes")
                        result["fixed_r_independent_blocks"] = [asdict(summarize_log_contributions(x))
                            for x in all_logs.chunk(spec["fixed_r_block_count"])]
                    result["precision_pass"] = result["summary"]["relative_se"] <= spec["maximum_relative_se"]
                    result["status"] = "completed_development"
                except (ValueError, FloatingPointError, RuntimeError, TimeoutError) as error:
                    result["status"] = "unresolved"
                    result["precision_pass"] = False
                    result["failure_reasons"].append(f"{type(error).__name__}: {error}")
                result["potential_evaluations"] = work-stage_work
                result["wall_seconds_including_failure"] = time.perf_counter()-stage
            row["status"] = "completed_development"
        except (ValueError, FloatingPointError, RuntimeError, TimeoutError) as error:
            row["status"] = "unresolved"
            row["failure_reasons"].append(f"{type(error).__name__}: {error}")
        row["potential_evaluations"] = work-row_work
        row["wall_seconds_including_failure"] = time.perf_counter()-row_start
        print(json.dumps({"finished": [old["cell"]["id"], old["parent_training_rep"], fit_rep], "status": row["status"],
            "risk_rse": next((e.get("summary", {}).get("relative_se") for e in row["estimators"]
                              if e["id"] == "risk-auxiliary"), None)}), flush=True)
    if source_tree_digest(ROOT) != source["source_tree_digest"]:
        raise RuntimeError("source changed during experiment")
    return {"schema": "npi.post-audit.r2-aux-risk.v1", "config": config, "source": source,
        "proposal_artifact_sha256": hashlib.sha256(raw).hexdigest(), "records": records,
        "guide_artifact_sha256": hashlib.sha256(guide_raw).hexdigest() if guide_raw else None,
        "inherited_guide_training_potential_evaluations": sum(t["potential_evaluations"]
            for r in guide_payload["records"] for t in r["training"]) if guide_payload else 0,
        "seed_ledger": ledger.to_dict(), "potential_evaluations": work, "physical_wall_seconds": time.perf_counter()-start,
        "model_q_changed": False, "performance_claim_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/post_audit/r2_auxiliary_risk_crosscheck_v1.yaml")
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
    result = run(config, source)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "work": result["potential_evaluations"], "wall_seconds": result["physical_wall_seconds"]}), flush=True)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    config, source = payload["config"], payload["source"]
    torch.set_num_threads(config["torch_threads"])
    if payload["schema"] != "npi.post-audit.r2-aux-risk.v1" or canonical_digest(config) != source["config_digest"]:
        raise ValueError("schema/config mismatch")
    archive_path = ROOT/source["snapshot_path"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["snapshot_sha256"]:
        raise ValueError("source snapshot mismatch")
    with zipfile.ZipFile(archive_path) as z:
        if json.loads(z.read("RUN_CONFIG.json")) != config:
            raise ValueError("archive config mismatch")
        for name, sha in source["snapshot_file_hashes"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("snapshot entry mismatch")
    raw = (ROOT/config["proposal_artifact_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != payload["proposal_artifact_sha256"]:
        raise ValueError("q source changed")
    original = json.loads(raw)
    guide_raw = (ROOT/config["guide_artifact_path"]).read_bytes() if config.get("guide_artifact_path") else None
    guide_payload = json.loads(guide_raw) if guide_raw else None
    if guide_raw and hashlib.sha256(guide_raw).hexdigest() != payload["guide_artifact_sha256"]:
        raise ValueError("guide artifact changed")
    if guide_payload and payload["inherited_guide_training_potential_evaluations"] != sum(
        t["potential_evaluations"] for r in guide_payload["records"] for t in r["training"]):
        raise ValueError("inherited guide training cost mismatch")
    expected = {(x["cell"]["id"], x["parent_training_rep"]): x for x in original["records"]}
    repetitions = config.get("training_repetitions", 1)
    if len(expected)*repetitions != len(payload["records"]):
        raise ValueError("missing parent")
    identities = set()
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    used_seeds: set[int] = set()
    for row in payload["records"]:
        key = (row["cell"]["id"], row["parent_training_rep"])
        identity = (*key, row.get("auxiliary_training_rep", 0))
        if key not in expected or identity in identities or not 0 <= identity[2] < repetitions:
            raise ValueError("duplicate/undeclared parent")
        identities.add(identity)
        q_digest = next(x["proposal_digest"] for x in expected[key]["candidates"] if x["method"] == config["method"])
        if q_digest != row["q_digest"]:
            raise ValueError("q changed")
        problem = _problem(original["config"]["model"], expected[key]["cell"], original["config"]["smc"]["steps"])
        static_guide = build_volterra_excursion_guide(problem, **config["volterra_guide"]) if config.get("volterra_guide") else None
        if static_guide is not None and (row["closed_form_guide_parameters"] != proposal_parameters(static_guide)
                or row["driver_basis_probe_paths"] != problem.local_dimension):
            raise ValueError("closed-form guide binding mismatch")

        def require_seed(role: str, regime: str, level: int, stream: str,
                         fixed_row: dict[str, Any] = row, task: str = problem.task_id) -> None:
            if "auxiliary_training_rep" in fixed_row:
                stream = f"fit-{fixed_row['auxiliary_training_rep']}-{stream}"
            value = ledger.lookup(SeedKey("r2-aux-risk-"+canonical_digest(config)[:16], role,
                regime, task, level, fixed_row["parent_training_rep"], stream))
            if value in used_seeds:
                raise ValueError("random stream reused")
            used_seeds.add(value)

        for training in row["training"]:
            if static_guide is not None and config["training"].get("independence_every", 0):
                if training["guide_parameters"] != proposal_parameters(static_guide):
                    raise ValueError("static global guide changed")
            if guide_payload is not None:
                members = [proposal_from_parameters(e["r_parameters"]) for g in guide_payload["records"]
                           if g["q_digest"] == q_digest for e in g["estimators"]
                           if e["id"] == training["kind"]+"-auxiliary"]
                guide = combine_defensive_gaussian_mixtures(tuple(members), tuple([1/len(members)]*len(members)))
                if proposal_parameters(guide) != training["guide_parameters"]:
                    raise ValueError("guide density/ensemble mismatch")
            for island in range(config["training"]["islands"]):
                require_seed("auxiliary-training", training["kind"], island, "whole")
        for estimator in row["estimators"]:
            params = estimator["r_parameters"]
            if canonical_digest(params) != estimator["r_digest"] or proposal_from_parameters(params).defensive_mass < .1-1e-12:
                raise ValueError("auxiliary density mismatch")
            if estimator["id"] == "direct-q" and estimator["r_digest"] != q_digest:
                raise ValueError("direct-q comparator changed")
            if estimator["id"] == "volterra-only" and (static_guide is None or params != proposal_parameters(static_guide)):
                raise ValueError("fixed static density changed")
            if estimator["status"] != "completed_development":
                if estimator["precision_pass"] or not estimator["failure_reasons"]:
                    raise ValueError("stale precision qualification")
                continue
            parts = estimator["batches"]
            for batch in range(len(parts)):
                require_seed("auxiliary-final", estimator["id"], batch, "path")
                require_seed("auxiliary-final", estimator["id"], batch, "label")
            n = sum(x["count"] for x in parts)
            lm = float(torch.logsumexp(torch.tensor([x["log_mean"]+math.log(x["count"]) for x in parts], dtype=torch.float64), 0)-math.log(n))
            l2 = float(torch.logsumexp(torch.tensor([x["log_second_moment"]+math.log(x["count"]) for x in parts], dtype=torch.float64), 0)-math.log(n))
            rse = math.sqrt(max(0., math.expm1(l2-2*lm))/(n-1))
            summary = estimator["summary"]
            expected_count = config["evaluation"].get("counts", {}).get(estimator["id"], config["evaluation"]["count"])
            if (n != expected_count or summary["count"] != n
                    or not math.isclose(lm, summary["log_mean"], abs_tol=1e-12)
                    or not math.isclose(rse, summary["relative_se"], rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("auxiliary IID moment mismatch")
            if config["evaluation"].get("fixed_r_block_count"):
                blocks = estimator["fixed_r_independent_blocks"]
                block_count = config["evaluation"]["fixed_r_block_count"]
                if len(blocks) != block_count or any(b["count"] != n//block_count for b in blocks):
                    raise ValueError("fixed-r independent block contract mismatch")
                if len(parts) % block_count:
                    raise ValueError("block boundaries must align with audited batches")
                width = len(parts)//block_count
                for i, block in enumerate(blocks):
                    block_parts = parts[i*width:(i+1)*width]
                    bn = sum(p["count"] for p in block_parts)
                    blm = float(torch.logsumexp(torch.tensor([p["log_mean"]+math.log(p["count"])
                        for p in block_parts], dtype=torch.float64), 0)-math.log(bn))
                    bl2 = float(torch.logsumexp(torch.tensor([p["log_second_moment"]+math.log(p["count"])
                        for p in block_parts], dtype=torch.float64), 0)-math.log(bn))
                    brse = math.sqrt(max(0., math.expm1(bl2-2*blm))/(bn-1))
                    if not math.isclose(blm, block["log_mean"], abs_tol=1e-12) or not math.isclose(
                        brse, block["relative_se"], rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError("fixed-r block moments mismatch")
            if estimator["precision_pass"] != (summary["relative_se"] <= config["evaluation"]["maximum_relative_se"]):
                raise ValueError("precision flag mismatch")
            if estimator["potential_evaluations"] != n:
                raise ValueError("inference cost mismatch")
        if row["status"] == "completed_development":
            expected_ids = {"direct-q"} if config.get("static_only") else {"direct-q", "event-auxiliary", "risk-auxiliary"}
            if config.get("static_control"):
                expected_ids.add("volterra-only")
            if {e["id"] for e in row["estimators"]} != expected_ids:
                raise ValueError("missing comparator")
            training_work = config["training"]["islands"]*smc_work(config["training"], config["training"]["particles"])
            expected_training = 0 if config.get("static_only") else 2
            if len(row["training"]) != expected_training or any(t["potential_evaluations"] != training_work for t in row["training"]):
                raise ValueError("unmatched training budget")
            if config.get("island_ensemble", False):
                for t in row["training"]:
                    members = tuple(proposal_from_parameters(p) for p in t["fit_member_parameters"])
                    expected_weights = [.5, *([.5/config["training"]["islands"]]*config["training"]["islands"])]
                    if len(members) != 1+config["training"]["islands"] or t["fit_member_weights"] != expected_weights:
                        raise ValueError("island ensemble construction mismatch")
                    combined = combine_defensive_gaussian_mixtures(members, tuple(expected_weights))
                    if static_guide is not None and config.get("guide_output_mass", 0):
                        mass = config["guide_output_mass"]
                        combined = combine_defensive_gaussian_mixtures((combined, static_guide), (1-mass, mass))
                    final = next(e for e in row["estimators"] if e["id"] == t["kind"]+"-auxiliary")
                    if proposal_parameters(combined) != final["r_parameters"]:
                        raise ValueError("island ensemble density mismatch")
            if sum(t["potential_evaluations"] for t in row["training"])+sum(e["potential_evaluations"] for e in row["estimators"]) != row["potential_evaluations"]:
                raise ValueError("row work mismatch")
    if sum(x["potential_evaluations"] for x in payload["records"]) != payload["potential_evaluations"]:
        raise ValueError("total work mismatch")
    if payload["model_q_changed"] or payload["performance_claim_authorized"] or payload["potential_evaluations"] > config["budget"]["max_potential_evaluations"]:
        raise ValueError("model/performance/budget violation")
    if all(r["status"] == "completed_development" and all(e["status"] == "completed_development"
           for e in r["estimators"]) for r in payload["records"]) and len(used_seeds) != len(ledger.records):
        raise ValueError("unused/undeclared random streams")
    return {"status": "source_and_arithmetic_pass_not_statistical_certificate", "parents": len(expected),
            "whole_training_jobs": 0 if config.get("static_only") else len(identities),
            "fixed_r_evaluation_repetitions": len(identities) if config.get("static_only") else 0,
            "seed_streams": len(ledger.records)}


if __name__ == "__main__":
    main()
