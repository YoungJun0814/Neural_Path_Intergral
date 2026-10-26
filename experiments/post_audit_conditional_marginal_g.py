"""One-shot bounded marginal-G risk evaluation; no event-q model replacement."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Any

import torch

from experiments.post_audit_analytic_residual_batch import Budget
from experiments.post_audit_analytic_residual_recovery import DIRECTORY as PARENT
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_residual_paired_pilot import DIRECTORY as Q1
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.analytic_residual_batch import batched_moments
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.conditional_marginal_g import (
    aggregate_error,
    blend,
    consistent,
    fit_marginal,
    json_compatible,
    moment_error,
    precision,
    risk_training_logs,
    select_alpha,
    selection_logs,
    training_gate,
)
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

DIRECTORY = "results/post_audit/conditional_marginal_g_v1/session_20261021"
PLAN = "docs/plans/MARGINAL_G_CONDITIONAL_RISK_DEVELOPMENT_2026-10-21_KO.md"
THEORY = "docs/theory/ANALYTIC_RESIDUAL_MARGINAL_G_SCOPE_ADDENDUM_2026-10-21_KO.md"
NAMESPACE = "conditional-marginal-G-development-v1"
EXPECTED = "a0a6a3b39c475fcf5fa92a2a6a652fdf21c3da1a311c56dbda2b0888f2896f54"


def evidence() -> dict[str, Any]:
    path = ROOT/PARENT
    result, receipt = [json.loads((path/name).read_text()) for name in ("result.json", "audit.json")]
    if sha(path/"result.json") != EXPECTED or receipt["result_sha256"] != EXPECTED or result["P2_qualified"]:
        raise ValueError("immutable audited diagnostic-only parent required")
    audit_snapshot(ROOT, result["source"], result["config"])
    if sha(ROOT/Q1/"result.json") != result["source"]["snapshot_file_hashes"][Q1+"/result.json"]:
        raise ValueError("fixed Q1 model/seed/config anchor changed")
    return json.loads((ROOT/Q1/"result.json").read_text())


def identity(cell: str, rep: int, phase: str, block: int) -> str:
    return f"{cell}/{rep}/{phase}/{block}"


def seed_layout(cells: list[str], prior: dict[str, Any]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    ledger, mapping = SeedLedger(), {}
    for cell in cells:
        requests = [(0, phase, b) for phase in ("ref-q", "ref-p") for b in range(8)]
        requests += [(rep, "train", b) for rep in range(2) for b in range(8)]
        requests += [(rep, "selection", 0) for rep in range(2)]
        requests += [(rep, phase, b) for rep in range(2) for phase in ("final-q", "final-selected") for b in range(8)]
        for rep, phase, block in requests:
            mapping[identity(cell, rep, phase, block)] = {
                role: ledger.allocate(SeedKey(NAMESPACE, phase, "development", cell, 32, rep, f"{block}/{role}"))
                for role in ("path", "label")}
    old = SeedLedger.from_dict(prior["seeds"]["ledger"])
    if {r.seed for r in ledger.records} & {r.seed for r in old.records}:
        raise ValueError("new/Q1 scoped seed collision")
    return mapping, {"ledger": ledger.to_dict(), "sha256": ledger.sha256,
        "previous_seeds_compared": len(old), "scope": "all new streams and immediately preceding Q1; not repository-wide",
        "collision_count": 0}


def evaluate_bank(cache: Any, q: GaussianMixtureShiftSpec,
                  sampler: DefensiveFiniteRankGaussianMixture, streams: dict[str, int],
                  count: int, conditional: bool, budget: Budget) -> dict[str, Any]:
    started = time.perf_counter()
    clock = time.perf_counter()
    draw = sampler.sample(count, path_seed=streams["path"], label_seed=streams["label"])
    z = draw.samples
    draw_wall = time.perf_counter()-clock
    clock = time.perf_counter()
    dq = torch.logsumexp(z @ q.means.T-.5*q.means.square().sum(1)+q.weights.log(), 1)
    ds = draw.log_q_over_p
    density_wall = time.perf_counter()-clock
    rows: list[dict[str, Any]] = []
    profile_wall, quadrature_wall, raw_wall = 0., 0., 0.
    scalar_before = budget.scalar
    if conditional:
        for offset in range(0, count, 128):
            group = z[offset:offset+128]
            profiles = []
            clock = time.perf_counter()
            for point in group:
                budget.physics(1)
                profiles.append(LastVarianceResidualSlice.create(cache, point[None]))
            profile_wall += time.perf_counter()-clock
            clock = time.perf_counter()
            calculated = batched_moments(profiles, charge=budget.charge)
            quadrature_wall += time.perf_counter()-clock
            for j, (profile, moment) in enumerate(zip(profiles, calculated["rows"], strict=True), offset):
                rows.append({"coordinates": z[j].tolist(), "label": int(draw.labels[j]),
                    "log_dq": float(dq[j]), "log_ds": float(ds[j]),
                    "profile": dataclasses.asdict(profile), "quadrature": moment,
                    **moment_error(moment)})
    else:
        clock = time.perf_counter()
        budget.physics(count)
        logg = cache.log_probability(z)
        raw_wall = time.perf_counter()-clock
        rows = [{"coordinates": z[j].tolist(), "label": int(draw.labels[j]),
            "log_dq": float(dq[j]), "log_ds": float(ds[j]), "log_g": float(logg[j]),
            "log_score": float(2*logg[j]-dq[j]-ds[j])} for j in range(count)]
    budget.charge(0)
    return json_compatible({"seeds": streams, "count": count, "conditional": conditional,
        "rows": rows, "sampler_digest": canonical_digest(proposal_parameters(sampler)),
        "physical_evaluations": count, "scalar_evaluations": budget.scalar-scalar_before,
        "cost": {"draw": draw_wall, "density": density_wall, "profile": profile_wall,
            "quadrature": quadrature_wall, "raw_payoff": raw_wall, "total": time.perf_counter()-started}})


def arrays(banks: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    rows = [r for b in banks for r in b["rows"]]
    keys = ("log_m2", "log_error_proxy", "log_dq", "log_ds") if banks[0]["conditional"] else ("log_score",)
    return {key: torch.tensor([r[key] for r in rows], dtype=torch.float64) for key in keys}


def train_summary(banks: list[dict[str, Any]]) -> dict[str, Any]:
    a = arrays(banks)
    denominator = a["log_dq"]+a["log_ds"]
    w = risk_training_logs(a["log_m2"], a["log_dq"], a["log_ds"])
    error = aggregate_error(a["log_m2"], a["log_error_proxy"], denominator)
    rows = [r for b in banks for r in b["rows"]]
    gate = training_gate(w, error, all(r["structural_numerics_pass"] for r in rows))
    natural = torch.tensor([r["label"] == 0 for r in rows])
    weights = w.softmax(0)
    return {"gate": gate, "ordinary_log_mean_f": float(w.logsumexp(0))-math.log(len(w)),
        "individual_relative_unresolved": sum(not r["individual_relative_resolved"] for r in rows),
        "natural_component_draws": int(natural.sum()), "risk_weight_fraction_in_natural_draws": float(weights[natural].sum()),
        "component_label_is_not_unique_physical_mode": True}


def endpoint(banks: list[dict[str, Any]]) -> dict[str, Any]:
    a = arrays(banks)
    if not banks[0]["conditional"]:
        return precision(a["log_score"])
    structural = all(r["structural_numerics_pass"] for b in banks for r in b["rows"])
    d = a["log_dq"]+a["log_ds"]
    result = precision(a["log_m2"]-d,
        numerical_proxy=aggregate_error(a["log_m2"], a["log_error_proxy"], d))
    result["structural_pass"] = structural
    result["qualified"] = result["qualified"] and structural
    return result


def final_summary(fitrecord: dict[str, Any], baseline: dict[str, Any], selected: dict[str, Any],
                  reference: dict[str, Any]) -> dict[str, Any]:
    scale = 2*(selected["log_mean"]-baseline["log_mean"])
    ratio_log = (scale+2*math.log(selected["block_rse"])-2*math.log(baseline["block_rse"])) if min(
        baseline["block_rse"], selected["block_rse"]) > 0 else None
    ratio = math.exp(min(690., ratio_log)) if ratio_log is not None else 1e300
    return {**fitrecord, "baseline": baseline, "selected": selected,
        "block_variance_ratio_proxy": ratio,
        "reference_consistency": all(consistent(e, r) for e in (baseline, selected) for r in reference["methods"].values())}


def selection_summary(selection: dict[str, Any], q: GaussianMixtureShiftSpec,
                      fitted: DefensiveFiniteRankGaussianMixture) -> tuple[list[float], list[float], bool]:
    a = arrays([selection])
    z = torch.tensor([r["coordinates"] for r in selection["rows"]], dtype=torch.float64)
    objectives, errors = [], []
    for alpha in (0., .25, .5):
        dg = blend(q, fitted, alpha).log_q_over_p(z)
        values = selection_logs(a["log_m2"], a["log_dq"], a["log_ds"], dg)
        objectives.append(float(values.logsumexp(0))-math.log(len(values)))
        errors.append(aggregate_error(a["log_m2"], a["log_error_proxy"], 2*a["log_dq"]+a["log_ds"]+dg, squared=True))
    passed = max(errors) <= 1e-6 and all(r["structural_numerics_pass"] for r in selection["rows"])
    return objectives, errors, passed


def decide(training: list[dict[str, Any]], references: list[dict[str, Any]],
           finals: list[dict[str, Any]]) -> dict[str, Any]:
    if len(training) != 4 or len(references) != 2:
        raise ValueError("complete four-bank/two-cell diagnostic required")
    passed = all(t["summary"]["gate"]["pass"] for t in training)
    if not passed:
        if finals:
            raise ValueError("training STOP forbids final production")
        decision = "STOP_MARGINAL_TRAINING_BANK_UNRELIABLE"
    else:
        if len(finals) != 4:
            raise ValueError("four complete final fits required after bank gate")
        qualified = all(r["qualified"] for r in references)
        qualified = qualified and all(f["baseline"]["qualified"] and f["selected"]["qualified"]
            and f["reference_consistency"] and f["selection_numerics_pass"] for f in finals)
        candidate = qualified and all(f["alpha"] > 0 and f["block_variance_ratio_proxy"] <= .8 for f in finals)
        decision = "CANDIDATE_FOR_SEPARATE_CONFIRMATION_ONLY" if candidate else "NO_ADOPTION_MARGINAL_G_DEVELOPMENT"
    return {"decision": decision, "training_gate_passed": passed, "new_event_q_adopted": False,
        "new_G_adopted": False, "P2_qualified": False, "D3_authorized": False,
        "large_training_authorized": False, "not_fixed_precision_or_population_speedup": True}


def run() -> dict[str, Any]:
    end_start = time.perf_counter()
    torch.set_num_threads(1)
    prior = evidence()
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("fresh one-shot marginal-G study; no overwrite")
    path.mkdir(parents=True, exist_ok=True)
    config = {"namespace": NAMESPACE, "model": prior["config"]["model"], "cells": prior["config"]["cells"],
        "parent_sha256": EXPECTED, "training_reps": 2, "training_blocks": 8, "training_block_count": 256,
        "reference_blocks": 8, "reference_block_count": 512, "selection_count": 512,
        "final_blocks": 8, "final_block_count": 128, "physics_cap": 40000,
        "scalar_cap": 120000000, "wall_cap": 1800, "alphas": [0., .25, .5]}
    clock = time.perf_counter()
    prepared, setup = prepare(config)
    setup_wall = time.perf_counter()-clock
    mapping, ledger = seed_layout(list(prepared), prior)
    clock = time.perf_counter()
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, THEORY,
        PARENT+"/result.json", PARENT+"/audit.json", Q1+"/result.json"))
    source_wall = time.perf_counter()-clock
    clock = time.perf_counter()
    write_json_atomic_nonoverwriting(path/"protocol.json", json_compatible({"config": config, "source": source,
        "seeds": ledger, "mapping": mapping}))
    protocol_io_wall = time.perf_counter()-clock
    budget = Budget(120000000, 40000, 1800)
    manifests: list[dict[str, Any]] = []
    io_wall = 0.
    statistics_wall = 0.

    def save(name: str, payload: dict[str, Any]) -> None:
        nonlocal io_wall
        clock = time.perf_counter()
        write_json_atomic_nonoverwriting(path/name, json_compatible(payload))
        io_wall += time.perf_counter()-clock
        manifests.append({"path": name, "sha256": sha(path/name)})

    training: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    train_banks: dict[tuple[str, int], list[dict[str, Any]]] = {}
    try:
        for cell, (cache, q) in prepared.items():
            sampler = blend(q, None, 0.)
            for rep in range(2):
                banks = []
                for b in range(8):
                    value = evaluate_bank(cache, q, sampler, mapping[identity(cell, rep, "train", b)], 256, True, budget)
                    save(f"train-{cell}-{rep}-{b}.json", value)
                    banks.append(value)
                clock = time.perf_counter()
                summary = train_summary(banks)
                statistics_wall += time.perf_counter()-clock
                training.append({"cell": cell, "rep": rep, "summary": summary})
                train_banks[cell, rep] = banks
                print(f"Training {cell}/{rep}: ESS={summary['gate']['ess']:.3f}, pass={summary['gate']['pass']}", flush=True)
        for cell, (cache, q) in prepared.items():
            methods: dict[str, Any] = {}
            for phase in ("ref-q", "ref-p"):
                sampler = blend(q, None, 0.) if phase == "ref-q" else DefensiveFiniteRankGaussianMixture(
                    (FiniteRankGaussianComponent.natural(64),), torch.ones(1, dtype=torch.float64))
                banks = []
                for b in range(8):
                    value = evaluate_bank(cache, q, sampler, mapping[identity(cell, 0, phase, b)], 512, False, budget)
                    save(f"{phase}-{cell}-{b}.json", value)
                    banks.append(value)
                clock = time.perf_counter()
                methods[phase] = endpoint(banks)
                statistics_wall += time.perf_counter()-clock
            agreement = consistent(methods["ref-q"], methods["ref-p"])
            references.append({"cell": cell, "methods": methods, "mean_consistency": agreement,
                "qualified": agreement and all(m["qualified"] for m in methods.values())})
        if all(t["summary"]["gate"]["pass"] for t in training):
            for t in training:
                cell, rep = t["cell"], t["rep"]
                cache, q = prepared[cell]
                banks = train_banks[cell, rep]
                a = arrays(banks)
                samples = torch.tensor([r["coordinates"] for bank in banks for r in bank["rows"]], dtype=torch.float64)
                clock = time.perf_counter()
                fitted, info = fit_marginal(samples, risk_training_logs(a["log_m2"], a["log_dq"], a["log_ds"]))
                fit_wall = time.perf_counter()-clock
                candidates = [blend(q, fitted, alpha) for alpha in (0., .25, .5)]
                selection = evaluate_bank(cache, q, candidates[-1], mapping[identity(cell, rep, "selection", 0)], 512, True, budget)
                clock = time.perf_counter()
                objectives, errors, selection_pass = selection_summary(selection, q, fitted)
                selection_score_wall = time.perf_counter()-clock
                selected_alpha = select_alpha(objectives)
                save(f"selection-{cell}-{rep}.json", selection)
                if not selection_pass:
                    raise FloatingPointError("frozen selection aggregate numerical gate failed")
                fitrecord = {"cell": cell, "rep": rep, "proposal": proposal_parameters(fitted), "info": info,
                    "fit_wall": fit_wall, "selection_score_wall": selection_score_wall,
                    "selection_objectives": objectives, "selection_error_proxies": errors,
                    "alpha": selected_alpha, "selection_numerics_pass": selection_pass}
                save(f"fit-{cell}-{rep}.json", fitrecord)
                endpoints = {}
                for phase, candidate in (("final-q", candidates[0]), ("final-selected", blend(q, fitted, selected_alpha))):
                    final_banks = []
                    for b in range(8):
                        value = evaluate_bank(cache, q, candidate, mapping[identity(cell, rep, phase, b)], 128, True, budget)
                        save(f"{phase}-{cell}-{rep}-{b}.json", value)
                        final_banks.append(value)
                    clock = time.perf_counter()
                    endpoints[phase] = endpoint(final_banks)
                    statistics_wall += time.perf_counter()-clock
                baseline, selected = endpoints["final-q"], endpoints["final-selected"]
                reference = next(r for r in references if r["cell"] == cell)
                finals.append(final_summary(fitrecord, baseline, selected, reference))
        clock = time.perf_counter()
        summary = decide(training, references, finals)
        decision_aggregation_wall = time.perf_counter()-clock
        result = json_compatible({"source": source, "config": config, "seeds": ledger, "artifacts": manifests,
            "training": training, "references": references, "finals": finals, "summary": summary,
            "physical_evaluations": budget.physical, "scalar_evaluations": budget.scalar,
            "cost": {"setup": setup, "setup_wall": setup_wall, "source_freeze_wall": source_wall,
                "protocol_io_wall": protocol_io_wall, "artifact_io_wall": io_wall,
                "bank_statistics_wall": statistics_wall,
                "decision_aggregation_wall": decision_aggregation_wall,
                "compute_and_artifact_io": time.perf_counter()-budget.started},
            "unexecuted_fit_selection_final_not_reallocated": not summary["training_gate_passed"]})
        clock = time.perf_counter()
        write_json_atomic_nonoverwriting(path/"result.json", result)
        final_io = time.perf_counter()-clock
        write_json_atomic_nonoverwriting(path/"end_to_end.json", {"result_sha256": sha(path/"result.json"),
            "end_to_end_wall_before_receipt": time.perf_counter()-end_start,
            "final_result_io_wall": final_io, "audit_and_tests_excluded": True})
    except Exception as error:
        write_json_atomic_nonoverwriting(path/"failure.json", {"error": type(error).__name__+": "+str(error),
            "physical": budget.physical, "scalar": budget.scalar, "completed_artifacts": manifests,
            "wall_before_failure_receipt": time.perf_counter()-end_start,
            "P2_qualified": False, "large_training_authorized": False})
        raise
    return result


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    prior = evidence()
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, digest in result["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("frozen marginal-G source changed")
    prepared, _ = prepare(result["config"])
    mapping, ledger = seed_layout(list(prepared), prior)
    if ledger != result["seeds"]:
        raise ValueError("sealed stream ledger changed")
    entries = {}
    for item in result["artifacts"]:
        if sha(path/item["path"]) != item["sha256"] or item["path"] in entries:
            raise ValueError("artifact missing/changed/duplicate")
        entries[item["path"]] = json.loads((path/item["path"]).read_text())
    schedule: dict[str, tuple[str, int, str, int]] = {}
    expected_names = set()
    fitted_proposals = {}
    for cell, (_, q) in prepared.items():
        for rep in range(2):
            for b in range(8):
                schedule[f"train-{cell}-{rep}-{b}.json"] = cell, rep, "train", b
        for phase in ("ref-q", "ref-p"):
            for b in range(8):
                schedule[f"{phase}-{cell}-{b}.json"] = cell, 0, phase, b
        if result["finals"]:
            for rep in range(2):
                name = f"fit-{cell}-{rep}.json"
                expected_names.add(name)
                saved = entries[name]
                banks = [entries[f"train-{cell}-{rep}-{b}.json"] for b in range(8)]
                a = arrays(banks)
                samples = torch.tensor([r["coordinates"] for bank in banks for r in bank["rows"]], dtype=torch.float64)
                fresh, info = fit_marginal(samples, risk_training_logs(a["log_m2"], a["log_dq"], a["log_ds"]))
                if proposal_parameters(fresh) != saved["proposal"] or info != saved["info"]:
                    raise ValueError("all fit geometry/parameters replay failed")
                fitted_proposals[cell, rep] = proposal_from_parameters(saved["proposal"])
                selection_name = f"selection-{cell}-{rep}.json"
                schedule[selection_name] = cell, rep, "selection", 0
                objectives, errors, valid = selection_summary(entries[selection_name], q, fresh)
                if (objectives != saved["selection_objectives"] or errors != saved["selection_error_proxies"]
                        or valid != saved["selection_numerics_pass"] or select_alpha(objectives) != saved["alpha"]):
                    raise ValueError("complete selection replay failed")
                for phase in ("final-q", "final-selected"):
                    for b in range(8):
                        schedule[f"{phase}-{cell}-{rep}-{b}.json"] = cell, rep, phase, b
                baseline = endpoint([entries[f"final-q-{cell}-{rep}-{b}.json"] for b in range(8)])
                selected = endpoint([entries[f"final-selected-{cell}-{rep}-{b}.json"] for b in range(8)])
                reference = next(r for r in result["references"] if r["cell"] == cell)
                final = next(f for f in result["finals"] if f["cell"] == cell and f["rep"] == rep)
                if final_summary(saved, baseline, selected, reference) != final:
                    raise ValueError("all final summaries changed")
    expected_names.update(schedule)
    if set(entries) != expected_names:
        raise ValueError("complete branch artifact inventory mismatch")
    budget = Budget(10000000, 3000, 600)
    replays = 0
    for cell, (cache, q) in prepared.items():
        for rep in range(2):
            banks = [entries[f"train-{cell}-{rep}-{b}.json"] for b in range(8)]
            saved_summary = next(t["summary"] for t in result["training"] if t["cell"] == cell and t["rep"] == rep)
            if train_summary(banks) != saved_summary:
                raise ValueError("all training statistic/gate replay failed")
        for phase in ("ref-q", "ref-p"):
            banks = [entries[f"{phase}-{cell}-{b}.json"] for b in range(8)]
            reference = next(r for r in result["references"] if r["cell"] == cell)
            if endpoint(banks) != reference["methods"][phase]:
                raise ValueError("all reference statistic replay failed")
            sampler = blend(q, None, 0.) if phase == "ref-q" else DefensiveFiniteRankGaussianMixture(
                (FiniteRankGaussianComponent.natural(64),), torch.ones(1, dtype=torch.float64))
            replay = evaluate_bank(cache, q, sampler, mapping[identity(cell, 0, phase, 0)], 512, False, budget)
            if any(replay[k] != banks[0][k] for k in replay if k != "cost"):
                raise ValueError("four raw reference block replays failed")
            replays += 512
        reference = next(r for r in result["references"] if r["cell"] == cell)
        agreement = consistent(reference["methods"]["ref-q"], reference["methods"]["ref-p"])
        if (reference["mean_consistency"] != agreement or reference["qualified"] != (
                agreement and all(m["qualified"] for m in reference["methods"].values()))):
            raise ValueError("reference qualification recomputation failed")
    # All stored paths/densities/scores, plus selected full profile reconstruction.
    for name, (cell, rep, phase, b) in schedule.items():
        bank = entries[name]
        cache, q = prepared[cell]
        z = torch.tensor([r["coordinates"] for r in bank["rows"]], dtype=torch.float64)
        if phase == "ref-p":
            sampler = DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(64),), torch.ones(1, dtype=torch.float64))
        elif phase in ("selection", "final-selected"):
            alpha = .5 if phase == "selection" else entries[f"fit-{cell}-{rep}.json"]["alpha"]
            sampler = blend(q, fitted_proposals[cell, rep], alpha)
        else:
            sampler = blend(q, None, 0.)
        streams = mapping[identity(cell, rep, phase, b)]
        count = 256 if phase == "train" else (128 if phase.startswith("final-") else 512)
        draw = sampler.sample(count, path_seed=streams["path"], label_seed=streams["label"])
        if (bank["seeds"] != streams or not torch.equal(draw.samples, z)
                or draw.labels.tolist() != [r["label"] for r in bank["rows"]]
                or draw.log_q_over_p.tolist() != [r["log_ds"] for r in bank["rows"]]
                or bank["sampler_digest"] != canonical_digest(proposal_parameters(sampler))
                or bank["physical_evaluations"] != count or bank["count"] != count
                or bank["conditional"] != (not phase.startswith("ref-"))):
            raise ValueError("all normalized sampler/density/seed/count bindings failed")
        dq = torch.logsumexp(z @ q.means.T-.5*q.means.square().sum(1)+q.weights.log(), 1)
        if not torch.equal(dq, torch.tensor([r["log_dq"] for r in bank["rows"]], dtype=torch.float64)):
            raise ValueError("all stored q densities changed")
        if bank["conditional"]:
            if any(moment_error(r["quadrature"]) != {k: r[k] for k in moment_error(r["quadrature"])} for r in bank["rows"]):
                raise ValueError("all stored numerical proxy/structural gates changed")
            selected_full = 16 if b == 0 and phase in ("train", "final-q", "final-selected") else (8 if phase == "selection" else 0)
            if selected_full:
                profiles = []
                for i in range(128):
                    if i >= selected_full:
                        profiles.append(LastVarianceResidualSlice(**bank["rows"][i]["profile"]))
                        continue
                    budget.physics(1)
                    profile = LastVarianceResidualSlice.create(cache, z[i:i+1])
                    if dataclasses.asdict(profile) != bank["rows"][i]["profile"]:
                        raise ValueError("prespecified full profile replay failed")
                    profiles.append(profile)
                replay = json_compatible(batched_moments(profiles, charge=budget.charge))
                if replay["rows"] != [r["quadrature"] for r in bank["rows"][:128]]:
                    raise ValueError("complete first 128 quadrature rows replay failed")
                replays += selected_full
        else:
            for row in bank["rows"]:
                if row["log_score"] != 2*row["log_g"]-row["log_dq"]-row["log_ds"]:
                    raise ValueError("all raw risk scores changed")
    if decide(result["training"], result["references"], result["finals"]) != result["summary"]:
        raise ValueError("complete gated decision changed")
    expected_physics = sum(b["physical_evaluations"] for b in entries.values() if "rows" in b)
    expected_scalar = sum(b["scalar_evaluations"] for b in entries.values() if "rows" in b)
    if expected_physics != result["physical_evaluations"] or expected_scalar != result["scalar_evaluations"]:
        raise ValueError("production budget accounting changed")
    end = json.loads((path/"end_to_end.json").read_text())
    if end["result_sha256"] != sha(path/"result.json"):
        raise ValueError("final I/O cost receipt result binding changed")
    return {"status": "all_artifacts_density_statistics_gate_and_prespecified_replays_passed",
        "result_sha256": sha(path/"result.json"), "audit_physical": budget.physical,
        "audit_scalar": budget.scalar, "full_replayed_paths": replays,
        "audit_wall": time.perf_counter()-started, "decision": result["summary"]["decision"], "P2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        output = audit()
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", output)
        print(json.dumps(output))
    else:
        output = run()
        print(json.dumps(output["summary"]))
