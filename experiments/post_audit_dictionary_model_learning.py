"""Same-dictionary event-KL versus direct-M2 original-event model development."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch

from experiments.post_audit_defensive_allocation import digest_bank, guard
from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_matched_window_mechanism import KWARGS
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import DIRECTORY as PARENT_DIRECTORY
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    sample_gaussian_mixture,
)
from src.path_integral.paired_model_risk import paired_risk_reduction
from src.path_integral.physical_dictionary_learning import (
    DictionaryObjective,
    event_training_diagnostics,
    fit_dictionary_weights,
)
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.risk_aware_model_correction import full_mixture_log_q_over_p
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, sensitivity

DIRECTORY = "results/post_audit/dictionary_model_learning_v1/session_20261020"
PLAN = "docs/plans/PHYSICAL_DICTIONARY_MODEL_LEARNING_2026-10-20_KO.md"
THEORY = "docs/theory/PHYSICAL_DICTIONARY_MODEL_LEARNING_CONTRACT_2026-10-20_KO.md"
ANCHORS = tuple(PARENT_DIRECTORY+"/"+n+".json" for n in ("result", "audit"))
NAMESPACE = "physical-dictionary-original-model-learning-v1"
ARMS = ("uniform", "event-kl", "risk-m2")
ALPHAS = (0., .25, .5)
TRAIN, SELECT, FINAL, HELDOUT, CHUNK = 16384, 16384, 32768, 131072, 2048
FIT_WORK, TOTAL_WORK = 32768, 9503232


def load_parent() -> dict[str, Any]:
    parent, receipt = [json.loads((ROOT/p).read_text()) for p in ANCHORS]
    audit_snapshot(ROOT, parent["source"], parent["config"])
    if (receipt["status"] != "small_grid_original_model_source_seed_fit_selection_final_risk_replay_pass"
            or receipt["result_sha256"] != sha(ROOT/ANCHORS[0])
            or parent["summary"]["development_candidate_arms"]
            or any(parent["summary"][k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("complete audited negative K2 decision required")
    return parent


def configuration(parent: dict[str, Any]) -> dict[str, Any]:
    return {"namespace": NAMESPACE, "model": parent["config"]["model"], "cells": parent["config"]["cells"],
        "steps": 32, "torch_threads": 1, "guide": KWARGS,
        "anchors": {p: sha(ROOT/p) for p in ANCHORS}, "repetitions": 5, "train": TRAIN, "selection": SELECT,
        "final_blocks": 8, "final_block": FINAL, "heldout": HELDOUT, "chunk": CHUNK,
        "arms": list(ARMS), "alphas": list(ALPHAS), "maximum_iterations": 256, "line_search_iterations": 40,
        "gap_tolerance": 1e-6, "paired_comparisons": 30,
        "natural_mass": .1, "uniform_shifted_floor": .1, "adaptive_mass": .8,
        "final_development_rse": .05, "maximum_block_to_raw_rse": 2., "successes_per_cell": 4}


def seed(j: dict[str, Any], stream: str, index: int = 0, role: str = "path") -> int:
    key = f"{NAMESPACE}/{j['cell']}/{j['rep']}/{stream}/{index}/{role}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "rep": r} for c in CELLS for r in range(5)]


def final_jobs() -> list[dict[str, Any]]:
    return [{**j, "arm": a, "block": b} for j in jobs() for a in ARMS for b in range(8)]


def fit_id(j: dict[str, Any]) -> str:
    return f"fit-{j['cell']}-{j['rep']}"


def final_id(j: dict[str, Any]) -> str:
    return f"final-{j['cell']}-{j['rep']}-{j['arm']}-{j['block']}"


def risk_id(j: dict[str, Any]) -> str:
    return f"heldout-{j['cell']}-{j['rep']}"


def stream_digest(j: dict[str, Any], stream: str, count: int, block: int = 0) -> str:
    return canonical_digest([seed(j, stream, 100*block+k, role)
        for k in range(count//CHUNK) for role in ("path", "label")])


def bank(prepared: Any, spec: GaussianMixtureShiftSpec, j: dict[str, Any], stream: str,
         count: int, block: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count % CHUNK:
        raise ValueError("complete fixed chunks required")
    cache, _ = prepared
    values: list[list[torch.Tensor]] = [[], [], []]
    with torch.no_grad():
        for k in range(count//CHUNK):
            z = sample_gaussian_mixture(spec, CHUNK,
                gaussian_generator=torch.Generator().manual_seed(seed(j, stream, 100*block+k)),
                label_generator=torch.Generator().manual_seed(seed(j, stream, 100*block+k, "label"))).target_coordinates
            g = cache.log_probability(z)
            d = full_mixture_log_q_over_p(z, spec)
            if not torch.isfinite(g).all() or (g > 1e-12).any():
                raise FloatingPointError("original conditional CDF contract failed")
            for dest, v in zip(values, (z, g, d), strict=True):
                dest.append(v)
            guard()
    return torch.cat(values[0]), torch.cat(values[1]), torch.cat(values[2])


def component_logs(z: torch.Tensor, spec: GaussianMixtureShiftSpec) -> torch.Tensor:
    return z @ spec.means.T-.5*spec.means.square().sum(1)


def select_weights(components: torch.Tensor, g: torch.Tensor, d: torch.Tensor,
                   uniform: torch.Tensor, fitted: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    identical = torch.equal(uniform, fitted)
    weights = [uniform if alpha == 0 or identical else (1-alpha)*uniform+alpha*fitted for alpha in ALPHAS]
    scores = [float(torch.logsumexp(2*g-d-torch.logsumexp(components+w.log(), 1), 0))-math.log(len(g)) for w in weights]
    selected = min(range(3), key=lambda k: (scores[k], ALPHAS[k]))
    return weights[selected], {"alpha": ALPHAS[selected], "selection_log_m2": scores,
        "no_op": bool(torch.equal(weights[selected], uniform)), "selection_not_final_estimate": True}


def fit_values(j: dict[str, Any], prepared: Any) -> dict[str, Any]:
    _, guide = prepared
    begin = time.perf_counter()
    training = bank(prepared, guide, j, "train", TRAIN)
    train_components = component_logs(training[0], guide)
    train_wall = time.perf_counter()-begin
    fitted, fit_diagnostics, fit_times = {}, {}, {}
    for arm in ARMS[1:]:
        start = time.perf_counter()
        fit = fit_dictionary_weights(train_components, training[1], training[2], objective=cast(DictionaryObjective, arm))
        fit_times[arm] = time.perf_counter()-start
        fitted[arm] = fit.weights
        fit_diagnostics[arm] = {"weights": fit.weights.tolist(), "initial_objective": fit.initial_objective,
            "final_objective": fit.final_objective, "empirical_gap": fit.relative_empirical_gap,
            "objective_units": fit.objective_units, "gap_units": fit.gap_units, "iterations": fit.iterations,
            "history": list(fit.history), "converged_empirical": fit.relative_empirical_gap <= 1e-6}
    start = time.perf_counter()
    selected_bank = bank(prepared, guide, j, "selection", SELECT)
    selection_components = component_logs(selected_bank[0], guide)
    selection_bank_wall = time.perf_counter()-start
    weights, selections, selection_times = {"uniform": guide.weights.tolist()}, {}, {}
    for arm in ARMS[1:]:
        start = time.perf_counter()
        chosen, selections[arm] = select_weights(selection_components, selected_bank[1], selected_bank[2], guide.weights, fitted[arm])
        selection_times[arm] = time.perf_counter()-start
        weights[arm] = chosen.tolist()
    # Dictionaries are never refitted: only their normalized mixing weights change.
    for arm in ARMS:
        w = torch.tensor(weights[arm], dtype=torch.float64)
        if (len(w) != 373 or abs(float(w[0])-.1) > 1e-12 or (w <= 0).any()
                or not torch.all(w >= .5*guide.weights-1e-14)):
            raise ValueError("same-dictionary/floor/baseline-mixture contract failed")
    initial_risk = 2*training[1]-2*training[2]
    return {**j, "actual_work": FIT_WORK, "weights": weights,
        "weight_digests": {a: canonical_digest(w) for a, w in weights.items()}, "selections": selections,
        "fit_diagnostics": fit_diagnostics, "event_bank_diagnostics": event_training_diagnostics(training[1], training[2]),
        "uniform_m2_bank_diagnostics": {"ess": float(initial_risk.softmax(0).square().sum().reciprocal()),
            "max_fraction": float(initial_risk.softmax(0).max())},
        "train_digest": digest_bank(training), "selection_digest": digest_bank(selected_bank),
        "train_seed_digest": stream_digest(j, "train", TRAIN), "selection_seed_digest": stream_digest(j, "selection", SELECT),
        "cost": {"train_bank_and_components": train_wall, "fit": fit_times,
            "selection_bank_and_components": selection_bank_wall, "selection": selection_times},
        "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-begin}


def proposal(prepared: Any, fit: dict[str, Any], arm: str) -> GaussianMixtureShiftSpec:
    return GaussianMixtureShiftSpec(prepared[1].means, torch.tensor(fit["weights"][arm], dtype=torch.float64))


def final_values(j: dict[str, Any], prepared: Any, fit: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    arrays = bank(prepared, proposal(prepared, fit, j["arm"]), j, "final-"+j["arm"], FINAL, j["block"])
    logs = arrays[1]-arrays[2]
    if (logs > math.log(10)+1e-10).any():
        raise FloatingPointError("defensive event bound violated")
    return {**j, "actual_work": FINAL, "moments": log_moments(logs), "sample_digest": digest_bank(arrays),
        "weight_digest": fit["weight_digests"][j["arm"]],
        "seed_digest": stream_digest(j, "final-"+j["arm"], FINAL, j["block"]),
        "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-begin}


def risk_values(j: dict[str, Any], prepared: Any, fit: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    z, g, d = bank(prepared, prepared[1], j, "heldout-risk", HELDOUT)
    components = component_logs(z, prepared[1])
    logs = {a: 2*g-d-torch.logsumexp(components+torch.tensor(fit["weights"][a], dtype=torch.float64).log(), 1) for a in ARMS}
    if any(not torch.isfinite(v).all() or (v > math.log(100)+1e-10).any() for v in logs.values()):
        raise FloatingPointError("raw risk contribution bound failed")
    blocks = {a: [log_moments(b)["log_mean"] for b in v.split(HELDOUT//8)] for a, v in logs.items()}
    comparisons = []
    for baseline, candidate in itertools.combinations(ARMS, 2):
        raw = paired_risk_reduction(logs[baseline], logs[candidate], comparisons=30)
        block = paired_risk_reduction(torch.tensor(blocks[baseline], dtype=torch.float64),
            torch.tensor(blocks[candidate], dtype=torch.float64), comparisons=30)
        noop = fit["weight_digests"][baseline] == fit["weight_digests"][candidate]
        comparisons.append({"baseline": baseline, "candidate": candidate, "raw": raw, "block": block,
            "same_proposal": noop, "development_reduction_signal": bool(not noop
                and raw["candidate_lower_m2_signal"] and block["candidate_lower_m2_signal"]
                and raw["candidate_lower_absolute_signal"] and block["candidate_lower_absolute_signal"])})
    return {**j, "actual_work": HELDOUT, "seed_digest": stream_digest(j, "heldout-risk", HELDOUT),
        "sample_digest": digest_bank((z, g, d)), "weight_digests": fit["weight_digests"],
        "risk_moments": {a: log_moments(v) for a, v in logs.items()}, "block_log_means": blocks,
        "sensitivity": {a: sensitivity(blocks[a], bootstrap_seed=seed(j, "bootstrap-risk-"+a)) for a in ARMS},
        "comparisons": comparisons, "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-begin}


def workflow_cost(arm: str, fit: dict[str, Any], setup: dict[str, Any], inference: float) -> dict[str, float]:
    offline = setup["simulator_setup"]+setup["geometry_setup"]
    cost = fit["cost"]
    training = cost["train_bank_and_components"]+cost["fit"][arm] if arm != "uniform" else 0.
    selection = cost["selection_bank_and_components"]+cost["selection"][arm] if arm != "uniform" else 0.
    return {"setup": offline, "train_bank_and_fit": training, "selection": selection, "inference": inference,
        "total_fixed_final_budget_wall": offline+training+selection+inference}


def summarize(fits: list[dict[str, Any]], finals: list[dict[str, Any]], risks: list[dict[str, Any]],
              setup: dict[str, Any]) -> dict[str, Any]:
    expected = {(j["cell"], j["rep"]) for j in jobs()}
    if (len(fits) != 10 or {(j["cell"], j["rep"]) for j in fits} != expected
            or len(finals) != 240 or {final_id(j) for j in finals} != {final_id(j) for j in final_jobs()}
            or len(risks) != 10 or {(j["cell"], j["rep"]) for j in risks} != expected):
        raise ValueError("complete same-dictionary family required; no prefix inference")
    rows = []
    for fit in fits:
        risk = next(r for r in risks if (r["cell"], r["rep"]) == (fit["cell"], fit["rep"]))
        for arm in ARMS:
            group = sorted((r for r in finals if (r["cell"], r["rep"], r["arm"]) == (fit["cell"], fit["rep"], arm)), key=lambda r: r["block"])
            moments = merge_retaining_maximum([r["moments"] for r in group])
            stability = sensitivity([r["moments"]["log_mean"] for r in group], bootstrap_seed=seed(fit, "bootstrap-final-"+arm))
            precise = moments["relative_se"] <= .05 and stability["between_unit_relative_se"] <= 2*moments["relative_se"]
            contrast = next((c for c in risk["comparisons"] if (c["baseline"], c["candidate"]) == ("uniform", arm)), None)
            rows.append({"cell": fit["cell"], "rep": fit["rep"], "arm": arm, "moments": moments,
                "sensitivity": stability, "alpha": fit["selections"].get(arm, {}).get("alpha"),
                "development_precision_only": bool(precise),
                "reduction_and_precision_signal": bool(precise and contrast and contrast["development_reduction_signal"]),
                "weight_digest": fit["weight_digests"][arm], "cost": workflow_cost(arm, fit, setup[fit["cell"]], sum(r["wall"] for r in group))})
    cells: list[dict[str, Any]] = []
    for cell in CELLS:
        methods = []
        for arm in ARMS:
            group = [r for r in rows if (r["cell"], r["arm"]) == (cell, arm)]
            methods.append({"arm": arm, "median_rse": statistics.median(r["moments"]["relative_se"] for r in group),
                "rse_range": [min(r["moments"]["relative_se"] for r in group), max(r["moments"]["relative_se"] for r in group)],
                "precision_passes": sum(r["development_precision_only"] for r in group),
                "reduction_and_precision_signals": sum(r["reduction_and_precision_signal"] for r in group),
                "selected_alphas": [r["alpha"] for r in group],
                "median_fixed_budget_total_wall": statistics.median(r["cost"]["total_fixed_final_budget_wall"] for r in group)})
        cells.append({"cell": cell, "methods": methods})
    candidates = [a for a in ARMS[1:] if all(next(m for m in c["methods"] if m["arm"] == a)["reduction_and_precision_signals"] >= 4 for c in cells)]
    return {"fits": fits, "rows": rows, "heldout_risk": risks, "cells": cells, "development_candidate_arms": candidates,
        "scope": "N32_same_dictionary_original_event_model_development_not_certified_speedup",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def manifest_for(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {"config": config, "source": source["snapshot_sha256"], "expected_units": ["prepare",
        *("preflight-"+c for c in CELLS), *(fit_id(j) for j in jobs()), *(final_id(j) for j in final_jobs()),
        *(risk_id(j) for j in jobs())], "limits": {"D": {"work": TOTAL_WORK, "wall": 7200.}}}


def validate_fit_record(fit: dict[str, Any], guide: GaussianMixtureShiftSpec) -> None:
    if set(fit["weights"]) != set(ARMS) or set(fit["fit_diagnostics"]) != set(ARMS[1:]):
        raise ValueError("incomplete fitted dictionary arms")
    if fit["event_bank_diagnostics"]["samples"] != TRAIN:
        raise ValueError("training bank count changed")
    for arm in ARMS:
        w = torch.tensor(fit["weights"][arm], dtype=torch.float64)
        if (w.shape != (373,) or not torch.isfinite(w).all() or abs(float(w.sum())-1) > 1e-12
                or abs(float(w[0])-.1) > 1e-12 or (w[1:] < .1/372-1e-14).any()
                or (w < .5*guide.weights-1e-14).any()
                or canonical_digest(fit["weights"][arm]) != fit["weight_digests"][arm]):
            raise ValueError("saved dictionary weights/floors/digest invalid")
        if arm == "uniform":
            if not torch.equal(w, guide.weights):
                raise ValueError("uniform dictionary changed")
            continue
        raw = torch.tensor(fit["fit_diagnostics"][arm]["weights"], dtype=torch.float64)
        selected = fit["selections"][arm]
        scores = selected["selection_log_m2"]
        if (raw.shape != (373,) or not torch.isfinite(raw).all() or abs(float(raw.sum())-1) > 1e-12
                or abs(float(raw[0])-.1) > 1e-12 or (raw[1:] < .1/372-1e-14).any()
                or len(scores) != len(ALPHAS) or any(not math.isfinite(x) for x in scores)):
            raise ValueError("raw fitted weights/selection scores invalid")
        alpha = ALPHAS[min(range(len(ALPHAS)), key=lambda k: (scores[k], ALPHAS[k]))]
        expected = guide.weights if alpha == 0 or torch.equal(raw, guide.weights) else (1-alpha)*guide.weights+alpha*raw
        if (selected["alpha"] != alpha or not torch.equal(expected, w)
                or selected["no_op"] != bool(torch.equal(w, guide.weights))
                or selected["selection_not_final_estimate"] is not True):
            raise ValueError("fitted/selected weights binding invalid")


def validate_moment_count(moment: dict[str, Any], count: int) -> None:
    if (type(moment["count"]) is not int or moment["count"] != count
            or any(not math.isfinite(moment[k]) for k in ("log_sum", "log_sum_squares", "log_mean", "relative_se"))
            or not math.isclose(moment["log_mean"], moment["log_sum"]-math.log(count), abs_tol=1e-12)
            or moment["relative_se"] < 0):
        raise ValueError("saved moment count/mean invalid")
    ratio = moment["log_sum_squares"]-math.log(count)-2*moment["log_mean"]
    expected = math.sqrt(max(0., math.expm1(ratio))/(count-1))
    if ratio < -1e-10 or not math.isclose(expected, moment["relative_se"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("saved moment relative SE inconsistent")


def validate_risk_record(risk: dict[str, Any], fit: dict[str, Any]) -> None:
    pairs = set(itertools.combinations(ARMS, 2))
    if (len(risk["comparisons"]) != 3
            or {(c["baseline"], c["candidate"]) for c in risk["comparisons"]} != pairs):
        raise ValueError("incomplete paired risk comparisons")
    for arm in ARMS:
        validate_moment_count(risk["risk_moments"][arm], HELDOUT)
        blocks = risk["block_log_means"][arm]
        if (len(blocks) != 8 or any(not math.isfinite(x) for x in blocks)
                or not math.isclose(float(torch.logsumexp(torch.tensor(blocks, dtype=torch.float64), 0))-math.log(8),
                    risk["risk_moments"][arm]["log_mean"], abs_tol=1e-12)
                or risk["sensitivity"][arm] != sensitivity(blocks, bootstrap_seed=seed(risk, "bootstrap-risk-"+arm))):
            raise ValueError("risk block count/mean/sensitivity invalid")
    for comparison in risk["comparisons"]:
        a, b = comparison["baseline"], comparison["candidate"]
        raw, block = comparison["raw"], comparison["block"]
        recomputed = paired_risk_reduction(torch.tensor(risk["block_log_means"][a], dtype=torch.float64),
            torch.tensor(risk["block_log_means"][b], dtype=torch.float64), comparisons=30)
        noop = fit["weight_digests"][a] == fit["weight_digests"][b]
        signal = bool(not noop and raw["candidate_lower_m2_signal"] and block["candidate_lower_m2_signal"]
            and raw["candidate_lower_absolute_signal"] and block["candidate_lower_absolute_signal"])
        if (raw["sample_count"] != HELDOUT or block["sample_count"] != 8 or block != recomputed
                or comparison["same_proposal"] != noop or comparison["development_reduction_signal"] != signal):
            raise ValueError("paired sample count/summary/gate invalid")


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    config = configuration(load_parent())
    directory = ROOT/DIRECTORY
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("one-shot protocol: preserve existing run; no unit/seed replacement or implicit resume")
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, *ANCHORS))
    manifest = manifest_for(config, source)
    journal, prepared = RecoveryJournal(directory, manifest), {}

    def preparation():
        begin = time.perf_counter()
        p, costs = prepare(config)
        prepared.update(p)
        return {"actual_work": 0, "setup": costs, "wall": time.perf_counter()-begin}

    prep = journal.unit("prepare", phase="D", work=0, maximum_wall=120., operation=preparation)
    for cell in CELLS:
        def preflight(c=cell):
            cache, guide = prepared[c]
            j = {"cell": c, "rep": 0}
            begin = time.perf_counter()
            z = sample_gaussian_mixture(guide, 256,
                gaussian_generator=torch.Generator().manual_seed(seed(j, "preflight")),
                label_generator=torch.Generator().manual_seed(seed(j, "preflight", role="label"))).target_coordinates
            g, d = cache.log_probability(z), full_mixture_log_q_over_p(z, guide)
            wall = time.perf_counter()-begin
            if wall*128 > 60.:
                raise TimeoutError("final block resource forecast exceeded")
            return {"cell": c, "actual_work": 256, "bank_digest": digest_bank((z, g, d)), "wall": wall}
        journal.unit("preflight-"+cell, phase="D", work=256, maximum_wall=30., operation=preflight)
    fits, finals, risks = [], [], []
    for j in jobs():
        fit = journal.unit(fit_id(j), phase="D", work=FIT_WORK, maximum_wall=300.,
            operation=partial(fit_values, j, prepared[j["cell"]]))
        fits.append(fit)
        for arm in ARMS:
            for block in range(8):
                job = {**j, "arm": arm, "block": block}
                finals.append(journal.unit(final_id(job), phase="D", work=FINAL, maximum_wall=60.,
                    operation=partial(final_values, job, prepared[j["cell"]], fit)))
        risks.append(journal.unit(risk_id(j), phase="D", work=HELDOUT, maximum_wall=180.,
            operation=partial(risk_values, j, prepared[j["cell"]], fit)))
        print(f"retained dictionary completed fit {len(fits)}/10", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "setup": prep["setup"],
        "summary": summarize(fits, finals, risks, prep["setup"]), "units": 263, "physical_evaluations": TOTAL_WORK,
        "phase_wall": time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-D.clock.json")["started_unix"] > 7200.:
        journal._write(directory/"phase-D.violation.json", {"manifest": journal.digest, "phase": "D",
            "status": "protocol_failure_after_final_output_io"})
        raise TimeoutError("final I/O budget exceeded")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config = configuration(load_parent())
    audit_snapshot(ROOT, output["source"], config)
    manifest = manifest_for(config, output["source"])
    if output["config"] != config or output["manifest"] != manifest:
        raise ValueError("protocol changed")
    for name, digest in output["source"]["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/")) and sha(ROOT/name) != digest:
            raise ValueError("frozen numerical runtime changed")
    journal, expected = RecoveryJournal(path.parent, manifest), set(manifest["expected_units"])
    for suffix in ("start", "finish"):
        if {p.name.removesuffix(f".{suffix}.json") for p in path.parent.glob(f"*.{suffix}.json")} != expected:
            raise ValueError("incomplete/unexpected family")
    for name in expected:
        start, finish = [journal._read(path.parent/(name+"."+s+".json")) for s in ("start", "finish")]
        work = 0 if name == "prepare" else 256 if name.startswith("preflight-") else FIT_WORK if name.startswith("fit-") else HELDOUT if name.startswith("heldout-") else FINAL
        cap = 120. if name == "prepare" else 30. if name.startswith("preflight-") else 300. if name.startswith("fit-") else 180. if name.startswith("heldout-") else 60.
        if (start["phase"] != "D" or start["reserved_work"] != work or start["reserved_wall"] != cap
                or finish["status"] != "complete" or finish["phase"] != "D" or finish["work"] != work
                or finish["result"]["actual_work"] != work or finish["wall"] > cap):
            raise ValueError("unit/cost contract failure")
    prepared, costs = prepare(config)
    saved_setup = journal._read(path.parent/"prepare.finish.json")["result"]["setup"]
    if any(costs[c][k] != saved_setup[c][k] for c in CELLS for k in ("static_components", "static_digest")):
        raise ValueError("dictionary geometry changed")
    fits, finals, risks = [], [], []
    for j in jobs():
        fit = journal._read(path.parent/(fit_id(j)+".finish.json"))["result"]
        if (any(fit[k] != v for k, v in j.items()) or fit["torch_threads"] != 1
                or fit["train_seed_digest"] != stream_digest(j, "train", TRAIN)
                or fit["selection_seed_digest"] != stream_digest(j, "selection", SELECT)
                or any(canonical_digest(w) != fit["weight_digests"][a] for a, w in fit["weights"].items())):
            raise ValueError("fit identity/seed/parameter mismatch")
        validate_fit_record(fit, prepared[j["cell"]][1])
        if j["rep"] == 0:
            replay = fit_values(j, prepared[j["cell"]])
            fields = ("weights", "selections", "fit_diagnostics", "event_bank_diagnostics", "uniform_m2_bank_diagnostics", "train_digest", "selection_digest")
            if any(replay[k] != fit[k] for k in fields):
                raise ValueError("fit/selection replay mismatch")
        fits.append(fit)
        risk = journal._read(path.parent/(risk_id(j)+".finish.json"))["result"]
        if (any(risk[k] != v for k, v in j.items()) or risk["torch_threads"] != 1
                or risk["seed_digest"] != stream_digest(j, "heldout-risk", HELDOUT)
                or risk["weight_digests"] != fit["weight_digests"]):
            raise ValueError("heldout identity/seed/binding mismatch")
        validate_risk_record(risk, fit)
        if j["rep"] == 0:
            replay = risk_values(j, prepared[j["cell"]], fit)
            if any(replay[k] != risk[k] for k in ("sample_digest", "risk_moments", "block_log_means", "sensitivity", "comparisons")):
                raise ValueError("heldout replay mismatch")
        risks.append(risk)
    for j in final_jobs():
        fit = next(f for f in fits if (f["cell"], f["rep"]) == (j["cell"], j["rep"]))
        record = journal._read(path.parent/(final_id(j)+".finish.json"))["result"]
        if (any(record[k] != v for k, v in j.items()) or record["torch_threads"] != 1
                or record["weight_digest"] != fit["weight_digests"][j["arm"]]
                or record["seed_digest"] != stream_digest(j, "final-"+j["arm"], FINAL, j["block"])):
            raise ValueError("final identity/seed/binding mismatch")
        validate_moment_count(record["moments"], FINAL)
        if j["rep"] == 0 and j["block"] == 0:
            replay = final_values(j, prepared[j["cell"]], fit)
            if any(replay[k] != record[k] for k in ("sample_digest", "moments")):
                raise ValueError("final replay mismatch")
        finals.append(record)
    if (saved_setup != output["setup"] or summarize(fits, finals, risks, saved_setup) != output["summary"]
            or journal.accounting("D")[0] != TOTAL_WORK or output["physical_evaluations"] != TOTAL_WORK
            or output["units"] != 263 or output["phase_wall"] > 7200. or (path.parent/"phase-D.violation.json").exists()):
        raise ValueError("complete summary/work/qualification mismatch")
    return {"status": "dictionary_model_source_geometry_seed_fit_selection_final_risk_replay_pass",
        "replayed_fits": 2, "replayed_final_blocks": 6, "replayed_heldout_banks": 2,
        "audit_physical_evaluations": 524288, "result_sha256": sha(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        path = ROOT/args.audit
        receipt = audit(path)
        write_json_atomic_nonoverwriting(path.parent/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        result = run()
        print(json.dumps({k: result[k] for k in ("units", "physical_evaluations", "phase_wall")}))
