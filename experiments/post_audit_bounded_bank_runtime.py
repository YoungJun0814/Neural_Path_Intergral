"""Shared frozen runtime for the bounded D2/D3 study, not a search loop."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import torch

from experiments.post_audit_bounded_direct_m2 import DIRECTORY as D0
from experiments.post_audit_bounded_direct_m2 import EXPECTED_AUDIT, EXPECTED_PARENT, PLAN
from experiments.post_audit_dictionary_model_learning import DIRECTORY as PARENT
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.balanced_mixture_bank import balanced_counts, draw_bank, stream_layout
from src.path_integral.bounded_decision_protocol import DESIGNS, precision_gate
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.physical_dictionary_learning import (
    event_training_diagnostics,
    fit_dictionary_weights,
)
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.stratified_paired_risk import estimate, merge_sufficient, sufficient
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import sensitivity

ORACLE = "results/post_audit/bounded_bank_oracles_v1/session_20261020"
CONFIG = "configs/post_audit/bounded_bank_decision_v1.yaml"


def anchors() -> dict[str, Any]:
    parent = json.loads((ROOT/PARENT/"result.json").read_text())
    if sha(ROOT/PARENT/"result.json") != EXPECTED_PARENT or sha(ROOT/PARENT/"audit.json") != EXPECTED_AUDIT:
        raise ValueError("immutable parent anchors changed")
    audit_snapshot(ROOT, parent["source"], parent["config"])
    for relative, gate in ((D0, "d0_evidence_gate"), (ORACLE, "d1_gate")):
        path = ROOT/relative/"result.json"
        result, receipt = [json.loads(p.read_text()) for p in (path, path.parent/"audit.json")]
        audit_snapshot(ROOT, result["source"], result["config"])
        if sha(path) != receipt["result_sha256"] or not result["summary"][gate]:
            raise ValueError("complete audited D0/D1 gate required")
    return parent


def begin_run(directory: str, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    path = ROOT/directory
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("one-shot protocol: existing evidence cannot be overwritten or resumed")
    path.mkdir(parents=True, exist_ok=True)
    extras = (PLAN, CONFIG, PARENT+"/result.json", PARENT+"/audit.json", D0+"/result.json", D0+"/audit.json",
        ORACLE+"/result.json", ORACLE+"/audit.json")
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=extras)
    write_json_atomic_nonoverwriting(path/"protocol.json", {"config": config, "source": source})
    return path, source


def key(phase: str, cell: str, design: str, *, rep: int = 0, role: str = "risk", block: int = 0) -> dict[str, Any]:
    return {"phase": phase, "cell": cell, "design": design, "rep": rep, "role": role, "block": block}


def seed_for(k: dict[str, Any], endpoint: str) -> int:
    return int(hashlib.sha256((canonical_digest(k)+"/"+endpoint).encode()).hexdigest()[:15], 16)


def evaluate(cache: Any, spec: GaussianMixtureShiftSpec, count: int, design: str,
             k: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    z, labels, seed_digest = draw_bank(spec, count, design, k)
    gparts, cparts = [], []
    from experiments.post_audit_defensive_allocation import guard
    for batch in z.split(2048):
        with torch.no_grad():
            g = cache.log_probability(batch)
            c = batch @ spec.means.T-.5*spec.means.square().sum(1)
        if not torch.isfinite(g).all() or (g > 1e-12).any():
            raise FloatingPointError("finite conditional CDF contract failed")
        gparts.append(g)
        cparts.append(c)
        guard()
    g, c = torch.cat(gparts), torch.cat(cparts)
    d = torch.logsumexp(c+spec.weights.log(), 1)
    digest = hashlib.sha256()
    for v in (z, labels, g, d):
        digest.update(v.contiguous().numpy().tobytes())
    return c, g, d, labels, seed_digest, digest.hexdigest()


def group_moments(logs: torch.Tensor, labels: torch.Tensor, spec: GaussianMixtureShiftSpec, design: str) -> list[dict[str, Any]]:
    if design == "iid":
        return [sufficient(logs)]
    counts = balanced_counts(spec.weights, len(logs))
    if labels.bincount(minlength=spec.components).tolist() != list(counts):
        raise ValueError("incomplete balanced strata")
    return [sufficient(logs[labels == j]) for j in range(spec.components)]


def summarize_blocks(blocks: list[dict[str, Any]], spec: GaussianMixtureShiftSpec,
                     design: str, k: dict[str, Any], *, probability_first: bool) -> dict[str, Any]:
    if len(blocks) != 8 or [b["block"] for b in blocks] != list(range(8)):
        raise ValueError("eight complete balanced blocks required")
    sizes = {b["actual_work"] for b in blocks}
    if len(sizes) != 1:
        raise ValueError("equal block sizes required")
    groups = 1 if design == "iid" else spec.components
    if any(len(b["strata"]) != groups for b in blocks):
        raise ValueError("saved stratum group count changed")
    masses = torch.ones(1, dtype=torch.float64) if design == "iid" else spec.weights
    merged = [merge_sufficient([b["strata"][j] for b in blocks]) for j in range(groups)]
    stats = estimate(merged, masses)
    block_logs = torch.tensor([b["estimator"]["log_mean"] for b in blocks], dtype=torch.float64)
    block_summary = estimate([sufficient(block_logs)], torch.ones(1, dtype=torch.float64))
    diagnostics = [sensitivity(block_logs[:, j].tolist(), bootstrap_seed=seed_for(k, "bootstrap-"+str(j)))
        for j in range(block_logs.shape[1])]
    return {"estimator": stats, "block_estimator": block_summary, "sensitivities": diagnostics,
        "gates": precision_gate(stats, diagnostics, probability_first=probability_first),
        "strata": merged, "block_log_means": block_logs.tolist()}


def risk_block(prepared: Any, weights: list[list[float]], design: str, count: int,
               k: dict[str, Any], *, include_probability: bool = True) -> dict[str, Any]:
    start = time.perf_counter()
    cache, spec = prepared
    c, g, d, labels, seed_digest, sample_digest = evaluate(cache, spec, count, design, k)
    columns = [g-d] if include_probability else []
    for w in weights:
        wv = torch.tensor(w, dtype=torch.float64)
        GaussianMixtureShiftSpec(spec.means, wv)
        dq = torch.logsumexp(c+wv.log(), 1)
        columns.append(2*g-d-dq)
    logs = torch.stack(columns, 1)
    if (logs > math.log(100)+1e-10).any():
        raise FloatingPointError("defensive risk bound failed")
    strata = group_moments(logs, labels, spec, design)
    stats = estimate(strata, torch.ones(1, dtype=torch.float64) if design == "iid" else spec.weights)
    return {**k, "actual_work": count, "strata": strata, "estimator": stats,
        "weights_digest": canonical_digest(weights), "seed_digest": seed_digest,
        "sample_digest": sample_digest, "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-start}


def fit_pair(prepared: Any, cell: str, rep: int) -> dict[str, Any]:
    start = time.perf_counter()
    cache, spec = prepared
    raw, fits, costs, seeds, samples, banks = {}, {}, {}, {}, {}, {}
    for design, arm in zip(DESIGNS, ("E-IID", "E-STRAT"), strict=True):
        begin = time.perf_counter()
        c, g, d, _, sd, digest = evaluate(cache, spec, 19840, design, key("D3", cell, design, rep=rep, role="train"))
        bank_wall = time.perf_counter()-begin
        begin = time.perf_counter()
        fit = fit_dictionary_weights(c, g, d, objective="event-kl")
        costs[arm] = {"train": bank_wall, "fit": time.perf_counter()-begin}
        raw[arm], seeds[arm], samples[arm] = fit.weights, sd, digest
        fits[arm] = {"weights": fit.weights.tolist(), "gap": fit.relative_empirical_gap,
            "iterations": fit.iterations, "converged": fit.relative_empirical_gap <= 1e-6,
            "initial": fit.initial_objective, "final": fit.final_objective}
        banks[arm] = event_training_diagnostics(g, d)
    begin = time.perf_counter()
    c, g, d, _, sd, digest = evaluate(cache, spec, 19840, "stratified", key("D3", cell, "stratified", rep=rep, role="selection"))
    selection_wall = time.perf_counter()-begin
    weights, selected = {"U": spec.weights.tolist()}, {}
    # Exact proportional sampling means each row has the same quadrature mass 1/N.
    from experiments.post_audit_dictionary_model_learning import select_weights
    for arm in ("E-IID", "E-STRAT"):
        begin = time.perf_counter()
        w, selection = select_weights(c, g, d, spec.weights, raw[arm])
        selected[arm] = selection
        weights[arm] = w.tolist()
        costs[arm]["selection"] = selection_wall+time.perf_counter()-begin
    output = {"cell": cell, "rep": rep, "actual_work": 59520, "weights": weights, "fits": fits,
        "selections": selected, "bank_diagnostics": banks, "costs": costs,
        "train_seed_digests": seeds, "train_sample_digests": samples,
        "selection_seed_digest": sd, "selection_sample_digest": digest, "wall": time.perf_counter()-start}
    validate_fit(output, spec)
    return output


def validate_fit(fit: dict[str, Any], guide: GaussianMixtureShiftSpec) -> None:
    if set(fit["weights"]) != {"U", "E-IID", "E-STRAT"} or fit["weights"]["U"] != guide.weights.tolist():
        raise ValueError("D3 uniform/family binding invalid")
    for design, arm in zip(DESIGNS, ("E-IID", "E-STRAT"), strict=True):
        raw = torch.tensor(fit["fits"][arm]["weights"], dtype=torch.float64)
        w = torch.tensor(fit["weights"][arm], dtype=torch.float64)
        for v in (raw, w):
            GaussianMixtureShiftSpec(guide.means, v)
            if abs(float(v[0])-.1) > 1e-12 or (v[1:] < .1/372-1e-14).any():
                raise ValueError("D3 defensive weight domain invalid")
        selection = fit["selections"][arm]
        scores = selection["selection_log_m2"]
        if len(scores) != 3 or any(not math.isfinite(s) for s in scores):
            raise ValueError("D3 complete finite selection scores required")
        alpha = (0., .25, .5)[min(range(3), key=lambda i: (scores[i], i))]
        expected = guide.weights if alpha == 0 or torch.equal(raw, guide.weights) else (1-alpha)*guide.weights+alpha*raw
        train_key = key("D3", fit["cell"], design, rep=fit["rep"], role="train")
        if (selection["alpha"] != alpha or not torch.equal(w, expected)
                or selection["no_op"] != bool(torch.equal(w, guide.weights))
                or fit["bank_diagnostics"][arm]["samples"] != 19840
                or fit["train_seed_digests"][arm] != canonical_digest(stream_layout(guide, 19840, design, train_key))):
            raise ValueError("D3 fit/selection/count/seed binding invalid")
    sk = key("D3", fit["cell"], "stratified", rep=fit["rep"], role="selection")
    if fit["selection_seed_digest"] != canonical_digest(stream_layout(guide, 19840, "stratified", sk)):
        raise ValueError("D3 selection seed identity invalid")


def probability_block(prepared: Any, weights: list[float], count: int, k: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    cache, guide = prepared
    spec = GaussianMixtureShiftSpec(guide.means, torch.tensor(weights, dtype=torch.float64))
    _, g, d, _, sd, digest = evaluate(cache, spec, count, "iid", k)
    x = g-d
    if (x > math.log(10)+1e-10).any():
        raise FloatingPointError("ordinary event bound failed")
    stats = [sufficient(torch.stack((x, 2*x), 1))]
    return {**k, "actual_work": count, "strata": stats,
        "estimator": estimate(stats, torch.ones(1, dtype=torch.float64)),
        "weight_digest": canonical_digest(weights), "seed_digest": sd, "sample_digest": digest,
        "wall": time.perf_counter()-start, "torch_threads": torch.get_num_threads()}


def validate_streams(prepared: dict[str, Any], requests: list[tuple[int, str, dict[str, Any]]]) -> int:
    seen: set[int] = set()
    for count, design, k in requests:
        layout = stream_layout(prepared[k["cell"]][1], count, design, k)
        for row in layout:
            for role in ("path_seed", "label_seed"):
                if role in row:
                    if row[role] in seen:
                        raise ValueError("study-wide seed collision")
                    seen.add(row[role])
    return len(seen)
