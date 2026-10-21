"""D3 conditional stage, frozen before D2; never bypasses a failed D2 gate."""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import torch

from experiments.post_audit_bounded_bank_runtime import (
    fit_pair,
    key,
    probability_block,
    risk_block,
    summarize_blocks,
    validate_fit,
    validate_streams,
)
from src.path_integral.balanced_mixture_bank import balanced_counts, stream_layout
from src.path_integral.bounded_decision_protocol import ARMS, DESIGNS, allocate, repeated_decision
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.stratified_paired_risk import paired


def assess(fit: dict[str, Any], probabilities: dict[str, Any], risks: dict[str, Any],
           prob_pilots: dict[str, Any], setup: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    j = {k: fit[k] for k in ("cell", "rep")}
    ident = j["cell"]+"-"+str(j["rep"])
    contrasts = {}
    if risks:
        for ia, ib in ((0, 1), (0, 2), (1, 2)):
            contrasts[ARMS[ia]+"/"+ARMS[ib]] = comparison(risks, ia, ib, fit["weights"][ARMS[ia]] == fit["weights"][ARMS[ib]])

    def precise(arms: tuple[str, ...]) -> bool:
        return all(a in probabilities and probabilities[a]["summary"]["estimator"]["relative_se"][0] <= .05
            and probabilities[a]["summary"]["sensitivities"][0]["between_unit_relative_se"] <= 2*probabilities[a]["summary"]["estimator"]["relative_se"][0] for a in arms)

    rows = []
    for arm in ARMS[1:]:
        interval = contrasts.get("U/"+arm, {})
        workflow = {}
        for a in ("U", arm):
            if a in probabilities:
                offline = setup[j["cell"]]["simulator_setup"]+setup[j["cell"]]["geometry_setup"]
                trained = sum(fit["costs"][a].values()) if a != "U" else 0.
                workflow[a] = offline+trained+prob_pilots[ident+a]["wall"]+probabilities[a]["wall"]
        rows.append({**j, "arm": arm, "evaluable": bool(precise(("U", arm)) and interval.get("evaluable", False)),
            "success": bool(precise(("U", arm)) and interval.get("primary_reduction", False)),
            "cost_ratio": workflow[arm]/workflow["U"] if len(workflow) == 2 else 1e300, "workflow_wall": workflow,
            "bank_effect": bool(precise(ARMS[1:]) and contrasts.get("E-IID/E-STRAT", {}).get("primary_reduction", False)),
            "dual_design": bool(precise(("U", arm)) and interval.get("dual_design_reduction", False))})
    return contrasts, rows


def eight_probability(prepared: Any, weights: list[float], count: int, k: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    blocks = [probability_block(prepared, weights, count//8, {**k, "block": b}) for b in range(8)]
    return {"blocks": blocks, "summary": summarize_blocks(blocks, prepared[1], "iid", k, probability_first=True),
        "actual_work": count, "wall": time.perf_counter()-begin}


def eight_risk(prepared: Any, fit: dict[str, Any], count: int, k: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    weights = [fit["weights"][a] for a in ARMS]
    blocks = [risk_block(prepared, weights, k["design"], count//8, {**k, "block": b}, include_probability=False) for b in range(8)]
    return {"blocks": blocks, "summary": summarize_blocks(blocks, prepared[1], k["design"], k, probability_first=False),
        "actual_work": count, "wall": time.perf_counter()-begin}


def risk_allocation(pilot: dict[str, Any]) -> dict[str, Any]:
    summary = pilot["summary"]
    allocations = [allocate(r, s["between_unit_relative_se"], kind="risk")
        for r, s in zip(summary["estimator"]["relative_se"], summary["sensitivities"], strict=True)]
    if any(a["status"] != "allocated" for a in allocations):
        return {"status": "unallocated_shared_risk_group", "count": None, "proposals": allocations}
    return {"status": "allocated", "count": max(a["count"] for a in allocations), "proposals": allocations}


def comparison(risks: dict[str, Any], a: int, b: int, same: bool) -> dict[str, Any]:
    intervals = {}
    for design in DESIGNS:
        summary = risks[design]["summary"]
        intervals[design] = {unit: paired(summary[field], a, b, comparisons=240)
            for unit, field in (("raw", "estimator"), ("block", "block_estimator"))}
    primary = intervals["stratified"]
    positive = all(c["positive_signal"] for c in primary.values()) and primary["raw"]["relative_reduction"] >= .2
    opposite = any(c["negative_signal"] for c in intervals["iid"].values())
    precision = all(risks["stratified"]["summary"]["gates"][i] for i in (a, b))
    dual = all(c["positive_signal"] for c in intervals["iid"].values()) and all(risks["iid"]["summary"]["gates"][i] for i in (a, b))
    return {"intervals": intervals, "evaluable": precision, "same_proposal": same,
        "primary_reduction": bool(not same and positive and not opposite and precision),
        "dual_design_reduction": bool(not same and positive and not opposite and precision and dual)}


def run(directory: Any, source: dict[str, Any], prepared: dict[str, Any], setup: dict[str, Any],
        d2: dict[str, Any]) -> dict[str, Any]:
    if not d2["decision"]["d2_gate"]:
        raise ValueError("D2 gate failed: conditional D3 is prohibited")
    cells = tuple(prepared)
    jobs: list[dict[str, Any]] = [{"cell": c, "rep": rep} for c in cells for rep in range(5)]
    requests: list[tuple[int, str, dict[str, Any]]] = []
    for j in jobs:
        for design in DESIGNS:
            requests.append((19840, design, key("D3", **j, design=design, role="train")))
            for b in range(8):
                requests.append((2480, design, key("D3", **j, design=design, role="risk-pilot", block=b)))
        requests.append((19840, "stratified", key("D3", **j, design="stratified", role="selection")))
        for arm in ARMS:
            for b in range(8):
                requests.append((2048, "iid", key("D3", **j, design="iid", role="prob-pilot-"+arm, block=b)))
    validate_streams(prepared, requests)
    manifest = {"source": source["snapshot_sha256"], "d2_digest": canonical_digest(d2),
        "jobs": jobs, "limits": {"D3": {"work": 12522240, "wall": 3600.}}}
    path = directory/"D3"
    path.mkdir()
    journal = RecoveryJournal(path, manifest)
    fits: list[dict[str, Any]] = []
    prob_pilots: dict[str, Any] = {}
    risk_pilots: dict[str, Any] = {}
    allocations: list[dict[str, Any]] = []
    for j in jobs:
        ident = j["cell"]+"-"+str(j["rep"])
        fit = journal.unit("fit-"+ident, phase="D3", work=59520, maximum_wall=180.,
            operation=partial(fit_pair, prepared[j["cell"]], j["cell"], j["rep"]))
        fits.append(fit)
        for arm in ARMS:
            k = key("D3", **j, design="iid", role="prob-pilot-"+arm)
            prob_pilots[ident+arm] = journal.unit("prob-pilot-"+ident+"-"+arm, phase="D3", work=16384,
                maximum_wall=60., operation=partial(eight_probability, prepared[j["cell"]], fit["weights"][arm], 16384, k))
        for design in DESIGNS:
            k = key("D3", **j, design=design, role="risk-pilot")
            risk_pilots[ident+design] = journal.unit("risk-pilot-"+ident+"-"+design, phase="D3", work=19840,
                maximum_wall=90., operation=partial(eight_risk, prepared[j["cell"]], fit, 19840, k))
        allocations.append({**j, "probability": {a: allocate(prob_pilots[ident+a]["summary"]["estimator"]["relative_se"][0],
            prob_pilots[ident+a]["summary"]["sensitivities"][0]["between_unit_relative_se"], kind="probability") for a in ARMS},
            "risk": risk_allocation(risk_pilots[ident+"stratified"]),
            "iid_risk_forecast": risk_allocation(risk_pilots[ident+"iid"])})
        print(f"D3 completed fit/pilot {len(fits)}/10", flush=True)
    write_json_atomic_nonoverwriting(path/"allocation.json", {"source": source["snapshot_sha256"], "allocations": allocations})
    final_requests: list[tuple[int, str, dict[str, Any]]] = []
    for allocation in allocations:
        j = {k: allocation[k] for k in ("cell", "rep")}
        for arm, record in allocation["probability"].items():
            if record["count"] is not None:
                final_requests.extend((record["count"]//8, "iid", key("D3", **j, design="iid", role="prob-confirm-"+arm, block=b)) for b in range(8))
        if allocation["risk"]["count"] is not None:
            final_requests.extend((allocation["risk"]["count"]//8, design, key("D3", **j, design=design, role="risk-confirm", block=b)) for design in DESIGNS for b in range(8))
    validate_streams(prepared, requests+final_requests)
    rows: list[dict[str, Any]] = []
    confirmations: list[dict[str, Any]] = []
    for fit, allocation in zip(fits, allocations, strict=True):
        j = {k: fit[k] for k in ("cell", "rep")}
        ident = j["cell"]+"-"+str(j["rep"])
        probabilities, risks = {}, {}
        for arm in ARMS:
            n = allocation["probability"][arm]["count"]
            if n is not None:
                probabilities[arm] = journal.unit("prob-confirm-"+ident+"-"+arm, phase="D3", work=n,
                    maximum_wall=120., operation=partial(eight_probability, prepared[j["cell"]], fit["weights"][arm], n,
                        key("D3", **j, design="iid", role="prob-confirm-"+arm)))
        n = allocation["risk"]["count"]
        if n is not None:
            for design in DESIGNS:
                risks[design] = journal.unit("risk-confirm-"+ident+"-"+design, phase="D3", work=n,
                    maximum_wall=180., operation=partial(eight_risk, prepared[j["cell"]], fit, n,
                        key("D3", **j, design=design, role="risk-confirm")))
        contrasts, assessed = assess(fit, probabilities, risks, prob_pilots, setup)
        confirmations.append({**j, "probability": probabilities, "risk": risks, "comparisons": contrasts})
        rows.extend(assessed)
    return {"fits": fits, "allocations": allocations, "probability_pilots": prob_pilots,
        "risk_pilots": risk_pilots, "confirmations": confirmations, "rows": rows,
        "decision": repeated_decision(rows, cells), "actual_work": journal.accounting("D3")[0], "manifest": manifest}


def audit(directory: Any, saved: dict[str, Any], source: dict[str, Any], prepared: dict[str, Any],
          setup: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """All sufficient moments/allocations, with the prespecified rep0 replays."""
    started = time.perf_counter()
    if not d2["decision"]["d2_gate"] or saved["manifest"]["source"] != source["snapshot_sha256"]:
        raise ValueError("D3 cannot qualify without bound D2 gate")
    journal = RecoveryJournal(directory, saved["manifest"])
    if len(saved["fits"]) != 10 or len(saved["confirmations"]) != 10 or len(saved["allocations"]) != 10:
        raise ValueError("complete ten-repetition conditional family required")
    expected: set[str] = set()
    replay_calls = 0

    def bound_unit(name: str, record: dict[str, Any], work: int, cap: float) -> None:
        expected.add(name)
        start, finish = [journal._read(directory/(name+"."+s+".json")) for s in ("start", "finish")]
        if (finish["status"] != "complete" or finish["result"] != record or start["reserved_work"] != work
                or start["reserved_wall"] != cap or finish["work"] != work or record["actual_work"] != work
                or start["phase"] != "D3" or finish["phase"] != "D3" or finish["wall"] > cap):
            raise ValueError("D3 unit work/count/binding changed")

    def bank_check(record: dict[str, Any], fit: dict[str, Any], design: str, count: int,
                   role: str, arm: str | None, replay_all: bool) -> None:
        nonlocal replay_calls
        cell, rep = fit["cell"], fit["rep"]
        k = key("D3", cell, design, rep=rep, role=role)
        guide = prepared[cell][1]
        if len(record["blocks"]) != 8 or record["actual_work"] != count:
            raise ValueError("D3 eight-bank count changed")
        for b, block in enumerate(record["blocks"]):
            bk = {**k, "block": b}
            n = count//8
            if any(block[field] != value for field, value in bk.items()) or block["actual_work"] != n or block["torch_threads"] != 1:
                raise ValueError("D3 block identity/count changed")
            if arm is not None:
                spec = GaussianMixtureShiftSpec(guide.means, torch.tensor(fit["weights"][arm], dtype=torch.float64))
                if block["weight_digest"] != canonical_digest(fit["weights"][arm]):
                    raise ValueError("D3 probability proposal binding")
            else:
                spec = guide
                if block["weights_digest"] != canonical_digest([fit["weights"][a] for a in ARMS]):
                    raise ValueError("D3 risk proposal binding")
            counts = (n,) if design == "iid" else balanced_counts(guide.weights, n)
            if (len(block["strata"]) != len(counts) or any(p["count"] != cnt for p, cnt in zip(block["strata"], counts, strict=True))
                    or block["seed_digest"] != canonical_digest(stream_layout(spec, n, design, bk))):
                raise ValueError("D3 stratum/seed binding")
            if rep == 0 and (replay_all or b == 0):
                rebuilt = (probability_block(prepared[cell], fit["weights"][arm], n, bk) if arm is not None
                    else risk_block(prepared[cell], [fit["weights"][a] for a in ARMS], design, n, bk, include_probability=False))
                if any(rebuilt[f] != block[f] for f in ("strata", "estimator", "seed_digest", "sample_digest")):
                    raise ValueError("D3 prespecified numerical block replay failed")
                replay_calls += n
        rebuilt_summary = summarize_blocks(record["blocks"], guide, design, k, probability_first=arm is not None)
        if rebuilt_summary != record["summary"]:
            raise ValueError("D3 all-record covariance/summary changed")

    rows = []
    for fit, allocation, confirmation in zip(saved["fits"], saved["allocations"], saved["confirmations"], strict=True):
        cell, rep = fit["cell"], fit["rep"]
        ident = cell+"-"+str(rep)
        bound_unit("fit-"+ident, fit, 59520, 180.)
        if any((r["cell"], r["rep"]) != (cell, rep) for r in (allocation, confirmation)):
            raise ValueError("D3 fit/allocation/confirmation identity")
        guide = prepared[cell][1]
        validate_fit(fit, guide)
        for arm in ARMS:
            w = torch.tensor(fit["weights"][arm], dtype=torch.float64)
            GaussianMixtureShiftSpec(guide.means, w)
            if abs(float(w[0])-.1) > 1e-12 or (w < .5*guide.weights-1e-14).any():
                raise ValueError("D3 defensive domain invalid")
        if fit["weights"]["U"] != guide.weights.tolist():
            raise ValueError("D3 uniform geometry changed")
        if rep == 0:
            rebuilt_fit = fit_pair(prepared[cell], cell, rep)
            fields = ("weights", "fits", "selections", "bank_diagnostics", "train_seed_digests", "train_sample_digests", "selection_seed_digest", "selection_sample_digest")
            if any(rebuilt_fit[f] != fit[f] for f in fields):
                raise ValueError("D3 training/selection replay mismatch")
            replay_calls += 59520
        for arm in ARMS:
            pilot = saved["probability_pilots"][ident+arm]
            bound_unit("prob-pilot-"+ident+"-"+arm, pilot, 16384, 60.)
            bank_check(pilot, fit, "iid", 16384, "prob-pilot-"+arm, arm, True)
            allocation_expected = allocate(pilot["summary"]["estimator"]["relative_se"][0],
                pilot["summary"]["sensitivities"][0]["between_unit_relative_se"], kind="probability")
            if allocation_expected != allocation["probability"][arm]:
                raise ValueError("D3 probability allocation policy changed")
            n = allocation_expected["count"]
            if n is not None:
                record = confirmation["probability"][arm]
                bound_unit("prob-confirm-"+ident+"-"+arm, record, n, 120.)
                bank_check(record, fit, "iid", n, "prob-confirm-"+arm, arm, False)
            elif arm in confirmation["probability"]:
                raise ValueError("unallocated probability improperly sampled")
        for design in DESIGNS:
            pilot = saved["risk_pilots"][ident+design]
            bound_unit("risk-pilot-"+ident+"-"+design, pilot, 19840, 90.)
            bank_check(pilot, fit, design, 19840, "risk-pilot", None, True)
        if allocation["risk"] != risk_allocation(saved["risk_pilots"][ident+"stratified"]):
            raise ValueError("shared primary risk allocation changed")
        n = allocation["risk"]["count"]
        if n is not None:
            for design in DESIGNS:
                record = confirmation["risk"][design]
                bound_unit("risk-confirm-"+ident+"-"+design, record, n, 180.)
                bank_check(record, fit, design, n, "risk-confirm", None, False)
        elif confirmation["risk"]:
            raise ValueError("unallocated shared risk group improperly sampled")
        contrasts, assessed = assess(fit, confirmation["probability"], confirmation["risk"], saved["probability_pilots"], setup)
        if contrasts != confirmation["comparisons"]:
            raise ValueError("D3 paired comparison policy changed")
        rows.extend(assessed)
    for suffix in ("start", "finish"):
        if {p.name.removesuffix("."+suffix+".json") for p in directory.glob("*."+suffix+".json")} != expected:
            raise ValueError("D3 incomplete or extra journal units")
    if (rows != saved["rows"] or repeated_decision(rows, tuple(prepared)) != saved["decision"]
            or journal.accounting("D3")[0] != saved["actual_work"] or saved["actual_work"] > 12522240
            or any(directory.glob("*.violation.json")) or time.perf_counter()-started > 1800 or replay_calls > 876000):
        raise ValueError("D3 complete decision/work/replay budget mismatch")
    return {"status": "D3_all_moments_allocations_decision_and_prespecified_replay_pass",
        "audit_volterra_evaluations": replay_calls, "decision": saved["decision"]["decision"]}
