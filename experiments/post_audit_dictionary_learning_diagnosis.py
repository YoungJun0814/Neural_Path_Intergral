"""Source-bound descriptive analysis of a complete dictionary-learning family.

No new paths, fitting, selection, hypothesis tests, or qualification changes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any

from experiments.post_audit_dictionary_model_learning import ARMS, CELLS
from experiments.post_audit_dictionary_model_learning import DIRECTORY as PARENT_DIRECTORY
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/dictionary_learning_diagnosis_v1/session_20261020"
ANCHORS = tuple(PARENT_DIRECTORY+"/"+n+".json" for n in ("result", "audit"))


def describe_weights(weights: list[float], steps: int = 32) -> dict[str, Any]:
    """Label masses using the frozen PhysicalWindowGuide enumeration, not modes."""
    if (steps != 32 or len(weights) != 373 or any(not math.isfinite(w) or w <= 0 for w in weights)
            or abs(sum(weights)-1) > 1e-12 or abs(weights[0]-.1) > 1e-12):
        raise ValueError("complete N32 defensive dictionary weights required")
    shifted = [w/(1-weights[0]) for w in weights[1:]]
    by_kind = {"point": 0., "window": 0.}
    by_amplitude = {str(a): 0. for a in (2, 4, 6, 8)}
    by_price = {str(b): 0. for b in (0, 2, 4)}
    template_mass = []
    energy = 0.
    for i in range(31):
        template_mass.append(sum(weights[1+12*i:1+12*(i+1)]))
        for ai, a in enumerate((2, 4, 6, 8)):
            for bi, b in enumerate((0, 2, 4)):
                w = weights[1+12*i+3*ai+bi]
                by_kind["point" if i < 15 else "window"] += w
                by_amplitude[str(a)] += w
                by_price[str(b)] += w
                energy += w*(a*a+b*b)
    centres = [2*k for k in range(1, 16)]+[round(4+i*24/15) for i in range(16)]
    by_time = {"early": 0., "middle": 0., "late": 0.}
    for centre, mass in zip(centres, template_mass, strict=True):
        by_time["early" if centre/steps < 1/3 else "middle" if centre/steps < 2/3 else "late"] += mass
    return {"natural_mass": weights[0], "shifted_conditional_entropy": -sum(p*math.log(p) for p in shifted),
        "shifted_effective_components": 1/sum(p*p for p in shifted),
        "largest_component_mass": max(weights[1:]), "template_masses": template_mass,
        "mass_by_template_kind": by_kind, "mass_by_vol_amplitude": by_amplitude,
        "mass_by_price_amplitude": by_price, "mass_by_time_bin": by_time,
        "weighted_squared_mean_norm": energy,
        "interpretation": "mixture_allocation_not_target_mode_probability_or_path_attribution"}


def diagnose(summary: dict[str, Any]) -> dict[str, Any]:
    expected = {(c, r) for c in CELLS for r in range(5)}
    if (len(summary["fits"]) != 10 or len(summary["rows"]) != 30 or len(summary["heldout_risk"]) != 10
            or {(f["cell"], f["rep"]) for f in summary["fits"]} != expected
            or {(f["cell"], f["rep"]) for f in summary["heldout_risk"]} != expected
            or {(r["cell"], r["rep"], r["arm"]) for r in summary["rows"]} != {(c, r, a) for c, r in expected for a in ARMS}):
        raise ValueError("complete audited dictionary-learning evidence required")
    banks, weights, rows = [], [], []
    comparisons: list[dict[str, Any]] = []
    for fit in summary["fits"]:
        event, risk = fit["event_bank_diagnostics"], fit["uniform_m2_bank_diagnostics"]
        banks.append({"cell": fit["cell"], "rep": fit["rep"], "event_ess": event["effective_sample_size"],
            "event_maximum_fraction": event["maximum_weight_fraction"], "risk_ess": risk["ess"],
            "risk_maximum_fraction": risk["max_fraction"], "training_count": event["samples"]})
        for arm in ARMS[1:]:
            uniform = fit["weights"]["uniform"]
            raw = fit["fit_diagnostics"][arm]
            selected = fit["selections"][arm]
            weights.append({"cell": fit["cell"], "rep": fit["rep"], "arm": arm,
                "alpha": selected["alpha"], "no_op": selected["no_op"],
                "raw": describe_weights(raw["weights"]), "selected": describe_weights(fit["weights"][arm]),
                "selected_total_variation_from_uniform": .5*sum(abs(w-u) for w, u in zip(fit["weights"][arm], uniform, strict=True)),
                "optimizer_iterations": raw["iterations"], "empirical_converged": raw["converged_empirical"],
                "empirical_gap": raw["empirical_gap"], "gap_units": raw["gap_units"]})
    for risk in summary["heldout_risk"]:
        for arm in ARMS:
            direct = next(r for r in summary["rows"] if (r["cell"], r["rep"], r["arm"]) == (risk["cell"], risk["rep"], arm))
            heldout = risk["risk_moments"][arm]
            rows.append({"cell": risk["cell"], "rep": risk["rep"], "arm": arm,
                "heldout_m2_rse": heldout["relative_se"], "heldout_maximum_fraction": heldout["maximum_fraction"],
                "direct_probability_rse": direct["moments"]["relative_se"],
                "heldout_over_direct_observed_m2": math.exp(heldout["log_mean"]-direct["moments"]["log_second_moment"]),
                "not_population_ratio_or_bias_proof": True,
                "direct_m2_uncertainty_not_available_from_merged_probability_moments": True})
        comparisons.extend({"cell": risk["cell"], "rep": risk["rep"], "baseline": c["baseline"],
            "candidate": c["candidate"], "point_m2_reduction": c["raw"]["relative_m2_reduction"],
            "absolute_relative_raw_block_signal": c["development_reduction_signal"],
            "same_proposal": c["same_proposal"]} for c in risk["comparisons"])
    cells = []
    for cell in CELLS:
        bank = [b for b in banks if b["cell"] == cell]
        methods = []
        for arm in ARMS:
            group = [r for r in rows if (r["cell"], r["arm"]) == (cell, arm)]
            methods.append({"arm": arm, "median_heldout_m2_rse": statistics.median(r["heldout_m2_rse"] for r in group),
                "maximum_heldout_single_fraction": max(r["heldout_maximum_fraction"] for r in group),
                "median_heldout_over_direct_observed_m2": statistics.median(r["heldout_over_direct_observed_m2"] for r in group)})
        cells.append({"cell": cell, "methods": methods,
            "training_ess_ranges": {k: [min(b[k+"_ess"] for b in bank), max(b[k+"_ess"] for b in bank)] for k in ("event", "risk")},
            "maximum_training_fraction": {k: max(b[k+"_maximum_fraction"] for b in bank) for k in ("event", "risk")}})
    return {"scope": "post_hoc_allocation_and_concentration_description_not_new_test",
        "banks": banks, "weights": weights, "rows": rows, "comparisons": comparisons, "cells": cells,
        "development_candidate_arms": summary["development_candidate_arms"],
        "additional_physical_evaluations": 0, "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def evidence() -> dict[str, Any]:
    parent, receipt = [json.loads((ROOT/p).read_text()) for p in ANCHORS]
    audit_snapshot(ROOT, parent["source"], parent["config"])
    if (receipt["status"] != "dictionary_model_source_geometry_seed_fit_selection_final_risk_replay_pass"
            or receipt["result_sha256"] != sha(ROOT/ANCHORS[0])):
        raise ValueError("complete audited dictionary-learning family required")
    return parent


def run() -> dict[str, Any]:
    parent = evidence()
    config = {"anchors": {p: sha(ROOT/p) for p in ANCHORS}, "additional_physics": 0}
    directory = ROOT/DIRECTORY
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("post-hoc evidence already exists; no overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=ANCHORS)
    output = {"config": config, "source": source, "summary": diagnose(parent["summary"])}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    return output


def audit() -> dict[str, Any]:
    parent = evidence()
    path = ROOT/DIRECTORY/"result.json"
    output = json.loads(path.read_text())
    config = {"anchors": {p: sha(ROOT/p) for p in ANCHORS}, "additional_physics": 0}
    audit_snapshot(ROOT, output["source"], config)
    if output["config"] != config or output["summary"] != diagnose(parent["summary"]):
        raise ValueError("descriptive analysis replay mismatch")
    return {"status": "dictionary_learning_zero_physics_source_bound_recompute_pass",
        "additional_physical_evaluations": 0, "result_sha256": sha(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        receipt = audit()
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        result = run()
        print(json.dumps(result["summary"]["cells"]))
