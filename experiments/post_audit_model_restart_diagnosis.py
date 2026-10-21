"""Zero-physics, source-bound post-hoc triage of the original-model restart."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import ARMS, CELLS
from experiments.post_audit_small_grid_model_restart import DIRECTORY as PARENT_DIRECTORY
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/model_restart_diagnosis_v1/session_20261020"
ANCHORS = tuple(PARENT_DIRECTORY+"/"+n+".json" for n in ("result", "audit"))


def diagnose(summary: dict[str, Any]) -> dict[str, Any]:
    if (len(summary["rows"]) != 40 or len(summary["fits"]) != 10
            or len(summary["heldout_risk"]) != 10):
        raise ValueError("complete model-development evidence required")
    banks, rows, comparisons = [], [], []
    for fit in summary["fits"]:
        for kind in ("event", "risk"):
            d = fit["fit_diagnostics"][kind]
            banks.append({"cell": fit["cell"], "rep": fit["rep"], "kind": kind,
                "bank_count": d["bank_count"], "ess": d["ess"], "ess_fraction": d["ess"]/d["bank_count"],
                "maximum_weight_fraction": d["max_fraction"],
                "alpha": fit["selections"][kind]["alpha"],
                "no_op": fit["selections"][kind]["no_op"]})
    for risk in summary["heldout_risk"]:
        for arm in ARMS:
            direct = next(r for r in summary["rows"] if (r["cell"], r["rep"], r["arm"]) == (risk["cell"], risk["rep"], arm))
            heldout = risk["risk_moments"][arm]
            rows.append({"cell": risk["cell"], "rep": risk["rep"], "arm": arm,
                "direct_probability_rse": direct["moments"]["relative_se"],
                "heldout_m2_rse": heldout["relative_se"],
                "heldout_m2_maximum_single_fraction": heldout["maximum_fraction"],
                "heldout_over_direct_observed_m2": math.exp(heldout["log_mean"]-direct["moments"]["log_second_moment"]),
                "not_population_ratio_or_bias_proof": True})
        for c in risk["contrasts"]:
            comparisons.append({"cell": risk["cell"], "rep": risk["rep"],
                "baseline": c["baseline"], "candidate": c["candidate"],
                "point_m2_reduction": c["raw"]["relative_m2_reduction"],
                "relative_only_signal": c["raw"]["candidate_lower_m2_signal"] and c["block"]["candidate_lower_m2_signal"],
                "absolute_and_relative_signal": c["development_reduction_signal"],
                "same_proposal": c["same_proposal"]})
    cells = []
    for cell in CELLS:
        details = []
        for arm in ARMS:
            group = [r for r in rows if (r["cell"], r["arm"]) == (cell, arm)]
            details.append({"arm": arm,
                "median_heldout_m2_rse": statistics.median(r["heldout_m2_rse"] for r in group),
                "maximum_heldout_single_fraction": max(r["heldout_m2_maximum_single_fraction"] for r in group),
                "median_heldout_over_direct_observed_m2": statistics.median(r["heldout_over_direct_observed_m2"] for r in group)})
        cells.append({"cell": cell, "methods": details,
            "bank_ess_ranges": {k: [min(r["ess"] for r in banks if r["cell"] == cell and r["kind"] == k),
                max(r["ess"] for r in banks if r["cell"] == cell and r["kind"] == k)] for k in ("event", "risk")}})
    return {"scope": "post_hoc_fixed_evidence_triage_not_new_hypothesis_test", "banks": banks, "rows": rows,
        "comparisons": comparisons, "cells": cells, "additional_physical_evaluations": 0,
        "development_candidate_arms": summary["development_candidate_arms"],
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    parent, receipt = [json.loads((ROOT/p).read_text()) for p in ANCHORS]
    audit_snapshot(ROOT, parent["source"], parent["config"])
    if (receipt["status"] != "small_grid_original_model_source_seed_fit_selection_final_risk_replay_pass"
            or receipt["result_sha256"] != sha(ROOT/ANCHORS[0])):
        raise ValueError("complete audited original-model evidence required")
    config = {"anchors": {p: sha(ROOT/p) for p in ANCHORS}, "additional_physics": 0}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=ANCHORS)
    output = {"config": config, "source": source, "summary": diagnose(parent["summary"])}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    return output


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["cells"]))
