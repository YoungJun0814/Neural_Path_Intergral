"""Zero-physics post-hoc concentration diagnosis, never a coverage certificate."""

from __future__ import annotations

import json
import math
from typing import Any

from experiments.post_audit_defensive_allocation import DIRECTORY as ALLOCATION_DIRECTORY
from experiments.post_audit_defensive_allocation import PLAN
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/defensive_allocation_tail_diagnosis_v1/session_20261020"


def diagnose(output: dict[str, Any]) -> dict[str, Any]:
    rows = []
    source_rows = output["summary"]["rows"]
    if len(source_rows) != 8 or any(r["summary"]["count"] != 32768 for r in source_rows):
        raise ValueError("complete fixed allocation family required")
    for r in source_rows:
        paths, moments = r["top_sixteen_paths"], r["summary"]
        if len(paths) != 16:
            raise ValueError("complete retained tail ledger required")
        first = math.exp(paths[0]["log_contribution"]-moments["log_sum"])
        if abs(first-moments["maximum_fraction"]) > 1e-10:
            raise ValueError("top path and raw moments mismatch")
        fractions = [math.exp(p["log_contribution"]-moments["log_sum"]) for p in paths]
        square_fractions = [math.exp(2*p["log_contribution"]-moments["log_sum_squares"]) for p in paths]
        time_mass = [sum(f for p, f in zip(paths, fractions, strict=True)
                         if q/4 <= p["maximum_left_variance_time"] < (q+1)/4) for q in range(4)]
        rows.append({"cell": r["cell"], "rep": r["rep"], "arm": r["arm"],
            "maximum_sample_share_of_observed_squared_contributions": square_fractions[0],
            "top_sixteen_observed_contribution_share": sum(fractions),
            "top_sixteen_share_of_observed_squared_contributions": sum(square_fractions),
            "top_sixteen_quarter_peak_time_observed_mass_only": time_mass,
            "largest_path": paths[0], "largest_path_integrated_variance": math.exp(paths[0]["log_integrated_variance"]),
            "largest_path_maximum_variance": math.exp(paths[0]["log_maximum_left_variance"]),
            "top_only_scope": "selected_16_contributing_samples_not_full_path_clusters_or_population_mode_probabilities"})
    fits = output["summary"]["fits"]
    overlap = []
    for cell in sorted({f["cell"] for f in fits}):
        a, b = sorted((f for f in fits if f["cell"] == cell), key=lambda f: f["rep"])
        total_variation = .5*sum(abs(x-y) for x, y in zip(a["weights"], b["weights"], strict=True))
        overlap.append({"cell": cell, "learned_weight_total_variation": total_variation,
                        "scope": "sampler_weight_instability_not_target_bias_or_physical_mode_distance"})
    return {"rows": rows, "between_fit_weight_diagnostics": overlap,
        "new_physical_evaluations": 0, "post_hoc_exploratory": True,
        "next_priority": "bank_generalization_and_geometry_coverage_before_precision_escalation",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    paths = [ALLOCATION_DIRECTORY+"/"+f+".json" for f in ("result", "audit")]
    output, receipt = [json.loads((ROOT/p).read_text()) for p in paths]
    audit_snapshot(ROOT, output["source"], output["config"])
    if (receipt["status"] != "defensive_allocation_source_bank_fit_final_replay_pass"
            or receipt["result_sha256"] != sha(ROOT/paths[0])
            or receipt["replayed_fits"] != 4 or receipt["replayed_final_blocks"] != 4):
        raise ValueError("complete allocation replay receipt required")
    config = {"anchors": {p: sha(ROOT/p) for p in paths}}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *paths))
    result = {"config": config, "source": source, "analysis": diagnose(output)}
    write_json_atomic_nonoverwriting(directory/"result.json", result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["analysis"], indent=2))
