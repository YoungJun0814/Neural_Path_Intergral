"""Zero-physics synthesis of the complete audited all-path ensemble experiment."""

from __future__ import annotations

import json
import math
from typing import Any

from experiments.post_audit_bank_ensemble_coverage import DIRECTORY as PARENT_DIRECTORY
from experiments.post_audit_bank_ensemble_coverage import PLAN, THEORY
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/bank_ensemble_full_path_analysis_v1/session_20261020"


def analyze(output: dict[str, Any]) -> dict[str, Any]:
    source = output["summary"]["rows"]
    if len(source) != 8 or len({(r["cell"], r["arm"]) for r in source}) != 8:
        raise ValueError("complete two-cell four-arm family required")
    rows = []
    for r in source:
        m, ledger = r["summary"], r["coverage"]
        if m["count"] != 65536 or ledger["count"] != 65536 or len(r["bin_summaries"]) != 96:
            raise ValueError("all fixed final samples/bins required")
        observed = [b for b in r["bin_summaries"] if b["sample_count"]]
        total = sum(b["observed_contribution_fraction"] for b in observed)
        square_total = sum(b["observed_squared_contribution_fraction"] for b in observed)
        if abs(total-1) > 1e-10 or abs(square_total-1) > 1e-10:
            raise ValueError("joint bins do not partition observed first/second contributions")
        groups = {}
        for key, length in (("intensity_bin", 6), ("time_bin", 4), ("window_bin", 4)):
            groups[key] = [{"bin": k,
                "sample_count": sum(b["sample_count"] for b in observed if b[key] == k),
                "observed_contribution_fraction": sum(b["observed_contribution_fraction"] for b in observed if b[key] == k),
                "observed_squared_contribution_fraction": sum(b["observed_squared_contribution_fraction"] for b in observed if b[key] == k)}
                for k in range(length)]
        top = ledger["top"]
        forecast = math.ceil(3*m["count"]*(m["relative_se"]/.015)**2/8192)*8192
        rows.append({"cell": r["cell"], "arm": r["arm"], "mean": math.exp(m["log_mean"]),
            "raw_relative_se": m["relative_se"],
            "between_block_relative_se": r["block_sensitivity"]["between_unit_relative_se"],
            "maximum_sample_fraction": m["maximum_fraction"],
            "maximum_sample_observed_square_fraction": math.exp(2*top[0]["log_contribution"]-m["log_sum_squares"]),
            "top_sixteen_observed_contribution_fraction": sum(math.exp(t["log_contribution"]-m["log_sum"]) for t in top),
            "unobserved_joint_bins": 96-len(observed), "physical_bin_groups": groups,
            "precision_1p5_pass": r["precision_1p5_pass"],
            "fresh_safety3_precision_forecast": {"draws": forecast, "unit_multiple": 8192,
                "estimated_score_wall": forecast/m["count"]*r["score_wall"],
                "authorized": False, "scope": "pilot_projection_not_precision_guarantee_or_optional_extension"}})
    return {"rows": rows, "comparisons": output["summary"]["comparisons"],
            "new_physical_evaluations": 0,
            "scope": "observed_all_path_partitions_not_physical_mode_clustering_or_unseen_tail_certificate",
            "next_priority": "fresh_fixed_precision_design_and_guide_independent_corroboration_not_spike_only_geometry",
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    paths = [PARENT_DIRECTORY+"/"+name+".json" for name in ("result", "audit")]
    parent, audit = [json.loads((ROOT/p).read_text()) for p in paths]
    audit_snapshot(ROOT, parent["source"], parent["config"])
    if (audit["status"] != "bank_ensemble_all_path_source_seed_feature_moment_replay_pass"
            or audit["result_sha256"] != sha(ROOT/paths[0]) or audit["replayed_blocks"] != 8):
        raise ValueError("complete source-bound parent and replay receipt required")
    config = {"anchors": {p: sha(ROOT/p) for p in paths}}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, *paths))
    result = {"config": config, "source": source, "analysis": analyze(parent)}
    write_json_atomic_nonoverwriting(directory/"result.json", result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"new_physical_evaluations": result["analysis"]["new_physical_evaluations"],
        "ensemble_forecasts": [{"cell": r["cell"], **r["fresh_safety3_precision_forecast"]}
            for r in result["analysis"]["rows"] if r["arm"] == "ensemble"]}))
