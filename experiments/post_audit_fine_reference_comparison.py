"""Complete-family SMC versus IID guide cross-check; zero new physics calls."""

from __future__ import annotations

import json
from typing import Any

from experiments.post_audit_p2_fine_resource_sweep import CELLS, DESIGNS
from experiments.post_audit_physical_window_recovery import DIRECTORY as GUIDE_DIRECTORY
from experiments.post_audit_physical_window_reference import METHODS
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_completion import DIRECTORY as RESOURCE_DIRECTORY
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import relative_equivalence

PLAN = "docs/plans/FINE_REFERENCE_CROSSMECHANISM_COMPARISON_2026-10-19_KO.md"
DIRECTORY = "results/post_audit/fine_reference_crosscheck_v1/session_20261019"


def interval_diagnosis(equivalence: dict[str, Any]) -> dict[str, Any]:
    """Interpret the existing adjusted development CI, not an extra test/gate."""
    if equivalence["status"] == "unresolved_numerical_range":
        return {"status": "unresolved_numerical_range", "interval": None}
    delta = equivalence["symmetric_relative_difference"]
    radius = equivalence["confidence_z"]*equivalence["standard_error"]
    lower, upper = delta-radius, delta+radius
    margin = equivalence["margin"]
    status = ("inside_equivalence_margin" if equivalence["pass"] else
              "interval_entirely_outside_margin" if lower > margin or upper < -margin else
              "uncertainty_crosses_equivalence_margin")
    return {"status": status, "interval": [lower, upper],
            "scope": "same_eight_adjusted_delta_method_intervals_not_an_additional_test"}


def compare(resource: dict[str, Any], guide: dict[str, Any]) -> dict[str, Any]:
    smc, iid = resource["summaries"]["rows"], guide["summaries"]["rows"]
    if (len(smc) != 4 or {(r["cell"], r["design"]) for r in smc} != {(c, d) for c in CELLS for d in DESIGNS}
            or len(iid) != 4 or {(r["cell"], r["method"]) for r in iid} != {(c, m) for c in CELLS for m in METHODS}):
        raise ValueError("complete fixed cross-mechanism grid required")
    rows = []
    for s in smc:
        for g in iid:
            if s["cell"] != g["cell"]:
                continue
            for r in (s, g):
                if [x["grid"] for x in r["components"]] != ["fine", "coarse"]:
                    raise ValueError("fine/coarse component convention changed")
            a, b = s["components"][0]["summary"], g["components"][0]["summary"]
            equivalence = relative_equivalence(a, b, comparisons=8)
            rows.append({"cell": s["cell"], "smc_design": s["design"], "iid_guide": g["method"],
                         "smc_fine_log_mean": a["log_mean"], "iid_fine_log_mean": b["log_mean"],
                         "smc_fine_rse": a["relative_se"], "iid_fine_rse": b["relative_se"],
                         "both_reported_rse_at_most_1_5_percent": max(a["relative_se"], b["relative_se"]) <= .015,
                         "equivalence": equivalence, "interval_diagnosis": interval_diagnosis(equivalence)})
    agreements = all(r["equivalence"]["pass"] for r in rows)
    precision = all(r["both_reported_rse_at_most_1_5_percent"] for r in rows)
    next_step = ("investigate_reference_disagreement_or_uncertainty" if not agreements else
                 "preregister_fresh_fixed_count_precision_study" if not precision else
                 "preregister_new_confirmation_and_tail_coverage_gate")
    return {"rows": rows, "comparisons": 8, "all_development_agreements_pass": agreements,
            "all_reported_precision_probes_pass": precision, "next_priority": next_step,
            "multiplicity_scope": "eight_crossmechanism_fine_mu_development_comparisons_only",
            "new_physical_evaluations": 0, "methods_certified_for_wall_comparison": False,
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    guide_path = GUIDE_DIRECTORY+"/result.json"
    paths = [RESOURCE_DIRECTORY+"/result.json", RESOURCE_DIRECTORY+"/audit.json",
             guide_path, guide_path.replace("result.json", "audit.json")]
    resource, resource_audit, guide, guide_audit = [json.loads((ROOT/p).read_text()) for p in paths]
    for output in (resource, guide):
        audit_snapshot(ROOT, output["source"], output["config"])
    if (resource_audit["status"] != "battery_independent_full_fixed_grid_replay_pass"
            or guide_audit["status"] != "physical_window_source_geometry_seed_moment_cost_replay_pass"
            or resource_audit["result_sha256"] != sha(ROOT/paths[0]) or guide_audit["result_sha256"] != sha(ROOT/paths[2])
            or resource_audit["replayed_wholes"] != 4 or guide_audit["replayed_blocks"] != 4):
        raise ValueError("complete numerical replay receipts required")
    config = {"anchors": {p: sha(ROOT/p) for p in paths}, "comparisons": 8, "margin": .1, "alpha": .05}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *paths))
    output = {"config": config, "source": source, "comparison": compare(resource, guide)}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    return output


if __name__ == "__main__":
    output = run()
    print(json.dumps({k: v for k, v in output["comparison"].items() if k != "rows"}))
