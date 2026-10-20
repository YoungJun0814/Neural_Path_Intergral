"""Source-bound zero-physics synthesis; observed labels are not physical modes."""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Any

from experiments.post_audit_bridge_mutation_allocation import DIRECTORY as ALLOCATION_DIRECTORY
from experiments.post_audit_matched_window_mechanism import ARMS, KWARGS
from experiments.post_audit_matched_window_mechanism import DIRECTORY as MATCHED_DIRECTORY
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/window_priority_analysis_v1/session_20261019"
PLAN = "docs/plans/WINDOW_PRIORITY_POSTPROCESSING_2026-10-19_KO.md"


def decode_label(label: int, centres: list[int]) -> dict[str, Any]:
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 373 or len(centres) != 31:
        raise ValueError("fixed 373-component dictionary label required")
    if label == 0:
        return {"kind": "natural", "template": None, "centre_step": None, "amplitude": 0., "price_amplitude": 0.}
    i, rem = divmod(label-1, 12)
    a, b = divmod(rem, 3)
    return {"kind": "shift", "template": i, "centre_step": centres[i],
            "amplitude": KWARGS["amplitudes"][a], "price_amplitude": KWARGS["price_amplitudes"][b]}


def observed_label_summary(diagnostic: dict[str, Any], centres: list[int]) -> dict[str, Any]:
    counts, sums, total = diagnostic["label_counts"], diagnostic["label_log_sums"], diagnostic["log_sum"]
    if len(counts) != 373 or len(sums) != 373 or sum(counts) != diagnostic["count"]:
        raise ValueError("incomplete component diagnostics")
    labels = [{"label": k, **decode_label(k, centres), "sample_count": counts[k],
               "observed_contribution_fraction": math.exp(v-total)} for k, v in enumerate(sums) if v is not None]
    if abs(sum(x["observed_contribution_fraction"] for x in labels)-1) > 1e-10:
        raise ValueError("observed label mass does not reconstruct total")
    price_mass = {str(b): sum(x["observed_contribution_fraction"] for x in labels
                             if x["kind"] == "shift" and x["price_amplitude"] == b) for b in (0., 2., 4.)}
    natural = next((x["observed_contribution_fraction"] for x in labels if x["label"] == 0), 0.)
    return {"top_ten_sampler_components": sorted(labels, key=lambda x: x["observed_contribution_fraction"], reverse=True)[:10],
            "natural_observed_fraction": natural, "shift_price_amplitude_observed_fractions": price_mass,
            "top_sixteen_sample_contribution_fraction": sum(math.exp(r["log_contribution"]-total) for r in diagnostic["top"]),
            "top_sixteen_features": diagnostic["top"], "feature_names": diagnostic["feature_names"],
            "observed_weighted_feature_means": diagnostic["observed_weighted_feature_means"],
            "scope": "sampler_label_contribution_diagnostics_not_path_clustering_or_unseen_tail_certificate"}


def analyze(allocation: dict[str, Any], matched: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for r in matched["summary"]["rows"]:
        g = matched["geometry"][f"{r['cell']}-{r['arm']}"]
        rows.append({"cell": r["cell"], "arm": r["arm"], "fine": r["components"][0]["summary"],
                     "coarse": r["components"][1]["summary"],
                     "diagnostics": observed_label_summary(r["diagnostics"], g["centres"])})
    if len(rows) != 6 or {r["arm"] for r in rows} != set(ARMS):
        raise ValueError("complete matched mechanism family required")
    for family, count in ((allocation["g1_crosscheck"], 4), (matched["summary"]["contrasts"], 6)):
        expected_z = NormalDist().inv_cdf(1-.05/(2*count))
        if any(abs(r["equivalence"]["confidence_z"]-expected_z) > 1e-12 for r in family):
            raise ValueError("actual multiplicity differs from preregistered family")
    return {"rows": rows, "contrasts": matched["summary"]["contrasts"],
            "smc_development_selection": allocation["summary"]["selected_development_arm"],
            "smc_rows": allocation["summary"]["rows"], "new_physical_evaluations": 0,
            "multiplicity": {"smc_g1_comparisons": 4, "matched_comparisons": 6,
                "numerical_z_verified": True,
                "historical_helper_scope_label_caveat": "interval_diagnosis scope string says eight; actual numeric z uses four or six; archives preserved"},
            "next_priority": "fresh_two_mechanism_precision_design_and_calibrated_path_geometry_not_large_training",
            "methods_certified_for_wall_comparison": False, "p1_qualified": False,
            "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    paths = [d+"/"+f+".json" for d in (ALLOCATION_DIRECTORY, MATCHED_DIRECTORY) for f in ("result", "audit")]
    allocation, aa, matched, ma = [json.loads((ROOT/p).read_text()) for p in paths]
    for output in (allocation, matched):
        audit_snapshot(ROOT, output["source"], output["config"])
    if (aa["status"] != "fixed_bridge_mutation_source_seed_moment_replay_pass" or aa["replayed_wholes"] != 4
            or ma["status"] != "matched_window_moment_diagnostic_geometry_seed_replay_pass" or ma["replayed_blocks"] != 6
            or aa["result_sha256"] != sha(ROOT/paths[0]) or ma["result_sha256"] != sha(ROOT/paths[2])):
        raise ValueError("source-bound complete replay receipts required")
    config = {"anchors": {p: sha(ROOT/p) for p in paths}}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *paths))
    output = {"config": config, "source": source, "analysis": analyze(allocation, matched)}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    return output


if __name__ == "__main__":
    output = run()
    print(json.dumps({k: v for k, v in output["analysis"].items() if k not in ("rows", "smc_rows", "contrasts")}))
