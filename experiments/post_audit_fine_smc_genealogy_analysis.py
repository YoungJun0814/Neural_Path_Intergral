"""Zero-physics diagnosis of the audited fine-only independent-reference failure."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

from experiments.post_audit_fresh_ensemble_fine_smc import (
    ARMS,
    CELLS,
    PLAN,
    THEORY,
    smc_id,
    smc_jobs,
    smc_work,
)
from experiments.post_audit_fresh_ensemble_fine_smc import (
    DIRECTORY as PARENT_DIRECTORY,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/fine_smc_genealogy_analysis_v1/session_20261020"


def analyze(output: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 32 or {smc_id(r) for r in records} != {smc_id(j) for j in smc_jobs()}:
        raise ValueError("complete independent whole family required")
    rows = []
    for cell in CELLS:
        for arm in ARMS:
            group = sorted((r for r in records if r["cell"] == cell and r["arm"] == arm), key=lambda r: r["rep"])
            summary = next(r for r in output["summary"]["smc_rows"] if r["cell"] == cell and r["arm"] == arm)
            crossing = {}
            for threshold in (64, 32, 16):
                values = [next((s["beta"] for s in r["stages"] if s["surviving_initial_ancestors"] <= threshold), None) for r in group]
                found = [v for v in values if v is not None]
                crossing[str(threshold)] = {"observed_wholes": len(found),
                    "first_crossing_beta_min": min(found) if found else None,
                    "first_crossing_beta_median": statistics.median(found) if found else None,
                    "first_crossing_beta_max": max(found) if found else None}
            count = math.ceil(3*8*(summary["summary"]["relative_se"]/.015)**2)
            rows.append({"cell": cell, "arm": arm, "initial_particles": 128,
                "ancestry_threshold_first_crossing": crossing,
                "independent_whole_rse": summary["summary"]["relative_se"],
                "initial_ancestors_final": summary["initial_ancestors"],
                "min_stage_ess_range": [min(summary["minimum_stage_ess"]), max(summary["minimum_stage_ess"])],
                "fresh_safety3_whole_forecast": {"wholes": count, "physical_evaluations": count*smc_work(arm),
                    "estimated_score_hours": count*summary["score_wall"]/8/3600, "authorized": False,
                    "scope": "small_pilot_projection_not_precision_guarantee_or_execution_authorization"}})
    return {"iid_precision_pass": {r["cell"]: r["precision_and_sensitivity"]["pass"] for r in output["summary"]["iid_rows"]},
        "smc_rows": rows, "new_physical_evaluations": 0,
        "diagnosis": "early_genealogical_collapse_observed_not_proven_unique_cause_or_target_bias",
        "next_priority": "separate_pilot_frozen_early_bridge_and_mutation_contrast_not_blind_whole_count_escalation",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    paths = [PARENT_DIRECTORY+"/"+name+".json" for name in ("result", "audit")]
    output, receipt = [json.loads((ROOT/p).read_text()) for p in paths]
    audit_snapshot(ROOT, output["source"], output["config"])
    if (receipt["status"] != "fresh_ensemble_fine_only_smc_source_seed_feature_stage_replay_pass"
            or receipt["result_sha256"] != sha(ROOT/paths[0]) or receipt["replayed_iid_blocks"] != 4
            or receipt["replayed_smc_wholes"] != 4):
        raise ValueError("complete source-bound numerical replay required")
    parent = ROOT/PARENT_DIRECTORY
    journal = RecoveryJournal(parent, output["manifest"])
    units = [PARENT_DIRECTORY+"/"+smc_id(j)+".finish.json" for j in smc_jobs()]
    records = [journal._read(ROOT/p)["result"] for p in units]
    config = {"anchors": {p: sha(ROOT/p) for p in (*paths, *units)}}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, *paths, *units))
    result = {"config": config, "source": source, "analysis": analyze(output, records)}
    write_json_atomic_nonoverwriting(directory/"result.json", result)
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"new_physical_evaluations": 0, "iid_precision_pass": output["analysis"]["iid_precision_pass"],
                      "next_priority": output["analysis"]["next_priority"]}))
