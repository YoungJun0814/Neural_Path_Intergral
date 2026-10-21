"""D0: source-bound, zero-new-physics reconstruction from all 240 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.post_audit_dictionary_model_learning import (
    DIRECTORY as PARENT,
)
from experiments.post_audit_dictionary_model_learning import (
    final_id,
    final_jobs,
    seed,
    stream_digest,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.direct_second_moment_diagnostics import direct_m2_blocks
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/bounded_direct_m2_v1/session_20261020_archive_v2"
STARTUP_FAILURE = "results/post_audit/bounded_direct_m2_v1/session_20261020/startup_failure.json"
PLAN = "docs/plans/BOUNDED_RESEARCH_CONTINUE_PIVOT_STOP_EXECUTION_PLAN_2026-10-20_KO.md"
EXPECTED_PARENT = "3964e74bb620da10b34a5c8256a46286b3a46a8eb5ab676031314fe647273033"
EXPECTED_AUDIT = "7183ff46af14cb4fcdd9b768fac9a6bf54063d43ca682f0bd618d0190e011de2"


def inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    directory = ROOT/PARENT
    if sha(directory/"result.json") != EXPECTED_PARENT or sha(directory/"audit.json") != EXPECTED_AUDIT:
        raise ValueError("immutable prior result/audit changed")
    parent, receipt = [json.loads((directory/(n+".json")).read_text()) for n in ("result", "audit")]
    audit_snapshot(ROOT, parent["source"], parent["config"])
    if receipt["result_sha256"] != EXPECTED_PARENT or not receipt["status"].endswith("replay_pass"):
        raise ValueError("audited parent required")
    if json.loads((directory/"manifest.json").read_text()) != parent["manifest"]:
        raise ValueError("parent journal manifest mismatch")
    journal = RecoveryJournal(directory, parent["manifest"])
    records, hashes = [], {}
    for j in final_jobs():
        name = final_id(j)
        start, finish = [journal._read(directory/(name+"."+s+".json")) for s in ("start", "finish")]
        r = finish["result"]
        fit = next(f for f in parent["summary"]["fits"] if (f["cell"], f["rep"]) == (j["cell"], j["rep"]))
        if (finish["status"] != "complete" or start["reserved_work"] != 32768 or finish["work"] != 32768
                or r["moments"]["count"] != 32768 or r["actual_work"] != 32768
                or any(r[k] != v for k, v in j.items())
                or r["seed_digest"] != stream_digest(j, "final-"+j["arm"], 32768, j["block"])
                or r["weight_digest"] != fit["weight_digests"][j["arm"]]):
            raise ValueError("complete bound final checkpoint required")
        records.append(r)
        for suffix in ("start", "finish"):
            path = directory/(name+"."+suffix+".json")
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    return parent, records, hashes


def analyze(parent: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 240 or len({final_id(r) for r in records}) != 240:
        raise ValueError("all 240 checkpoint blocks required")
    rows = []
    for row in parent["summary"]["rows"]:
        group = sorted((r for r in records if (r["cell"], r["rep"], r["arm"]) == (row["cell"], row["rep"], row["arm"])), key=lambda r: r["block"])
        direct = direct_m2_blocks([r["moments"] for r in group], bootstrap_seed=seed(row, "bounded-direct-x2"),
            expected_log_second=row["moments"]["log_second_moment"])
        heldout = next(r for r in parent["summary"]["heldout_risk"] if (r["cell"], r["rep"]) == (row["cell"], row["rep"]))
        h = heldout["risk_moments"][row["arm"]]
        rows.append({"cell": row["cell"], "rep": row["rep"], "arm": row["arm"], "direct": direct,
            "probability_rse": row["moments"]["relative_se"], "heldout_m2_raw_rse": h["relative_se"],
            "heldout_m2_block_rse": heldout["sensitivity"][row["arm"]]["between_unit_relative_se"],
            "heldout_over_observed_direct_m2": math.exp(h["log_mean"]-direct["log_mean_m2"]),
            "ratio_is_not_population_or_bias_certificate": True})
    return {"rows": rows, "estimators": len(rows), "checkpoints": len(records),
        "additional_physical_evaluations": 0, "d0_evidence_gate": len(rows) == 30,
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    parent, records, hashes = inputs()
    config = {"parent": EXPECTED_PARENT, "audit": EXPECTED_AUDIT, "checkpoint_hashes": hashes, "physics": 0,
        "startup_failure": sha(ROOT/STARTUP_FAILURE), "archive_protocol": "v2_hash_bound_checkpoints_no_oversized_argv"}
    directory = ROOT/DIRECTORY
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("one-shot evidence; no overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config,
        extra_snapshot_paths=(PLAN, PARENT+"/result.json", PARENT+"/audit.json", PARENT+"/manifest.json", STARTUP_FAILURE))
    result = {"config": config, "source": source, "summary": analyze(parent, records)}
    write_json_atomic_nonoverwriting(directory/"result.json", result)
    return result


def audit(path: Path) -> dict[str, Any]:
    saved = json.loads(path.read_text())
    parent, records, hashes = inputs()
    config = {"parent": EXPECTED_PARENT, "audit": EXPECTED_AUDIT, "checkpoint_hashes": hashes, "physics": 0,
        "startup_failure": sha(ROOT/STARTUP_FAILURE), "archive_protocol": "v2_hash_bound_checkpoints_no_oversized_argv"}
    audit_snapshot(ROOT, saved["source"], config)
    if saved["config"] != config or saved["summary"] != analyze(parent, records):
        raise ValueError("zero-physics complete recomputation mismatch")
    return {"status": "bounded_D0_all240_checkpoint_binding_recompute_pass", "result_sha256": sha(path),
        "additional_physical_evaluations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        receipt = audit(ROOT/DIRECTORY/"result.json")
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        result = run()
        print(json.dumps({k: v for k, v in result["summary"].items() if k != "rows"}))
