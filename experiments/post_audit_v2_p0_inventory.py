"""P0 only: read-only historical audits, additive contracts, no target sampling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import psutil
import yaml

from experiments.post_audit_r2_family_diagnosis import freeze_source
from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.structural_v2_contract import RULES, SCHEMA
from src.path_integral.structural_v2_inventory import (
    audit_snapshot,
    file_sha256,
    scan_legacy,
    workspace_file,
)

ROOT = Path(__file__).resolve().parents[1]


def run(config: dict[str, Any], source: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    if config["schema"] != "npi.structural-v2.p0-config.v1":
        raise ValueError("unsupported P0 configuration")
    paths = config["historical_artifacts"]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("empty/duplicate evidence grid")
    started, process = time.perf_counter(), psutil.Process()
    peak_rss = process.memory_info().rss
    current = source_tree_digest(root)
    if source["config_digest"] != canonical_digest(config) or source["source_tree_digest"] != current:
        raise ValueError("P0 runtime/config binding changed after source freeze")
    budget = config["budget"]
    if any(budget[k] != 0 for k in ("max_new_target_samples", "max_potential_evaluations",
                                   "max_new_model_candidates", "max_whole_training_runs")):
        raise ValueError("P0 must not authorize target sampling or model training")

    def check() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        if time.perf_counter()-started >= budget["max_wall_seconds"]:
            raise TimeoutError("unresolved_inventory_wall_budget")
        if peak_rss > budget["max_process_rss_bytes"]:
            raise TimeoutError("unresolved_inventory_memory_budget")

    records = []
    for path in paths:
        try:
            check()
            item = scan_legacy(root, path, current_source_digest=current, check_limits=check)
            item["status"] = "binding_pass_not_performance_validation"
            records.append(item)
        except (ValueError, KeyError, OSError, TimeoutError) as error:
            records.append({"path": path, "status": "unresolved" if isinstance(error, TimeoutError) else "fail",
                            "failure_reason": f"{type(error).__name__}: {error}"})
            if isinstance(error, TimeoutError):
                break
    documents = []
    for name in config["evidence_documents"]:
        path = workspace_file(root, name)
        sha, size = file_sha256(path)
        documents.append({"path": name, "sha256": sha, "bytes": size})
    passed = len(records) == len(paths) and all(r["status"] == "binding_pass_not_performance_validation" for r in records)
    return {"schema": "npi.structural-v2.p0-inventory.v1", "config": config, "source": source,
            "records": records, "documents": documents,
            "measurement_schema": SCHEMA,
            "measurement_rules": [{"estimand_kind": a, "estimator_kind": b, "se_unit": se,
                                   "allowed_sample_roles": sorted(roles)}
                                  for (a, b), (se, roles) in sorted(RULES.items())],
            "status": "pass_with_historical_limitations" if passed else "unresolved",
            "p0_binding_grid_complete": passed, "historical_source_mismatches": sum(
                r.get("current_runtime_source_matches") is False for r in records),
            "new_target_samples": 0, "new_potential_evaluations": 0, "new_model_training_runs": 0,
            "validation_test_internal_calls_measured": False,
            "physical_wall_seconds": time.perf_counter()-started, "sampled_peak_process_rss_bytes": peak_rss,
            "scientific_gates": {"independent_reference": "not_run", "mesh": "not_run",
                                 "stein_correctness": "not_run", "performance_dominance": "not_authorized",
                                 "confirmation": "not_run", "novelty": "review_pending"},
            "limitations": ["Historical binding audits are not full numerical replay or statistical certification.",
                            "Historical runtime-source mismatch is disclosed, not repaired by rewriting evidence.",
                            "V2 control declarations do not establish global bounds or field correctness.",
                            "Tests and source inventory do not prove rare-tail coverage or independence."]}


def audit(payload: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    if payload["schema"] != "npi.structural-v2.p0-inventory.v1":
        raise ValueError("unsupported P0 inventory")
    config, source = payload["config"], payload["source"]
    audit_snapshot(root, source, config)
    if [r["path"] for r in payload["records"]] != config["historical_artifacts"]:
        raise ValueError("incomplete/changed P0 evidence grid")
    for old in payload["records"]:
        new = scan_legacy(root, old["path"], current_source_digest=source["source_tree_digest"])
        if {k: v for k, v in old.items() if k != "status"} != new or old["status"] != "binding_pass_not_performance_validation":
            raise ValueError("stale or altered inventory row")
    for document in payload["documents"]:
        if file_sha256(workspace_file(root, document["path"])) != (document["sha256"], document["bytes"]):
            raise ValueError("evidence document changed after P0 freeze")
    if [d["path"] for d in payload["documents"]] != config["evidence_documents"]:
        raise ValueError("missing/undeclared evidence documents")
    expected_rules = [{"estimand_kind": a, "estimator_kind": b, "se_unit": se,
                       "allowed_sample_roles": sorted(roles)}
                      for (a, b), (se, roles) in sorted(RULES.items())]
    if (payload["measurement_schema"] != SCHEMA or payload["measurement_rules"] != expected_rules
            or payload["historical_source_mismatches"] != sum(
                r["current_runtime_source_matches"] is False for r in payload["records"])):
        raise ValueError("V2 declaration or source-mismatch summary changed")
    if (payload["status"] != "pass_with_historical_limitations" or payload["p0_binding_grid_complete"] is not True
            or any(payload[k] != 0 for k in ("new_target_samples", "new_potential_evaluations", "new_model_training_runs"))
            or payload["scientific_gates"] != {"independent_reference": "not_run", "mesh": "not_run",
                 "stein_correctness": "not_run", "performance_dominance": "not_authorized",
                 "confirmation": "not_run", "novelty": "review_pending"}):
        raise ValueError("unauthorized P0 claim/sampling status")
    return {"status": "p0_fresh_binding_audit_pass_not_statistical_certificate",
            "historical_artifacts": len(payload["records"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/post_audit/structural_v2_p0_v1.yaml")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(json.loads(args.audit.read_text(encoding="utf-8"))), allow_nan=False))
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT/config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    source = freeze_source(output, config, extra_snapshot_paths=tuple(config["evidence_documents"]))
    result = run(config, source)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "status": result["status"],
                      "artifacts": len(result["records"]), "wall_seconds": result["physical_wall_seconds"]}))
    if not result["p0_binding_grid_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
