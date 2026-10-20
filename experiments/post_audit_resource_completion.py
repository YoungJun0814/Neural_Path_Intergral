"""Finish original fixed-grid numerical records without battery-based stopping."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.post_audit_p2_fine_resource_sweep import (
    check_unit,
    identity,
    joint_work,
    prepare,
    unit_values,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, recovery_summary, sha
from experiments.post_audit_resource_stable_recovery import DIRECTORY as PREVIOUS
from experiments.post_audit_resource_stable_recovery import load_inputs as previous_inputs
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/p2_fine_resource_completion_v3/session_20261019"
PLAN = "docs/plans/BATTERY_INDEPENDENT_RECOVERY_AND_GUIDE_EXECUTION_2026-10-19_KO.md"
THIS = "experiments/post_audit_resource_completion.py"
PREVIOUS_LOADER = "experiments/post_audit_resource_stable_recovery.py"
POLICY = "battery_checks_disabled_by_explicit_user_request_2026-10-19"


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    original, records, pending, bindings = previous_inputs()
    directory = ROOT/PREVIOUS
    source = json.loads((directory/"source.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    config = manifest["config"]
    audit_snapshot(ROOT, source, config)
    if (config["anchors"] != bindings or config["original_config"] != original
            or manifest["source"] != source["snapshot_sha256"]
            or config["reused_units"] != [identity(r) for r in records]
            or config["computed_units"] != [identity(j) for j in pending]
            or manifest["expected_units"] != config["computed_units"]
            or any(sha(ROOT/p) != digest for p, digest in config["runtime_hashes"].items())
            or sha(ROOT/PREVIOUS_LOADER) != source["snapshot_file_hashes"][PREVIOUS_LOADER]
            or (directory/"result.json").exists()):
        raise ValueError("changed previous runtime/grid/source")
    journal = RecoveryJournal(directory, manifest)
    expected_ids = {"preparation", identity(pending[0])}
    for suffix in ("start", "finish"):
        if {p.name.removesuffix("."+suffix+".json") for p in directory.glob("*."+suffix+".json")} != expected_ids:
            raise ValueError("unexpected previous unit membership")
    finish = journal._read(directory/(identity(pending[0])+".finish.json"))
    journal._read(directory/"preparation.finish.json")
    if finish["status"] != "complete" or finish["work"] != 2*joint_work(pending[0]):
        raise ValueError("previous numerical whole is incomplete")
    check_unit(pending[0], finish["result"])
    if finish["result"]["torch_threads"] != 8:
        raise ValueError("changed backend")
    failure = json.loads((directory/"power_failure.json").read_text())
    if failure["unit"] != identity(pending[0]) or failure["error"] != "battery declined by at least two percentage points during one whole":
        raise ValueError("unexpected historical failure")
    for name in ("source.json", "manifest.json", "run.source.zip", "power_failure.json",
                 "preparation.finish.json", identity(pending[0])+".finish.json"):
        bindings[PREVIOUS+"/"+name] = sha(directory/name)
    records.append(finish["result"])
    if len(records) != 23 or sum(2*joint_work(j) for j in pending[1:]) != 1149184:
        raise ValueError("changed 23-reused/9-computed grid")
    return original, records, pending[1:], bindings


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, records, pending, bindings = load_inputs()
    config = {"original_config": original, "anchors": bindings, "battery_policy": POLICY,
              "reused_units": [identity(r) for r in records], "computed_units": [identity(j) for j in pending],
              "runtime_hashes": {p: sha(ROOT/p) for p in (THIS, PREVIOUS_LOADER)}, "torch_threads": 8}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *bindings))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": config["computed_units"],
                "limits": {"C": {"work": 1200000, "wall": 21600.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}

    def preparation() -> dict[str, Any]:
        prepared.update(prepare(original))
        return {"actual_work": 0}

    journal.unit("preparation", phase="C", work=0, maximum_wall=30., operation=preparation)
    for j in pending:
        def operation(job: dict[str, Any] = j) -> dict[str, Any]:
            cpu = time.process_time()
            r = unit_values(job, prepared[job["cell"]])
            check_unit(job, r)
            r["cpu_process_seconds"] = time.process_time()-cpu
            if r["torch_threads"] != 8 or psutil.Process().memory_info().rss > 4*1024**3:
                raise RuntimeError("changed backend or RSS guard")
            return r

        records.append(journal.unit(identity(j), phase="C", work=2*joint_work(j), maximum_wall=1200., operation=operation))
        print(f"fixed-grid completion {len(records)}/32 (23 reused, {len(records)-23}/9 computed)", flush=True)
    work = journal.accounting("C")[0]
    output = {"config": config, "source": source, "manifest": manifest, "summaries": recovery_summary(records),
              "units": 32, "reused_units": 23, "newly_computed_units": 9,
              "new_physical_forward": work, "new_joint_backward": work//2, "new_timestep_forward": work//2*6144,
              "full_grid_physical_forward": 4108288, "discarded_failed_reserved_physical": 258560,
              "historical_operational_protocols_pass": False,
              "phase_wall": time.time()-journal._read(directory/"phase-C.clock.json")["started_unix"]}
    if work != 1149184 or output["phase_wall"] > 21600:
        raise ValueError("changed cost or phase cap")
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-C.clock.json")["started_unix"] > 21600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    original, records, pending, bindings = load_inputs()
    if (config["battery_policy"] != POLICY or config["torch_threads"] != 8 or config["anchors"] != bindings
            or config["original_config"] != original or config["reused_units"] != [identity(r) for r in records]
            or config["computed_units"] != [identity(j) for j in pending] or manifest["config"] != config
            or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["expected_units"] != config["computed_units"]
            or manifest["limits"] != {"C": {"work": 1200000, "wall": 21600.}}
            or any(sha(ROOT/p) != h for p, h in config["runtime_hashes"].items())):
        raise ValueError("changed source/grid/policy")
    journal = RecoveryJournal(path.parent, manifest)
    for suffix in ("start", "finish"):
        if {p.name.removesuffix("."+suffix+".json") for p in path.parent.glob("*."+suffix+".json")} != {"preparation", *config["computed_units"]}:
            raise ValueError("unexpected new journal membership")
    for j in pending:
        f = journal._read(path.parent/(identity(j)+".finish.json"))
        if (f["status"] != "complete" or f["phase"] != "C" or f["work"] != 2*joint_work(j)
                or f["reserved_work"] != 2*joint_work(j) or f["wall"] > 1200.):
            raise ValueError("invalid unit completion/cost")
        check_unit(j, f["result"])
        if f["result"]["torch_threads"] != 8:
            raise ValueError("changed backend")
        records.append(f["result"])
    if (recovery_summary(records) != output["summaries"] or journal.accounting("C")[0] != 1149184
            or output["units"] != 32 or output["reused_units"] != 23 or output["newly_computed_units"] != 9
            or output["new_physical_forward"] != 1149184 or output["new_joint_backward"] != 574592
            or output["new_timestep_forward"] != 3530293248 or output["full_grid_physical_forward"] != 4108288
            or output["discarded_failed_reserved_physical"] != 258560 or output["historical_operational_protocols_pass"]
            or output["phase_wall"] > 21600 or any((path.parent/p).exists() for p in ("phase-C.violation.json", "final_io_failure.json"))):
        raise ValueError("changed summary/cost or operational failure")
    prepared, work = prepare(original), 0
    for r in records:
        if r["rep"] == 0:
            replay = unit_values(r, prepared[r["cell"]])
            if any(replay[k] != r[k] for k in ("log_normalizer", "log_component_normalizers", "stages")):
                raise ValueError("whole numerical replay mismatch")
            work += replay["actual_work"]
    if work != 513536:
        raise ValueError("changed replay work")
    return {"status": "battery_independent_full_fixed_grid_replay_pass", "replayed_wholes": 4,
            "audit_physical_forward": work, "audit_joint_backward": work//2,
            "historical_operational_protocols_pass": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        path = ROOT/args.audit
        receipt = {**audit(path), "result_sha256": sha(path)}
        write_json_atomic_nonoverwriting(path.parent/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        print(json.dumps({k: v for k, v in run().items() if k in ("units", "new_physical_forward", "phase_wall")}))
