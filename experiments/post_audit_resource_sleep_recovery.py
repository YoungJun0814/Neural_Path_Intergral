"""Complete the original fixed 32-seed grid in a separate recovery artifact.

The failed original protocol remains failed. This is not fresh confirmation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.post_audit_p2_fine_resource_sweep import (
    check_unit,
    identity,
    jobs,
    joint_work,
    prepare,
    summarize,
    unit_values,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "results/post_audit/p2_fine_resource_v2/session_20261016"
DIRECTORY = "results/post_audit/p2_fine_resource_sleep_recovery_v1/session_20261016"
PLAN = "docs/plans/RESOURCE_V2_SLEEP_RECOVERY_2026-10-16_KO.md"
NAMESPACE = "paired-fine-resources-v2"
FAILED = "high-eta200-k1-more-temperatures-4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power_status() -> dict[str, int]:
    """Read-only Windows status; never change critical battery or sleep policy."""
    if os.name != "nt":
        raise RuntimeError("local Windows AC status required")

    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flags", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("reserved", ctypes.c_ubyte), ("remaining", ctypes.c_uint32), ("full", ctypes.c_uint32)]

    value = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)):
        raise OSError("GetSystemPowerStatus failed")
    status = {"ac_connected": int(value.ac), "battery_percent": int(value.percent)}
    validate_power(status)
    return status


def validate_power(status: dict[str, int]) -> None:
    if status["ac_connected"] != 1 or not 20 <= status["battery_percent"] <= 100:
        raise RuntimeError("connect AC power and charge battery to at least 20 percent")


def load_original() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    directory = ROOT/ORIGINAL
    source = json.loads((directory/"source.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    config = manifest["config"]
    audit_snapshot(ROOT, source, config)
    if (manifest["source"] != source["snapshot_sha256"] or config["seed_namespace"] != NAMESPACE
            or config["torch_threads"] != 8 or manifest["expected_units"] != [identity(j) for j in jobs(NAMESPACE)]
            or (directory/"result.json").exists()):
        raise ValueError("unexpected original protocol binding/completion")
    # Archive identity is separate from equality of the currently executed runtime.
    for name, digest in source["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/", "configs/")) and sha(ROOT/name) != digest:
            raise ValueError(f"original runtime changed: {name}")
    journal = RecoveryJournal(directory, manifest)
    reused, pending = [], []
    bindings = {name: sha(directory/name) for name in ("source.json", "manifest.json", "run.source.zip")}
    failed_seen = False
    for j in jobs(NAMESPACE):
        name = identity(j)+".finish.json"
        path = directory/name
        if path.exists():
            finish = journal._read(path)
            bindings[name] = sha(path)
            if finish["status"] == "complete":
                check_unit(j, finish["result"])
                if finish["result"]["torch_threads"] != 8 or failed_seen:
                    raise ValueError("non-prefix or changed backend")
                reused.append(finish["result"])
            elif identity(j) == FAILED and finish["status"] == "protocol_failure":
                if finish["error"] != "TimeoutError: complete unit exceeded wall reservation":
                    raise ValueError("unexpected original failure")
                failed_seen = True
                pending.append(j)
            else:
                raise ValueError("unexpected failed unit")
        else:
            if not failed_seen or (directory/(identity(j)+".start.json")).exists():
                raise ValueError("unexpected missing/interrupted original unit")
            pending.append(j)
    if len(reused) != 20 or len(pending) != 12 or not failed_seen:
        raise ValueError("only the documented 20+12 fixed-grid recovery is allowed")
    return config, reused, pending, bindings


def recovery_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize(records)
    for row in result["rows"]:
        forecast = row["fresh_precision_forecast"]
        forecast["mixed_power_state_wall_forecast_unqualified"] = forecast.pop("wall_shared_laptop_seconds")
    result["scope"] = "complete_original_fixed_seed_grid_recovery_not_fresh_confirmation_or_wall_benchmark"
    return result


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    power = power_status()
    original, reused, pending, bindings = load_original()
    config = {"original_config": original, "original_bindings": bindings, "power_at_start": power,
              "reused_units": [identity(r) for r in reused], "computed_units": [identity(j) for j in pending],
              "torch_threads": 8, "unit_wall": 1200., "phase_wall": 21600.}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config,
                           extra_snapshot_paths=(PLAN, *(ORIGINAL+"/"+name for name in bindings)))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": config["computed_units"],
                "limits": {"S": {"work": 1600000, "wall": 21600.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}

    def preparation() -> dict[str, Any]:
        prepared.update(prepare(original))
        return {"actual_work": 0}

    journal.unit("preparation", phase="S", work=0, maximum_wall=30., operation=preparation)
    records = list(reused)
    for j in pending:
        status = power_status()

        def operation(job: dict[str, Any] = j, before: dict[str, int] = status) -> dict[str, Any]:
            cpu = time.process_time()
            r = unit_values(job, prepared[job["cell"]])
            check_unit(job, r)
            r.update({"power_before": before, "power_after": power_status(), "cpu_process_seconds": time.process_time()-cpu})
            if r["torch_threads"] != 8 or psutil.Process().memory_info().rss > 4*1024**3:
                raise RuntimeError("recovery backend/RSS guard")
            return r

        records.append(journal.unit(identity(j), phase="S", work=2*joint_work(j), maximum_wall=1200., operation=operation))
        print(f"fixed-seed sleep recovery {len(records)}/32 (20 reused, {len(records)-20}/12 computed)", flush=True)
    work = journal.accounting("S")[0]
    output = {"config": config, "source": source, "manifest": manifest, "summaries": recovery_summary(records),
              "units": 32, "reused_units": 20, "newly_computed_units": 12, "new_physical_forward": work,
              "new_joint_backward": work//2, "new_timestep_forward": work//2*6144,
              "full_grid_physical_forward": 4108288, "discarded_original_failure_reserved_physical": 129280,
              "original_protocol_pass": False, "phase_wall": time.time()-journal._read(directory/"phase-S.clock.json")["started_unix"]}
    if work != 1537024 or output["phase_wall"] > 21600:
        raise ValueError("recovery count/wall mismatch")
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-S.clock.json")["started_unix"] > 21600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("recovery final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config = output["config"]
    audit_snapshot(ROOT, output["source"], config)
    original, records, pending, bindings = load_original()
    if (bindings != config["original_bindings"] or original != config["original_config"]
            or output["manifest"]["source"] != output["source"]["snapshot_sha256"]
            or output["manifest"]["config"] != config
            or config["reused_units"] != [identity(r) for r in records]
            or config["computed_units"] != [identity(j) for j in pending]
            or output["manifest"]["expected_units"] != config["computed_units"]):
        raise ValueError("changed original/recovery bindings")
    journal = RecoveryJournal(path.parent, output["manifest"])
    for j in pending:
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("incomplete recovery grid")
        check_unit(j, finish["result"])
        if finish["result"]["torch_threads"] != 8:
            raise ValueError("changed recovery backend")
        validate_power(finish["result"]["power_before"])
        validate_power(finish["result"]["power_after"])
        records.append(finish["result"])
    if (recovery_summary(records) != output["summaries"] or journal.accounting("S")[0] != 1537024
            or output["new_physical_forward"] != 1537024 or output["new_joint_backward"] != 768512
            or output["new_timestep_forward"] != 4721737728 or output["full_grid_physical_forward"] != 4108288
            or output["units"] != 32 or output["reused_units"] != 20 or output["newly_computed_units"] != 12
            or output["original_protocol_pass"] or output["phase_wall"] > 21600
            or (path.parent/"phase-S.violation.json").exists() or (path.parent/"final_io_failure.json").exists()):
        raise ValueError("changed summary/count or false qualification")
    prepared = prepare(original)
    work = 0
    for r in records:
        if r["rep"] != 0:
            continue
        replay = unit_values(r, prepared[r["cell"]])
        if any(replay[k] != r[k] for k in ("log_normalizer", "log_component_normalizers", "stages")):
            raise ValueError("original-seed numerical replay mismatch")
        work += replay["actual_work"]
    return {"status": "full_fixed_grid_sleep_recovery_replay_pass", "replayed_wholes": 4,
            "audit_physical_forward": work, "audit_joint_backward": work//2, "original_protocol_pass": False}


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
