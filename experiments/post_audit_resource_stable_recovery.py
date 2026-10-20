"""Append-only completion of the original grid after verified stable AC power."""

from __future__ import annotations

import argparse
import ctypes
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
    joint_work,
    prepare,
    unit_values,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import (
    DIRECTORY as PREVIOUS,
)
from experiments.post_audit_resource_sleep_recovery import (
    ORIGINAL,
    ROOT,
    load_original,
    recovery_summary,
    sha,
)
from experiments.post_audit_resource_sleep_recovery import (
    validate_power as validate_previous_power,
)
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/p2_fine_resource_stable_recovery_v2/session_20261019"
PLAN = "docs/plans/STABLE_POWER_FIXED_GRID_RECOVERY_2026-10-19_KO.md"
THIS = "experiments/post_audit_resource_stable_recovery.py"
OLD_LOADER = "experiments/post_audit_resource_sleep_recovery.py"


def read_power() -> dict[str, int]:
    if os.name != "nt":
        raise RuntimeError("Windows AC telemetry required")

    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flags", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("reserved", ctypes.c_ubyte), ("remaining", ctypes.c_uint32), ("full", ctypes.c_uint32)]

    status = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise OSError("GetSystemPowerStatus failed")
    return {"ac_connected": int(status.ac), "battery_percent": int(status.percent)}


def validate_power(status: dict[str, int], *, minimum: int = 25) -> None:
    if status["ac_connected"] != 1 or not minimum <= status["battery_percent"] <= 100:
        raise RuntimeError("stable AC and adequate battery reserve required")


def check_power_pair(before: dict[str, int], after: dict[str, int]) -> None:
    validate_power(before)
    validate_power(after)
    if after["battery_percent"] < before["battery_percent"]-1:
        raise RuntimeError("battery declined by at least two percentage points during one whole")


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    original, records, pending, original_bindings = load_original()
    directory = ROOT/PREVIOUS
    source = json.loads((directory/"source.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    audit_snapshot(ROOT, source, manifest["config"])
    if (manifest["source"] != source["snapshot_sha256"] or manifest["config"]["original_config"] != original
            or manifest["config"]["original_bindings"] != original_bindings
            or manifest["config"]["computed_units"] != [identity(j) for j in pending]
            or manifest["expected_units"] != [identity(j) for j in pending]
            or sha(ROOT/OLD_LOADER) != source["snapshot_file_hashes"][OLD_LOADER]
            or (directory/"result.json").exists()):
        raise ValueError("changed previous recovery binding/runtime/completion")
    journal = RecoveryJournal(directory, manifest)
    bindings = {ORIGINAL+"/"+name: digest for name, digest in original_bindings.items()}
    for name in ("source.json", "manifest.json", "run.source.zip", "preparation.finish.json"):
        bindings[PREVIOUS+"/"+name] = sha(directory/name)
    journal._read(directory/"preparation.finish.json")
    for k, j in enumerate(pending):
        path = directory/(identity(j)+".finish.json")
        if k < 3:
            finish = journal._read(path)
            bindings[PREVIOUS+"/"+path.name] = sha(path)
            if k < 2:
                if finish["status"] != "complete":
                    raise ValueError("documented reused whole is not complete")
                r = finish["result"]
                check_unit(j, r)
                if r["torch_threads"] != 8:
                    raise ValueError("changed previous backend")
                validate_previous_power(r["power_before"])
                validate_previous_power(r["power_after"])
                records.append(r)
            elif (finish["status"] != "protocol_failure" or finish["error"] !=
                  "RuntimeError: connect AC power and charge battery to at least 20 percent"):
                raise ValueError("unexpected prior power failure")
        elif path.exists() or (directory/(identity(j)+".start.json")).exists():
            raise ValueError("unexpected further previous units")
    if len(records) != 22 or sum(2*joint_work(j) for j in pending[2:]) != 1278464:
        raise ValueError("changed 22-reused/10-computed fixed grid")
    return original, records, pending[2:], bindings


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    power = read_power()
    validate_power(power, minimum=30)
    original, records, pending, bindings = load_inputs()
    config = {"original_config": original, "anchors": bindings, "power_at_start": power,
              "reused_units": [identity(r) for r in records], "computed_units": [identity(j) for j in pending],
              "runtime_hashes": {p: sha(ROOT/p) for p in (THIS, OLD_LOADER)}, "torch_threads": 8,
              "unit_wall": 1200., "phase_wall": 21600., "minimum_battery": 25, "maximum_unit_drop": 1}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *bindings))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": config["computed_units"],
                "limits": {"T": {"work": 1300000, "wall": 21600.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}

    def preparation() -> dict[str, Any]:
        prepared.update(prepare(original))
        return {"actual_work": 0}

    journal.unit("preparation", phase="T", work=0, maximum_wall=30., operation=preparation)
    for j in pending:
        before = read_power()
        validate_power(before)

        def operation(job: dict[str, Any] = j, power_before: dict[str, int] = before) -> dict[str, Any]:
            cpu = time.process_time()
            r = unit_values(job, prepared[job["cell"]])
            check_unit(job, r)
            r.update({"power_before": power_before, "power_after": read_power(),
                      "cpu_process_seconds": time.process_time()-cpu})
            if r["torch_threads"] != 8 or psutil.Process().memory_info().rss > 4*1024**3:
                raise RuntimeError("changed backend or RSS guard")
            return r

        r = journal.unit(identity(j), phase="T", work=2*joint_work(j), maximum_wall=1200., operation=operation)
        try:
            check_power_pair(r["power_before"], r["power_after"])
        except RuntimeError as error:
            write_json_atomic_nonoverwriting(directory/"power_failure.json", {
                "unit": identity(j), "before": r["power_before"], "after": r["power_after"], "error": str(error)})
            raise
        records.append(r)
        print(f"stable-power grid {len(records)}/32 (22 reused, {len(records)-22}/10 computed); battery {r['power_after']['battery_percent']}%", flush=True)
    work = journal.accounting("T")[0]
    output = {"config": config, "source": source, "manifest": manifest, "summaries": recovery_summary(records),
              "units": 32, "reused_units": 22, "newly_computed_units": 10,
              "new_physical_forward": work, "new_joint_backward": work//2, "new_timestep_forward": work//2*6144,
              "full_grid_physical_forward": 4108288, "discarded_prior_failure_reserved_physical": 258560,
              "original_protocol_pass": False, "previous_recovery_protocol_pass": False,
              "phase_wall": time.time()-journal._read(directory/"phase-T.clock.json")["started_unix"]}
    if work != 1278464 or output["phase_wall"] > 21600:
        raise ValueError("changed work or phase cap")
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-T.clock.json")["started_unix"] > 21600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    original, records, pending, bindings = load_inputs()
    if (bindings != config["anchors"] or config["original_config"] != original
            or config["reused_units"] != [identity(r) for r in records]
            or config["computed_units"] != [identity(j) for j in pending]
            or manifest["config"] != config or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["expected_units"] != config["computed_units"]
            or any(sha(ROOT/p) != digest for p, digest in config["runtime_hashes"].items())
            or config["torch_threads"] != 8 or config["unit_wall"] != 1200. or config["phase_wall"] != 21600.
            or config["minimum_battery"] != 25 or config["maximum_unit_drop"] != 1
            or manifest["limits"] != {"T": {"work": 1300000, "wall": 21600.}}):
        raise ValueError("changed source/original grid/runtime")
    validate_power(config["power_at_start"], minimum=30)
    journal = RecoveryJournal(path.parent, manifest)
    expected_ids = {"preparation", *config["computed_units"]}
    for suffix in ("start", "finish"):
        if {p.name.removesuffix("."+suffix+".json") for p in path.parent.glob("*."+suffix+".json")} != expected_ids:
            raise ValueError("unexpected journal membership")
    for j in pending:
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if (finish["status"] != "complete" or finish["work"] != 2*joint_work(j)
                or finish["reserved_work"] != 2*joint_work(j) or finish["phase"] != "T" or finish["wall"] > 1200.):
            raise ValueError("incomplete new grid")
        r = finish["result"]
        check_unit(j, r)
        if r["torch_threads"] != 8:
            raise ValueError("changed numerical backend")
        check_power_pair(r["power_before"], r["power_after"])
        records.append(r)
    if (recovery_summary(records) != output["summaries"] or journal.accounting("T")[0] != 1278464
            or output["new_physical_forward"] != 1278464 or output["new_joint_backward"] != 639232
            or output["new_timestep_forward"] != 3927441408 or output["full_grid_physical_forward"] != 4108288
            or output["discarded_prior_failure_reserved_physical"] != 258560
            or output["units"] != 32 or output["reused_units"] != 22 or output["newly_computed_units"] != 10
            or output["original_protocol_pass"] or output["previous_recovery_protocol_pass"]
            or output["phase_wall"] > 21600 or any((path.parent/name).exists() for name in
                 ("power_failure.json", "phase-T.violation.json", "final_io_failure.json"))):
        raise ValueError("changed summary/cost or operational failure")
    prepared, work = prepare(original), 0
    for r in records:
        if r["rep"] != 0:
            continue
        replay = unit_values(r, prepared[r["cell"]])
        if any(replay[k] != r[k] for k in ("log_normalizer", "log_component_normalizers", "stages")):
            raise ValueError("fixed-original-seed replay mismatch")
        work += replay["actual_work"]
    if work != 513536:
        raise ValueError("replay grid changed")
    return {"status": "stable_power_full_fixed_grid_replay_pass", "replayed_wholes": 4,
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
