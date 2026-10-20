"""Recover one interrupted fixed block; preserve the original incomplete protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.post_audit_physical_window_reference import (
    DRAW_COUNT,
    WORK,
    execution_paths,
    geometry_binding,
    identity,
    jobs,
    prepare,
    seed,
    summarize,
    unit_values,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

ORIGINAL = execution_paths("battery_independent")[2]
DIRECTORY = "results/post_audit/physical_window_reference_recovery_v1/session_20261019"
PLAN = "docs/plans/PHYSICAL_WINDOW_INTERRUPTION_RECOVERY_2026-10-19_KO.md"


def validate_record(j: dict[str, Any], r: dict[str, Any], geometries: dict[str, Any]) -> None:
    digest = hashlib.sha256(json.dumps([seed(j, k, role) for k in range(32)
                                      for role in ("path", "label", "augmentation")]).encode()).hexdigest()
    if (any(r[k] != v for k, v in j.items()) or r["seed_digest"] != digest
            or r["actual_work"] != WORK or r["density_evaluations"] != DRAW_COUNT
            or r["time_step_forward_evaluations"] != DRAW_COUNT*6144 or r["torch_threads"] != 8
            or r["geometry"] != geometries[f"{j['cell']}-{j['method']}"]
            or any(k in r for k in ("power_before", "power_after"))):
        raise ValueError("fixed record identity/cost/geometry changed")


def load_original() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    directory = ROOT/ORIGINAL
    source = json.loads((directory/"source.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    config = manifest["config"]
    audit_snapshot(ROOT, source, config)
    if (config["execution_profile"] != "battery_independent" or config["torch_threads"] != 8
            or manifest["source"] != source["snapshot_sha256"]
            or manifest["expected_units"] != [identity(j) for j in jobs()]
            or any((directory/p).exists() for p in ("result.json", "phase-W.violation.json", "final_io_failure.json", "power_failure.json"))
            or any(sha(ROOT/p) != d for p, d in config["anchors"].items())):
        raise ValueError("unexpected original protocol or anchors")
    # Comparator changes are unrelated postprocessing, never imported by the producer.
    runtime = {p: d for p, d in source["snapshot_file_hashes"].items()
               if p.startswith(("src/", "experiments/"))
               and p != "experiments/post_audit_fine_reference_comparison.py"}
    if any(sha(ROOT/p) != d for p, d in runtime.items()):
        raise ValueError("original numerical runtime changed")
    journal = RecoveryJournal(directory, manifest)
    paths = ["source.json", "manifest.json", "run.source.zip", "preparation.start.json", "preparation.finish.json", "phase-W.clock.json"]
    prep = journal._read(directory/"preparation.finish.json")
    if (prep["status"] != "complete" or prep["result"]["linear_geometry_rows"] != 124
            or prep["work"] != 0 or prep["wall"] > 60.):
        raise ValueError("incomplete geometry preparation")
    geometries = prep["result"]["geometry"]
    if {p.name for p in directory.glob("*.start.json")} != {"preparation.start.json", *(identity(j)+".start.json" for j in jobs())}:
        raise ValueError("unexpected original start membership")
    if {p.name for p in directory.glob("*.finish.json")} != {"preparation.finish.json", *(identity(j)+".finish.json" for j in jobs()[:-1])}:
        raise ValueError("only documented 31 complete + final interrupted allowed")
    records = []
    for j in jobs():
        start_name = identity(j)+".start.json"
        start = journal._read(directory/start_name)
        if start["phase"] != "W" or start["reserved_work"] != WORK or start["reserved_wall"] != 180.:
            raise ValueError("changed reservation")
        paths.append(start_name)
        if j == jobs()[-1]:
            continue
        name = identity(j)+".finish.json"
        f = journal._read(directory/name)
        if f["status"] != "complete" or f["work"] != WORK or f["wall"] > 180.:
            raise ValueError("invalid original completion")
        validate_record(j, f["result"], geometries)
        records.append(f["result"])
        paths.append(name)
    return config, records, {ORIGINAL+"/"+p: sha(directory/p) for p in paths}


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, records, bindings = load_original()
    config = {"original_config": original, "bindings": bindings,
              "runtime_hashes": {p: sha(ROOT/p) for p in ("experiments/post_audit_physical_window_recovery.py",)},
              "reused_units": [identity(r) for r in records], "computed_unit": identity(jobs()[-1])}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *bindings))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "limits": {"R": {"work": WORK, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    prepared = {}

    def preparation():
        prepared.update(prepare(original))
        return {"actual_work": 0, "linear_geometry_rows": 124,
                "geometry": {f"{c}-{m}": geometry_binding(v[0]) for (c, m), v in prepared.items()}}

    prep = journal.unit("preparation", phase="R", work=0, maximum_wall=60., operation=preparation)
    j = jobs()[-1]
    r = journal.unit(identity(j), phase="R", work=WORK, maximum_wall=180.,
                     operation=lambda: unit_values(j, prepared[(j["cell"], j["method"])]))
    validate_record(j, r, prep["geometry"])
    records.append(r)
    output = {"config": config, "source": source, "manifest": manifest, "geometry": prep["geometry"],
              "summaries": summarize(records), "units": 32, "reused_units": 31, "new_units": 1,
              "new_physical_evaluations": WORK, "new_density_evaluations": DRAW_COUNT,
              "new_timestep_forward_evaluations": DRAW_COUNT*6144,
              "discarded_interrupted_reserved_physical": WORK, "full_grid_physical": 32*WORK,
              "original_protocol_pass": False, "scope": "fixed_grid_numerical_recovery_not_confirmation_or_wall_benchmark",
              "phase_wall": time.time()-journal._read(directory/"phase-R.clock.json")["started_unix"]}
    if journal.accounting("R")[0] != WORK or output["phase_wall"] > 1800.:
        raise ValueError("changed recovery budget")
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-R.clock.json")["started_unix"] > 1800.:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final recovery I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    original, records, bindings = load_original()
    if (config["original_config"] != original or config["bindings"] != bindings
            or config["reused_units"] != [identity(r) for r in records] or config["computed_unit"] != identity(jobs()[-1])
            or any(sha(ROOT/p) != d for p, d in config["runtime_hashes"].items())
            or manifest != {"config": config, "source": output["source"]["snapshot_sha256"],
                            "limits": {"R": {"work": WORK, "wall": 1800.}}}):
        raise ValueError("changed recovery source/membership")
    journal = RecoveryJournal(path.parent, manifest)
    for suffix in ("start", "finish"):
        if {p.name for p in path.parent.glob("*."+suffix+".json")} != {"preparation."+suffix+".json", identity(jobs()[-1])+"."+suffix+".json"}:
            raise ValueError("unexpected recovery journal membership")
    finish = journal._read(path.parent/(identity(jobs()[-1])+".finish.json"))
    if finish["status"] != "complete" or finish["work"] != WORK or finish["wall"] > 180.:
        raise ValueError("failed recovery block")
    records.append(finish["result"])
    prepared = prepare(original)
    geometries = {f"{c}-{m}": geometry_binding(v[0]) for (c, m), v in prepared.items()}
    for j, r in zip(jobs(), records, strict=True):
        validate_record(j, r, geometries)
        if j["rep"] == 0 and unit_values(j, prepared[(j["cell"], j["method"])])["moments"] != r["moments"]:
            raise ValueError("numerical replay mismatch")
    if (summarize(records) != output["summaries"] or output["geometry"] != geometries
            or output["units"] != 32 or output["reused_units"] != 31 or output["new_units"] != 1
            or output["new_physical_evaluations"] != WORK or output["new_density_evaluations"] != DRAW_COUNT
            or output["new_timestep_forward_evaluations"] != DRAW_COUNT*6144
            or output["discarded_interrupted_reserved_physical"] != WORK or output["full_grid_physical"] != 32*WORK
            or output["original_protocol_pass"] or output["phase_wall"] > 1800.
            or journal.accounting("R")[0] != WORK or (path.parent/"phase-R.violation.json").exists()
            or (path.parent/"final_io_failure.json").exists()):
        raise ValueError("changed recovery summary/cost")
    return {"status": "physical_window_source_geometry_seed_moment_cost_replay_pass", "replayed_blocks": 4,
            "audit_physical_evaluations": 4*WORK, "audit_density_evaluations": 4*DRAW_COUNT,
            "original_protocol_pass": False}


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
        print(json.dumps({k: v for k, v in run().items() if k in ("units", "new_physical_evaluations", "phase_wall")}))
