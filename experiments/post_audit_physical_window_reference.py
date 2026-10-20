"""Matched-373-component point/window paired IID reference development pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Literal

import psutil
import torch

from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_completion import DIRECTORY as COMPLETION_DIRECTORY
from experiments.post_audit_resource_completion import POLICY as BATTERY_POLICY
from experiments.post_audit_resource_stable_recovery import (
    check_power_pair,
    read_power,
    validate_power,
)
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "results/post_audit/p2_fine_resource_stable_recovery_v2/session_20261019/result.json"
RESOURCE_AUDIT = "results/post_audit/p2_fine_resource_stable_recovery_v2/session_20261019/audit.json"
BACKEND = "results/post_audit/fine_backend_calibration_v1/session_20261016/result.json"
PLAN = "docs/plans/PHYSICAL_WINDOW_REFERENCE_EXECUTION_2026-10-19_KO.md"
DIRECTORY = "results/post_audit/physical_window_reference_v1/session_20261019"
METHODS: tuple[Literal["point"], Literal["mixed-window"]] = ("point", "mixed-window")
DRAW_COUNT, CHUNK, WORK = 16384, 512, 32768


def execution_paths(profile: str) -> tuple[str, str, str, str, str]:
    if profile == "stable_power":
        return ANCHOR, RESOURCE_AUDIT, DIRECTORY, PLAN, "stable_power_full_fixed_grid_replay_pass"
    if profile == "battery_independent":
        return (COMPLETION_DIRECTORY+"/result.json", COMPLETION_DIRECTORY+"/audit.json",
                "results/post_audit/physical_window_reference_v1/session_20261019_battery_independent",
                "docs/plans/BATTERY_INDEPENDENT_RECOVERY_AND_GUIDE_EXECUTION_2026-10-19_KO.md",
                "battery_independent_full_fixed_grid_replay_pass")
    raise ValueError("unknown execution profile")


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "method": m, "rep": r} for c in CELLS for m in METHODS for r in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['method']}-{j['rep']}"


def seed(j: dict[str, Any], chunk: int, role: str) -> int:
    return int(hashlib.sha256(f"physical-window-reference-v1/{identity(j)}/{chunk}/{role}".encode()).hexdigest()[:15], 16)


def geometry_binding(guide: PhysicalWindowGuide) -> dict[str, Any]:
    return {"mean_sha256": hashlib.sha256(guide.spec.means.numpy().tobytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(guide.spec.weights.numpy().tobytes()).hexdigest(),
            "components": guide.spec.components, "centres": list(guide.centres),
            "price_window_steps": list(guide.price_window_steps), "vol_window_steps": list(guide.vol_window_steps),
            "maximum_energy": float(.5*guide.spec.means.square().sum(1).max()),
            "maximum_causal_roundoff": guide.maximum_causal_roundoff}


def prepare(config: dict[str, Any]) -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for c in config["cells"]:
        p = _problem(config["model"], c, 4096)
        pair = PairedVolterraProbability.create(p)
        for method in METHODS:
            guide = PhysicalWindowGuide.create(p, kind=method, **config["guide"])
            out[(c["id"], method)] = (guide, pair)
    return out


def unit_values(j: dict[str, Any], prepared: Any) -> dict[str, Any]:
    begin = time.perf_counter()
    guide, pair = prepared
    parts: list[list[dict[str, Any]]] = [[], []]
    for chunk in range(32):
        z = guide.sample(CHUNK, path_seed=seed(j, chunk, "path"), label_seed=seed(j, chunk, "label"))
        eps = torch.randn((CHUNK, 2048), dtype=torch.float64, generator=torch.Generator().manual_seed(seed(j, chunk, "augmentation")))
        values = pair.log_components(torch.cat((z, eps), 1))-guide.log_q_over_p(z)[:, None]
        if (values > math.log(10)+1e-9).any():
            raise ValueError("defensive probability contribution bound failed")
        for k in (0, 1):
            parts[k].append(log_moments(values[:, k]))
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("chunk RSS guard")
    return {**j, "actual_work": WORK, "density_evaluations": DRAW_COUNT,
            "time_step_forward_evaluations": DRAW_COUNT*6144, "moments": [merge_retaining_maximum(p) for p in parts],
            "geometry": geometry_binding(guide), "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-begin,
            "seed_digest": hashlib.sha256(json.dumps([seed(j, k, role) for k in range(32) for role in ("path", "label", "augmentation")]).encode()).hexdigest()}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 32 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete matched-geometry grid required")
    rows: list[dict[str, Any]] = []
    for c in CELLS:
        for method in METHODS:
            group = sorted((r for r in records if r["cell"] == c and r["method"] == method), key=lambda r: r["rep"])
            logs = torch.tensor([[r["moments"][k]["log_mean"] for k in (0, 1)] for r in group], dtype=torch.float64)
            components: list[dict[str, Any]] = [{"grid": name, "summary": merge_retaining_maximum([r["moments"][k] for r in group]),
                           "block_sensitivity": sensitivity(logs[:, k].tolist(), bootstrap_seed=seed({"cell": c, "method": method, "rep": 1000}, k, "bootstrap"))}
                          for k, name in enumerate(("fine", "coarse"))]
            worst = max(r["summary"]["relative_se"] for r in components)
            required = max(262144, 512*math.ceil(3*(8*DRAW_COUNT)*(worst/.015)**2/512))
            rows.append({"cell": c, "method": method, "components": components,
                         "paired_difference": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=4),
                         "fresh_precision_forecast": {"target_rse": .015, "safety": 3, "draws": required},
                         "scope": "reference_development_not_original_q_performance_or_continuum_certificate"})
    agreements = []
    for c in CELLS:
        group = [r for r in rows if r["cell"] == c]
        agreements.append({"cell": c, "equivalence": relative_equivalence(group[0]["components"][0]["summary"],
                                                                           group[1]["components"][0]["summary"], comparisons=2)})
    return {"rows": rows, "within_guide_mean_agreements": agreements,
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run(*, profile: str = "stable_power") -> dict[str, Any]:
    torch.set_num_threads(8)
    anchor_name, audit_name, directory_name, plan_name, receipt_status = execution_paths(profile)
    enforce_power = profile == "stable_power"
    power = read_power() if enforce_power else None
    if power is not None:
        validate_power(power, minimum=30)
    if not (ROOT/anchor_name).is_file():
        raise RuntimeError("complete and audit the original fixed resource grid recovery first")
    anchor = json.loads((ROOT/anchor_name).read_text())
    if (anchor["units"] != 32 or anchor.get("original_protocol_pass", False)
            or anchor.get("historical_operational_protocols_pass", False)):
        raise ValueError("incomplete resource anchor")
    audit_snapshot(ROOT, anchor["source"], anchor["config"])
    resource_audit = json.loads((ROOT/audit_name).read_text())
    if (resource_audit["status"] != receipt_status or resource_audit["replayed_wholes"] != 4
            or resource_audit["audit_physical_forward"] != 513536 or resource_audit["audit_joint_backward"] != 256768
            or resource_audit["result_sha256"] != hashlib.sha256((ROOT/anchor_name).read_bytes()).hexdigest()):
        raise ValueError("full resource recovery replay receipt required")
    backend = json.loads((ROOT/BACKEND).read_text())
    audit_snapshot(ROOT, backend["source"], backend["config"])
    if backend["selected_threads"] != 8 or not all(r["numerical_equivalence_pass"] for r in backend["rows"]):
        raise ValueError("eight-thread backend calibration required")
    original_config = anchor["config"]["original_config"]
    config = {"model": original_config["model"], "cells": original_config["cells"],
              "guide": {"amplitudes": [2., 4., 6., 8.], "price_amplitudes": [0., 2., 4.], "defensive_mass": .1},
              "anchors": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (anchor_name, audit_name, BACKEND)},
              "torch_threads": 8, "execution_profile": profile,
              "battery_policy": "enforced" if enforce_power else BATTERY_POLICY,
              "power_at_start": power, "minimum_battery": 25 if enforce_power else None,
              "maximum_unit_drop": 1 if enforce_power else None}
    directory = ROOT/directory_name
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(
        plan_name, anchor_name, audit_name, BACKEND, "docs/plans/PHYSICAL_WINDOW_GUIDE_ORACLE_2026-10-16_KO.md",
        "docs/plans/PHYSICAL_WINDOW_REFERENCE_PILOT_2026-10-16_KO.md",
        "docs/theory/PHYSICAL_WINDOW_DENSITY_CONSISTENCY_2026-10-16_KO.md"))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    seeds = [seed(j, k, role) for j in grid for k in range(32) for role in ("path", "label", "augmentation")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "limits": {"W": {"work": 1200000, "wall": 7200.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}
    def preparation():
        prepared.update(prepare(config))
        return {"actual_work": 0, "linear_geometry_rows": 124,
                "geometry": {f"{c}-{m}": geometry_binding(v[0]) for (c, m), v in prepared.items()}}
    prep = journal.unit("preparation", phase="W", work=0, maximum_wall=60., operation=preparation)
    records = []
    for j in grid:
        before = read_power() if enforce_power else None
        if before is not None:
            validate_power(before)
        def operation(job: dict[str, Any] = j, power_before: dict[str, int] | None = before) -> dict[str, Any]:
            r = unit_values(job, prepared[(job["cell"], job["method"])])
            if power_before is not None:
                r.update({"power_before": power_before, "power_after": read_power()})
            if r["actual_work"] != WORK or r["density_evaluations"] != DRAW_COUNT or r["torch_threads"] != 8:
                raise ValueError("unit work/backend changed")
            return r
        r = journal.unit(identity(j), phase="W", work=WORK, maximum_wall=180., operation=operation)
        if enforce_power:
            try:
                check_power_pair(r["power_before"], r["power_after"])
            except RuntimeError as error:
                write_json_atomic_nonoverwriting(directory/"power_failure.json", {
                    "unit": identity(j), "before": r["power_before"], "after": r["power_after"], "error": str(error)})
                raise
        records.append(r)
        print(f"physical-window reference pilot {len(records)}/32", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "geometry": prep["geometry"],
              "summaries": summarize(records), "units": 32, "work": journal.accounting("W")[0],
              "density_evaluations": 524288, "time_step_forward_evaluations": 3221225472,
              "linear_geometry_rows": 124,
              "phase_wall": time.time()-journal._read(directory/"phase-W.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-W.clock.json")["started_unix"] > 7200:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    profile = config["execution_profile"]
    anchor_name, audit_name, _, _, _ = execution_paths(profile)
    enforce_power = profile == "stable_power"
    if enforce_power:
        validate_power(config["power_at_start"], minimum=30)
    elif config["power_at_start"] is not None:
        raise ValueError("battery-independent run unexpectedly sampled battery state")
    if (manifest["source"] != output["source"]["snapshot_sha256"] or manifest["config"] != config
            or manifest["expected_units"] != [identity(j) for j in jobs()]
            or config["minimum_battery"] != (25 if enforce_power else None)
            or config["maximum_unit_drop"] != (1 if enforce_power else None)
            or config["battery_policy"] != ("enforced" if enforce_power else BATTERY_POLICY)
            or set(config["anchors"]) != {anchor_name, audit_name, BACKEND}
            or any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != d for p, d in config["anchors"].items())):
        raise ValueError("changed source/anchors/grid")
    prepared, journal = prepare(config), RecoveryJournal(path.parent, manifest)
    geometries = {f"{c}-{m}": geometry_binding(v[0]) for (c, m), v in prepared.items()}
    if geometries != output["geometry"]:
        raise ValueError("changed geometry")
    records, replayed = [], 0
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed block")
        r = finish["result"]
        if enforce_power:
            check_power_pair(r["power_before"], r["power_after"])
        elif any(k in r for k in ("power_before", "power_after")):
            raise ValueError("battery-independent block unexpectedly sampled battery state")
        digest = hashlib.sha256(json.dumps([seed(j, k, role) for k in range(32) for role in ("path", "label", "augmentation")]).encode()).hexdigest()
        if (any(r[k] != v for k, v in j.items()) or r["seed_digest"] != digest or r["actual_work"] != WORK
                or r["density_evaluations"] != DRAW_COUNT or r["time_step_forward_evaluations"] != DRAW_COUNT*6144
                or r["geometry"] != geometries[f"{j['cell']}-{j['method']}"] or r["torch_threads"] != 8):
            raise ValueError("changed block binding/cost")
        if j["rep"] == 0:
            replay = unit_values(j, prepared[(j["cell"], j["method"])])
            if replay["moments"] != r["moments"]:
                raise ValueError("numerical replay mismatch")
            replayed += 1
        records.append(r)
    if (summarize(records) != output["summaries"] or output["units"] != 32
            or journal.accounting("W")[0] != output["work"] or output["work"] != 1048576
            or output["density_evaluations"] != 524288 or output["time_step_forward_evaluations"] != 3221225472
            or output["linear_geometry_rows"] != 124 or (path.parent/"phase-W.violation.json").exists()
            or (path.parent/"final_io_failure.json").exists() or (path.parent/"power_failure.json").exists()):
        raise ValueError("changed summaries/cost or operational failure")
    return {"status": "physical_window_source_geometry_seed_moment_cost_replay_pass", "replayed_blocks": replayed,
            "audit_physical_evaluations": replayed*WORK, "audit_density_evaluations": replayed*DRAW_COUNT}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    parser.add_argument("--battery-independent", action="store_true")
    args = parser.parse_args()
    if args.audit:
        path = ROOT/args.audit
        receipt = {**audit(path), "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        write_json_atomic_nonoverwriting(path.parent/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        print(json.dumps({k: v for k, v in run(profile="battery_independent" if args.battery_independent else "stable_power").items()
                          if k in ("units", "work", "phase_wall")}))
