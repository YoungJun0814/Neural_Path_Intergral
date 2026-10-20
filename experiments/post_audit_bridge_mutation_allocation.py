"""Fixed near-work-matched guide-free SMC allocation development, no qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.post_audit_fine_reference_comparison import interval_diagnosis
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

DIRECTORY = "results/post_audit/fine_bridge_mutation_allocation_v1/session_20261019"
PLAN = "docs/plans/FINE_BRIDGE_MUTATION_ALLOCATION_2026-10-19_KO.md"
NAMESPACE = "fine-bridge-mutation-allocation-v1"
ARMS = {"64points-4moves": (64, 4, 4), "128points-2moves": (128, 2, 8)}
ANCHORS = tuple(f"results/post_audit/{d}/session_20261019/{f}.json" for d in (
    "p2_fine_resource_completion_v3", "physical_window_reference_recovery_v1") for f in ("result", "audit"))


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "arm": a, "rep": r} for c in CELLS for a in ARMS for r in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['arm']}-{j['rep']}"


def seed(j: dict[str, Any]) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}/{identity(j)}".encode()).hexdigest()[:15], 16)


def joint_work(arm: str) -> int:
    points, moves, _ = ARMS[arm]
    return 64*(1+(points-2)*moves)


def unit_values(j: dict[str, Any], pair: PairedVolterraProbability) -> dict[str, Any]:
    begin = time.perf_counter()
    points, moves, resample = ARMS[j["arm"]]
    cfg = WeightedSMCConfig(64, tuple((k/(points-1))**2 for k in range(points)), moves, .35, 1,
                            seed(j), resample_every=resample, resampling_scheme="stratified")
    r = estimate_gradient_umbrella_normalizers(pair.log_components, dimension=pair.dimension,
            log_mixture_weights=torch.full((2,), .5, dtype=torch.float64).log(), config=cfg,
            maximum_drift_norm=20.)
    if r.potential_evaluations != joint_work(j["arm"]) or r.gradient_evaluations != r.potential_evaluations:
        raise ValueError("unexpected forward/backward work")
    values = r.log_component_normalizers
    if not torch.isfinite(values).all() or abs(float(torch.logsumexp(values-math.log(2), 0))-r.log_normalizer) > 1e-10:
        raise ValueError("unnormalized component identity failed")
    if psutil.Process().memory_info().rss > 4*1024**3:
        raise MemoryError("whole RSS guard")
    return {**j, "seed": seed(j), "actual_work": 2*r.potential_evaluations,
            "joint_backward": r.gradient_evaluations, "timestep_forward": 6144*r.potential_evaluations,
            "torch_threads": torch.get_num_threads(), "log_normalizer": r.log_normalizer,
            "log_component_normalizers": values.tolist(), "stages": list(r.stages), "wall": time.perf_counter()-begin}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 32 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete allocation grid required")
    rows: list[dict[str, Any]] = []
    for c in CELLS:
        for arm in ARMS:
            group = sorted((r for r in records if r["cell"] == c and r["arm"] == arm), key=lambda r: r["rep"])
            logs = torch.tensor([r["log_component_normalizers"] for r in group], dtype=torch.float64)
            components: list[dict[str, Any]] = [{"grid": grid, "summary": log_moments(logs[:, k]),
                           "sensitivity": sensitivity(logs[:, k].tolist(), bootstrap_seed=seed({"cell": c+grid, "arm": arm, "rep": 1000}))}
                          for k, grid in enumerate(("fine", "coarse"))]
            worst = max(x["summary"]["relative_se"] for x in components)
            count = max(32, math.ceil(3*8*(worst/.015)**2))
            rows.append({"cell": c, "arm": arm, "components": components,
                         "paired_difference": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=4),
                         "initial_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in group],
                         "mean_mutation_acceptance": [sum(s["mutation_acceptance"] for s in r["stages"][:-1])/(len(r["stages"])-1) for r in group],
                         "minimum_stage_ess": [min(s["ess_before_resampling"] for s in r["stages"]) for r in group],
                         "fresh_forecast": {"wholes": count, "physical": 2*joint_work(arm)*count,
                                            "timestep": 6144*joint_work(arm)*count, "safety": 3, "target_rse": .015}})
    return {"rows": rows, "selected_development_arm": {
        c: min((r for r in rows if r["cell"] == c), key=lambda r: (r["fresh_forecast"]["timestep"], r["arm"]))["arm"] for c in CELLS},
        "selection_rule": "lowest_safety3_fresh_timestep_forecast_not_qualification",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def load_anchors() -> tuple[dict[str, Any], dict[str, Any]]:
    resource, ra, guide, ga = [json.loads((ROOT/p).read_text()) for p in ANCHORS]
    for x in (resource, guide):
        audit_snapshot(ROOT, x["source"], x["config"])
    if (ra["result_sha256"] != sha(ROOT/ANCHORS[0]) or ga["result_sha256"] != sha(ROOT/ANCHORS[2])
            or ra["replayed_wholes"] != 4 or ga["replayed_blocks"] != 4
            or ra["status"] != "battery_independent_full_fixed_grid_replay_pass"
            or ga["status"] != "physical_window_source_geometry_seed_moment_cost_replay_pass"):
        raise ValueError("complete replay anchors required")
    return resource["config"]["original_config"], guide


def crosscheck(summary: dict[str, Any], guide: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in summary["rows"]:
        g = next(x for x in guide["summaries"]["rows"] if x["cell"] == r["cell"] and x["method"] == "mixed-window")
        e = relative_equivalence(r["components"][0]["summary"], g["components"][0]["summary"], comparisons=4)
        out.append({"cell": r["cell"], "arm": r["arm"], "equivalence": e, "interval_diagnosis": interval_diagnosis(e)})
    return out


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, guide = load_anchors()
    config = {"original": original, "arms": {k: list(v) for k, v in ARMS.items()}, "namespace": NAMESPACE,
              "anchors": {p: sha(ROOT/p) for p in ANCHORS}, "torch_threads": 8}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *ANCHORS))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in jobs()],
                "limits": {"B": {"work": 1100000, "wall": 14400.}}}
    journal = RecoveryJournal(directory, manifest)
    prepared: dict[str, Any] = {}
    def preparation():
        prepared.update({c["id"]: PairedVolterraProbability.create(_problem(original["model"], c, 4096)) for c in original["cells"]})
        return {"actual_work": 0}
    journal.unit("preparation", phase="B", work=0, maximum_wall=30., operation=preparation)
    records = []
    for j in jobs():
        def operation(job: dict[str, Any] = j) -> dict[str, Any]:
            return unit_values(job, prepared[job["cell"]])
        records.append(journal.unit(identity(j), phase="B", work=2*joint_work(j["arm"]), maximum_wall=600.,
                       operation=operation))
        print(f"bridge/mutation allocation {len(records)}/32", flush=True)
    summary = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summary": summary,
              "g1_crosscheck": crosscheck(summary, guide), "units": 32, "physical": journal.accounting("B")[0],
              "joint_backward": sum(r["joint_backward"] for r in records),
              "timestep_forward": sum(r["timestep_forward"] for r in records),
              "phase_wall": time.time()-journal._read(directory/"phase-B.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-B.clock.json")["started_unix"] > 14400.:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    original, guide = load_anchors()
    if (config["original"] != original or config["arms"] != {k: list(v) for k, v in ARMS.items()}
            or config["namespace"] != NAMESPACE or config["torch_threads"] != 8
            or config["anchors"] != {p: sha(ROOT/p) for p in ANCHORS}
            or manifest != {"config": config, "source": output["source"]["snapshot_sha256"],
                "expected_units": [identity(j) for j in jobs()], "limits": {"B": {"work": 1100000, "wall": 14400.}}}):
        raise ValueError("changed fixed source/config")
    for p, digest in output["source"]["snapshot_file_hashes"].items():
        if (p.startswith("src/") or p == "experiments/post_audit_bridge_mutation_allocation.py") and sha(ROOT/p) != digest:
            raise ValueError("changed numerical runtime")
    journal = RecoveryJournal(path.parent, manifest)
    prepared = {c["id"]: PairedVolterraProbability.create(_problem(original["model"], c, 4096)) for c in original["cells"]}
    records, replay_work = [], 0
    for suffix in ("start", "finish"):
        if {p.name for p in path.parent.glob("*."+suffix+".json")} != {"preparation."+suffix+".json", *(identity(j)+"."+suffix+".json" for j in jobs())}:
            raise ValueError("unexpected journal membership")
    for j in jobs():
        f = journal._read(path.parent/(identity(j)+".finish.json"))
        r = f["result"]
        if (f["status"] != "complete" or f["work"] != 2*joint_work(j["arm"]) or f["wall"] > 600.
                or any(r[k] != v for k, v in j.items()) or r["seed"] != seed(j) or r["torch_threads"] != 8
                or r["actual_work"] != f["work"] or r["joint_backward"] != joint_work(j["arm"])
                or r["timestep_forward"] != 6144*joint_work(j["arm"])):
            raise ValueError("changed numerical unit")
        if j["rep"] == 0:
            replay = unit_values(j, prepared[j["cell"]])
            if any(replay[k] != r[k] for k in ("log_normalizer", "log_component_normalizers", "stages")):
                raise ValueError("whole replay mismatch")
            replay_work += r["actual_work"]
        records.append(r)
    s = summarize(records)
    if (s != output["summary"] or crosscheck(s, guide) != output["g1_crosscheck"]
            or output["units"] != 32 or output["physical"] != 1028096 or journal.accounting("B")[0] != 1028096
            or output["joint_backward"] != 514048 or output["timestep_forward"] != 3158310912
            or output["phase_wall"] > 14400. or any((path.parent/p).exists() for p in ("phase-B.violation.json", "final_io_failure.json"))):
        raise ValueError("changed summary/cost")
    return {"status": "fixed_bridge_mutation_source_seed_moment_replay_pass", "replayed_wholes": 4,
            "audit_physical": replay_work, "audit_joint_backward": replay_work//2}


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
        print(json.dumps({k: v for k, v in run().items() if k in ("units", "physical", "joint_backward", "phase_wall")}))
