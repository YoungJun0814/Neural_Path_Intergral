"""Three-arm matched price-window development with observed tail diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.post_audit_bridge_mutation_allocation import DIRECTORY as ALLOCATION_DIRECTORY
from experiments.post_audit_bridge_mutation_allocation import load_anchors
from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_fine_reference_comparison import interval_diagnosis
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_physical_window_reference import geometry_binding
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.gaussian_span_marginalization import sample_gaussian_mixture
from src.path_integral.matched_window_diagnostics import (
    contribution_diagnostics,
    merge_contribution_diagnostics,
    paired_contributions_with_features,
    price_only_window_guide,
)
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity

DIRECTORY = "results/post_audit/matched_window_mechanism_v1/session_20261019"
PLAN = "docs/plans/MATCHED_WINDOW_MECHANISM_PILOT_2026-10-19_KO.md"
THEORY = "docs/theory/MATCHED_WINDOW_AND_CONTRIBUTION_CONTRACT_2026-10-19_KO.md"
CALIBRATION = "results/post_audit/matched_window_calibration_v1/session_20261019/result.json"
NAMESPACE = "matched-window-mechanism-v1"
ARMS = ("A-point", "B-price-only-window", "C-mixed-window")
DRAW_COUNT, CHUNK, WORK = 16384, 512, 32768
KWARGS: dict[str, Any] = {"amplitudes": [2., 4., 6., 8.], "price_amplitudes": [0., 2., 4.], "defensive_mass": .1}


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "arm": a, "rep": r} for c in CELLS for a in ARMS for r in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{j['cell']}-{j['arm']}-{j['rep']}"


def seed(j: dict[str, Any], chunk: int, role: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}/{identity(j)}/{chunk}/{role}".encode()).hexdigest()[:15], 16)


def prepare(config: dict[str, Any]) -> dict[tuple[str, str], Any]:
    out = {}
    for c in config["original"]["cells"]:
        p = _problem(config["original"]["model"], c, 4096)
        pair = PairedVolterraProbability.create(p)
        guides = (PhysicalWindowGuide.create(p, kind="point", **KWARGS),
                  price_only_window_guide(p, **KWARGS),
                  PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS))
        for a, guide in zip(ARMS, guides, strict=True):
            out[(c["id"], a)] = guide, pair
    return out


def unit_values(j: dict[str, Any], prepared: Any) -> dict[str, Any]:
    begin = time.perf_counter()
    guide, pair = prepared
    parts: list[list[dict[str, Any]]] = [[], []]
    diagnostics = []
    with torch.no_grad():
        for k in range(32):
            sample = sample_gaussian_mixture(guide.spec, CHUNK,
                gaussian_generator=torch.Generator().manual_seed(seed(j, k, "path")),
                label_generator=torch.Generator().manual_seed(seed(j, k, "label")))
            eps = torch.randn((CHUNK, 2048), dtype=torch.float64, generator=torch.Generator().manual_seed(seed(j, k, "augmentation")))
            values, features = paired_contributions_with_features(pair, guide, sample.target_coordinates, eps, sample.labels)
            for m in (0, 1):
                parts[m].append(log_moments(values[:, m]))
            diagnostics.append(contribution_diagnostics(values[:, 0], sample.labels, features, components=373))
            if psutil.Process().memory_info().rss > 4*1024**3:
                raise MemoryError("diagnostic chunk RSS guard")
    return {**j, "actual_work": WORK, "density_evaluations": DRAW_COUNT, "timestep_forward": DRAW_COUNT*6144,
            "torch_threads": torch.get_num_threads(), "moments": [merge_retaining_maximum(p) for p in parts],
            "diagnostics": merge_contribution_diagnostics(diagnostics), "geometry": geometry_binding(guide),
            "seed_digest": hashlib.sha256(json.dumps([seed(j, k, role) for k in range(32) for role in ("path", "label", "augmentation")]).encode()).hexdigest(),
            "wall": time.perf_counter()-begin}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 48 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete three-arm fixed grid required")
    rows: list[dict[str, Any]] = []
    for c in CELLS:
        for a in ARMS:
            group = sorted((r for r in records if r["cell"] == c and r["arm"] == a), key=lambda r: r["rep"])
            logs = torch.tensor([[r["moments"][k]["log_mean"] for k in (0, 1)] for r in group], dtype=torch.float64)
            components: list[dict[str, Any]] = [{"grid": grid, "summary": merge_retaining_maximum([r["moments"][k] for r in group]),
                "block_sensitivity": sensitivity(logs[:, k].tolist(), bootstrap_seed=seed({"cell": c, "arm": a, "rep": 1000}, k, "bootstrap"))}
                for k, grid in enumerate(("fine", "coarse"))]
            diagnostic = merge_contribution_diagnostics([r["diagnostics"] for r in group])
            if abs(diagnostic["log_sum"]-components[0]["summary"]["log_sum"]) > 1e-10 or sum(diagnostic["label_counts"]) != 8*DRAW_COUNT:
                raise ValueError("diagnostic/estimator mass or count mismatch")
            rows.append({"cell": c, "arm": a, "components": components, "diagnostics": diagnostic,
                         "paired_difference": paired_normalizer_difference(logs[:, 0], logs[:, 1], comparisons=6)})
    contrasts = []
    for c in CELLS:
        group = [r for r in rows if r["cell"] == c]
        for i, k in ((0, 1), (1, 2), (0, 2)):
            a, b = group[i]["components"][0]["summary"], group[k]["components"][0]["summary"]
            e = relative_equivalence(a, b, comparisons=6)
            contrasts.append({"cell": c, "left": ARMS[i], "right": ARMS[k], "equivalence": e,
                "interval_diagnosis": interval_diagnosis(e),
                "observed_rse_squared_ratio_not_certified_variance_ratio": (a["relative_se"]/b["relative_se"])**2 if b["relative_se"] > 0 else None})
    return {"rows": rows, "contrasts": contrasts, "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, _ = load_anchors()
    ap, receipt_path = ALLOCATION_DIRECTORY+"/result.json", ALLOCATION_DIRECTORY+"/audit.json"
    allocation, receipt = json.loads((ROOT/ap).read_text()), json.loads((ROOT/receipt_path).read_text())
    audit_snapshot(ROOT, allocation["source"], allocation["config"])
    if (receipt["status"] != "fixed_bridge_mutation_source_seed_moment_replay_pass" or receipt["replayed_wholes"] != 4
            or receipt["result_sha256"] != sha(ROOT/ap)):
        raise ValueError("allocation numerical audit must complete first")
    calibration = json.loads((ROOT/CALIBRATION).read_text())
    audit_snapshot(ROOT, calibration["source"], calibration["config"])
    if not calibration["pass"] or calibration["physical"] != 8192:
        raise ValueError("matched-window numeric/resource preflight required")
    config = {"original": original, "guide": KWARGS, "arms": list(ARMS), "namespace": NAMESPACE,
              "anchors": {p: sha(ROOT/p) for p in (ap, receipt_path, CALIBRATION)}, "torch_threads": 8}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, ap, receipt_path, CALIBRATION))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    seeds = [seed(j, k, role) for j in jobs() for k in range(32) for role in ("path", "label", "augmentation")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in jobs()],
                "limits": {"M": {"work": 1600000, "wall": 7200.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}
    def preparation():
        prepared.update(prepare(config))
        return {"actual_work": 0, "linear_geometry_rows": 186,
                "geometry": {f"{c}-{a}": geometry_binding(v[0]) for (c, a), v in prepared.items()}}
    prep = journal.unit("preparation", phase="M", work=0, maximum_wall=60., operation=preparation)
    records = []
    for j in jobs():
        def operation(job: dict[str, Any] = j) -> dict[str, Any]:
            return unit_values(job, prepared[(job["cell"], job["arm"])])
        records.append(journal.unit(identity(j), phase="M", work=WORK, maximum_wall=180., operation=operation))
        print(f"matched-window mechanism {len(records)}/48", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "geometry": prep["geometry"],
              "summary": summarize(records), "units": 48, "physical": journal.accounting("M")[0],
              "density": 786432, "timestep_forward": 4831838208,
              "phase_wall": time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"] > 7200.:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["arms"] != list(ARMS) or config["guide"] != KWARGS or config["namespace"] != NAMESPACE
            or config["torch_threads"] != 8 or any(sha(ROOT/p) != d for p, d in config["anchors"].items())
            or manifest != {"config": config, "source": output["source"]["snapshot_sha256"],
                "expected_units": [identity(j) for j in jobs()], "limits": {"M": {"work": 1600000, "wall": 7200.}}}):
        raise ValueError("changed fixed config/source")
    for p, digest in output["source"]["snapshot_file_hashes"].items():
        if (p.startswith("src/") or p == "experiments/post_audit_matched_window_mechanism.py") and sha(ROOT/p) != digest:
            raise ValueError("changed numerical runtime")
    prepared, journal = prepare(config), RecoveryJournal(path.parent, manifest)
    geometry = {f"{c}-{a}": geometry_binding(v[0]) for (c, a), v in prepared.items()}
    records = []
    for suffix in ("start", "finish"):
        if {p.name for p in path.parent.glob("*."+suffix+".json")} != {"preparation."+suffix+".json", *(identity(j)+"."+suffix+".json" for j in jobs())}:
            raise ValueError("changed journal membership")
    for j in jobs():
        f = journal._read(path.parent/(identity(j)+".finish.json"))
        r = f["result"]
        digest = hashlib.sha256(json.dumps([seed(j, k, role) for k in range(32) for role in ("path", "label", "augmentation")]).encode()).hexdigest()
        if (f["status"] != "complete" or f["work"] != WORK or f["wall"] > 180.
                or any(r[k] != v for k, v in j.items()) or r["seed_digest"] != digest
                or r["actual_work"] != WORK or r["density_evaluations"] != DRAW_COUNT or r["timestep_forward"] != DRAW_COUNT*6144
                or r["torch_threads"] != 8 or r["geometry"] != geometry[f"{j['cell']}-{j['arm']}"]):
            raise ValueError("changed block cost/identity")
        if j["rep"] == 0:
            replay = unit_values(j, prepared[(j["cell"], j["arm"])])
            if any(replay[k] != r[k] for k in ("moments", "diagnostics", "geometry", "seed_digest")):
                raise ValueError("moment/diagnostic numerical replay mismatch")
        records.append(r)
    if (summarize(records) != output["summary"] or output["geometry"] != geometry or output["units"] != 48
            or output["physical"] != 1572864 or journal.accounting("M")[0] != 1572864
            or output["density"] != 786432 or output["timestep_forward"] != 4831838208 or output["phase_wall"] > 7200.
            or any((path.parent/p).exists() for p in ("phase-M.violation.json", "final_io_failure.json"))):
        raise ValueError("changed summary/cost")
    return {"status": "matched_window_moment_diagnostic_geometry_seed_replay_pass", "replayed_blocks": 6,
            "audit_physical": 196608, "audit_density": 98304}


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
        print(json.dumps({k: v for k, v in run().items() if k in ("units", "physical", "phase_wall")}))
