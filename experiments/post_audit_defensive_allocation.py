"""Fresh train/validation/final fixed-dictionary allocation, no P2 authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.post_audit_bridge_mutation_allocation import load_anchors
from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_matched_window_mechanism import KWARGS
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.defensive_mixture_allocation import (
    fit_mixture_allocation,
    log_second_moment_objective,
)
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    sample_gaussian_mixture,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity

DIRECTORY = "results/post_audit/defensive_allocation_v1/session_20261020"
PLAN = "docs/plans/DEFENSIVE_MIXTURE_ALLOCATION_RECOVERY_2026-10-20_KO.md"
NAMESPACE = "defensive-allocation-v1"
TRAIN, VALIDATION, FINAL, CHUNK = 4096, 2048, 4096, 256
ARMS = ("uniform", "learned")


def seed(cell: str, rep: int, stream: str, block: int, chunk: int, role: str) -> int:
    key = f"{NAMESPACE}/{cell}/{rep}/{stream}/{block}/{chunk}/{role}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "rep": rep, "arm": a, "block": b}
            for c in CELLS for rep in range(2) for a in ARMS for b in range(8)]


def identity(job: dict[str, Any]) -> str:
    return f"{job['cell']}-{job['rep']}-{job['arm']}-{job['block']}"


def final_seed_digest(job: dict[str, Any]) -> str:
    values = [seed(job["cell"], job["rep"], "final-"+job["arm"], job["block"], k, role)
              for k in range(FINAL//CHUNK) for role in ("path", "label")]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def guard() -> None:
    if psutil.Process().memory_info().rss > 4*1024**3:
        raise MemoryError("four GiB RSS guard")


def sample(spec: GaussianMixtureShiftSpec, cell: str, rep: int, stream: str, block: int, chunk: int):
    return sample_gaussian_mixture(spec, CHUNK,
        gaussian_generator=torch.Generator().manual_seed(seed(cell, rep, stream, block, chunk, "path")),
        label_generator=torch.Generator().manual_seed(seed(cell, rep, stream, block, chunk, "label")))


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for cell in config["original"]["cells"]:
        p = _problem(config["original"]["model"], cell, 4096)
        out[cell["id"]] = (PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS),
                            CachedVolterraTerminal.create(p))
    guard()
    return out


def bank(prepared: Any, cell: str, rep: int, stream: str, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    guide, cache = prepared
    arrays: list[list[torch.Tensor]] = [[], [], []]
    for k in range(count//CHUNK):
        draw = sample(guide.spec, cell, rep, stream, 0, k)
        z = draw.target_coordinates
        components = z @ guide.spec.means.T-.5*guide.spec.means.square().sum(1)
        g = cache.log_probability(z)
        d = torch.logsumexp(components+guide.spec.weights.log(), 1)
        for dest, values in zip(arrays, (components, g, d), strict=True):
            dest.append(values)
        guard()
    return torch.cat(arrays[0]), torch.cat(arrays[1]), torch.cat(arrays[2])


def digest_bank(arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fit_values(prepared: Any, cell: str, rep: int) -> dict[str, Any]:
    begin = time.perf_counter()
    train = bank(prepared, cell, rep, "train", TRAIN)
    physics_wall = time.perf_counter()-begin
    start = time.perf_counter()
    fitted = fit_mixture_allocation(*train)
    fit_wall = time.perf_counter()-start
    validation = bank(prepared, cell, rep, "validation", VALIDATION)
    original = prepared[0].spec.weights
    objectives = [log_second_moment_objective(*validation, w) for w in (original, fitted.weights)]
    return {"cell": cell, "rep": rep, "actual_work": TRAIN+VALIDATION,
            "weights": fitted.weights.tolist(), "training_bank_digest": digest_bank(train),
            "validation_bank_digest": digest_bank(validation),
            "training": {"initial_log_objective": fitted.initial_log_objective,
                "final_log_objective": fitted.final_log_objective,
                "relative_empirical_fw_gap": fitted.relative_empirical_fw_gap,
                "iterations": fitted.iterations},
            "validation_log_objectives": objectives,
            "validation_learned_over_uniform_second_moment": math.exp(objectives[1]-objectives[0]),
            "training_physics_wall": physics_wall, "fit_wall": fit_wall,
            "total_wall": time.perf_counter()-begin, "torch_threads": torch.get_num_threads()}


def unit_values(job: dict[str, Any], prepared: Any, weights: torch.Tensor) -> dict[str, Any]:
    guide, cache = prepared
    spec = GaussianMixtureShiftSpec(guide.spec.means, weights)
    parts, top = [], []
    begin = time.perf_counter()
    for k in range(FINAL//CHUNK):
        draw = sample(spec, job["cell"], job["rep"], "final-"+job["arm"], job["block"], k)
        z = draw.target_coordinates
        terminal, integrated, variance = cache.quantities(z)
        task = cache.problem.task
        assert isinstance(task, TerminalThresholdTask)
        g = torch.special.log_ndtr((math.log(task.level)-terminal)/((1-cache.problem.rho**2)*integrated).sqrt())
        d = torch.logsumexp(z @ spec.means.T-.5*spec.means.square().sum(1)+weights.log(), 1)
        values = g-d
        if not torch.isfinite(values).all() or (values > math.log(10)+1e-10).any():
            raise FloatingPointError("ordinary full-mixture contribution contract failed")
        parts.append(log_moments(values))
        max_variance, indices = variance[:, :-1].max(1)
        for i in torch.argsort(values, descending=True, stable=True)[:16]:
            top.append({"log_contribution": float(values[i]), "sampler_label": int(draw.labels[i]),
                "log_integrated_variance": float(integrated[i].log()),
                "log_maximum_left_variance": float(max_variance[i].log()),
                "maximum_left_variance_time": float(indices[i]*cache.problem.step_dt)})
        top = sorted(top, key=lambda x: x["log_contribution"], reverse=True)[:16]
        guard()
    return {**job, "actual_work": FINAL, "moments": merge_retaining_maximum(parts),
            "top_sixteen_paths": top, "wall": time.perf_counter()-begin,
            "torch_threads": torch.get_num_threads(),
            "seed_digest": final_seed_digest(job)}


def summarize(records: list[dict[str, Any]], fits: list[dict[str, Any]]) -> dict[str, Any]:
    if (len(records) != 64 or {identity(r) for r in records} != {identity(j) for j in jobs()}
            or len(fits) != 4 or {(f["cell"], f["rep"]) for f in fits} != {(c, r) for c in CELLS for r in range(2)}):
        raise ValueError("complete two-fit final family required; no prefix inference")
    rows, contrasts = [], []
    for c in CELLS:
        for rep in range(2):
            summaries = []
            for arm in ARMS:
                group = sorted((r for r in records if r["cell"] == c and r["rep"] == rep and r["arm"] == arm), key=lambda r: r["block"])
                moment = merge_retaining_maximum([r["moments"] for r in group])
                summaries.append(moment)
                rows.append({"cell": c, "rep": rep, "arm": arm, "summary": moment,
                    "block_sensitivity": sensitivity([r["moments"]["log_mean"] for r in group],
                        bootstrap_seed=seed(c, rep, "bootstrap-"+arm, 0, 0, "bootstrap")),
                    "top_sixteen_paths": sorted((p for r in group for p in r["top_sixteen_paths"]),
                        key=lambda p: p["log_contribution"], reverse=True)[:16],
                    "score_wall": sum(r["wall"] for r in group), "precision_1p5_pass": moment["relative_se"] <= .015})
            contrasts.append({"cell": c, "rep": rep,
                "equivalence": relative_equivalence(*summaries, comparisons=4),
                "observed_raw_rse_squared_ratio_not_efficiency_certificate": (summaries[0]["relative_se"]/summaries[1]["relative_se"])**2,
                "observed_rse_reduced": summaries[1]["relative_se"] < summaries[0]["relative_se"]})
    return {"rows": rows, "contrasts": contrasts, "fits": fits,
            "scope": "independent_final_conditional_on_each_bank_same_geometry_two_fit_development_only",
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, _ = load_anchors()
    config = {"original": original, "namespace": NAMESPACE, "guide": KWARGS, "torch_threads": 8,
              "train": TRAIN, "validation": VALIDATION, "final": FINAL, "chunk": CHUNK,
              "independent_fits": 2, "final_blocks": 8, "natural_mass": .1, "shifted_floor_mass": .1,
              "maximum_iterations": 256, "relative_gap_tolerance": 1e-6, "line_search_iterations": 40}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    manifest = {"config": config, "source": source["snapshot_sha256"],
                "expected_units": ["prepare", *("preflight-"+c for c in CELLS),
                    *(f"fit-{c}-{rep}" for c in CELLS for rep in range(2)), *(identity(j) for j in jobs())],
                "limits": {"A": {"work": 287232, "wall": 7200.}}}
    journal, prepared = RecoveryJournal(directory, manifest), {}

    def preparation():
        begin = time.perf_counter()
        prepared.update(prepare(config))
        return {"actual_work": 0, "wall": time.perf_counter()-begin}

    with torch.no_grad():
        prep = journal.unit("prepare", phase="A", work=0, maximum_wall=120., operation=preparation)
        fixtures = []
        for cell in CELLS:
            def preflight(c: str = cell):
                start = time.perf_counter()
                arrays = bank(prepared[c], c, 1000, "preflight", CHUNK)
                wall = time.perf_counter()-start
                if wall*16 > 180.:
                    raise TimeoutError("final unit forecast exceeds preflight budget")
                return {"cell": c, "actual_work": CHUNK, "wall": wall, "bank_digest": digest_bank(arrays)}
            fixtures.append(journal.unit("preflight-"+cell, phase="A", work=CHUNK, maximum_wall=30., operation=preflight))
        fits = []
        for cell in CELLS:
            for rep in range(2):
                fit = journal.unit(f"fit-{cell}-{rep}", phase="A", work=TRAIN+VALIDATION, maximum_wall=180.,
                    operation=partial(fit_values, prepared[cell], cell, rep))
                fits.append(fit)
                print(f"fit {cell}/{rep} validation M2 ratio={fit['validation_learned_over_uniform_second_moment']:.4f}", flush=True)
        records = []
        for job in jobs():
            f = next(f for f in fits if f["cell"] == job["cell"] and f["rep"] == job["rep"])
            weights = (prepared[job["cell"]][0].spec.weights if job["arm"] == "uniform"
                       else torch.tensor(f["weights"], dtype=torch.float64))
            records.append(journal.unit(identity(job), phase="A", work=FINAL, maximum_wall=180.,
                operation=partial(unit_values, job, prepared[job["cell"]], weights)))
            print(f"allocation final {len(records)}/64", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "preparation": prep,
              "preflight": fixtures, "summary": summarize(records, fits), "units": 71,
              "physical_evaluations": 287232, "final_physical": 262144,
              "phase_wall": time.time()-journal._read(directory/"phase-A.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if output["phase_wall"] > 7200.:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final phase wall failure")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    audit_snapshot(ROOT, output["source"], output["config"])
    for p, digest in output["source"]["snapshot_file_hashes"].items():
        if (p.startswith("src/") or p == "experiments/post_audit_defensive_allocation.py") and sha(ROOT/p) != digest:
            raise ValueError("changed allocation runtime")
    journal = RecoveryJournal(path.parent, output["manifest"])
    expected = {"prepare", *("preflight-"+c for c in CELLS),
                *(f"fit-{c}-{rep}" for c in CELLS for rep in range(2)), *(identity(j) for j in jobs())}
    if (set(output["manifest"]["expected_units"]) != expected
            or output["manifest"]["limits"] != {"A": {"work": 287232, "wall": 7200.}}
            or output["manifest"]["config"] != output["config"]
            or output["manifest"]["source"] != output["source"]["snapshot_sha256"]):
        raise ValueError("changed family manifest")
    for suffix in ("start", "finish"):
        if {p.name.removesuffix(f".{suffix}.json") for p in path.parent.glob(f"*.{suffix}.json")} != expected:
            raise ValueError("unexpected journal membership")
    for name in expected:
        f = journal._read(path.parent/(name+".finish.json"))
        work = 0 if name == "prepare" else CHUNK if name.startswith("preflight-") else TRAIN+VALIDATION if name.startswith("fit-") else FINAL
        cap = 120. if name == "prepare" else 30. if name.startswith("preflight-") else 180.
        if f["status"] != "complete" or f["work"] != work or f["wall"] > cap or f["result"]["actual_work"] != work:
            raise ValueError("failed unit or changed work/cap in allocation family")
    prepared = prepare(output["config"])
    fits, records = [], []
    with torch.no_grad():
        for c in CELLS:
            for rep in range(2):
                f = journal._read(path.parent/f"fit-{c}-{rep}.finish.json")["result"]
                replay = fit_values(prepared[c], c, rep)
                for key in ("weights", "training_bank_digest", "validation_bank_digest", "training", "validation_log_objectives"):
                    if f[key] != replay[key]:
                        raise ValueError("training/validation replay mismatch")
                fits.append(f)
        for job in jobs():
            r = journal._read(path.parent/(identity(job)+".finish.json"))["result"]
            if (any(r[k] != v for k, v in job.items()) or r["actual_work"] != FINAL
                    or r["torch_threads"] != 8 or r["seed_digest"] != final_seed_digest(job)):
                raise ValueError("final unit specification mismatch")
            if job["block"] == 0 and job["arm"] == "learned":
                fitted = next(f for f in fits if (f["cell"], f["rep"]) == (job["cell"], job["rep"]))
                replay = unit_values(job, prepared[job["cell"]], torch.tensor(fitted["weights"], dtype=torch.float64))
                if any(replay[k] != r[k] for k in ("moments", "top_sixteen_paths", "seed_digest")):
                    raise ValueError("learned final replay mismatch")
            records.append(r)
    if (summarize(records, fits) != output["summary"] or journal.accounting("A")[0] != 287232
            or output["phase_wall"] > 7200. or output["physical_evaluations"] != 287232
            or any((path.parent/p).exists() for p in ("phase-A.violation.json", "final_io_failure.json"))):
        raise ValueError("summary/work/phase mismatch")
    return {"status": "defensive_allocation_source_bank_fit_final_replay_pass", "replayed_fits": 4,
            "replayed_final_blocks": 4, "audit_physical_evaluations": 40960, "result_sha256": sha(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        path = ROOT/args.audit
        receipt = audit(path)
        write_json_atomic_nonoverwriting(path.parent/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        print(json.dumps({k: v for k, v in run().items() if k in ("units", "physical_evaluations", "phase_wall")}))
