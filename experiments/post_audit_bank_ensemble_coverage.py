"""Fresh four-arm bank-ensemble evaluation with every path in fixed physical bins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch

from experiments.post_audit_defensive_allocation import DIRECTORY as ALLOCATION_DIRECTORY
from experiments.post_audit_defensive_allocation import guard, prepare
from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.bank_ensemble_coverage import (
    coverage_ledger,
    mean_defensive_weights,
    merge_coverage,
    physical_path_features,
    summarize_bins,
)
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    sample_gaussian_mixture,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity

DIRECTORY = "results/post_audit/bank_ensemble_full_path_v1/session_20261020"
PLAN = "docs/plans/BANK_ENSEMBLE_FULL_PATH_COVERAGE_2026-10-20_KO.md"
THEORY = "docs/theory/BANK_ENSEMBLE_FULL_PATH_COVERAGE_REVIEW_2026-10-20_KO.md"
NAMESPACE = "bank-ensemble-full-path-v1"
DRAW_COUNT, CHUNK = 8192, 256
ARMS = ("uniform", "fit0", "fit1", "ensemble")
ANCHORS = tuple(ALLOCATION_DIRECTORY+"/"+name+".json" for name in ("result", "audit"))


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "arm": a, "block": b} for c in CELLS for a in ARMS for b in range(8)]


def identity(job: dict[str, Any]) -> str:
    return f"{job['cell']}-{job['arm']}-{job['block']}"


def seed(cell: str, arm: str, block: int, chunk: int, role: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}/{cell}/{arm}/{block}/{chunk}/{role}".encode()).hexdigest()[:15], 16)


def seed_digest(job: dict[str, Any], count: int) -> str:
    values = [seed(job["cell"], job["arm"], job["block"], k, role)
              for k in range(count//CHUNK) for role in ("path", "label")]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def load_anchor() -> dict[str, Any]:
    original, audit = [json.loads((ROOT/p).read_text()) for p in ANCHORS]
    audit_snapshot(ROOT, original["source"], original["config"])
    if (audit["status"] != "defensive_allocation_source_bank_fit_final_replay_pass"
            or audit["result_sha256"] != sha(ROOT/ANCHORS[0]) or audit["replayed_fits"] != 4
            or audit["replayed_final_blocks"] != 4
            or any(original["summary"][key] is not False for key in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("complete unchanged allocation evidence required")
    return original


def configuration(anchor: dict[str, Any]) -> dict[str, Any]:
    fits = anchor["summary"]["fits"]
    if {(f["cell"], f["rep"]) for f in fits} != {(c, r) for c in CELLS for r in range(2)} or len(fits) != 4:
        raise ValueError("both independent banks per cell required")
    return {"original": anchor["config"]["original"], "namespace": NAMESPACE,
        "anchors": {p: sha(ROOT/p) for p in ANCHORS}, "torch_threads": 8,
        "fixed_bank_weights": {c: [next(f["weights"] for f in fits if (f["cell"], f["rep"]) == (c, r)) for r in range(2)] for c in CELLS},
        "draw_count": DRAW_COUNT, "chunk": CHUNK, "arms": list(ARMS), "blocks": 8,
        "bins": {"integrated_variance": [.25, .5, 1., 2., 4.], "peak_time_fraction": [.25, .5, .75],
                 "T32_energy_fraction": [.25, .5, .75], "boundary_equality": "right"}}


def prepare_all(config: dict[str, Any]) -> dict[str, Any]:
    basic = prepare(config)
    out = {}
    for c in CELLS:
        guide, cache = basic[c]
        weights = [torch.tensor(w, dtype=torch.float64) for w in config["fixed_bank_weights"][c]]
        ensemble = mean_defensive_weights(weights)
        if guide.spec.components != 373:
            raise ValueError("same 373-component fixed dictionary required")
        specs = {a: GaussianMixtureShiftSpec(guide.spec.means, w) for a, w in zip(ARMS,
            [guide.spec.weights, *weights, ensemble], strict=True)}
        out[c] = cache, specs
    return out


def unit_values(job: dict[str, Any], prepared: Any, *, count: int = DRAW_COUNT) -> dict[str, Any]:
    if count not in (CHUNK, DRAW_COUNT):
        raise ValueError("only fixed preflight or whole final unit allowed")
    cache, specs = prepared
    spec = specs["uniform" if job["arm"] == "preflight" else job["arm"]]
    begin = time.perf_counter()
    parts, ledgers = [], []
    with torch.no_grad():
        for k in range(count//CHUNK):
            draw = sample_gaussian_mixture(spec, CHUNK,
                gaussian_generator=torch.Generator().manual_seed(seed(job["cell"], job["arm"], job["block"], k, "path")),
                label_generator=torch.Generator().manual_seed(seed(job["cell"], job["arm"], job["block"], k, "label")))
            z = draw.target_coordinates
            terminal, integrated, variance = cache.quantities(z)
            p, task = cache.problem, cache.problem.task
            assert isinstance(task, TerminalThresholdTask)
            g = torch.special.log_ndtr((math.log(task.level)-terminal)/((1-p.rho**2)*integrated).sqrt())
            d = torch.logsumexp(z @ spec.means.T-.5*spec.means.square().sum(1)+spec.weights.log(), 1)
            values = g-d
            if not torch.isfinite(values).all() or (values > math.log(10)+1e-10).any():
                raise FloatingPointError("original full-mixture contribution contract failed")
            features = physical_path_features(variance[:, :-1], maturity=p.maturity)
            torch.testing.assert_close(features[:, 0], integrated.log(), rtol=1e-12, atol=1e-12)
            parts.append(log_moments(values))
            ledgers.append(coverage_ledger(values, features))
            guard()
    moment, ledger = merge_retaining_maximum(parts), merge_coverage(ledgers)
    if abs(moment["log_sum"]-ledger["log_sum"]) > 1e-10 or abs(moment["log_sum_squares"]-ledger["log_sum_squares"]) > 1e-10:
        raise ValueError("all-path bins do not reconstruct ordinary estimate")
    return {**job, "actual_work": count, "moments": moment, "coverage": ledger,
            "seed_digest": seed_digest(job, count), "torch_threads": torch.get_num_threads(),
            "wall": time.perf_counter()-begin}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 64 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete four-arm all-path family required")
    rows, comparisons = [], []
    for c in CELLS:
        moments = {}
        for a in ARMS:
            group = sorted((r for r in records if r["cell"] == c and r["arm"] == a), key=lambda r: r["block"])
            m = merge_retaining_maximum([r["moments"] for r in group])
            moments[a] = m
            coverage = merge_coverage([r["coverage"] for r in group])
            rows.append({"cell": c, "arm": a, "summary": m, "coverage": coverage,
                "bin_summaries": summarize_bins(coverage),
                "block_sensitivity": sensitivity([r["moments"]["log_mean"] for r in group],
                    bootstrap_seed=seed(c, a, 1000, 0, "bootstrap")),
                "precision_1p5_pass": m["relative_se"] <= .015,
                "score_wall": sum(r["wall"] for r in group)})
        for a in ARMS[:-1]:
            comparisons.append({"cell": c, "baseline": a, "candidate": "ensemble",
                "equivalence": relative_equivalence(moments[a], moments["ensemble"], comparisons=6),
                "observed_rse_squared_ratio_not_efficiency_certificate": (moments[a]["relative_se"]/moments["ensemble"]["relative_se"])**2})
    return {"rows": rows, "comparisons": comparisons,
        "scope": "fresh_fixed_bank_weights_all_sample_finite_grid_development_not_population_coverage",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def manifest_for(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {"config": config, "source": source["snapshot_sha256"],
            "expected_units": ["prepare", *("preflight-"+c for c in CELLS), *(identity(j) for j in jobs())],
            "limits": {"E": {"work": 524800, "wall": 7200.}}}


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    config = configuration(load_anchor())
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, *ANCHORS))
    manifest = manifest_for(config, source)
    journal, prepared = RecoveryJournal(directory, manifest), {}

    def preparation():
        start = time.perf_counter()
        prepared.update(prepare_all(config))
        return {"actual_work": 0, "wall": time.perf_counter()-start}

    prep = journal.unit("prepare", phase="E", work=0, maximum_wall=120., operation=preparation)
    preflights = []
    for c in CELLS:
        def preflight(cell: str = c):
            values = unit_values({"cell": cell, "arm": "preflight", "block": 0}, prepared[cell], count=CHUNK)
            if values["wall"]*32 > 180.:
                raise TimeoutError("whole scoring forecast exceeds fixed budget")
            return values
        preflights.append(journal.unit("preflight-"+c, phase="E", work=CHUNK, maximum_wall=30., operation=preflight))
    records = []
    for job in jobs():
        records.append(journal.unit(identity(job), phase="E", work=DRAW_COUNT, maximum_wall=180.,
            operation=partial(unit_values, job, prepared[job["cell"]])))
        print(f"bank ensemble full-path {len(records)}/64", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "preparation": prep,
        "preflight": preflights, "summary": summarize(records), "units": 67,
        "physical_evaluations": 524800, "final_physical": 524288,
        "phase_wall": time.time()-journal._read(directory/"phase-E.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-E.clock.json")["started_unix"] > 7200.:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O phase limit")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output = json.loads(path.read_text())
    config = configuration(load_anchor())
    audit_snapshot(ROOT, output["source"], config)
    manifest = manifest_for(config, output["source"])
    if output["config"] != config or output["manifest"] != manifest:
        raise ValueError("changed fixed bank/feature/seed configuration")
    for name, digest in output["source"]["snapshot_file_hashes"].items():
        if (name.startswith("src/") or name == "experiments/post_audit_bank_ensemble_coverage.py") and sha(ROOT/name) != digest:
            raise ValueError("changed numerical runtime")
    journal = RecoveryJournal(path.parent, manifest)
    expected = set(manifest["expected_units"])
    for suffix in ("start", "finish"):
        if {p.name.removesuffix(f".{suffix}.json") for p in path.parent.glob(f"*.{suffix}.json")} != expected:
            raise ValueError("unexpected completed family membership")
    for name in expected:
        f = journal._read(path.parent/(name+".finish.json"))
        work = 0 if name == "prepare" else CHUNK if name.startswith("preflight-") else DRAW_COUNT
        cap = 120. if name == "prepare" else 30. if name.startswith("preflight-") else 180.
        if f["status"] != "complete" or f["work"] != work or f["wall"] > cap or f["result"]["actual_work"] != work:
            raise ValueError("failed unit or changed work/cap")
    prepared, records = prepare_all(config), []
    for job in jobs():
        r = journal._read(path.parent/(identity(job)+".finish.json"))["result"]
        if (any(r[k] != v for k, v in job.items()) or r["actual_work"] != DRAW_COUNT
                or r["seed_digest"] != seed_digest(job, DRAW_COUNT) or r["torch_threads"] != 8):
            raise ValueError("changed final unit identity")
        if job["block"] == 0:
            replay = unit_values(job, prepared[job["cell"]])
            if any(replay[k] != r[k] for k in ("moments", "coverage", "seed_digest")):
                raise ValueError("full-path feature/moment/seed replay mismatch")
        records.append(r)
    if (summarize(records) != output["summary"] or journal.accounting("E")[0] != 524800
            or output["physical_evaluations"] != 524800 or output["units"] != 67
            or output["phase_wall"] > 7200. or any((path.parent/p).exists() for p in ("phase-E.violation.json", "final_io_failure.json"))):
        raise ValueError("changed aggregate/qualification/work")
    return {"status": "bank_ensemble_all_path_source_seed_feature_moment_replay_pass",
            "replayed_blocks": 8, "audit_physical_evaluations": 65536, "result_sha256": sha(path)}


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
