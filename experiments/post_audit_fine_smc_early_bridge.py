"""Frozen equal-work early-beta power contrast; paired independent whole runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch

from experiments.post_audit_defensive_allocation import guard
from experiments.post_audit_fresh_ensemble_fine_smc import DIRECTORY as PARENT_DIRECTORY
from experiments.post_audit_fresh_ensemble_fine_smc import (
    fine_components,
    interval_status,
    precision_gate,
)
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

DIRECTORY = "results/post_audit/fine_smc_early_bridge_v1/session_20261020"
PLAN = "docs/plans/FINE_SMC_EARLY_BRIDGE_CONTRAST_2026-10-20_KO.md"
THEORY = "docs/theory/FINE_SMC_EARLY_BRIDGE_CONTRACT_2026-10-20_KO.md"
NAMESPACE = "fine-smc-early-bridge-power-v1"
ANCHORS = tuple(PARENT_DIRECTORY+"/"+n+".json" for n in ("result", "audit"))
POWERS = (2, 3, 4)
MILESTONES = (.01, .04, .1, .5, 1.)


def seed(cell: str, rep: int, stream: str = "final") -> int:
    if stream not in ("final", "preflight"):
        raise ValueError("frozen stream required")
    return int(hashlib.sha256(f"{NAMESPACE}/{stream}/{cell}/{rep}".encode()).hexdigest()[:15], 16)


def jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "rep": r, "power": p} for c in CELLS for r in range(8) for p in POWERS]


def identity(job: dict[str, Any], *, preflight: bool = False) -> str:
    return f"{'preflight' if preflight else 'final'}-{job['cell']}-p{job['power']}-{job['rep']}"


def smc_config(job: dict[str, Any], *, preflight: bool = False) -> WeightedSMCConfig:
    if job["power"] not in POWERS:
        raise ValueError("frozen power required")
    return WeightedSMCConfig(32 if preflight else 128,
        tuple((k/63)**job["power"] for k in range(64)), 2, .35, 1,
        seed(job["cell"], job["rep"], "preflight" if preflight else "final"),
        resample_every=4, resampling_scheme="stratified")


def load_parent() -> dict[str, Any]:
    parent, receipt = [json.loads((ROOT/p).read_text()) for p in ANCHORS]
    audit_snapshot(ROOT, parent["source"], parent["config"])
    if (receipt["status"] != "fresh_ensemble_fine_only_smc_source_seed_feature_stage_replay_pass"
            or receipt["result_sha256"] != sha(ROOT/ANCHORS[0])
            or receipt["replayed_smc_wholes"] != 4 or receipt["replayed_iid_blocks"] != 4
            or len(parent["summary"]["iid_rows"]) != 2
            or not all(r["precision_and_sensitivity"]["pass"] for r in parent["summary"]["iid_rows"])):
        raise ValueError("complete audited probability reference required")
    return parent


def configuration(parent: dict[str, Any]) -> dict[str, Any]:
    return {"original": parent["config"]["base"]["original"], "namespace": NAMESPACE,
        "anchors": {p: sha(ROOT/p) for p in ANCHORS}, "powers": list(POWERS),
        "particles": 128, "points": 64, "moves": 2, "resample_every": 4,
        "resampling_scheme": "stratified", "scale": .35, "drift_cap": 20., "wholes": 8,
        "preflight_particles": 32, "steps": 4096, "torch_threads": 8,
        "coupling": "same_seed_within_cell_replicate_triplet_not_independent_arms",
        "milestones": list(MILESTONES), "maximum_rse": .015, "maximum_unit_share": .15,
        "maximum_loo_shift": .05, "equivalence_margin": .1,
        "reference_comparisons": 6, "paired_comparisons": 4}


def prepare(config: dict[str, Any]) -> dict[str, CachedVolterraTerminal]:
    original = config["original"]
    return {c: CachedVolterraTerminal.create(_problem(original["model"],
        next(v for v in original["cells"] if v["id"] == c), 4096)) for c in CELLS}


def values(job: dict[str, Any], cache: CachedVolterraTerminal, *, preflight: bool = False) -> dict[str, Any]:
    cfg = smc_config(job, preflight=preflight)
    begin = time.perf_counter()
    result = estimate_gradient_umbrella_normalizers(partial(fine_components, cache),
        dimension=cache.problem.local_dimension, log_mixture_weights=torch.zeros(1, dtype=torch.float64),
        config=cfg, maximum_drift_norm=20.)
    expected = cfg.particles*125
    if (result.potential_evaluations != expected or result.gradient_evaluations != expected
            or abs(float(result.log_component_normalizers[0])-result.log_normalizer) > 1e-10):
        raise ValueError("single original-target product/work mismatch")
    guard()
    return {**job, "preflight": preflight, "seed": cfg.seed, "actual_work": expected,
        "backward_evaluations": expected, "dimension": cache.problem.local_dimension,
        "log_mu": result.log_normalizer, "stages": list(result.stages),
        "guide_independent_kernel": True, "torch_threads": torch.get_num_threads(),
        "wall": time.perf_counter()-begin}


def ancestry_at(record: dict[str, Any], beta: float) -> dict[str, Any]:
    eligible = [s for s in record["stages"] if s["beta"] <= beta]
    stage = eligible[-1] if eligible else {"beta": 0., "surviving_initial_ancestors": 128}
    return {"requested_beta": beta, "recorded_beta": stage["beta"],
            "ancestors": stage["surviving_initial_ancestors"]}


def summarize(records: list[dict[str, Any]], parent: dict[str, Any]) -> dict[str, Any]:
    if len(records) != 48 or {identity(r) for r in records} != {identity(j) for j in jobs()}:
        raise ValueError("complete fixed 48-whole family required")
    rows, references, paired = [], [], []
    for cell in CELLS:
        vectors = {}
        iid = next(r for r in parent["summary"]["iid_rows"] if r["cell"] == cell)
        for power in POWERS:
            group = sorted((r for r in records if (r["cell"], r["power"]) == (cell, power)), key=lambda r: r["rep"])
            logs = torch.tensor([r["log_mu"] for r in group], dtype=torch.float64)
            vectors[power] = logs
            stats = log_moments(logs)
            stability = sensitivity(logs.tolist(), bootstrap_seed=seed(cell, 100+power))
            gate = precision_gate(stats, stability, iid=False)
            milestones = [{"requested_beta": beta, "wholes": [ancestry_at(r, beta) for r in group],
                "median_ancestors": statistics.median(ancestry_at(r, beta)["ancestors"] for r in group)} for beta in MILESTONES]
            eq = relative_equivalence(iid["summary"], stats, comparisons=6)
            rows.append({"cell": cell, "power": power, "summary": stats, "sensitivity": stability,
                "precision_and_sensitivity": gate, "milestones": milestones,
                "final_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in group],
                "minimum_stage_ess": [min(s["ess_before_resampling"] for s in r["stages"]) for r in group],
                "mean_mutation_acceptance": [statistics.mean(s["mutation_acceptance"] for s in r["stages"][:-1]) for r in group],
                "score_wall": sum(r["wall"] for r in group)})
            references.append({"cell": cell, "power": power, "equivalence": eq,
                "interval_status": interval_status(eq),
                "scoped_development_corroboration_pass": bool(iid["precision_and_sensitivity"]["pass"] and gate["pass"] and eq["pass"])})
        for power in (3, 4):
            paired.append({"cell": cell, "power": power, "denominator": "power2_mean",
                "contrast": paired_normalizer_difference(vectors[2], vectors[power], comparisons=4)})
    candidates = []
    for power in (3, 4):
        checks = []
        for cell in CELLS:
            base = next(r for r in rows if (r["cell"], r["power"]) == (cell, 2))
            variant = next(r for r in rows if (r["cell"], r["power"]) == (cell, power))
            checks.append(variant["summary"]["relative_se"] < base["summary"]["relative_se"]
                and statistics.median(variant["final_ancestors"]) >= statistics.median(base["final_ancestors"])
                and variant["milestones"][1]["median_ancestors"] >= base["milestones"][1]["median_ancestors"])
        if all(checks):
            candidates.append(power)
    return {"rows": rows, "reference_comparisons": references, "paired_mean_diagnostics": paired,
        "development_candidate_powers": candidates,
        "scope": "equal_work_schedule_development_not_confirmation_variance_proof_or_continuum",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def preflight_jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "rep": 0, "power": p} for c in CELLS for p in POWERS]


def manifest_for(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {"config": config, "source": source["snapshot_sha256"],
        "expected_units": ["prepare", *(identity(j, preflight=True) for j in preflight_jobs()), *(identity(j) for j in jobs())],
        "limits": {"B": {"work": 792000, "wall": 14400.}}}


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    parent = load_parent()
    config = configuration(parent)
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, *ANCHORS))
    manifest = manifest_for(config, source)
    journal, prepared = RecoveryJournal(directory, manifest), {}

    def preparation():
        begin = time.perf_counter()
        prepared.update(prepare(config))
        return {"actual_work": 0, "wall": time.perf_counter()-begin}

    journal.unit("prepare", phase="B", work=0, maximum_wall=120., operation=preparation)
    for j in preflight_jobs():
        def preflight(job=j):
            r = values(job, prepared[job["cell"]], preflight=True)
            if r["wall"]*4 > 300.:
                raise TimeoutError("fixed whole resource forecast exceeded")
            return r
        journal.unit(identity(j, preflight=True), phase="B", work=4000, maximum_wall=60., operation=preflight)
    records = []
    for j in jobs():
        records.append(journal.unit(identity(j), phase="B", work=16000, maximum_wall=300.,
            operation=partial(values, j, prepared[j["cell"]])))
        print(f"equal-work early bridge {len(records)}/48", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summary": summarize(records, parent),
        "units": 55, "final_forward": 768000, "physical_evaluations": 792000, "backward_evaluations": 792000,
        "phase_wall": time.time()-journal._read(directory/"phase-B.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if output["phase_wall"] > 14400.:
        raise TimeoutError("final phase budget exceeded")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output, parent = json.loads(path.read_text()), load_parent()
    config = configuration(parent)
    audit_snapshot(ROOT, output["source"], config)
    manifest = manifest_for(config, output["source"])
    if output["config"] != config or output["manifest"] != manifest:
        raise ValueError("changed protocol")
    for name, digest in output["source"]["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/")) and sha(ROOT/name) != digest:
            raise ValueError("changed numerical runtime")
    journal, expected = RecoveryJournal(path.parent, manifest), set(manifest["expected_units"])
    for suffix in ("start", "finish"):
        if {p.name.removesuffix(f".{suffix}.json") for p in path.parent.glob(f"*.{suffix}.json")} != expected:
            raise ValueError("incomplete/unexpected family")
    for name in expected:
        start = journal._read(path.parent/(name+".start.json"))
        finish = journal._read(path.parent/(name+".finish.json"))
        work = 0 if name == "prepare" else 4000 if name.startswith("preflight-") else 16000
        cap = 120. if name == "prepare" else 60. if name.startswith("preflight-") else 300.
        if (start["phase"] != "B" or start["reserved_work"] != work or start["reserved_wall"] != cap
                or finish["phase"] != "B" or finish["status"] != "complete" or finish["work"] != work
                or finish["result"]["actual_work"] != work or finish["wall"] > cap):
            raise ValueError("unit/budget violation")
    prepared, records = prepare(config), []
    for is_preflight, family in ((True, preflight_jobs()), (False, jobs())):
        for j in family:
            record = journal._read(path.parent/(identity(j, preflight=is_preflight)+".finish.json"))["result"]
            if (any(record[k] != v for k, v in j.items()) or record["preflight"] != is_preflight
                    or record["seed"] != seed(j["cell"], j["rep"], "preflight" if is_preflight else "final")
                    or record["dimension"] != 8192 or record["torch_threads"] != 8
                    or record["backward_evaluations"] != (4000 if is_preflight else 16000)
                    or not record["guide_independent_kernel"] or len(record["stages"]) != 63
                    or (is_preflight and record["wall"]*4 > 300.)
                    or [s["beta"] for s in record["stages"]] != list(smc_config(j).temperatures[1:])):
                raise ValueError("identity/seed/target/stages changed")
            if not is_preflight:
                if j["rep"] == 0:
                    replay = values(j, prepared[j["cell"]])
                    if any(record[k] != replay[k] for k in ("log_mu", "stages", "actual_work", "backward_evaluations")):
                        raise ValueError("whole replay mismatch")
                records.append(record)
    if (summarize(records, parent) != output["summary"] or journal.accounting("B")[0] != 792000
            or output["physical_evaluations"] != 792000 or output["backward_evaluations"] != 792000
            or output["units"] != 55 or output["final_forward"] != 768000
            or output["phase_wall"] > 14400. or (path.parent/"phase-B.violation.json").exists()):
        raise ValueError("aggregate/work/summary mismatch")
    return {"status": "fine_smc_early_bridge_source_seed_crn_stage_work_replay_pass", "replayed_wholes": 6,
        "audit_physical_evaluations": 96000, "audit_backward_evaluations": 96000,
        "result_sha256": sha(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        target = ROOT/args.audit
        receipt = audit(target)
        write_json_atomic_nonoverwriting(target.parent/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        result = run()
        print(json.dumps({k: result[k] for k in ("units", "physical_evaluations", "backward_evaluations", "phase_wall")}))
