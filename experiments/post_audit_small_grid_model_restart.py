"""Fresh original-q development, separate from unresolved fine-grid certification."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch

from experiments.post_audit_defensive_allocation import digest_bank, guard
from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_matched_window_mechanism import KWARGS
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    sample_gaussian_mixture,
)
from src.path_integral.paired_model_risk import paired_risk_reduction
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.risk_aware_model_correction import (
    blend_model_correction,
    fit_model_correction,
    full_mixture_log_q_over_p,
    log_second_moment_bank,
    shift_spec_from_parameters,
    shift_spec_parameters,
)
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, sensitivity
from src.path_integral.weighted_bank_mixture import fit_weighted_bank_mixture
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

DIRECTORY = "results/post_audit/small_grid_model_restart_v1/session_20261020"
PLAN = "docs/plans/SMALL_GRID_MODEL_RESTART_2026-10-20_KO.md"
THEORY = "docs/theory/SMALL_GRID_MODEL_RESTART_CONTRACT_2026-10-20_KO.md"
NAMESPACE = "small-grid-original-model-restart-v1"
ARMS = ("parent", "static", "event", "risk")
ALPHAS = (0., .25, .5)
TRAIN, SELECTION, FINAL_BLOCK, HELDOUT, CHUNK = 8192, 16384, 32768, 131072, 2048
PARENT_WORK, FIT_WORK, TOTAL_WORK = 72960, 97536, 12772352


def configuration() -> dict[str, Any]:
    return {"namespace": NAMESPACE, "torch_threads": 1, "steps": 32,
        "model": {"spot": 100., "maturity": 1., "hurst": .05, "eta": 1.5, "xi": .04, "rho": -.7},
        "cells": [{"id": CELLS[0], "threshold": 1.}, {"id": CELLS[1], "threshold": 1., "eta": 2.}],
        "parent_repetitions": 5, "parent_islands": 3, "particles": 256, "points": 49,
        "bridge_power": 4, "moves": 2, "pcn_scale": .35, "resample_every": 4,
        "natural_mass": .1, "shift_clusters": 2, "covariance_rank": 0, "guide": KWARGS,
        "train": TRAIN, "selection": SELECTION, "alphas": list(ALPHAS), "arms": list(ARMS),
        "final_blocks": 8, "final_block_count": FINAL_BLOCK, "heldout": HELDOUT, "chunk": CHUNK,
        "paired_comparisons": 60, "development_final_rse": .05, "maximum_block_to_raw_rse": 2.,
        "development_successes_per_cell": 4, "global_qualification_unchanged": True}


def seed(cell: str, rep: int, stream: str, block: int = 0, role: str = "path") -> int:
    key = f"{NAMESPACE}/{cell}/{rep}/{stream}/{block}/{role}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def fit_jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "rep": r} for c in CELLS for r in range(5)]


def final_jobs() -> list[dict[str, Any]]:
    return [{**j, "arm": a, "block": b} for j in fit_jobs() for a in ARMS for b in range(8)]


def fit_id(j: dict[str, Any]) -> str:
    return f"fit-{j['cell']}-{j['rep']}"


def final_id(j: dict[str, Any]) -> str:
    return f"final-{j['cell']}-{j['rep']}-{j['arm']}-{j['block']}"


def heldout_id(j: dict[str, Any]) -> str:
    return f"heldout-{j['cell']}-{j['rep']}"


def stream_digest(j: dict[str, Any], stream: str, count: int, block: int = 0) -> str:
    seeds = [seed(j["cell"], j["rep"], stream, block*100+k, role)
             for k in range(count//CHUNK) for role in ("path", "label")]
    return canonical_digest(seeds)


def prepare(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared, costs = {}, {}
    for cell in config["cells"]:
        start = time.perf_counter()
        cache = CachedVolterraTerminal.create(_problem(config["model"], cell, 32))
        simulator_wall = time.perf_counter()-start
        start = time.perf_counter()
        guide = PhysicalWindowGuide.create(cache.problem, kind="mixed-window", **KWARGS)
        geometry_wall = time.perf_counter()-start
        if guide.spec.components != 373:
            raise ValueError("fixed physical dictionary changed")
        prepared[cell["id"]] = cache, guide.spec
        costs[cell["id"]] = {"simulator_setup": simulator_wall, "geometry_setup": geometry_wall,
            "static_components": guide.spec.components, "static_digest": canonical_digest(shift_spec_parameters(guide.spec))}
    guard()
    return prepared, costs


def iid_bank(cache: CachedVolterraTerminal, spec: GaussianMixtureShiftSpec, j: dict[str, Any],
             stream: str, count: int, block: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count % CHUNK:
        raise ValueError("complete fixed chunks required")
    z_parts, g_parts, d_parts = [], [], []
    with torch.no_grad():
        for k in range(count//CHUNK):
            draw = sample_gaussian_mixture(spec, CHUNK,
                gaussian_generator=torch.Generator().manual_seed(seed(j["cell"], j["rep"], stream, block*100+k, "path")),
                label_generator=torch.Generator().manual_seed(seed(j["cell"], j["rep"], stream, block*100+k, "label")))
            z = draw.target_coordinates
            g, d = cache.log_probability(z), full_mixture_log_q_over_p(z, spec)
            if not torch.isfinite(g).all() or (g > 1e-12).any():
                raise FloatingPointError("original conditional probability contract failed")
            z_parts.append(z)
            g_parts.append(g)
            d_parts.append(d)
            guard()
    return torch.cat(z_parts), torch.cat(g_parts), torch.cat(d_parts)


def choose_alpha(z: torch.Tensor, g: torch.Tensor, d: torch.Tensor,
                 q0: GaussianMixtureShiftSpec, correction: GaussianMixtureShiftSpec) -> tuple[GaussianMixtureShiftSpec, dict[str, Any]]:
    candidates = [blend_model_correction(q0, correction, alpha) for alpha in ALPHAS]
    scores = [log_second_moment_bank(z, g, d, q) for q in candidates]
    selected = min(range(len(ALPHAS)), key=lambda k: (scores[k], ALPHAS[k]))
    return candidates[selected], {"alpha": ALPHAS[selected], "selection_log_m2": scores,
        "candidate_digests": [canonical_digest(shift_spec_parameters(q)) for q in candidates],
        "no_op": ALPHAS[selected] == 0., "selection_not_final_risk_estimate": True}


def fit_values(j: dict[str, Any], prepared: Any) -> dict[str, Any]:
    cache, guide = prepared
    begin = time.perf_counter()
    particles, weights, diagnostics = [], [], []
    for island in range(3):
        bank = estimate_weighted_tempered_normalizer(cache.log_probability, dimension=64,
            config=WeightedSMCConfig(256, tuple((k/48)**4 for k in range(49)), 2, .35, 1,
                seed(j["cell"], j["rep"], "parent-island", island), resample_every=4,
                resampling_scheme="stratified", retain_final_particles=True))
        if bank.final_particles is None or bank.final_weights is None or bank.potential_evaluations != PARENT_WORK//3:
            raise ValueError("parent bank contract mismatch")
        particles.append(bank.final_particles)
        weights.append(bank.final_weights/3)
        diagnostics.append(bank.replicate_diagnostics[0])
    parent_smc_wall = time.perf_counter()-begin
    start = time.perf_counter()
    parent = fit_weighted_bank_mixture(torch.cat(particles), torch.cat(weights), clusters=2,
        covariance_rank=0, defensive_mass=.1).proposal
    q0 = GaussianMixtureShiftSpec(torch.stack([c.mean for c in parent.components]), parent.weights.clone())
    parent_fit_wall = time.perf_counter()-start
    start = time.perf_counter()
    training = iid_bank(cache, guide, j, "train", TRAIN)
    train_wall = time.perf_counter()-start
    corrections, fit_wall, fit_diagnostics = {}, {}, {}
    for kind in ("event", "risk"):
        start = time.perf_counter()
        correction = fit_model_correction(*training, q0, kind=kind)
        fit_wall[kind] = time.perf_counter()-start
        corrections[kind] = correction.proposal
        fit_diagnostics[kind] = {**asdict(correction.diagnostics), "cluster_masses": list(correction.cluster_masses),
            "parameters": shift_spec_parameters(correction.proposal)}
    start = time.perf_counter()
    selection = iid_bank(cache, guide, j, "selection", SELECTION)
    selection_bank_wall = time.perf_counter()-start
    proposals, selections, selection_wall = {"parent": q0, "static": guide}, {}, {}
    for kind in ("event", "risk"):
        start = time.perf_counter()
        proposals[kind], selections[kind] = choose_alpha(*selection, q0, corrections[kind])
        selection_wall[kind] = time.perf_counter()-start
    parameters = {a: shift_spec_parameters(q) for a, q in proposals.items()}
    return {**j, "actual_work": FIT_WORK, "torch_threads": torch.get_num_threads(),
        "parameters": parameters, "proposal_digests": {a: canonical_digest(p) for a, p in parameters.items()},
        "component_counts": {a: q.components for a, q in proposals.items()}, "selections": selections,
        "fit_diagnostics": fit_diagnostics, "parent_smc_diagnostics": diagnostics,
        "parent_seed_digest": canonical_digest([seed(j["cell"], j["rep"], "parent-island", i) for i in range(3)]),
        "train_digest": digest_bank(training), "selection_digest": digest_bank(selection),
        "train_seed_digest": stream_digest(j, "train", TRAIN), "selection_seed_digest": stream_digest(j, "selection", SELECTION),
        "cost": {"parent_smc": parent_smc_wall, "parent_fit": parent_fit_wall, "train_bank": train_wall,
            "correction_fit": fit_wall, "selection_bank": selection_bank_wall, "selection_density": selection_wall},
        "wall": time.perf_counter()-begin}


def final_values(j: dict[str, Any], prepared: Any, fit: dict[str, Any]) -> dict[str, Any]:
    cache, _ = prepared
    spec = shift_spec_from_parameters(fit["parameters"][j["arm"]])
    begin = time.perf_counter()
    arrays = iid_bank(cache, spec, j, "final-"+j["arm"], FINAL_BLOCK, j["block"])
    logs = arrays[1]-arrays[2]
    if (logs > math.log(10)+1e-10).any():
        raise FloatingPointError("defensive event bound violated")
    return {**j, "actual_work": FINAL_BLOCK, "moments": log_moments(logs),
        "seed_digest": stream_digest(j, "final-"+j["arm"], FINAL_BLOCK, j["block"]),
        "proposal_digest": fit["proposal_digests"][j["arm"]], "sample_digest": digest_bank(arrays),
        "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-begin}


def heldout_values(j: dict[str, Any], prepared: Any, fit: dict[str, Any]) -> dict[str, Any]:
    cache, guide = prepared
    begin = time.perf_counter()
    z, g, d = iid_bank(cache, guide, j, "heldout-risk", HELDOUT)
    logs = {a: 2*g-d-full_mixture_log_q_over_p(z, shift_spec_from_parameters(fit["parameters"][a])) for a in ARMS}
    if any((v > math.log(100)+1e-10).any() or not torch.isfinite(v).all() for v in logs.values()):
        raise FloatingPointError("defensive raw risk bound violated")
    blocks = {a: [log_moments(v)["log_mean"] for v in values.split(HELDOUT//8)] for a, values in logs.items()}
    contrasts = []
    for baseline, candidate in itertools.combinations(ARMS, 2):
        raw = paired_risk_reduction(logs[baseline], logs[candidate], comparisons=60)
        block = paired_risk_reduction(torch.tensor(blocks[baseline], dtype=torch.float64),
            torch.tensor(blocks[candidate], dtype=torch.float64), comparisons=60)
        contrasts.append({"baseline": baseline, "candidate": candidate, "raw": raw, "block": block,
            "same_proposal": fit["proposal_digests"][baseline] == fit["proposal_digests"][candidate],
            "development_reduction_signal": bool(raw["candidate_lower_m2_signal"] and block["candidate_lower_m2_signal"]
                and raw["candidate_lower_absolute_signal"] and block["candidate_lower_absolute_signal"]
                and fit["proposal_digests"][baseline] != fit["proposal_digests"][candidate])})
    return {**j, "actual_work": HELDOUT, "seed_digest": stream_digest(j, "heldout-risk", HELDOUT),
        "sample_digest": digest_bank((z, g, d)), "proposal_digests": fit["proposal_digests"],
        "risk_moments": {a: log_moments(v) for a, v in logs.items()}, "block_log_means": blocks,
        "risk_sensitivity": {a: sensitivity(blocks[a], bootstrap_seed=seed(j["cell"], j["rep"], "bootstrap-risk-"+a)) for a in ARMS},
        "contrasts": contrasts, "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-begin}


def workflow_cost(arm: str, fit: dict[str, Any], setup: dict[str, Any], inference: float) -> dict[str, float]:
    cost = fit["cost"]
    offline = setup["simulator_setup"]+(0. if arm == "parent" else setup["geometry_setup"])
    parent = 0. if arm == "static" else cost["parent_smc"]+cost["parent_fit"]
    training = cost["train_bank"]+cost["correction_fit"][arm] if arm in ("event", "risk") else 0.
    selection = cost["selection_bank"]+cost["selection_density"][arm] if arm in ("event", "risk") else 0.
    return {"offline_setup": offline, "parent_training_and_fit": parent, "correction_bank_and_fit": training,
        "selection": selection, "inference": inference, "total_fixed_final_budget_wall": offline+parent+training+selection+inference}


def summarize(fits: list[dict[str, Any]], finals: list[dict[str, Any]], risks: list[dict[str, Any]],
              setup: dict[str, Any]) -> dict[str, Any]:
    expected = {(j["cell"], j["rep"]) for j in fit_jobs()}
    if (len(fits) != 10 or {(f["cell"], f["rep"]) for f in fits} != expected
            or len(finals) != 320 or {final_id(f) for f in finals} != {final_id(j) for j in final_jobs()}
            or len(risks) != 10 or {(r["cell"], r["rep"]) for r in risks} != expected):
        raise ValueError("complete fresh whole-training family required; no prefix inference")
    rows = []
    for fit in fits:
        for arm in ARMS:
            group = sorted((r for r in finals if (r["cell"], r["rep"], r["arm"]) == (fit["cell"], fit["rep"], arm)), key=lambda r: r["block"])
            moments = merge_retaining_maximum([r["moments"] for r in group])
            stable = sensitivity([r["moments"]["log_mean"] for r in group],
                bootstrap_seed=seed(fit["cell"], fit["rep"], "bootstrap-final-"+arm))
            rows.append({"cell": fit["cell"], "rep": fit["rep"], "arm": arm, "moments": moments,
                "block_sensitivity": stable, "proposal_digest": fit["proposal_digests"][arm],
                "components": fit["component_counts"][arm], "alpha": fit["selections"].get(arm, {}).get("alpha"),
                "development_precision_only": moments["relative_se"] <= .05 and stable["between_unit_relative_se"] <= 2*moments["relative_se"],
                "cost": workflow_cost(arm, fit, setup[fit["cell"]], sum(r["wall"] for r in group))})
    cells: list[dict[str, Any]] = []
    for cell in CELLS:
        methods: list[dict[str, Any]] = []
        for arm in ARMS:
            group = [r for r in rows if r["cell"] == cell and r["arm"] == arm]
            success = 0
            for row in group:
                risk = next(r for r in risks if (r["cell"], r["rep"]) == (cell, row["rep"]))
                if arm != "parent":
                    c = next(c for c in risk["contrasts"] if (c["baseline"], c["candidate"]) == ("parent", arm))
                    success += int(c["development_reduction_signal"] and row["development_precision_only"])
            methods.append({"arm": arm, "median_rse": statistics.median(r["moments"]["relative_se"] for r in group),
                "rse_range": [min(r["moments"]["relative_se"] for r in group), max(r["moments"]["relative_se"] for r in group)],
                "development_precision_passes": sum(r["development_precision_only"] for r in group),
                "parent_reduction_and_precision_signals": success,
                "median_fixed_budget_total_wall": statistics.median(r["cost"]["total_fixed_final_budget_wall"] for r in group),
                "selected_alphas": [r["alpha"] for r in group]})
        cells.append({"cell": cell, "methods": methods})
    candidates = [a for a in ARMS[1:] if all(next(m for m in c["methods"] if m["arm"] == a)["parent_reduction_and_precision_signals"] >= 4 for c in cells)]
    return {"rows": rows, "cells": cells, "heldout_risk": risks, "fits": fits,
        "development_candidate_arms": candidates, "scope": "N32_original_q_development_not_fine_reference_or_speedup_certificate",
        "static_repetitions": "fresh_sampling_repeats_of_fixed_geometry_not_training_repeats",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def manifest_for(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {"config": config, "source": source["snapshot_sha256"],
        "expected_units": ["prepare", *("preflight-"+c for c in CELLS), *(fit_id(j) for j in fit_jobs()),
            *(final_id(j) for j in final_jobs()), *(heldout_id(j) for j in fit_jobs())],
        "limits": {"M": {"work": TOTAL_WORK, "wall": 7200.}}}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    config = configuration()
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY))
    manifest = manifest_for(config, source)
    journal, prepared, setup = RecoveryJournal(directory, manifest), {}, {}

    def preparation():
        begin = time.perf_counter()
        p, c = prepare(config)
        prepared.update(p)
        setup.update(c)
        return {"actual_work": 0, "setup": c, "wall": time.perf_counter()-begin}

    saved = journal.unit("prepare", phase="M", work=0, maximum_wall=120., operation=preparation)
    setup.update(saved["setup"])
    for cell in CELLS:
        def preflight(c=cell):
            cache, guide = prepared[c]
            begin = time.perf_counter()
            z = sample_gaussian_mixture(guide, 256,
                gaussian_generator=torch.Generator().manual_seed(seed(c, 0, "preflight")),
                label_generator=torch.Generator().manual_seed(seed(c, 0, "preflight", role="label"))).target_coordinates
            g = cache.log_probability(z)
            d = full_mixture_log_q_over_p(z, guide)
            wall = time.perf_counter()-begin
            if wall*128 > 60.:
                raise TimeoutError("fixed final block resource forecast failed")
            return {"cell": c, "actual_work": 256, "digest": digest_bank((z, g, d)), "wall": wall}
        journal.unit("preflight-"+cell, phase="M", work=256, maximum_wall=30., operation=preflight)
    fits, finals, risks = [], [], []
    for j in fit_jobs():
        fit = journal.unit(fit_id(j), phase="M", work=FIT_WORK, maximum_wall=180.,
            operation=partial(fit_values, j, prepared[j["cell"]]))
        fits.append(fit)
        for arm in ARMS:
            for block in range(8):
                job = {**j, "arm": arm, "block": block}
                finals.append(journal.unit(final_id(job), phase="M", work=FINAL_BLOCK, maximum_wall=60.,
                    operation=partial(final_values, job, prepared[j["cell"]], fit)))
        risks.append(journal.unit(heldout_id(j), phase="M", work=HELDOUT, maximum_wall=180.,
            operation=partial(heldout_values, j, prepared[j["cell"]], fit)))
        print(f"original-q complete parent {len(fits)}/10", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "setup": setup,
        "summary": summarize(fits, finals, risks, setup), "units": 343, "physical_evaluations": TOTAL_WORK,
        "phase_wall": time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if output["phase_wall"] > 7200.:
        raise TimeoutError("final I/O phase budget exceeded")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config = configuration()
    audit_snapshot(ROOT, output["source"], config)
    manifest = manifest_for(config, output["source"])
    if output["config"] != config or output["manifest"] != manifest:
        raise ValueError("changed model-development protocol")
    for name, digest in output["source"]["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/")) and sha(ROOT/name) != digest:
            raise ValueError("changed numerical runtime")
    journal, expected = RecoveryJournal(path.parent, manifest), set(manifest["expected_units"])
    for suffix in ("start", "finish"):
        if {p.name.removesuffix(f".{suffix}.json") for p in path.parent.glob(f"*.{suffix}.json")} != expected:
            raise ValueError("incomplete/unexpected family")
    for name in expected:
        start, finish = [journal._read(path.parent/(name+"."+s+".json")) for s in ("start", "finish")]
        work = 0 if name == "prepare" else 256 if name.startswith("preflight-") else FIT_WORK if name.startswith("fit-") else HELDOUT if name.startswith("heldout-") else FINAL_BLOCK
        cap = 120. if name == "prepare" else 30. if name.startswith("preflight-") else 180. if name.startswith(("fit-", "heldout-")) else 60.
        if (start["phase"] != "M" or start["reserved_work"] != work or start["reserved_wall"] != cap
                or finish["status"] != "complete" or finish["phase"] != "M" or finish["work"] != work
                or finish["result"]["actual_work"] != work or finish["wall"] > cap):
            raise ValueError("unit budget/status mismatch")
    prepared, _ = prepare(config)
    fits, finals, risks = [], [], []
    for j in fit_jobs():
        fit = journal._read(path.parent/(fit_id(j)+".finish.json"))["result"]
        if (any(fit[k] != v for k, v in j.items()) or fit["torch_threads"] != 1
                or fit["train_seed_digest"] != stream_digest(j, "train", TRAIN)
                or fit["selection_seed_digest"] != stream_digest(j, "selection", SELECTION)
                or fit["parent_seed_digest"] != canonical_digest([seed(j["cell"], j["rep"], "parent-island", i) for i in range(3)])):
            raise ValueError("fit identity/seed changed")
        if any(canonical_digest(p) != fit["proposal_digests"][a] for a, p in fit["parameters"].items()):
            raise ValueError("proposal binding changed")
        if j["rep"] == 0:
            replay = fit_values(j, prepared[j["cell"]])
            fields = ("parameters", "proposal_digests", "selections", "fit_diagnostics", "parent_smc_diagnostics", "train_digest", "selection_digest")
            if any(replay[k] != fit[k] for k in fields):
                raise ValueError("whole parent/fit/selection replay mismatch")
        fits.append(fit)
        risk = journal._read(path.parent/(heldout_id(j)+".finish.json"))["result"]
        if (any(risk[k] != v for k, v in j.items()) or risk["seed_digest"] != stream_digest(j, "heldout-risk", HELDOUT)
                or risk["proposal_digests"] != fit["proposal_digests"] or risk["torch_threads"] != 1):
            raise ValueError("heldout identity/binding mismatch")
        if j["rep"] == 0:
            replay = heldout_values(j, prepared[j["cell"]], fit)
            if any(replay[k] != risk[k] for k in ("sample_digest", "risk_moments", "block_log_means", "risk_sensitivity", "contrasts")):
                raise ValueError("heldout numerical replay mismatch")
        risks.append(risk)
    for j in final_jobs():
        fit = next(f for f in fits if (f["cell"], f["rep"]) == (j["cell"], j["rep"]))
        record = journal._read(path.parent/(final_id(j)+".finish.json"))["result"]
        if (any(record[k] != v for k, v in j.items()) or record["torch_threads"] != 1
                or record["seed_digest"] != stream_digest(j, "final-"+j["arm"], FINAL_BLOCK, j["block"])
                or record["proposal_digest"] != fit["proposal_digests"][j["arm"]]):
            raise ValueError("final identity/seed/binding mismatch")
        if j["rep"] == 0 and j["block"] == 0:
            replay = final_values(j, prepared[j["cell"]], fit)
            if any(replay[k] != record[k] for k in ("moments", "sample_digest")):
                raise ValueError("final numerical replay mismatch")
        finals.append(record)
    saved_setup = journal._read(path.parent/"prepare.finish.json")["result"]["setup"]
    if (saved_setup != output["setup"] or summarize(fits, finals, risks, saved_setup) != output["summary"]
            or journal.accounting("M")[0] != TOTAL_WORK or output["physical_evaluations"] != TOTAL_WORK
            or output["units"] != 343 or output["phase_wall"] > 7200. or (path.parent/"phase-M.violation.json").exists()):
        raise ValueError("complete aggregate mismatch")
    return {"status": "small_grid_original_model_source_seed_fit_selection_final_risk_replay_pass",
        "replayed_parents": 2, "replayed_final_blocks": 8, "replayed_heldout_banks": 2,
        "audit_physical_evaluations": 719360, "result_sha256": sha(path)}


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
        result = run()
        print(json.dumps({k: result[k] for k in ("units", "physical_evaluations", "phase_wall")}))
