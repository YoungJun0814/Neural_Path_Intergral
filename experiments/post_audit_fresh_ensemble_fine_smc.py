"""Fresh fixed-count ensemble precision and independent fine-only SMC diagnostics."""

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

from experiments.post_audit_bank_ensemble_coverage import DIRECTORY as PARENT_DIRECTORY
from experiments.post_audit_bank_ensemble_coverage import prepare_all
from experiments.post_audit_defensive_allocation import guard
from experiments.post_audit_fft_static_fine_reference import merge_retaining_maximum
from experiments.post_audit_p2_paired_gradient import CELLS
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.bank_ensemble_coverage import (
    coverage_ledger,
    merge_coverage,
    physical_path_features,
    summarize_bins,
)
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gaussian_span_marginalization import sample_gaussian_mixture
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

DIRECTORY = "results/post_audit/fresh_ensemble_fine_smc_v1/session_20261020"
PLAN = "docs/plans/FRESH_ENSEMBLE_PRECISION_AND_FINE_ONLY_SMC_2026-10-20_KO.md"
THEORY = "docs/theory/FINE_ONLY_REFERENCE_AND_PRECISION_CONTRACT_2026-10-20_KO.md"
NAMESPACE = "fresh-ensemble-precision-fine-smc-v1"
ANCHORS = tuple(PARENT_DIRECTORY+"/"+name+".json" for name in ("result", "audit"))
COUNTS = {CELLS[0]: 59, CELLS[1]: 43}
DRAW_COUNT, CHUNK = 8192, 256
ARMS = {"64points-2moves": (128, 64, 2, 4), "128points-1move": (128, 128, 1, 8)}


def seed(*parts: Any) -> int:
    return int(hashlib.sha256((NAMESPACE+"/"+"/".join(map(str, parts))).encode()).hexdigest()[:15], 16)


def iid_jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "block": b} for c in CELLS for b in range(COUNTS[c])]


def smc_jobs() -> list[dict[str, Any]]:
    return [{"cell": c, "arm": a, "rep": rep} for c in CELLS for a in ARMS for rep in range(8)]


def iid_id(job: dict[str, Any]) -> str:
    return f"iid-{job['cell']}-{job['block']}"


def smc_id(job: dict[str, Any]) -> str:
    return f"smc-{job['cell']}-{job['arm']}-{job['rep']}"


def smc_work(arm: str) -> int:
    particles, points, moves, _ = (32, 16, 1, 4) if arm == "preflight" else ARMS[arm]
    return particles*(1+(points-2)*moves)


def iid_seed_digest(job: dict[str, Any], count: int, stream: str) -> str:
    values = [seed(stream, job["cell"], job["block"], k, role)
              for k in range(count//CHUNK) for role in ("path", "label")]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def load_parent() -> dict[str, Any]:
    parent, receipt = [json.loads((ROOT/p).read_text()) for p in ANCHORS]
    audit_snapshot(ROOT, parent["source"], parent["config"])
    if (receipt["status"] != "bank_ensemble_all_path_source_seed_feature_moment_replay_pass"
            or receipt["result_sha256"] != sha(ROOT/ANCHORS[0]) or receipt["replayed_blocks"] != 8):
        raise ValueError("complete unchanged parent with feature replay required")
    return parent


def configuration(parent: dict[str, Any]) -> dict[str, Any]:
    predicted = {}
    for cell in CELLS:
        r = next(r for r in parent["summary"]["rows"] if r["cell"] == cell and r["arm"] == "ensemble")
        if r["summary"]["count"] != 65536:
            raise ValueError("changed pilot count")
        worst = max(r["summary"]["relative_se"], r["block_sensitivity"]["between_unit_relative_se"])
        predicted[cell] = math.ceil(3*65536*(worst/.015)**2/DRAW_COUNT)
    if predicted != COUNTS:
        raise ValueError("changed source-bound fixed-count projection")
    return {"base": parent["config"], "counts": predicted, "namespace": NAMESPACE,
        "anchors": {p: sha(ROOT/p) for p in ANCHORS}, "torch_threads": 8,
        "draw_count": DRAW_COUNT, "chunk": CHUNK, "smc_arms": {a: list(v) for a, v in ARMS.items()},
        "smc_wholes": 8, "smc_preflight": [32, 16, 1, 4], "pcn_scale": .35, "maximum_drift_norm": 20.,
        "qualification": {"maximum_relative_se": .015, "maximum_unit_share": .15,
            "maximum_loo_shift": .05, "maximum_block_to_iid_ratio": 2., "equivalence_margin": .1}}


def iid_values(job: dict[str, Any], prepared: Any, *, count: int = DRAW_COUNT,
               stream: str = "iid-final") -> dict[str, Any]:
    if count not in (CHUNK, DRAW_COUNT) or stream not in ("iid-final", "iid-preflight"):
        raise ValueError("fixed final/preflight dimensions required")
    cache, specs = prepared
    spec, parts, ledgers = specs["ensemble"], [], []
    begin = time.perf_counter()
    with torch.no_grad():
        for k in range(count//CHUNK):
            draw = sample_gaussian_mixture(spec, CHUNK,
                gaussian_generator=torch.Generator().manual_seed(seed(stream, job["cell"], job["block"], k, "path")),
                label_generator=torch.Generator().manual_seed(seed(stream, job["cell"], job["block"], k, "label")))
            z = draw.target_coordinates
            terminal, integrated, variance = cache.quantities(z)
            p, task = cache.problem, cache.problem.task
            assert isinstance(task, TerminalThresholdTask)
            log_g = torch.special.log_ndtr((math.log(task.level)-terminal)/((1-p.rho**2)*integrated).sqrt())
            log_d = torch.logsumexp(z @ spec.means.T-.5*spec.means.square().sum(1)+spec.weights.log(), 1)
            values = log_g-log_d
            if not torch.isfinite(values).all() or (values > math.log(10)+1e-10).any():
                raise FloatingPointError("full-mixture ordinary contribution contract failed")
            features = physical_path_features(variance[:, :-1], maturity=p.maturity)
            torch.testing.assert_close(features[:, 0], integrated.log(), rtol=1e-12, atol=1e-12)
            parts.append(log_moments(values))
            ledgers.append(coverage_ledger(values, features))
            guard()
    moments, coverage = merge_retaining_maximum(parts), merge_coverage(ledgers)
    if any(abs(moments[k]-coverage[k]) > 1e-10 for k in ("log_sum", "log_sum_squares")):
        raise ValueError("path partitions changed ordinary estimate")
    return {**job, "actual_work": count, "moments": moments, "coverage": coverage,
            "seed_digest": iid_seed_digest(job, count, stream), "stream": stream,
            "torch_threads": torch.get_num_threads(), "wall": time.perf_counter()-begin}


def fine_components(cache: CachedVolterraTerminal, z: torch.Tensor) -> torch.Tensor:
    return cache.log_probability(z)[:, None]


def smc_values(job: dict[str, Any], cache: CachedVolterraTerminal) -> dict[str, Any]:
    particles, points, moves, every = (32, 16, 1, 4) if job["arm"] == "preflight" else ARMS[job["arm"]]
    whole_seed = seed("smc-whole", job["cell"], job["arm"], job["rep"])
    cfg = WeightedSMCConfig(particles, tuple((i/(points-1))**2 for i in range(points)), moves, .35, 1,
                           whole_seed, resample_every=every, resampling_scheme="stratified")
    begin = time.perf_counter()
    result = estimate_gradient_umbrella_normalizers(partial(fine_components, cache),
        dimension=cache.problem.local_dimension, log_mixture_weights=torch.zeros(1, dtype=torch.float64),
        config=cfg, maximum_drift_norm=20.)
    expected = smc_work(job["arm"])
    if (result.potential_evaluations != expected or result.gradient_evaluations != expected
            or abs(float(result.log_component_normalizers[0])-result.log_normalizer) > 1e-10):
        raise ValueError("fine-only normalizer/product or forward/backward accounting mismatch")
    guard()
    return {**job, "seed": whole_seed, "actual_work": expected, "backward_evaluations": expected,
        "timestep_forward": cache.problem.steps*expected, "log_mu": result.log_normalizer,
        "stages": list(result.stages), "dimension": cache.problem.local_dimension,
        "guide_independent_kernel": True, "torch_threads": torch.get_num_threads(),
        "wall": time.perf_counter()-begin}


def precision_gate(summary: dict[str, Any], stability: dict[str, Any], *, iid: bool) -> dict[str, bool]:
    checks = {"rse": summary["relative_se"] <= .015,
              "maximum_inferential_unit_share": stability["maximum_unit_contribution_fraction"] <= .15,
              "leave_one_out": stability["maximum_leave_one_out_relative_shift"] <= .05,
              "block_to_raw_consistency": not iid or stability["between_unit_relative_se"] <= 2*summary["relative_se"]}
    return {**checks, "pass": all(checks.values())}


def interval_status(equivalence: dict[str, Any]) -> str:
    difference, se, z = (equivalence["symmetric_relative_difference"], equivalence["standard_error"], equivalence["confidence_z"])
    low, high = difference-z*se, difference+z*se
    if equivalence["pass"]:
        return "development_equivalence_pass"
    if low > .1 or high < -.1:
        return "detected_development_disagreement_not_bias_attribution"
    return "unresolved_equivalence_uncertainty"


def iid_stage_summary(iid_records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(iid_records) != 102 or {iid_id(r) for r in iid_records} != {iid_id(j) for j in iid_jobs()}:
        raise ValueError("complete IID stage required; no prefix inference")
    rows = []
    for c in CELLS:
        group = sorted((r for r in iid_records if r["cell"] == c), key=lambda r: r["block"])
        moments = merge_retaining_maximum([r["moments"] for r in group])
        coverage = merge_coverage([r["coverage"] for r in group])
        stability = sensitivity([r["moments"]["log_mean"] for r in group], bootstrap_seed=seed("bootstrap-iid", c))
        rows.append({"cell": c, "method": "frozen-bank-ensemble-iid", "summary": moments,
            "sensitivity": stability, "coverage": coverage, "bin_summaries": summarize_bins(coverage),
            "precision_and_sensitivity": precision_gate(moments, stability, iid=True),
            "score_wall": sum(r["wall"] for r in group)})
    return {"iid_rows": rows, "scope": "complete_I_stage_not_independent_corroboration",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def summarize(iid_records: list[dict[str, Any]], smc_records: list[dict[str, Any]],
              parent: dict[str, Any]) -> dict[str, Any]:
    if (len(iid_records) != 102 or {iid_id(r) for r in iid_records} != {iid_id(j) for j in iid_jobs()}
            or len(smc_records) != 32 or {smc_id(r) for r in smc_records} != {smc_id(j) for j in smc_jobs()}):
        raise ValueError("complete fresh fixed-count two-mechanism family required")
    iid_rows = iid_stage_summary(iid_records)["iid_rows"]
    smc_rows, comparisons, pilot_comparisons = [], [], []
    for c in CELLS:
        row = next(r for r in iid_rows if r["cell"] == c)
        moments, iid_gate = row["summary"], row["precision_and_sensitivity"]
        old = next(r for r in parent["summary"]["rows"] if r["cell"] == c and r["arm"] == "ensemble")
        pilot_comparisons.append({"cell": c, "equivalence": relative_equivalence(moments, old["summary"], comparisons=2),
            "scope": "same_frozen_weights_historical_pilot_development_not_independent_mechanism"})
        for arm in ARMS:
            wholes = sorted((r for r in smc_records if r["cell"] == c and r["arm"] == arm), key=lambda r: r["rep"])
            logs = torch.tensor([r["log_mu"] for r in wholes], dtype=torch.float64)
            stats = log_moments(logs)
            sens = sensitivity(logs.tolist(), bootstrap_seed=seed("bootstrap-smc", c, arm))
            gate = precision_gate(stats, sens, iid=False)
            eq = relative_equivalence(moments, stats, comparisons=4)
            smc_rows.append({"cell": c, "arm": arm, "summary": stats, "sensitivity": sens,
                "precision_and_sensitivity": gate,
                "initial_ancestors": [r["stages"][-1]["surviving_initial_ancestors"] for r in wholes],
                "minimum_stage_ess": [min(s["ess_before_resampling"] for s in r["stages"]) for r in wholes],
                "mean_mutation_acceptance": [sum(s["mutation_acceptance"] for s in r["stages"][:-1])/len(r["stages"][:-1]) for r in wholes],
                "score_wall": sum(r["wall"] for r in wholes), "guide_independent_kernel": True})
            comparisons.append({"cell": c, "smc_arm": arm, "equivalence": eq,
                "interval_status": interval_status(eq), "both_accuracy_qualified": iid_gate["pass"] and gate["pass"],
                "scoped_development_corroboration_pass": bool(iid_gate["pass"] and gate["pass"] and eq["pass"])})
    return {"iid_rows": iid_rows, "smc_rows": smc_rows, "comparisons": comparisons,
        "pilot_comparisons": pilot_comparisons,
        "scope": "fresh_fine_probability_development_not_all_q_mesh_or_continuum_confirmation",
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def manifest_for(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {"config": config, "source": source["snapshot_sha256"],
        "expected_units": ["prepare", *("iid-preflight-"+c for c in CELLS), *(iid_id(j) for j in iid_jobs()),
                           *("smc-preflight-"+c for c in CELLS), *(smc_id(j) for j in smc_jobs())],
        "limits": {"I": {"work": 836096, "wall": 7200.}, "G": {"work": 517056, "wall": 10800.}}}


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
        prepared.update(prepare_all(config["base"]))
        return {"actual_work": 0, "wall": time.perf_counter()-begin}

    journal.unit("prepare", phase="I", work=0, maximum_wall=120., operation=preparation)
    for cell in CELLS:
        def iid_preflight(c: str = cell):
            r = iid_values({"cell": c, "block": 0}, prepared[c], count=CHUNK, stream="iid-preflight")
            if r["wall"]*32 > 180.:
                raise TimeoutError("IID whole-unit resource forecast failed")
            return r
        journal.unit("iid-preflight-"+cell, phase="I", work=CHUNK, maximum_wall=30., operation=iid_preflight)
    iid_records = []
    for j in iid_jobs():
        iid_records.append(journal.unit(iid_id(j), phase="I", work=DRAW_COUNT, maximum_wall=180.,
            operation=partial(iid_values, j, prepared[j["cell"]])))
        print(f"fresh ensemble precision {len(iid_records)}/102", flush=True)
    write_json_atomic_nonoverwriting(directory/"iid-stage.json", {"config": config, "source": source,
        "manifest": manifest, "summary": iid_stage_summary(iid_records), "iid_final_physical": 835584})
    for cell in CELLS:
        def smc_preflight(c: str = cell):
            r = smc_values({"cell": c, "arm": "preflight", "rep": 0}, prepared[c][0])
            if r["wall"]*16256/480 > 300.:
                raise TimeoutError("SMC whole-unit resource forecast failed")
            return r
        journal.unit("smc-preflight-"+cell, phase="G", work=480, maximum_wall=30., operation=smc_preflight)
    smc_records = []
    for j in smc_jobs():
        smc_records.append(journal.unit(smc_id(j), phase="G", work=smc_work(j["arm"]), maximum_wall=300.,
            operation=partial(smc_values, j, prepared[j["cell"]][0])))
        print(f"fine-only guide-free SMC {len(smc_records)}/32", flush=True)
    output: dict[str, Any] = {"config": config, "source": source, "manifest": manifest,
        "summary": summarize(iid_records, smc_records, parent), "units": 139,
        "iid_final_physical": 835584, "smc_final_physical": 516096,
        "physical_evaluations": 1353152, "backward_evaluations": 517056,
        "phase_wall": {p: time.time()-journal._read(directory/f"phase-{p}.clock.json")["started_unix"] for p in ("I", "G")}}
    # Phase I is sequentially followed by G; exclude G from its recorded active runtime.
    first_g = journal._read(directory/"phase-G.clock.json")["started_unix"]
    output["phase_wall"]["I"] = first_g-journal._read(directory/"phase-I.clock.json")["started_unix"]
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if any(output["phase_wall"][p] > manifest["limits"][p]["wall"] for p in ("I", "G")):
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O phase budget failure")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    output, parent = json.loads(path.read_text()), load_parent()
    config = configuration(parent)
    audit_snapshot(ROOT, output["source"], config)
    manifest = manifest_for(config, output["source"])
    if output["config"] != config or output["manifest"] != manifest:
        raise ValueError("changed fixed configuration")
    for name, digest in output["source"]["snapshot_file_hashes"].items():
        if (name.startswith("src/") or name == "experiments/post_audit_fresh_ensemble_fine_smc.py") and sha(ROOT/name) != digest:
            raise ValueError("changed numerical runtime")
    journal, expected = RecoveryJournal(path.parent, manifest), set(manifest["expected_units"])
    for suffix in ("start", "finish"):
        if {p.name.removesuffix(f".{suffix}.json") for p in path.parent.glob(f"*.{suffix}.json")} != expected:
            raise ValueError("incomplete/unexpected family membership")
    for name in expected:
        f = journal._read(path.parent/(name+".finish.json"))
        work = 0 if name == "prepare" else 256 if name.startswith("iid-preflight-") else 8192 if name.startswith("iid-") else 480 if name.startswith("smc-preflight-") else smc_work("64points-2moves" if "64points-2moves" in name else "128points-1move")
        cap = 120. if name == "prepare" else 30. if "preflight-" in name else 180. if name.startswith("iid-") else 300.
        if f["status"] != "complete" or f["work"] != work or f["result"]["actual_work"] != work or f["wall"] > cap:
            raise ValueError("failed unit or work/cap mutation")
    prepared, iid_records, smc_records = prepare_all(config["base"]), [], []
    for j in iid_jobs():
        r = journal._read(path.parent/(iid_id(j)+".finish.json"))["result"]
        if (any(r[k] != v for k, v in j.items()) or r["seed_digest"] != iid_seed_digest(j, DRAW_COUNT, "iid-final")
                or r["stream"] != "iid-final" or r["torch_threads"] != 8):
            raise ValueError("IID identity/seed changed")
        if j["block"] in (0, COUNTS[j["cell"]]-1):
            replay = iid_values(j, prepared[j["cell"]])
            if any(replay[k] != r[k] for k in ("moments", "coverage", "seed_digest")):
                raise ValueError("IID numerical replay mismatch")
        iid_records.append(r)
    for j in smc_jobs():
        r = journal._read(path.parent/(smc_id(j)+".finish.json"))["result"]
        if (any(r[k] != v for k, v in j.items()) or r["seed"] != seed("smc-whole", j["cell"], j["arm"], j["rep"])
                or r["torch_threads"] != 8 or r["backward_evaluations"] != smc_work(j["arm"])
                or r["dimension"] != 8192 or not r["guide_independent_kernel"]):
            raise ValueError("SMC identity/work/target changed")
        if j["rep"] == 0:
            replay = smc_values(j, prepared[j["cell"]][0])
            if any(replay[k] != r[k] for k in ("log_mu", "stages", "backward_evaluations")):
                raise ValueError("SMC whole replay mismatch")
        smc_records.append(r)
    intermediate = json.loads((path.parent/"iid-stage.json").read_text())
    if (intermediate != {"config": config, "source": output["source"], "manifest": manifest,
                        "summary": iid_stage_summary(iid_records), "iid_final_physical": 835584}
            or summarize(iid_records, smc_records, parent) != output["summary"]
            or journal.accounting("I")[0] != 836096 or journal.accounting("G")[0] != 517056
            or output["physical_evaluations"] != 1353152 or output["backward_evaluations"] != 517056
            or output["units"] != 139 or any(output["phase_wall"][p] > manifest["limits"][p]["wall"] for p in ("I", "G"))
            or any((path.parent/p).exists() for p in ("phase-I.violation.json", "phase-G.violation.json", "final_io_failure.json"))):
        raise ValueError("aggregate/work/operational mismatch")
    return {"status": "fresh_ensemble_fine_only_smc_source_seed_feature_stage_replay_pass",
        "replayed_iid_blocks": 4, "replayed_smc_wholes": 4, "audit_physical_evaluations": 97280,
        "audit_backward_evaluations": 64512, "result_sha256": sha(path)}


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
        print(json.dumps({k: v for k, v in run().items() if k in ("units", "physical_evaluations", "backward_evaluations", "phase_wall")}))
