"""Fresh canonical validation combining exact probability conditioning and MH."""

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
import yaml

from experiments.post_audit_canonical_gradient_recovery import family_weights
from experiments.post_audit_high_eta_family_validation import TARGETS
from experiments.post_audit_high_eta_family_validation import decision as high_decision
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_umbrella_reference_pilot import prepare
from experiments.post_audit_umbrella_reference_pilot import seed as umbrella_seed
from experiments.post_audit_umbrella_reference_pilot import unit_values as umbrella_unit
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.gaussian_mixture_marginal import ShiftMixtureMarginal
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, merge_moments, sensitivity
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
CELL = "canonical-h005-k1"
PLAN = "docs/plans/CANONICAL_FIVE_Q_FRESH_REFERENCE_VALIDATION_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/canonical_family_validation_v1/session_20261010"
NAMESPACE = "canonical-family-precision-v1-"
ANCHORS = {"risk": "results/post_audit/canonical_gradient_resource_v1/session_20261010/result.json",
           "probability": "results/post_audit/canonical_last_pair_probability_v1/session_20261010/result.json",
           "umbrella": "results/post_audit/umbrella_reference_pilot_v1/session_20261010/result.json",
           "gradient": "results/post_audit/canonical_gradient_recovery_v1/session_20261010/result.json"}


def jobs() -> list[dict[str, Any]]:
    return [{"cell": CELL, "namespace": NAMESPACE, "method": method, "family": family, "rep": rep}
            for method, family, count in (("guide-iid", "all", 256), ("free-full", "risk", 99),
                                          ("free-full", "mu", 71), ("envelope", "all", 154)) for rep in range(count)]


def identity(j: dict[str, Any]) -> str:
    return f"{CELL}-{j['method']}-{j['family']}-{j['rep']}"


def seed(j: dict[str, Any], stream: str) -> int:
    if j["method"] == "envelope" and stream == "whole":
        return umbrella_seed(NAMESPACE+f"{CELL}-envelope-{j['rep']}")
    return int(hashlib.sha256((NAMESPACE+identity(j)+"/"+stream).encode()).hexdigest()[:15], 16)


def expected_work(j: dict[str, Any]) -> tuple[int, int]:
    if j["method"] == "guide-iid":
        return 32768, 0
    if j["method"] == "envelope":
        return 254976, 0
    work = 517120 if j["family"] == "risk" else 254976
    return work, work


def unit_values(j: dict[str, Any], cells: dict[str, Any]) -> dict[str, Any]:
    c = cells[CELL]
    if j["method"] == "envelope":
        return {**umbrella_unit(j, cells), "backward_evaluations": 0}
    begin = time.perf_counter()
    if j["method"] == "guide-iid":
        streams = {s: seed(j, s) for s in ("path", "label")}
        z = c["guide"].sample(32768, path_seed=streams["path"], label_seed=streams["label"]).samples
        values = []
        for batch in z.split(4096):
            u = batch[:, :-2]
            cache = c["evaluator"].last_pair(u, c["guide"])
            full_g = torch.special.log_ndtr((cache.a-cache.b*batch[:, -2])/torch.sqrt(cache.c_squared))
            full_r = c["gdensity"].log_q_over_p(batch)
            rb = cache.log_mu_mean()-c["marginal"].log_q_over_p(u)
            values.append(torch.stack([rb]+[2*full_g-q.log_q_over_p(batch)-full_r for q in c["qdensities"]], 1))
        vector = torch.cat(values)
        return {**j, "seeds": streams, "actual_work": 32768, "backward_evaluations": 0,
                "moments": [log_moments(vector[:, k]) for k in range(6)], "component_targets": TARGETS,
                "q_digests": c["q_digests"], "wall": time.perf_counter()-begin,
                "guide_independent_kernel": False, "probability_contract": "exact_last_pair_marginal_ordinary_is"}
    risk = j["family"] == "risk"
    weights = family_weights("risks-pcnl" if risk else "probability-pcnl", c["log_weights"].tolist())
    def components(z):
        g = c["evaluator"].log_probability(z)
        return (torch.stack([log_risk_potential(g, q.log_q_over_p(z), defensive_mass=d)
                             for q, d in zip(c["qdensities"], c["deltas"], strict=True)], 1) if risk else g[:, None])
    points = 128 if risk else 64
    cfg = WeightedSMCConfig(1024, tuple((k/(points-1))**2 for k in range(points)), 4, .35,
                            1, seed(j, "whole"), resample_every=4, resampling_scheme="stratified")
    r = estimate_gradient_umbrella_normalizers(components, dimension=c["dimension"], log_mixture_weights=weights,
                                               config=cfg, maximum_drift_norm=20.)
    correction = torch.tensor([math.log(d) for d in c["deltas"]] if risk else [0.], dtype=torch.float64)
    return {**j, "seed": seed(j, "whole"), "actual_work": r.potential_evaluations,
            "backward_evaluations": r.gradient_evaluations, "log_normalizer": r.log_normalizer,
            "log_scaled_component_normalizers": r.log_component_normalizers.tolist(),
            "log_estimand_values": (r.log_component_normalizers-correction).tolist(),
            "component_targets": TARGETS[1:] if risk else TARGETS[:1], "log_mixture_weights": weights.tolist(),
            "q_digests": c["q_digests"], "stages": list(r.stages), "wall": time.perf_counter()-begin,
            "guide_independent_kernel": True, "kernel": "capped-gradient-exact-mh"}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in ("guide-iid", "free-full", "envelope"):
        for target in TARGETS:
            group = [r for r in records if r["method"] == method and target in r["component_targets"]]
            expected = 256 if method == "guide-iid" else 154 if method == "envelope" else 71 if target == "mu" else 99
            if len(group) != expected or {r["rep"] for r in group} != set(range(expected)):
                raise ValueError("incomplete fixed method-target grid")
            k = group[0]["component_targets"].index(target)
            if method == "guide-iid":
                stats = merge_moments([r["moments"][k] for r in group])
                logs = [r["moments"][k]["log_mean"] for r in group]
            else:
                logs = [r["log_estimand_values"][k] for r in group]
                stats = log_moments(torch.tensor(logs, dtype=torch.float64))
            out.append({"method": method, "target": target, "summary": stats,
                        "whole_or_equal_size_block_log_means": logs,
                        "sensitivity": sensitivity(logs, bootstrap_seed=seed({"method": method, "family": target, "rep": 1000}, "bootstrap")),
                        "guide_independent_kernel": method == "free-full"})
    return out


def decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    d = high_decision(summaries)
    success = d.pop("high_eta_five_q_development_reference_pass")
    return {**d, "canonical_five_q_development_reference_pass": success,
            "scope": "canonical_original_five_q_finite_N32_development_only"}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchors = {k: json.loads((ROOT/p).read_text()) for k, p in ANCHORS.items()}
    if anchors["risk"]["selected_design"] != "more-temperatures":
        raise ValueError("unexpected selected risk design")
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    weights = {CELL: anchors["umbrella"]["config"]["log_mixture_weights"][CELL]}
    config = {"input": input_config, "anchors": {k: {"path": p, "sha256": hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for k, p in ANCHORS.items()},
              "log_mixture_weights": weights, "counts": {"iid_blocks": 256, "risk_gradient": 99, "mu_gradient": 71, "envelope": 154},
              "backward_cap": 80000000, "scope": "canonical_original_five_q_development_reference_validation"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, *ANCHORS.values()))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    streams = [seed(j, s) for j in grid for s in (("path", "label") if j["method"] == "guide-iid" else ("whole",))]
    if len(grid) != 580 or len(streams) != len(set(streams)):
        raise ValueError("unexpected grid/seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"C": {"work": 160000000, "wall": 3600.}}}
    journal = RecoveryJournal(directory, manifest)
    cells: dict[str, Any] = {}
    def preparation():
        cells.update(prepare(metadata, rows, weights))
        c = cells[CELL]
        c["guide"] = build_volterra_excursion_guide(c["evaluator"].problem, **input_config["guide"])
        c["gdensity"] = CachedShiftDensity.create(c["guide"])
        c["marginal"] = ShiftMixtureMarginal.from_full(c["guide"], tuple(range(62)))
        return {"actual_work": 128}
    journal.unit("preparation", phase="C", work=128, maximum_wall=30., operation=preparation)
    records, backward_used = [], 0
    for j in grid:
        forward, backward = expected_work(j)
        if backward_used+backward > config["backward_cap"]:
            raise RuntimeError("backward reservation cap")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            r = unit_values(job, cells)
            if (r["actual_work"], r["backward_evaluations"]) != expected_work(job):
                raise ValueError("unexpected forward/backward accounting")
            return r
        records.append(journal.unit(identity(j), phase="C", work=forward, maximum_wall=30., operation=operation))
        backward_used += backward
        if len(records) % 32 == 0:
            print(f"canonical five-q validation {len(records)}/580", flush=True)
    summaries = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "decision": decision(summaries), "units": len(grid), "work": journal.accounting("C")[0],
              "backward_evaluations": backward_used,
              "phase_wall": time.time()-journal._read(directory/"phase-C.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-C.clock.json")["started_unix"] > 3600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    for k, p in ANCHORS.items():
        if config["anchors"][k] != {"path": p, "sha256": hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}:
            raise ValueError("changed anchor")
    if (manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()
            or manifest["expected_units"] != [identity(j) for j in jobs()]):
        raise ValueError("changed source/input/grid")
    weights = {CELL: json.loads((ROOT/ANCHORS["umbrella"]).read_text())["config"]["log_mixture_weights"][CELL]}
    if config["log_mixture_weights"] != weights:
        raise ValueError("changed fixed weights")
    metadata, rows = inputs(config["input"])
    cells = prepare(metadata, rows, weights)
    c = cells[CELL]
    c["guide"] = build_volterra_excursion_guide(c["evaluator"].problem, **config["input"]["guide"])
    c["gdensity"] = CachedShiftDensity.create(c["guide"])
    c["marginal"] = ShiftMixtureMarginal.from_full(c["guide"], tuple(range(62)))
    journal = RecoveryJournal(path.parent, manifest)
    records, replay_draws, replay_wholes, forward_audit, backward_audit = [], 0, 0, 128, 0
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed unit")
        r = finish["result"]
        iid = j["method"] == "guide-iid"
        if (any(r[k] != v for k, v in j.items()) or r["q_digests"] != c["q_digests"]
                or (r["actual_work"], r["backward_evaluations"]) != expected_work(j)
                or r["guide_independent_kernel"] != (j["method"] == "free-full")):
            raise ValueError("changed unit binding")
        if iid:
            if r["seeds"] != {s: seed(j, s) for s in ("path", "label")}:
                raise ValueError("changed IID seeds")
        else:
            if r["seed"] != seed(j, "whole"):
                raise ValueError("changed SMC seed")
            targets = r["component_targets"]
            w = c["log_weights"] if j["method"] == "envelope" else family_weights("risks-pcnl" if j["family"] == "risk" else "probability-pcnl", weights[CELL])
            if r["log_mixture_weights"] != w.tolist():
                raise ValueError("changed family weights")
            vector = torch.tensor(r["log_scaled_component_normalizers"], dtype=torch.float64)
            if abs(float(torch.logsumexp(vector+w, 0))-r["log_normalizer"]) > 1e-10:
                raise ValueError("broken unnormalized identity")
            correction = torch.tensor(([0.] if "mu" in targets else [])+([math.log(d) for d in c["deltas"]] if len(targets) > 1 else []), dtype=torch.float64)
            if (vector-correction).tolist() != r["log_estimand_values"]:
                raise ValueError("risk correction mismatch")
        if (iid and j["rep"] in (0, 255)) or (not iid and j["rep"] == 0):
            replay = unit_values(j, cells)
            fields = ("moments",) if iid else ("log_normalizer", "log_estimand_values", "stages")
            if any(replay[k] != r[k] for k in fields):
                raise ValueError("numerical replay mismatch")
            replay_draws += 32768 if iid else 0
            replay_wholes += int(not iid)
            forward_audit += replay["actual_work"]
            backward_audit += replay["backward_evaluations"]
        records.append(r)
    z = c["guide"].sample(1024, path_seed=seed({"method": "audit", "family": "all", "rep": 300}, "path"),
                          label_seed=seed({"method": "audit", "family": "all", "rep": 300}, "label")).samples
    cache = c["evaluator"].last_pair(z[:, :-2], c["guide"])
    torch.testing.assert_close(torch.special.log_ndtr((cache.a-cache.b*z[:, -2])/torch.sqrt(cache.c_squared)),
                               c["evaluator"].log_probability(z), rtol=2e-11, atol=2e-10)
    forward_audit += 2048
    summaries = summarize(records)
    if (summaries != output["summaries"] or decision(summaries) != output["decision"]
            or journal.accounting("C")[0] != output["work"] or output["work"] != 116953216
            or output["units"] != 580 or output["backward_evaluations"] != 69298176
            or sum(r["backward_evaluations"] for r in records) != output["backward_evaluations"]):
        raise ValueError("changed summary/decision/cost")
    if (path.parent/"phase-C.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_anchors_q_seed_identity_moment_cost_replay_pass_not_continuum_certificate",
            "replayed_draws": replay_draws, "replayed_wholes": replay_wholes,
            "audit_forward_evaluations": forward_audit, "audit_backward_evaluations": backward_audit,
            "decision": output["decision"]["canonical_five_q_development_reference_pass"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "units", "phase_wall", "backward_evaluations", "decision")}))
