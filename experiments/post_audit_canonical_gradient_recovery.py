"""Fresh canonical target separation pilot; no relaxation of P1/P2 gates."""

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

from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_umbrella_reference_pilot import prepare
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, precision_count, sensitivity
from src.path_integral.umbrella_normalizer_smc import estimate_umbrella_normalizers
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/CANONICAL_TARGET_SEPARATION_AND_GRADIENT_RECOVERY_2026-10-10_KO.md"
ANCHOR = "results/post_audit/umbrella_reference_pilot_v1/session_20261010/result.json"
DIRECTORY = "results/post_audit/canonical_gradient_recovery_v1/session_20261010"
CELL = "canonical-h005-k1"
ARMS = ("family-pcn", "risks-pcn", "risks-pcnl", "probability-pcnl")


def jobs() -> list[dict[str, Any]]:
    return [{"arm": arm, "rep": rep} for arm in ARMS for rep in range(8)]


def identity(j: dict[str, Any]) -> str:
    return f"{CELL}-{j['arm']}-{j['rep']}"


def seed(j: dict[str, Any]) -> int:
    return int(hashlib.sha256(("canonical-separation-gradient-v1/"+identity(j)).encode()).hexdigest()[:15], 16)


def expected_work(j: dict[str, Any]) -> tuple[int, int]:
    return (254976, 254976) if j["arm"].endswith("pcnl") else (508928, 0)


def family_weights(arm: str, frozen: list[float]) -> torch.Tensor:
    if arm not in ARMS or len(frozen) != 6:
        raise ValueError("unknown arm or incomplete family")
    w = torch.tensor(frozen, dtype=torch.float64)
    if not torch.isfinite(w).all() or abs(float(torch.logsumexp(w, 0))) > 1e-12:
        raise ValueError("invalid frozen family weights")
    if arm == "probability-pcnl":
        return torch.zeros(1, dtype=torch.float64)
    if arm.startswith("risks"):
        w = w[1:]
        w = w-torch.logsumexp(w, 0)
    return w


def component_targets(arm: str) -> list[str]:
    return ((["mu"] if arm in ("family-pcn", "probability-pcnl") else [])
            +([f"risk-q{j}" for j in range(5)] if arm != "probability-pcnl" else []))


def unit_values(j: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    arm = j["arm"]
    weights = family_weights(arm, c["log_weights"].tolist())
    targets = component_targets(arm)
    def components(z):
        g = c["evaluator"].log_probability(z)
        values = [g] if "mu" in targets else []
        if arm != "probability-pcnl":
            values += [log_risk_potential(g, q.log_q_over_p(z), defensive_mass=d)
                       for q, d in zip(c["qdensities"], c["deltas"], strict=True)]
        return torch.stack(values, 1)
    gradient = arm.endswith("pcnl")
    cfg = WeightedSMCConfig(1024, tuple((k/63)**2 for k in range(64)), 4 if gradient else 8,
                            .35, 1, seed(j), resample_every=4, resampling_scheme="stratified")
    if gradient:
        r = estimate_gradient_umbrella_normalizers(components, dimension=c["dimension"],
            log_mixture_weights=weights, config=cfg, maximum_drift_norm=20.)
        backward = r.gradient_evaluations
    else:
        p = estimate_umbrella_normalizers(components, dimension=c["dimension"],
            log_mixture_weights=weights, config=cfg, use_blocks=False)
        # Common record contract; the core result types remain distinct.
        r = None
        backward = 0
    result = r if gradient else p
    assert result is not None
    correction = torch.tensor(([0.] if "mu" in targets else [])
                              +([math.log(d) for d in c["deltas"]] if arm != "probability-pcnl" else []), dtype=torch.float64)
    return {**j, "seed": seed(j), "actual_work": result.potential_evaluations,
            "backward_evaluations": backward, "q_digests": c["q_digests"],
            "component_targets": targets, "log_mixture_weights": weights.tolist(),
            "log_normalizer": result.log_normalizer,
            "log_scaled_component_normalizers": result.log_component_normalizers.tolist(),
            "log_estimand_values": (result.log_component_normalizers-correction).tolist(),
            "stages": list(result.stages), "wall": time.perf_counter()-begin,
            "guide_independent_kernel": True, "kernel": "capped-gradient-exact-mh" if gradient else "full-pcn"}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for arm in ARMS:
        group = [r for r in records if r["arm"] == arm]
        if len(group) != 8 or {r["rep"] for r in group} != set(range(8)):
            raise ValueError("incomplete whole-run arm")
        normalizers = torch.tensor([r["log_normalizer"] for r in group], dtype=torch.float64)
        components: list[dict[str, Any]] = []
        for j, target in enumerate(component_targets(arm)):
            logs = [r["log_estimand_values"][j] for r in group]
            stats = log_moments(torch.tensor(logs, dtype=torch.float64))
            ratios = torch.tensor([r["log_scaled_component_normalizers"][j]-r["log_normalizer"] for r in group], dtype=torch.float64)
            correlation = (float(torch.corrcoef(torch.stack((normalizers, ratios)))[0, 1])
                           if float(normalizers.std()) > 1e-12 and float(ratios.std()) > 1e-12 else None)
            components.append({"target": target, "summary": stats,
                "whole_log_means": logs, "endpoint_component_log_ratios": ratios.tolist(),
                "log_normalizer_ratio_correlation": correlation,
                "sensitivity": sensitivity(logs, bootstrap_seed=seed({"arm": arm+target, "rep": 200})),
                "allocation": precision_count(stats, target_rse=.015, safety_factor=2,
                                               minimum=32, maximum=256, multiple=1)})
        count = max(c["allocation"]["required_count"] for c in components)
        forward, backward = expected_work(group[0])
        wall = sum(r["wall"] for r in group)
        out.append({"arm": arm, "components": components,
                    "normalizer_summary": log_moments(normalizers), "shared_whole_required": count,
                    "allocation_feasible": count <= 256, "forecast_forward_evaluations": count*forward,
                    "forecast_backward_evaluations": count*backward, "forecast_wall": count*wall/8,
                    "whole_wall": wall, "scope": "development_pilot_only_not_precision_or_speed_certificate"})
    return out


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    weights = {CELL: anchor["config"]["log_mixture_weights"][CELL]}
    config = {"input": input_config, "anchor": ANCHOR,
              "anchor_sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest(),
              "log_mixture_weights": weights, "arms": list(ARMS), "wholes_per_arm": 8,
              "backward_cap": 5000000, "maximum_drift_norm": 20.}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    grid = jobs()
    if len(grid) != 32 or len({seed(j) for j in grid}) != 32:
        raise ValueError("unexpected grid or seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [identity(j) for j in grid],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"G": {"work": 16000000, "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    cells: dict[str, Any] = {}
    def preparation():
        cells.update(prepare(metadata, rows, weights))
        return {"actual_work": 128}
    journal.unit("preparation", phase="G", work=1024, maximum_wall=30., operation=preparation)
    records: list[dict[str, Any]] = []
    backward_used = 0
    for j in grid:
        forward, backward = expected_work(j)
        if backward_used+backward > config["backward_cap"]:
            raise RuntimeError("backward reservation cap")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard exceeded")
        def operation(job=j):
            r = unit_values(job, cells[CELL])
            if (r["actual_work"], r["backward_evaluations"]) != expected_work(job):
                raise ValueError("unexpected forward/backward accounting")
            return r
        records.append(journal.unit(identity(j), phase="G", work=forward, maximum_wall=60., operation=operation))
        backward_used += backward
        if len(records) % 8 == 0:
            print(f"canonical separation/gradient pilot {len(records)}/32", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records),
              "units": len(records), "work": journal.accounting("G")[0], "backward_evaluations": backward_used,
              "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-G.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-G.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["anchor"] != ANCHOR
            or hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest() != config["anchor_sha256"]
            or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()
            or manifest["expected_units"] != [identity(j) for j in jobs()]):
        raise ValueError("changed anchor/source/input/grid")
    anchor = json.loads((ROOT/ANCHOR).read_text())
    weights = {CELL: anchor["config"]["log_mixture_weights"][CELL]}
    if weights != config["log_mixture_weights"]:
        raise ValueError("changed frozen weights")
    metadata, rows = inputs(config["input"])
    c = prepare(metadata, rows, weights)[CELL]
    journal = RecoveryJournal(path.parent, manifest)
    records, audit_forward, audit_backward, replayed = [], 128, 0, 0
    for j in jobs():
        finish = journal._read(path.parent/(identity(j)+".finish.json"))
        if finish["status"] != "complete":
            raise ValueError("failed whole")
        r = finish["result"]
        w = family_weights(j["arm"], weights[CELL])
        targets = component_targets(j["arm"])
        if (any(r[k] != v for k, v in j.items()) or r["seed"] != seed(j)
                or (r["actual_work"], r["backward_evaluations"]) != expected_work(j)
                or r["q_digests"] != c["q_digests"] or not r["guide_independent_kernel"]
                or r["log_mixture_weights"] != w.tolist() or r["component_targets"] != targets):
            raise ValueError("changed unit binding")
        vector = torch.tensor(r["log_scaled_component_normalizers"], dtype=torch.float64)
        if abs(float(torch.logsumexp(vector+w, 0))-r["log_normalizer"]) > 1e-10:
            raise ValueError("broken unnormalized identity")
        correction = torch.tensor(([0.] if "mu" in targets else [])
                                  +([math.log(d) for d in c["deltas"]] if j["arm"] != "probability-pcnl" else []), dtype=torch.float64)
        if (vector-correction).tolist() != r["log_estimand_values"]:
            raise ValueError("risk correction mismatch")
        if j["rep"] == 0:
            replay = unit_values(j, c)
            if any(replay[k] != r[k] for k in ("log_normalizer", "log_scaled_component_normalizers", "stages")):
                raise ValueError("whole replay mismatch")
            audit_forward += replay["actual_work"]
            audit_backward += replay["backward_evaluations"]
            replayed += 1
        records.append(r)
    if (summarize(records) != output["summaries"] or output["units"] != 32
            or journal.accounting("G")[0] != output["work"] or output["work"] != 12222592
            or sum(r["backward_evaluations"] for r in records) != output["backward_evaluations"]
            or output["backward_evaluations"] != 4079616
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("changed summary/accounting/gates")
    if (path.parent/"phase-G.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_anchor_q_seed_identity_accounting_whole_replay_pass_not_precision_certificate",
            "replayed_wholes": replayed, "audit_forward_evaluations": audit_forward,
            "audit_backward_evaluations": audit_backward}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "backward_evaluations", "units", "phase_wall")}))
