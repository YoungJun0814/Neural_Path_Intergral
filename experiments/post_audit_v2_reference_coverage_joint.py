"""One sealed guide-independent coverage diagnostic, with paired μ/risk targets."""

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

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from experiments.post_audit_v2_overnight_reference_recovery import TOYS
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.local_variance_reference import LocalVarianceReference
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import source_tree_digest
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/REFERENCE_COVERAGE_AND_JOINT_CONDITIONAL_PROTOCOL_2026-10-09_KO.md"
DEFAULT = "configs/post_audit/reference_coverage_joint_v1.yaml"


def seed(identity: str, stream: str) -> int:
    return int(hashlib.sha256(("reference-coverage-joint-v1/"+identity+"/"+stream).encode()).hexdigest()[:15], 16)


def validate_config(c: dict[str, Any]) -> None:
    expected = {"schema": "npi.reference-coverage-joint.v1", "replications": 8,
                "unit_count": 32768, "batch_size": 4096, "torch_threads": 1,
                "variance": 25., "defensive_mass": .1, "method": "parent-as-is"}
    if any(isinstance(c.get(k), bool) or c.get(k) != v for k, v in expected.items()):
        raise ValueError("sealed coverage design changed")
    if any(not isinstance(c[k], int) for k in ("replications", "unit_count", "batch_size", "torch_threads")):
        raise ValueError("integer counts required")
    if c["limits"] != {"D": {"work": 3000000, "wall": 900.}}:
        raise ValueError("sealed coverage budget changed")
    if c["guide"] != {"amplitudes": [2., 4., 6., 8.], "price_amplitudes": [0., 2., 4.], "defensive_mass": .1}:
        raise ValueError("baseline guide changed")


def identities(rows: list[dict[str, Any]]) -> list[tuple[str, str, str, int, int]]:
    cells = [("toy", name, method) for name in TOYS for method in ("natural", "local-scale")]
    cells += [("volterra", r["cell"]["id"], method) for r in rows
              if r["parent_training_rep"] == 0 for method in ("guide", "local-scale")]
    return [(family, cell, method, rep, batch) for family, cell, method in cells
            for rep in range(8) for batch in range(8)]


def identifier(job: tuple[str, str, str, int, int]) -> str:
    return "-".join(map(str, job))


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        for target in ("mu", "risk"):
            groups.setdefault((r["family"], r["cell"], r["method"], target), []).append(r)
    result = []
    for (family, cell, method, target), parts in sorted(groups.items()):
        logs = torch.tensor([x for r in parts for x in r["logs"][target]], dtype=torch.float64)
        stats = log_moments(logs)
        reps = []
        for rep in range(8):
            v = torch.tensor([x for r in parts if r["replication"] == rep for x in r["logs"][target]], dtype=torch.float64)
            reps.append(float(torch.logsumexp(v, 0))-math.log(len(v)))
        stability = sensitivity(reps, bootstrap_seed=seed(f"{family}-{cell}-{method}-{target}", "bootstrap"))
        total = torch.logsumexp(logs, 0)
        bins = torch.tensor([x for r in parts for x in r["geometry_bin"]])
        shares = [float(torch.exp(torch.logsumexp(logs[bins == b], 0)-total)) for b in range(4)]
        row: dict[str, Any] = {"family": family, "cell": cell, "method": method, "target": target,
            "summary": stats, "whole_replicate_log_means": reps, "stability": stability,
            "geometry_contribution_shares": shares,
            "evaluation_wall_seconds_paired_targets": sum(r["evaluation_wall"] for r in parts),
            "accuracy_development_pass": stats["relative_se"] <= .02 and stats["maximum_fraction"] <= .01
                and stability["maximum_leave_one_out_relative_shift"] <= .1}
        if family == "toy":
            row["oracle_log_mean"] = TOYS[cell].log_endpoint(1 if target == "mu" else 2)
            row["oracle_relative_error"] = math.expm1(stats["log_mean"]-row["oracle_log_mean"])
        result.append(row)
    return result


def comparisons(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cell in sorted({r["cell"] for r in summaries if r["family"] == "volterra"}):
        for target in ("mu", "risk"):
            a, b = [next(r for r in summaries if r["cell"] == cell and r["target"] == target
                         and r["method"] == m and r["family"] == "volterra") for m in ("guide", "local-scale")]
            equivalence = relative_equivalence(a["summary"], b["summary"], comparisons=4)
            eligible = a["accuracy_development_pass"] and b["accuracy_development_pass"] and equivalence["pass"]
            result.append({"cell": cell, "target": target, "mean_equivalence": equivalence,
                           "accuracy_eligible_for_development_cost_comparison": eligible,
                           "fixed_precision_winner": None, "production_reference_qualified": False})
    return result


def run(config: dict[str, Any]) -> dict[str, Any]:
    total_started = time.perf_counter()
    validate_config(config)
    torch.set_num_threads(1)
    directory = (ROOT/config["run_directory"]).resolve()
    if not directory.is_relative_to((ROOT/"results/post_audit").resolve()):
        raise ValueError("invalid research output directory")
    directory.mkdir(parents=True, exist_ok=True)
    metadata, rows = inputs(config)
    jobs = identities(rows)
    streams = [seed(identifier(j), s) for j in jobs for s in ("path", "label")]
    if len(set(streams)) != len(streams):
        raise ValueError("seed collision")
    source_path = directory/"source.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
        audit_snapshot(ROOT, source, config)
        if source["source_tree_digest"] != source_tree_digest(ROOT):
            raise ValueError("source changed; cannot resume")
    else:
        source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
        write_json_atomic_nonoverwriting(source_path, source)
    manifest = {"config": config, "source": source["snapshot_sha256"],
                "source_tree_digest": source_tree_digest(ROOT), "limits": config["limits"],
                "expected_units": [identifier(j) for j in jobs],
                "input_sha256": hashlib.sha256((ROOT/config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "q_bindings": [{"cell": r["cell"], "parent": 0, "q": r["q_digest"]}
                               for r in rows if r["parent_training_rep"] == 0],
                "target_pairing": "same_draw_mu_and_original_M2_not_independent_targets",
                "p2_authorized": False}
    journal = RecoveryJournal(directory, manifest)
    prepared: dict[str, Any] = {}
    # Preparation is inside the phase clock, not excluded from its wall budget.
    def prepare() -> dict[str, Any]:
        work = 0
        for r in rows:
            if r["parent_training_rep"] != 0:
                continue
            p = _problem(metadata["config"]["model"], r["cell"], metadata["config"]["smc"]["steps"])
            prepared[r["cell"]["id"]] = (p, CachedVolterraTerminal.create(p),
                CachedShiftDensity.create(proposal_from_parameters(r["q_parameters"])),
                build_volterra_excursion_guide(p, **config["guide"]))
            work += p.local_dimension
        return {"actual_work": work}
    preparation = journal.unit("preparation", phase="D", work=1024, maximum_wall=30., operation=prepare)
    if not prepared:
        # Cached preparation is process-local and must be rebuilt for a resumed run.
        raise RuntimeError("resume requires explicit prepared-cache protocol; no silent unaccounted rebuild")
    records = []
    for job in jobs:
        family, cell, method, rep, batch = job
        identity = identifier(job)
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("coverage RSS cap")
        def operation(f=family, c=cell, m=method, replication=rep, b=batch, ident=identity):
            started = time.perf_counter()
            seeds = {s: seed(ident, s) for s in ("path", "label")}
            count = config["batch_size"]
            if f == "toy":
                oracle = TOYS[c]
                if m == "natural":
                    z = torch.randn((count, oracle.dimension), dtype=torch.float64,
                                    generator=torch.Generator().manual_seed(seeds["path"]))
                    density = torch.zeros(count, dtype=torch.float64)
                else:
                    r = LocalVarianceReference(oracle.dimension, 1)
                    z = r.sample(count, path_seed=seeds["path"], label_seed=seeds["label"])
                    density = r.log_q_over_p(z)
                g = oracle.log_value(z)
                mu, risk = g-density, 2*g-density
                bins = ((z[:, 0] >= 2.5) & (z[:, 0] <= 7.5) & (z[:, 1].abs() <= 2)).long()
                binding = None
                oracle_difference = None
            else:
                p, evaluator, q, guide = prepared[c]
                if m == "guide":
                    z = guide.sample(count, path_seed=seeds["path"], label_seed=seeds["label"]).samples
                    density = CachedShiftDensity.create(guide).log_q_over_p(z)
                else:
                    r = LocalVarianceReference(p.local_dimension)
                    z = r.sample(count, path_seed=seeds["path"], label_seed=seeds["label"])
                    density = r.log_q_over_p(z)
                terminal, integrated, variance = evaluator.quantities(z)
                if not isinstance(p.task, TerminalThresholdTask):
                    raise ValueError("terminal threshold task required")
                g = torch.special.log_ndtr((math.log(p.task.level)-terminal)/torch.sqrt((1-p.rho**2)*integrated))
                mu, risk = g-density, 2*g-q.log_q_over_p(z)-density
                indices = variance[:, 1:-1].argmax(1)+1
                bins = torch.clamp(4*indices//p.steps, max=3)
                binding = next(r["q_digest"] for r in rows if r["cell"]["id"] == c and r["parent_training_rep"] == 0)
                oracle_difference = None
                if replication == 0 and b == 0:
                    expected = evaluate_rbergomi_conditional_terminal(p, z[:16], epsilon=1.).payoffs.log_left_probability
                    oracle_difference = float(torch.max(torch.abs(expected-g[:16])))
                    if oracle_difference > 2e-10:
                        raise ValueError("original simulator mismatch")
            wall = time.perf_counter()-started
            if not torch.isfinite(mu).all() or not torch.isfinite(risk).all():
                raise FloatingPointError("nonfinite contributions; no clipping/replacement")
            return {"family": f, "cell": c, "method": m, "replication": replication,
                "batch": b, "seeds": seeds, "q_digest": binding, "evaluation_wall": wall,
                "logs": {"mu": mu.tolist(), "risk": risk.tolist()}, "geometry_bin": bins.tolist(),
                "original_simulator_maximum_log_difference": oracle_difference,
                "actual_work": count+(16 if oracle_difference is not None else 0)}
        records.append(journal.unit(identity, phase="D", work=4112, maximum_wall=10., operation=operation))
        if len(records) % 64 == 0:
            print(f"completed {len(records)}/{len(jobs)}", flush=True)
    summaries = summarize(records)
    clocks = journal._read(directory/"phase-D.clock.json")
    result = {"status": "completed_development_not_reference_qualification", "config": config,
        "source": source, "manifest": manifest, "summaries": summaries,
        "comparisons": comparisons(summaries), "completed_units": len(records),
        "wall_including_preparation_io_and_summary": time.time()-clocks["started_unix"],
        "actual_toy_calls": sum(r["actual_work"] for r in records if r["family"] == "toy"),
        "actual_volterra_calls_including_preparation_oracle": sum(r["actual_work"] for r in records if r["family"] == "volterra")+preparation["actual_work"],
        "total_wall_including_input_archive_summary_excluding_final_result_write": time.perf_counter()-total_started,
        "p2_authorized": False, "production_reference_qualified": False}
    if result["wall_including_preparation_io_and_summary"] > 900:
        raise TimeoutError("summary-inclusive phase wall exceeded; do not qualify result")
    write_json_atomic_nonoverwriting(directory/"result.json", result)
    if time.time()-clocks["started_unix"] > 900:
        write_json_atomic_nonoverwriting(directory/"final_io_violation.json", {"status": "phase_wall_exceeded_in_final_io"})
        raise TimeoutError("final result I/O exceeded phase wall; preserve failed artifacts")
    return result


def audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    c = payload["config"]
    validate_config(c)
    torch.set_num_threads(c["torch_threads"])
    audit_snapshot(ROOT, payload["source"], c)
    metadata, rows = inputs(c)
    expected = identities(rows)
    m = payload["manifest"]
    if m["expected_units"] != [identifier(j) for j in expected] or payload["completed_units"] != len(expected):
        raise ValueError("incomplete or changed grid")
    if m["input_sha256"] != hashlib.sha256((ROOT/c["proposal_artifact_path"]).read_bytes()).hexdigest():
        raise ValueError("input changed")
    journal = RecoveryJournal(path.parent, m)
    if (path.parent/"final_io_violation.json").exists() or (path.parent/"phase-D.violation.json").exists():
        raise ValueError("operational failure; result not qualified")
    if payload["wall_including_preparation_io_and_summary"] > 900:
        raise ValueError("phase budget exceeded")
    fixed_bindings = [{"cell": r["cell"], "parent": 0, "q": r["q_digest"]}
                      for r in rows if r["parent_training_rep"] == 0]
    if m["q_bindings"] != fixed_bindings:
        raise ValueError("q binding list changed")
    records = [journal._read(path.parent/(identifier(j)+".finish.json"))["result"] for j in expected]
    for j, r in zip(expected, records, strict=True):
        f, cell, method, rep, batch = j
        if (r["family"], r["cell"], r["method"], r["replication"], r["batch"]) != j:
            raise ValueError("unit identity changed")
        if r["seeds"] != {s: seed(identifier(j), s) for s in ("path", "label")}:
            raise ValueError("seed binding changed")
        if any(len(r["logs"][t]) != c["batch_size"] for t in ("mu", "risk")):
            raise ValueError("sample count changed")
        q = None if f == "toy" else next(x["q_digest"] for x in rows if x["cell"]["id"] == cell and x["parent_training_rep"] == 0)
        if r["q_digest"] != q:
            raise ValueError("original q changed")
    summaries = summarize(records)
    if payload["summaries"] != summaries or payload["comparisons"] != comparisons(summaries):
        raise ValueError("saved arithmetic changed")
    if payload["p2_authorized"] or payload["production_reference_qualified"]:
        raise ValueError("unauthorized qualification")
    if len(list(path.parent.glob("*.finish.json"))) != len(expected)+1:
        raise ValueError("unexpected units")
    preparation = journal._read(path.parent/"preparation.finish.json")
    expected_volterra = sum(r["actual_work"] for r in records if r["family"] == "volterra")+preparation["result"]["actual_work"]
    expected_toy = sum(r["actual_work"] for r in records if r["family"] == "toy")
    if (expected_volterra != payload["actual_volterra_calls_including_preparation_oracle"]
            or expected_toy != payload["actual_toy_calls"] or expected_volterra > 1050000
            or journal.accounting("D")[0] != expected_volterra+expected_toy):
        raise ValueError("saved work mismatch")
    # A bounded path replay uses the ORIGINAL simulator and original q density,
    # not the cached evaluator that generated the saved contributions.
    error = 0.
    replay_work = 0
    guide_preparation_work = 0
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        p = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
        q = proposal_from_parameters(row["q_parameters"])
        guide = build_volterra_excursion_guide(p, **c["guide"])
        guide_preparation_work += p.local_dimension
        for method in ("guide", "local-scale"):
            j = ("volterra", row["cell"]["id"], method, 0, 0)
            path_seed, label_seed = seed(identifier(j), "path"), seed(identifier(j), "label")
            if method == "guide":
                z = guide.sample(c["batch_size"], path_seed=path_seed, label_seed=label_seed).samples[:16]
                density = guide.log_q_over_p(z)
            else:
                r = LocalVarianceReference(p.local_dimension)
                z = r.sample(c["batch_size"], path_seed=path_seed, label_seed=label_seed)[:16]
                density = r.log_q_over_p(z)
            log_g = evaluate_rbergomi_conditional_terminal(p, z).payoffs.log_left_probability
            expected_logs = {"mu": log_g-density, "risk": 2*log_g-q.log_q_over_p(z)-density}
            saved = records[expected.index(j)]
            for target, values in expected_logs.items():
                error = max(error, float(torch.max(torch.abs(values-torch.tensor(saved["logs"][target][:16], dtype=torch.float64)))))
            replay_work += len(z)
    if error > 2e-10:
        raise ValueError("original-simulator contribution replay failed")
    return {"status": "source_seed_q_arithmetic_path_audit_pass_not_tail_certificate", "units": len(records),
            "original_simulator_replay_calls": replay_work, "guide_preparation_calls": guide_preparation_work,
            "maximum_log_contribution_difference": error,
            "run_source_sha256": payload["source"]["snapshot_sha256"],
            "audit_runtime_source_tree_digest": source_tree_digest(ROOT)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(ROOT/args.audit)))
    else:
        output = run(yaml.safe_load((ROOT/args.config).read_text()))
        print(json.dumps({k: output[k] for k in ("status", "completed_units", "p2_authorized")}))


if __name__ == "__main__":
    main()
