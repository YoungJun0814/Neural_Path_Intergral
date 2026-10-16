"""Fresh canonical probability revalidation; immutable old failure remains."""

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

from experiments.post_audit_canonical_family_validation import decision as family_decision
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "results/post_audit/canonical_family_validation_v1/session_20261010/result.json"
PLAN = "docs/plans/POST_RECOVERY_NEXT_PRIORITIES_2026-10-16_KO.md"
DIRECTORY = "results/post_audit/canonical_mu_revalidation_v1/session_20261016"
COUNT, WORK = 215, 254976


def seed(rep: int) -> int:
    return int(hashlib.sha256(f"canonical-mu-revalidation-v1/{rep}".encode()).hexdigest()[:15], 16)


def unit_values(rep: int, evaluator: CachedVolterraTerminal) -> dict[str, Any]:
    begin = time.perf_counter()
    cfg = WeightedSMCConfig(1024, tuple((k/63)**2 for k in range(64)), 4, .35, 1, seed(rep),
                            resample_every=4, resampling_scheme="stratified")
    r = estimate_gradient_umbrella_normalizers(lambda z: evaluator.log_probability(z)[:, None], dimension=64,
                                               log_mixture_weights=torch.zeros(1, dtype=torch.float64),
                                               config=cfg, maximum_drift_norm=20.)
    if (r.potential_evaluations != WORK or r.gradient_evaluations != WORK
            or abs(float(r.log_component_normalizers[0])-r.log_normalizer) > 1e-10):
        raise ValueError("probability endpoint or work mismatch")
    return {"rep": rep, "seed": seed(rep), "actual_work": r.potential_evaluations,
            "backward_evaluations": r.gradient_evaluations, "log_mu": r.log_normalizer,
            "stages": list(r.stages), "wall": time.perf_counter()-begin}


def summarize(records: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any]:
    if len(records) != COUNT or {r["rep"] for r in records} != set(range(COUNT)):
        raise ValueError("complete fresh fixed count required")
    records = sorted(records, key=lambda r: r["rep"])
    logs = [r["log_mu"] for r in records]
    stats = log_moments(torch.tensor(logs, dtype=torch.float64))
    sens = sensitivity(logs, bootstrap_seed=seed(1000))
    fresh = {"method": "free-full", "target": "mu", "summary": stats, "sensitivity": sens,
             "whole_or_equal_size_block_log_means": logs, "guide_independent_kernel": True}
    anchor_mu = next(r for r in anchor["summaries"] if r["method"] == "guide-iid" and r["target"] == "mu")
    eq = relative_equivalence(stats, anchor_mu["summary"], comparisons=2)
    precision = (stats["relative_se"] <= .015 and stats["maximum_fraction"] <= .15
                 and sens["maximum_leave_one_out_relative_shift"] <= .05)
    rows = [fresh if r["method"] == "free-full" and r["target"] == "mu" else r for r in anchor["summaries"]]
    inventory = family_decision(rows)
    component_pass = inventory.pop("canonical_five_q_development_reference_pass")
    return {"fresh_mu": fresh, "static_mu_equivalence": eq, "mu_precision_pass": precision,
            "fresh_mu_development_reference_pass": bool(precision and eq["pass"]),
            "canonical_componentwise_development_recovery_pass": bool(component_pass),
            "componentwise_inventory": inventory,
            "old_validation_pass": anchor["decision"]["canonical_five_q_development_reference_pass"],
            "scope": "new_mu_with_immutable_old_risk_components_not_one_new_full_confirmation",
            "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def prepare(config: dict[str, Any]) -> CachedVolterraTerminal:
    return CachedVolterraTerminal.create(_problem(config["model"], config["cell"], 32))


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    anchor = json.loads((ROOT/ANCHOR).read_text())
    old_mu = next(r for r in anchor["summaries"] if r["method"] == "free-full" and r["target"] == "mu")
    if (anchor["decision"]["canonical_five_q_development_reference_pass"]
            or math.ceil(3*71*(old_mu["summary"]["relative_se"]/.015)**2) != COUNT):
        raise ValueError("unexpected frozen allocation anchor")
    metadata, rows = inputs(anchor["config"]["input"])
    config = {"model": metadata["config"]["model"], "cell": next(r["cell"] for r in rows if r["cell"]["id"] == "canonical-h005-k1"),
              "anchor": {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()},
              "count": COUNT, "forward_cap": 60000000, "backward_cap": 60000000}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    if len({seed(r) for r in range(COUNT)}) != COUNT:
        raise ValueError("seed collision")
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [f"mu-{r}" for r in range(COUNT)],
                "limits": {"P": {"work": config["forward_cap"], "wall": 1800.}}}
    journal = RecoveryJournal(directory, manifest)
    prepared: dict[str, CachedVolterraTerminal] = {}
    def preparation():
        prepared["evaluator"] = prepare(config)
        return {"actual_work": 0}
    journal.unit("preparation", phase="P", work=0, maximum_wall=30., operation=preparation)
    records, backward = [], 0
    for rep in range(COUNT):
        if backward+WORK > config["backward_cap"]:
            raise RuntimeError("backward reservation cap")
        if psutil.Process().memory_info().rss > 4*1024**3:
            raise MemoryError("RSS guard")
        def operation(replica: int = rep) -> dict[str, Any]:
            return unit_values(replica, prepared["evaluator"])
        records.append(journal.unit(f"mu-{rep}", phase="P", work=WORK, maximum_wall=30.,
                                    operation=operation))
        backward += WORK
        if len(records) % 32 == 0:
            print(f"canonical fresh mu {len(records)}/{COUNT}", flush=True)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summarize(records, anchor),
              "units": len(records), "work": journal.accounting("P")[0], "backward_evaluations": backward,
              "phase_wall": time.time()-journal._read(directory/"phase-P.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-P.clock.json")["started_unix"] > 1800:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (config["anchor"] != {"path": ANCHOR, "sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest()}
            or manifest["config"] != config or manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["expected_units"] != [f"mu-{r}" for r in range(COUNT)]):
        raise ValueError("changed source/anchor/grid")
    anchor, evaluator, journal = json.loads((ROOT/ANCHOR).read_text()), prepare(config), RecoveryJournal(path.parent, manifest)
    records, replayed = [], 0
    for rep in range(COUNT):
        finish = journal._read(path.parent/f"mu-{rep}.finish.json")
        if finish["status"] != "complete":
            raise ValueError("failed whole")
        r = finish["result"]
        if r["rep"] != rep or r["seed"] != seed(rep) or r["actual_work"] != WORK or r["backward_evaluations"] != WORK:
            raise ValueError("changed whole binding/accounting")
        if rep in (0, COUNT-1):
            replay = unit_values(rep, evaluator)
            if any(replay[k] != r[k] for k in ("log_mu", "stages")):
                raise ValueError("whole replay mismatch")
            replayed += 1
        records.append(r)
    if (summarize(records, anchor) != output["summaries"] or output["units"] != COUNT
            or journal.accounting("P")[0] != output["work"] or output["work"] != COUNT*WORK
            or output["backward_evaluations"] != COUNT*WORK or (path.parent/"phase-P.violation.json").exists()
            or (path.parent/"final_io_failure.json").exists()):
        raise ValueError("changed summary/cost or operational failure")
    return {"status": "fresh_canonical_mu_source_anchor_seed_cost_replay_pass", "replayed_wholes": replayed,
            "audit_forward_evaluations": replayed*WORK, "audit_backward_evaluations": replayed*WORK,
            "mu_development_pass": output["summaries"]["fresh_mu_development_reference_pass"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("units", "work", "phase_wall")}))
