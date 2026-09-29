"""Frozen-workflow, fixed-precision wall-time development comparison.

All stage time is measured sequentially in one process after a small warmup.
The policy and counts are fixed before this runner reads any new final draws.
Hyperparameter discovery/pilot search is disclosed separately, not included
as a zero-cost claim for a newly invented workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from experiments.post_audit_r15_reference_design import temperatures
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions
from src.path_integral.research_result_contract import canonical_digest, source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_bank_mixture import fit_weighted_bank_mixture
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed(ledger: SeedLedger, task: str, role: str, rep: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r15", role, "fixed-precision-total-work", task, 0, rep, stream,
    ))


def _qualified(mean: float, rse: float, anchor_mean: float, anchor_rse: float,
               policy: dict[str, Any]) -> dict[str, float | bool]:
    upper = abs(mean - anchor_mean) + policy["confidence_z"] * math.hypot(
        mean * rse, anchor_mean * anchor_rse,
    )
    margin = policy["relative_equivalence_margin"] * anchor_mean
    return {
        "upper_difference": upper,
        "margin": margin,
        "pass": bool(rse <= policy["maximum_relative_se"]
                     and anchor_rse <= policy["maximum_anchor_relative_se"]
                     and upper <= margin),
    }


def _smc_config(design: dict[str, Any], candidate: dict[str, Any],
                replicates: int, seed: int, retain: bool = False) -> WeightedSMCConfig:
    return WeightedSMCConfig(
        particles=int(design["config"]["design"]["particles"]),
        temperatures=temperatures(int(design["config"]["design"]["levels"]),
                                  int(candidate["bridge_power"])),
        mutation_steps=int(candidate["mutation_steps"]),
        pcn_scale=float(candidate["pcn_scale"]),
        replicates=replicates, seed=seed,
        resample_every=int(candidate["resample_every"]),
        resampling_scheme=str(candidate["resampling_scheme"]),
        retain_final_particles=retain,
    )


def _is_final(problem: Any, proposal: Any, spec: dict[str, Any],
              ledger: SeedLedger) -> dict[str, Any]:
    clusters = int(spec["clusters"])
    count = int(spec["units_per_cluster"])
    log_chunks = []
    cluster_logmeans = []
    started = time.perf_counter()
    for cluster in range(clusters):
        draw = proposal.sample(
            count,
            path_seed=_seed(ledger, problem.task_id, "total-work-is-final", cluster, "path"),
            label_seed=_seed(ledger, problem.task_id, "total-work-is-final", cluster, "label"),
        )
        values = _log_potential(problem, draw.samples) + draw.log_p_over_q
        log_chunks.append(values)
        cluster_logmeans.append(summarize_log_contributions(values).log_mean)
    wall = time.perf_counter() - started
    summary = summarize_log_contributions(torch.cat(log_chunks))
    if summary.log_mean is None or summary.relative_se is None:
        raise FloatingPointError("total-work IS estimator had no contribution")
    scale = max(x for x in cluster_logmeans if x is not None)
    scaled = [math.exp(x - scale) if x is not None else 0.0 for x in cluster_logmeans]
    between_rse = statistics.stdev(scaled) / math.sqrt(clusters) / statistics.mean(scaled)
    return {
        "mean": math.exp(summary.log_mean),
        "log_mean": summary.log_mean,
        "iid_relative_se": summary.relative_se,
        "between_cluster_relative_se": between_rse,
        "relative_se": max(summary.relative_se, between_rse),
        "samples": clusters * count,
        "cluster_logmeans": cluster_logmeans,
        "inference_wall_seconds": wall,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    inputs = {}
    hashes = {}
    for name in ("smc_design", "mixture_design", "refined_reference", "is_precision"):
        raw = (ROOT / config[f"{name}_path"]).read_bytes()
        inputs[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    design = inputs["smc_design"]
    mixture_design = inputs["mixture_design"]
    refined = inputs["refined_reference"]
    is_precision = inputs["is_precision"]
    selected = next(x for x in design["config"]["design"]["candidates"]
                    if x["id"] == design["selected_candidate_id"])
    mixture_selected = next(x for x in mixture_design["config"]["training"]["candidates"]
                            if x["id"] == mixture_design["selected_candidate_id"])
    common = dict(mixture_design["config"]["training"])
    common.pop("independent_smc_banks")
    common.pop("candidates")
    source = source_manifest(ROOT, config=config)
    ledger = SeedLedger()
    cells = []
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        # Warmup cost is reported but excluded from both frozen workflows.
        warm_started = time.perf_counter()
        _log_potential(problem, torch.zeros((256, problem.local_dimension), dtype=torch.float64))
        warm_wall = time.perf_counter() - warm_started
        anchor_smc = next(x for x in refined["cells"]
                          if x["cell"]["id"] == cell["id"])["new_reference"]
        anchor_is = next(x for x in is_precision["cells"]
                         if x["cell"]["id"] == cell["id"])
        if not anchor_is["qualified"] or anchor_smc["relative_se"] > 0.05:
            raise ValueError("no independently qualified development anchors")
        smc_started = time.perf_counter()
        smc = estimate_weighted_tempered_normalizer(
            partial(_log_potential, problem), dimension=problem.local_dimension,
            config=_smc_config(
                design, selected, int(config["smc_replicates"][cell["id"]]),
                _seed(ledger, problem.task_id, "total-work-smc-final", 0, "replicates"),
            ),
        )
        smc_wall = time.perf_counter() - smc_started
        smc_rse = smc.standard_error / smc.mean
        smc_accuracy = _qualified(
            smc.mean, smc_rse, anchor_is["mean"], anchor_is["relative_se"],
            config["qualification"],
        )
        bank_started = time.perf_counter()
        bank_samples = []
        bank_weights = []
        bank_ancestry = []
        for rep in range(int(config["is_training_banks"])):
            bank = estimate_weighted_tempered_normalizer(
                partial(_log_potential, problem), dimension=problem.local_dimension,
                config=_smc_config(
                    design, selected, 1,
                    _seed(ledger, problem.task_id, "total-work-is-bank", rep, "particles"),
                    retain=True,
                ),
            )
            if bank.final_particles is None or bank.final_weights is None:
                raise RuntimeError("SMC training particles missing")
            bank_samples.append(bank.final_particles)
            bank_weights.append(bank.final_weights / config["is_training_banks"])
            bank_ancestry.append(bank.replicate_diagnostics[0][
                "final_unique_initial_ancestors"
            ])
        bank_wall = time.perf_counter() - bank_started
        fit_started = time.perf_counter()
        fit = fit_weighted_bank_mixture(
            torch.cat(bank_samples), torch.cat(bank_weights),
            clusters=int(mixture_selected["clusters"]),
            covariance_rank=int(mixture_selected["covariance_rank"]), **common,
        )
        fit_wall = time.perf_counter() - fit_started
        independent_is = _is_final(
            problem, fit.proposal, config["is_final"], ledger,
        )
        is_accuracy = _qualified(
            independent_is["mean"], independent_is["relative_se"],
            anchor_smc["mean"], anchor_smc["relative_se"], config["qualification"],
        )
        pair_accuracy = _qualified(
            independent_is["mean"], independent_is["relative_se"],
            smc.mean, smc_rse, config["qualification"],
        )
        both_qualified = bool(smc_accuracy["pass"] and is_accuracy["pass"]
                              and pair_accuracy["pass"])
        smc_stages = {
            "offline_wall_seconds": 0.0,
            "fit_wall_seconds": 0.0,
            "selection_wall_seconds": 0.0,
            "inference_wall_seconds": smc_wall,
        }
        is_stages = {
            "offline_wall_seconds": bank_wall,
            "fit_wall_seconds": fit_wall,
            "selection_wall_seconds": 0.0,
            "inference_wall_seconds": independent_is["inference_wall_seconds"],
        }
        smc_total = sum(smc_stages.values())
        is_total = sum(is_stages.values())
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "warmup_excluded_seconds": warm_wall,
            "selected_smc": design["selected_candidate_id"],
            "selected_is_mixture": mixture_design["selected_candidate_id"],
            "smc": {
                "mean": smc.mean,
                "relative_se": smc_rse,
                "replicates": int(config["smc_replicates"][cell["id"]]),
                "log_replicate_estimates": smc.log_replicate_estimates.tolist(),
                "median_initial_ancestors": statistics.median(
                    x["final_unique_initial_ancestors"]
                    for x in smc.replicate_diagnostics
                ),
                "stages": smc_stages,
                "total_wall_seconds": smc_total,
                "accuracy": smc_accuracy,
            },
            "is": {
                **independent_is,
                "proposal_digest": canonical_digest(proposal_parameters(fit.proposal)),
                "training_bank_initial_ancestors": bank_ancestry,
                "weighted_smc_bank_ess": fit.weighted_bank_ess,
                "stages": is_stages,
                "total_wall_seconds": is_total,
                "accuracy": is_accuracy,
            },
            "pair_accuracy": pair_accuracy,
            "both_qualified": both_qualified,
            "fixed_precision_total_wall_ratio_smc_over_is": (
                smc_total / is_total if both_qualified else None
            ),
        })
        print(json.dumps({"finished_cell": cell["id"], "both_qualified": both_qualified},
                         allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r15-fixed-precision-total-work.v1",
        "role": "development_frozen_workflow_timing_not_confirmation",
        "source": source,
        "config": config,
        "input_artifact_sha256": hashes,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "Frozen algorithms have no runtime hyperparameter selection; selection time is zero by protocol.",
            "SMC-bank generation is charged to IS offline time, not counted as IID inference.",
            "Original research/design searches and failed development attempts are not amortized in this per-task frozen workflow; their costs must be disclosed separately.",
            "Single local sequential timing is not a hardware-controlled performance confidence interval.",
            "Comparison is withheld whenever either estimator or their pair fails the common accuracy gate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_fixed_precision_total_work_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"fixed-precision total-work output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
