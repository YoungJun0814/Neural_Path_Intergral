"""Independent fixed-count SMC and exact defensive-IS reference cross-check."""

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
from experiments.post_audit_r15_reference_design import temperatures
from src.path_integral.baselines.weighted_conditional_ce import (
    WeightedConditionalCEConfig,
    proposal_parameters,
    train_weighted_conditional_ce,
)
from src.path_integral.finite_rank_gaussian_transport import (
    combine_defensive_gaussian_mixtures,
)
from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions
from src.path_integral.research_result_contract import canonical_digest, source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed(ledger: SeedLedger, task: str, role: str, replicate: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r15", role, "reference", task, 0, replicate, stream,
    ))


def _log_potential(problem: Any, samples: torch.Tensor) -> torch.Tensor:
    return evaluate_rbergomi_conditional_terminal(
        problem, samples,
    ).payoffs.log_left_probability


def _is_reference(
    problem: Any, config: dict[str, Any], ledger: SeedLedger,
) -> dict[str, Any]:
    ce_spec = dict(config["is_training"])
    number_fits = int(ce_spec.pop("independent_ce_fits"))
    ce_config = WeightedConditionalCEConfig(**ce_spec)
    proposals = []
    fits = []
    started_fit = time.perf_counter()
    for fit_rep in range(number_fits):
        seed = _seed(ledger, problem.task_id, "is-training", fit_rep, "ce")
        result = train_weighted_conditional_ce(
            problem, training_seed=seed, config=ce_config,
        )
        proposals.append(result.proposal)
        fits.append({
            "training_seed": seed,
            "proposal_digest": result.proposal_digest,
            "target_ess_history": result.target_ess_history,
            "target_reached": result.target_reached,
            "wall_seconds": result.cost.wall_seconds,
        })
    fit_wall = time.perf_counter() - started_fit
    proposal = combine_defensive_gaussian_mixtures(
        tuple(proposals), tuple([1.0 / number_fits] * number_fits),
    )
    chunks = []
    cluster_scaled_means = []
    cluster_records = []
    eval_spec = config["is_evaluation"]
    sample_count = int(eval_spec["units_per_cluster"])
    started_eval = time.perf_counter()
    for cluster in range(int(eval_spec["clusters"])):
        draw = proposal.sample(
            sample_count,
            path_seed=_seed(ledger, problem.task_id, "is-final", cluster, "path"),
            label_seed=_seed(ledger, problem.task_id, "is-final", cluster, "label"),
        )
        log_g = _log_potential(problem, draw.samples)
        log_values = log_g + draw.log_p_over_q
        chunks.append(log_values)
        summary = summarize_log_contributions(log_values)
        normalizer = summarize_log_contributions(draw.log_p_over_q)
        cluster_records.append({
            "cluster": cluster,
            "conditional": summary.__dict__,
            "normalization": normalizer.__dict__,
        })
        cluster_scaled_means.append(summary.log_mean)
    eval_wall = time.perf_counter() - started_eval
    all_logs = torch.cat(chunks)
    summary = summarize_log_contributions(all_logs)
    if summary.log_mean is None or summary.relative_se is None:
        raise FloatingPointError("IS reference produced no nonzero contributions")
    log_scale = max(x for x in cluster_scaled_means if x is not None)
    scaled_cluster_means = [
        math.exp(x - log_scale) if x is not None else 0.0
        for x in cluster_scaled_means
    ]
    between_rse = (
        statistics.stdev(scaled_cluster_means)
        / math.sqrt(len(scaled_cluster_means))
        / statistics.mean(scaled_cluster_means)
    )
    robust_rse = max(summary.relative_se, between_rse)
    mean = math.exp(summary.log_mean)
    log_total = torch.logsumexp(all_logs, dim=0)
    ordered = torch.sort(all_logs, descending=True).values
    top_one_count = max(1, math.ceil(0.01 * ordered.numel()))
    top_tenth_count = max(1, math.ceil(0.10 * ordered.numel()))
    top_one_share = float(torch.exp(torch.logsumexp(
        ordered[:top_one_count], dim=0,
    ) - log_total))
    top_tenth_share = float(torch.exp(torch.logsumexp(
        ordered[:top_tenth_count], dim=0,
    ) - log_total))
    return {
        "mean": mean,
        "log_mean": summary.log_mean,
        "standard_error": mean * robust_rse,
        "relative_se": robust_rse,
        "iid_relative_se": summary.relative_se,
        "between_cluster_relative_se": between_rse,
        "training_fit_count": number_fits,
        "training_payoff_evaluations": number_fits * ce_config.iterations * ce_config.samples_per_iteration,
        "training_wall_seconds": fit_wall,
        "evaluation_wall_seconds": eval_wall,
        "evaluation_samples": ordered.numel(),
        "proposal_digest": canonical_digest(proposal_parameters(proposal)),
        "proposal_parameters": proposal_parameters(proposal),
        "defensive_mass": proposal.defensive_mass,
        "top_one_percent_contribution_share": top_one_share,
        "top_ten_percent_contribution_share": top_tenth_share,
        "maximum_contribution_fraction": summary.maximum_fraction,
        "cluster_records": cluster_records,
        "fits": fits,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    design_path = ROOT / config["design_path"]
    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    if design["schema"] != "npi.post-audit.r15-reference-design.v1":
        raise ValueError("wrong design artifact")
    if design["selection_status"] != "diversity_gate_pass":
        raise ValueError("no qualified SMC design was selected")
    selected_id = design["selected_candidate_id"]
    candidate = next(
        x for x in design["config"]["design"]["candidates"] if x["id"] == selected_id
    )
    ledger = SeedLedger()
    cells = []
    source = source_manifest(ROOT, config=config)
    for cell in design["config"]["cells"]:
        problem = _problem(
            design["config"]["model"], cell,
            int(design["config"]["design"]["steps"]),
        )
        count = int(design["prespecified_reference_counts"][cell["id"]])
        smc_seed = _seed(ledger, problem.task_id, "smc-reference", 0, selected_id)
        smc_config = WeightedSMCConfig(
            particles=int(design["config"]["design"]["particles"]),
            temperatures=temperatures(
                int(design["config"]["design"]["levels"]),
                int(candidate["bridge_power"]),
            ),
            mutation_steps=int(candidate["mutation_steps"]),
            pcn_scale=float(candidate["pcn_scale"]),
            replicates=count, seed=smc_seed,
            resample_every=int(candidate["resample_every"]),
            resampling_scheme=str(candidate["resampling_scheme"]),
        )
        started = time.perf_counter()
        smc = estimate_weighted_tempered_normalizer(
            partial(_log_potential, problem),
            dimension=problem.local_dimension, config=smc_config,
        )
        smc_wall = time.perf_counter() - started
        smc_record = {
            "mean": smc.mean,
            "log_mean": math.log(smc.mean) if smc.mean > 0 else None,
            "standard_error": smc.standard_error,
            "relative_se": smc.standard_error / smc.mean if smc.mean > 0 else None,
            "log_replicate_estimates": smc.log_replicate_estimates.tolist(),
            "replicates": count,
            "seed": smc_seed,
            "potential_evaluations": smc.potential_evaluations,
            "wall_seconds": smc_wall,
            "median_unique_initial_ancestors": statistics.median(
                int(x["final_unique_initial_ancestors"]) for x in smc.replicate_diagnostics
            ),
            "median_final_weight_ess_fraction": statistics.median(
                float(x["final_weight_ess_fraction"]) for x in smc.replicate_diagnostics
            ),
        }
        independent_is = _is_reference(problem, config, ledger)
        policy = config["accuracy"]
        smc_precise = (
            smc_record["relative_se"] is not None
            and smc_record["relative_se"] <= policy["maximum_reference_relative_se"]
        )
        is_precise = independent_is["relative_se"] <= policy["maximum_reference_relative_se"]
        combined_se = math.hypot(smc.standard_error, independent_is["standard_error"])
        difference_upper = abs(smc.mean - independent_is["mean"]) + policy["confidence_z"] * combined_se
        equivalence = difference_upper <= policy["relative_equivalence_margin"] * smc.mean
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "smc_reference": smc_record,
            "independent_defensive_is": independent_is,
            "agreement": {
                "both_precise": bool(smc_precise and is_precise),
                "equivalence_upper_difference": difference_upper,
                "equivalence_margin": policy["relative_equivalence_margin"] * smc.mean,
                "interval_equivalent": bool(equivalence),
                "reference_gate_pass": bool(smc_precise and is_precise and equivalence),
            },
        })
        print(json.dumps({"finished_cell": cell["id"],
                          "reference_gate_pass": cells[-1]["agreement"]["reference_gate_pass"]},
                         allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r15-reference-crosscheck.v1",
        "role": "development_reference_crosscheck_not_confirmation",
        "source": source,
        "config": config,
        "design_artifact_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "selected_candidate_id": selected_id,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "Gaussian-mixture IS training is independent of SMC reference but may still miss rare modes.",
            "SMC production counts were fixed from independent design data before reference generation.",
            "Finite cluster count and heavy tails limit nominal standard-error reliability.",
            "No reference disagreement is concealed; a failed gate prohibits performance claims.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_reference_crosscheck_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"reference output already exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
