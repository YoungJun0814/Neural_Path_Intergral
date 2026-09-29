"""Fresh SMC-trained Gaussian-mixture IS cross-check on independent IID paths."""

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
from src.path_integral.weighted_bank_mixture import (
    assign_weighted_bank_clusters,
    fit_weighted_bank_mixture,
)
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed(ledger: SeedLedger, task: str, role: str, replicate: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r15", role, "independent-mixture", task, 0, replicate, stream,
    ))


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    design_bytes = (ROOT / config["smc_design_path"]).read_bytes()
    mixture_bytes = (ROOT / config["mixture_design_path"]).read_bytes()
    reference_bytes = (ROOT / config["refined_reference_path"]).read_bytes()
    design = json.loads(design_bytes)
    mixture = json.loads(mixture_bytes)
    reference = json.loads(reference_bytes)
    if mixture["selection_status"] != "pilot_constraints_pass":
        raise ValueError("no mixture passed the pilot constraints")
    if design["selected_candidate_id"] != reference["selected_candidate_id"]:
        raise ValueError("SMC reference design changed")
    smc_candidate = next(x for x in design["config"]["design"]["candidates"]
                         if x["id"] == design["selected_candidate_id"])
    mixture_candidate = next(x for x in mixture["config"]["training"]["candidates"]
                             if x["id"] == mixture["selected_candidate_id"])
    ledger = SeedLedger()
    source = source_manifest(ROOT, config=config)
    cells = []
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        target_reference = next(x for x in reference["cells"]
                                if x["cell"]["id"] == cell["id"])["new_reference"]
        bank_samples = []
        bank_weights = []
        bank_diagnostics = []
        started_fit = time.perf_counter()
        for bank_rep in range(int(config["independent_smc_banks"])):
            smc = estimate_weighted_tempered_normalizer(
                partial(_log_potential, problem), dimension=problem.local_dimension,
                config=WeightedSMCConfig(
                    particles=int(design["config"]["design"]["particles"]),
                    temperatures=temperatures(int(design["config"]["design"]["levels"]),
                                              int(smc_candidate["bridge_power"])),
                    mutation_steps=int(smc_candidate["mutation_steps"]),
                    pcn_scale=float(smc_candidate["pcn_scale"]),
                    replicates=1,
                    seed=_seed(ledger, problem.task_id, "fresh-smc-bank", bank_rep, "particles"),
                    resample_every=int(smc_candidate["resample_every"]),
                    resampling_scheme=str(smc_candidate["resampling_scheme"]),
                    retain_final_particles=True,
                ),
            )
            if smc.final_particles is None or smc.final_weights is None:
                raise RuntimeError("fresh SMC bank missing")
            bank_samples.append(smc.final_particles)
            bank_weights.append(smc.final_weights / config["independent_smc_banks"])
            bank_diagnostics.append({
                "unique_initial_ancestors": smc.replicate_diagnostics[0][
                    "final_unique_initial_ancestors"
                ],
                "final_weight_ess_fraction": smc.replicate_diagnostics[0][
                    "final_weight_ess_fraction"
                ],
                "log_normalizer": float(smc.log_replicate_estimates[0]),
                "potential_evaluations": smc.potential_evaluations,
            })
        pooled = torch.cat(bank_samples)
        pooled_weights = torch.cat(bank_weights)
        common = dict(mixture["config"]["training"])
        common.pop("independent_smc_banks")
        common.pop("candidates")
        learned = fit_weighted_bank_mixture(
            pooled, pooled_weights,
            clusters=int(mixture_candidate["clusters"]),
            covariance_rank=int(mixture_candidate["covariance_rank"]), **common,
        )
        mode_fit = fit_weighted_bank_mixture(
            pooled, pooled_weights, clusters=4, covariance_rank=0, **common,
        )
        fit_wall = time.perf_counter() - started_fit
        proposal = learned.proposal
        eval_spec = config["evaluation"]
        count = int(eval_spec["units_per_cluster"])
        cluster_count = int(eval_spec["clusters"])
        log_chunks = []
        normalization_chunks = []
        cluster_logmeans = []
        cluster_records = []
        mode_parts: list[list[torch.Tensor]] = [[] for _ in range(mode_fit.centers.shape[0])]
        started_eval = time.perf_counter()
        for cluster in range(cluster_count):
            draw = proposal.sample(
                count,
                path_seed=_seed(ledger, problem.task_id, "independent-is-final", cluster, "path"),
                label_seed=_seed(ledger, problem.task_id, "independent-is-final", cluster, "label"),
            )
            log_values = _log_potential(problem, draw.samples) + draw.log_p_over_q
            log_chunks.append(log_values)
            normalization_chunks.append(draw.log_p_over_q)
            cluster_logmeans.append(summarize_log_contributions(log_values).log_mean)
            cluster_records.append({
                "cluster": cluster,
                "conditional": summarize_log_contributions(log_values).__dict__,
                "normalization": summarize_log_contributions(draw.log_p_over_q).__dict__,
            })
            labels = assign_weighted_bank_clusters(draw.samples, mode_fit)
            for index in range(len(mode_parts)):
                mode_parts[index].append(log_values[labels == index])
        eval_wall = time.perf_counter() - started_eval
        all_log_values = torch.cat(log_chunks)
        summary = summarize_log_contributions(all_log_values)
        normalization = summarize_log_contributions(torch.cat(normalization_chunks))
        if summary.log_mean is None or summary.relative_se is None:
            raise FloatingPointError("independent mixture IS produced no contribution")
        scaled_max = max(x for x in cluster_logmeans if x is not None)
        scaled_means = [math.exp(x - scaled_max) if x is not None else 0.0
                        for x in cluster_logmeans]
        between_rse = statistics.stdev(scaled_means) / math.sqrt(cluster_count) / statistics.mean(scaled_means)
        robust_rse = max(summary.relative_se, between_rse)
        mu = math.exp(summary.log_mean)
        ordered = torch.sort(all_log_values, descending=True).values
        log_total = torch.logsumexp(ordered, dim=0)
        top_count = max(1, math.ceil(0.01 * ordered.numel()))
        top_one_share = float(torch.exp(torch.logsumexp(ordered[:top_count], dim=0) - log_total))
        mode_shares = []
        for parts in mode_parts:
            joined = torch.cat(parts)
            mode_shares.append(float(torch.exp(torch.logsumexp(joined, dim=0) - log_total))
                               if joined.numel() else 0.0)
        ref_mu = float(target_reference["mean"])
        ref_se = float(target_reference["standard_error"])
        upper = abs(mu - ref_mu) + config["qualification"]["confidence_z"] * math.hypot(
            mu * robust_rse, ref_se,
        )
        margin = config["qualification"]["relative_equivalence_margin"] * ref_mu
        qualified = (
            robust_rse <= config["qualification"]["maximum_relative_se"]
            and target_reference["relative_se"]
            <= config["qualification"]["maximum_reference_relative_se"]
            and upper <= margin
        )
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "fresh_bank_diagnostics": bank_diagnostics,
            "pooled_weighted_bank_ess": learned.weighted_bank_ess,
            "cluster_masses": learned.cluster_masses,
            "proposal_digest": canonical_digest(proposal_parameters(proposal)),
            "proposal_parameters": proposal_parameters(proposal),
            "fit_wall_seconds": fit_wall,
            "inference_wall_seconds": eval_wall,
            "inference_samples": all_log_values.numel(),
            "is_log_mean": summary.log_mean,
            "is_mean": mu,
            "is_relative_se": robust_rse,
            "is_iid_relative_se": summary.relative_se,
            "is_between_cluster_relative_se": between_rse,
            "cluster_records": cluster_records,
            "normalization_log_mean": normalization.log_mean,
            "normalization_relative_se": normalization.relative_se,
            "top_one_percent_contribution_share": top_one_share,
            "maximum_contribution_fraction": summary.maximum_fraction,
            "mode_contribution_shares": mode_shares,
            "reference_mean": ref_mu,
            "reference_relative_se": target_reference["relative_se"],
            "equivalence_upper_difference": upper,
            "equivalence_margin": margin,
            "qualified": qualified,
            "timing_context": "single_CPU_run_not_power_or_warmup_controlled",
        })
        print(json.dumps({"finished_cell": cell["id"], "qualified": qualified},
                         allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r15-mixture-independent.v1",
        "role": "independent_development_crosscheck_not_confirmation",
        "source": source,
        "config": config,
        "smc_design_artifact_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "mixture_design_artifact_sha256": hashlib.sha256(mixture_bytes).hexdigest(),
        "refined_reference_artifact_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "selected_smc_candidate": design["selected_candidate_id"],
        "selected_mixture_candidate": mixture["selected_candidate_id"],
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "Mixture training banks are correlated SMC particles and are disjoint from final IID IS samples.",
            "Small RSE can still miss unseen modes; agreement with an independent SMC mechanism is required.",
            "Mode clusters are fitted from the training bank and cannot certify complete mode discovery.",
            "Timing is descriptive; fixed-precision total wall comparisons require qualified methods.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_mixture_independent_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"independent mixture output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
