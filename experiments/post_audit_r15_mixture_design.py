"""SMC-bank mixture pilot with held-out contribution and mode-cluster diagnostics."""

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
    WeightedBankMixtureFit,
    assign_weighted_bank_clusters,
    fit_weighted_bank_mixture,
    proposal_from_parameters,
)
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed(ledger: SeedLedger, task: str, role: str, replicate: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r15", role, "mixture-design", task, 0, replicate, stream,
    ))


def _profile_proposal(
    problem: Any, proposal: Any, mode_fit: WeightedBankMixtureFit,
    config: dict[str, Any], ledger: SeedLedger, *, name: str,
    anchor_mean: float, anchor_se: float,
) -> dict[str, Any]:
    spec = config["pilot_evaluation"]
    eval_clusters = int(spec["clusters"])
    count = int(spec["units_per_cluster"])
    cluster_logs = []
    cluster_log_means = []
    mode_log_contributions: list[list[torch.Tensor]] = [
        [] for _ in range(mode_fit.centers.shape[0])
    ]
    started = time.perf_counter()
    for cluster in range(eval_clusters):
        draw = proposal.sample(
            count,
            path_seed=_seed(ledger, problem.task_id, "pilot-final", cluster, f"{name}-path"),
            label_seed=_seed(ledger, problem.task_id, "pilot-final", cluster, f"{name}-label"),
        )
        log_g = _log_potential(problem, draw.samples)
        values = log_g + draw.log_p_over_q
        cluster_logs.append(values)
        cluster_log_means.append(summarize_log_contributions(values).log_mean)
        labels = assign_weighted_bank_clusters(draw.samples, mode_fit)
        for index in range(len(mode_log_contributions)):
            mode_log_contributions[index].append(values[labels == index])
    all_logs = torch.cat(cluster_logs)
    summary = summarize_log_contributions(all_logs)
    if summary.log_mean is None or summary.relative_se is None:
        raise FloatingPointError("mixture pilot had no nonzero contribution")
    mu = math.exp(summary.log_mean)
    scale = max(x for x in cluster_log_means if x is not None)
    scaled = [math.exp(x - scale) if x is not None else 0.0 for x in cluster_log_means]
    between_rse = statistics.stdev(scaled) / math.sqrt(eval_clusters) / statistics.mean(scaled)
    robust_rse = max(summary.relative_se, between_rse)
    log_total = torch.logsumexp(all_logs, dim=0)
    sorted_logs = torch.sort(all_logs, descending=True).values
    top_count = max(1, math.ceil(0.01 * sorted_logs.numel()))
    top_one_share = float(torch.exp(torch.logsumexp(sorted_logs[:top_count], dim=0) - log_total))
    mode_shares = []
    for parts in mode_log_contributions:
        joined = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)
        mode_shares.append(
            float(torch.exp(torch.logsumexp(joined, dim=0) - log_total))
            if joined.numel() else 0.0
        )
    probe = proposal.sample(
        int(spec["bank_probe_count"]),
        path_seed=_seed(ledger, problem.task_id, "bank-probe", 0, f"{name}-path"),
        label_seed=_seed(ledger, problem.task_id, "bank-probe", 0, f"{name}-label"),
    )
    probe_log_g = _log_potential(problem, probe.samples)
    bank_weights = torch.softmax(probe_log_g + probe.log_p_over_q, dim=0)
    bank_ess = 1.0 / float(torch.sum(bank_weights.square()))
    se = mu * robust_rse
    descriptive_z = abs(mu - anchor_mean) / math.hypot(se, anchor_se)
    return {
        "log_mean": summary.log_mean,
        "relative_se": robust_rse,
        "iid_relative_se": summary.relative_se,
        "between_cluster_relative_se": between_rse,
        "descriptive_anchor_z": descriptive_z,
        "maximum_contribution_fraction": summary.maximum_fraction,
        "top_one_percent_contribution_share": top_one_share,
        "mode_contribution_shares": mode_shares,
        "exact_q_bank_target_ess": bank_ess,
        "evaluation_samples": all_logs.numel(),
        "wall_seconds": time.perf_counter() - started,
        "proposal_digest": canonical_digest(proposal_parameters(proposal)),
        "proposal_parameters": proposal_parameters(proposal),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    design_bytes = (ROOT / config["design_path"]).read_bytes()
    reference_bytes = (ROOT / config["reference_crosscheck_path"]).read_bytes()
    design = json.loads(design_bytes)
    reference = json.loads(reference_bytes)
    if design["selected_candidate_id"] != reference["selected_candidate_id"]:
        raise ValueError("selected SMC design changed between artifacts")
    selected = next(x for x in design["config"]["design"]["candidates"]
                    if x["id"] == design["selected_candidate_id"])
    source = source_manifest(ROOT, config=config)
    ledger = SeedLedger()
    cells = []
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        ref = next(x for x in reference["cells"] if x["cell"]["id"] == cell["id"])
        anchor_mean = float(ref["smc_reference"]["mean"])
        anchor_se = float(ref["smc_reference"]["standard_error"])
        samples = []
        weights = []
        bank_diagnostics = []
        started_training = time.perf_counter()
        for rep in range(int(config["training"]["independent_smc_banks"])):
            smc = estimate_weighted_tempered_normalizer(
                partial(_log_potential, problem), dimension=problem.local_dimension,
                config=WeightedSMCConfig(
                    particles=int(design["config"]["design"]["particles"]),
                    temperatures=temperatures(int(design["config"]["design"]["levels"]),
                                              int(selected["bridge_power"])),
                    mutation_steps=int(selected["mutation_steps"]),
                    pcn_scale=float(selected["pcn_scale"]),
                    replicates=1,
                    seed=_seed(ledger, problem.task_id, "smc-bank", rep, "particles"),
                    resample_every=int(selected["resample_every"]),
                    resampling_scheme=str(selected["resampling_scheme"]),
                    retain_final_particles=True,
                ),
            )
            if smc.final_particles is None or smc.final_weights is None:
                raise RuntimeError("retained weighted SMC bank is missing")
            samples.append(smc.final_particles)
            weights.append(smc.final_weights / int(config["training"]["independent_smc_banks"]))
            bank_diagnostics.append({
                "final_unique_initial_ancestors": smc.replicate_diagnostics[0][
                    "final_unique_initial_ancestors"
                ],
                "final_weight_ess_fraction": smc.replicate_diagnostics[0][
                    "final_weight_ess_fraction"
                ],
                "log_normalizer": float(smc.log_replicate_estimates[0]),
                "potential_evaluations": smc.potential_evaluations,
            })
        smc_training_wall = time.perf_counter() - started_training
        pooled = torch.cat(samples)
        pooled_weights = torch.cat(weights)
        shared = dict(config["training"])
        shared.pop("independent_smc_banks")
        shared.pop("candidates")
        mode_fit = fit_weighted_bank_mixture(
            pooled, pooled_weights, clusters=4, covariance_rank=0, **shared,
        )
        candidate_results = []
        for candidate in config["training"]["candidates"]:
            started_fit = time.perf_counter()
            fit = fit_weighted_bank_mixture(
                pooled, pooled_weights,
                clusters=int(candidate["clusters"]),
                covariance_rank=int(candidate["covariance_rank"]), **shared,
            )
            fit_wall = time.perf_counter() - started_fit
            profile = _profile_proposal(
                problem, fit.proposal, mode_fit, config, ledger,
                name=str(candidate["id"]), anchor_mean=anchor_mean, anchor_se=anchor_se,
            )
            candidate_results.append({
                "candidate": candidate,
                "fit_wall_seconds": fit_wall,
                "cluster_masses": fit.cluster_masses,
                "weighted_smc_bank_ess": fit.weighted_bank_ess,
                "profile": profile,
            })
        ce_proposal = proposal_from_parameters(ref["independent_defensive_is"]["proposal_parameters"])
        ce_profile = _profile_proposal(
            problem, ce_proposal, mode_fit, config, ledger,
            name="previous-ce-portfolio", anchor_mean=anchor_mean, anchor_se=anchor_se,
        )
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "smc_training_wall_seconds": smc_training_wall,
            "smc_banks": bank_diagnostics,
            "pooled_weighted_smc_bank_ess": mode_fit.weighted_bank_ess,
            "fixed_mode_geometry": {
                "feature_mean": mode_fit.feature_mean.tolist(),
                "feature_directions": mode_fit.feature_directions.tolist(),
                "feature_scales": mode_fit.feature_scales.tolist(),
                "centers": mode_fit.centers.tolist(),
            },
            "candidates": candidate_results,
            "previous_ce_portfolio": ce_profile,
        })
        print(json.dumps({"finished_cell": cell["id"]}, allow_nan=False), flush=True)
    eligible = []
    for candidate in config["training"]["candidates"]:
        per_cell = [next(x for x in cell["candidates"] if x["candidate"]["id"] == candidate["id"])
                    for cell in cells]
        constraints_pass = all(
            x["profile"]["relative_se"] <= config["selection"]["maximum_pilot_relative_se"]
            and x["profile"]["descriptive_anchor_z"]
            <= config["selection"]["maximum_descriptive_reference_z"]
            for x in per_cell
        )
        eligible.append({
            "id": candidate["id"],
            "constraints_pass": constraints_pass,
            "worst_cell_pilot_relative_se": max(x["profile"]["relative_se"] for x in per_cell),
            "minimum_cell_bank_target_ess": min(
                x["profile"]["exact_q_bank_target_ess"] for x in per_cell
            ),
        })
    accepted = [x for x in eligible if x["constraints_pass"]]
    selected_candidate = min(
        accepted if accepted else eligible,
        key=lambda x: (x["worst_cell_pilot_relative_se"], x["id"]),
    )
    return {
        "schema": "npi.post-audit.r15-mixture-design.v1",
        "role": "development_selection_not_reference",
        "source": source,
        "config": config,
        "design_artifact_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "reference_artifact_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "selection_table": eligible,
        "selected_candidate_id": selected_candidate["id"],
        "selection_status": "pilot_constraints_pass" if accepted else "pilot_constraints_failed",
        "limitations": [
            "Fixed clustering on the pooled training bank may miss modes absent from that bank.",
            "Pilot IS estimates and their apparent relative SE cannot validate a selected proposal.",
            "Final reference needs independently trained mixture and independent held-out paths.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_mixture_design_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"mixture design output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output), "selection": payload["selected_candidate_id"]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
