"""Fresh-bank, fresh-IID-final rank/basis/KL-versus-M2 development ablation.

SMC particles are never treated as IID importance-sampling observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _dct, _problem
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.r1_bottleneck_diagnostics import (
    fit_projected_mean_shift,
    summarize_log_contributions,
    weighted_target_directions,
)
from src.path_integral.research_result_contract import canonical_digest, source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_bank_mixture import (
    assign_saved_cluster_geometry,
    proposal_from_parameters,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed(ledger: SeedLedger, task: str, role: str, rep: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r15", role, "fresh-ablation", task, 0, rep, stream,
    ))


def _qualification(mean: float, rse: float, reference: dict[str, Any],
                   policy: dict[str, Any]) -> dict[str, float | bool]:
    ref_mean = float(reference["mean"])
    upper = abs(mean - ref_mean) + float(policy["confidence_z"]) * math.hypot(
        mean * rse, float(reference["standard_error"]),
    )
    margin = float(policy["relative_equivalence_margin"]) * ref_mean
    return {
        "relative_difference": abs(mean - ref_mean) / ref_mean,
        "equivalence_upper_difference": upper,
        "equivalence_margin": margin,
        "qualified": bool(
            rse <= policy["maximum_relative_se"]
            and reference["relative_se"] <= policy["maximum_reference_relative_se"]
            and upper <= margin
        ),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    inputs = {}
    hashes = {}
    for name in ("smc_design", "first_crosscheck", "mixture_crosscheck",
                 "mixture_design", "refined_reference"):
        raw = (ROOT / config[f"{name}_path"]).read_bytes()
        inputs[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    design = inputs["smc_design"]
    first = inputs["first_crosscheck"]
    mixture = inputs["mixture_crosscheck"]
    mode_design = inputs["mixture_design"]
    refined = inputs["refined_reference"]
    if not all(x["selected_candidate_id"] == design["selected_candidate_id"]
               for x in (first, refined)):
        raise ValueError("reference design mismatch")
    ledger = SeedLedger()
    source = source_manifest(ROOT, config=config)
    cells = []
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        first_cell = next(x for x in first["cells"] if x["cell"]["id"] == cell["id"])
        mixture_cell = next(x for x in mixture["cells"] if x["cell"]["id"] == cell["id"])
        mode_cell = next(x for x in mode_design["cells"] if x["cell"]["id"] == cell["id"])
        geometry = mode_cell["fixed_mode_geometry"]
        reference = next(x for x in refined["cells"] if x["cell"]["id"] == cell["id"])[
            "new_reference"
        ]
        parents = {
            "ce_portfolio": proposal_from_parameters(
                first_cell["independent_defensive_is"]["proposal_parameters"]
            ),
            "smc_trained_mixture": proposal_from_parameters(
                mixture_cell["proposal_parameters"]
            ),
        }
        repetitions = []
        for rep in range(int(config["independent_training_repetitions"])):
            for parent_name, parent in parents.items():
                bank_started = time.perf_counter()
                draw = parent.sample(
                    int(config["bank_count"]),
                    path_seed=_seed(ledger, problem.task_id, "fresh-bank", rep,
                                    f"{parent_name}-path"),
                    label_seed=_seed(ledger, problem.task_id, "fresh-bank", rep,
                                     f"{parent_name}-label"),
                )
                log_g = _log_potential(problem, draw.samples)
                log_weights = log_g + draw.log_p_over_q
                weights = torch.softmax(log_weights, dim=0)
                bank_ess = 1.0 / float(torch.sum(weights.square()))
                bank_wall = time.perf_counter() - bank_started
                methods = []
                for basis in config["bases"]:
                    for rank in config["ranks"]:
                        directions = (
                            _dct(problem.steps, int(rank)) if basis == "dct"
                            else weighted_target_directions(
                                draw.samples, log_g, draw.log_p_over_q,
                                rank=int(rank), kind="target_pca",
                            )
                        )
                        for objective in config["objectives"]:
                            key = f"{basis}-r{rank}-{objective}"
                            fit_started = time.perf_counter()
                            proposal, fit_loss = fit_projected_mean_shift(
                                draw.samples, log_g, draw.log_p_over_q,
                                directions, objective=objective,
                                defensive_mass=float(config["defensive_mass"]),
                                steps=int(config["optimizer_steps"]),
                            )
                            fit_wall = time.perf_counter() - fit_started
                            final_started = time.perf_counter()
                            final = proposal.sample(
                                int(config["final_count"]),
                                path_seed=_seed(ledger, problem.task_id, "fresh-final", rep,
                                                f"{parent_name}-{key}-path"),
                                label_seed=_seed(ledger, problem.task_id, "fresh-final", rep,
                                                 f"{parent_name}-{key}-label"),
                            )
                            log_values = _log_potential(problem, final.samples) + final.log_p_over_q
                            summary = summarize_log_contributions(log_values)
                            normalization = summarize_log_contributions(final.log_p_over_q)
                            final_wall = time.perf_counter() - final_started
                            if summary.log_mean is None or summary.relative_se is None:
                                raise FloatingPointError("all final contributions vanished")
                            ordered = torch.sort(log_values, descending=True).values
                            top_count = max(1, math.ceil(0.01 * ordered.numel()))
                            top_share = float(torch.exp(
                                torch.logsumexp(ordered[:top_count], dim=0)
                                - torch.logsumexp(ordered, dim=0)
                            ))
                            labels = assign_saved_cluster_geometry(final.samples, geometry)
                            log_total = torch.logsumexp(log_values, dim=0)
                            mode_shares = []
                            mode_sample_fractions = []
                            for mode in range(len(geometry["centers"])):
                                selected = labels == mode
                                mode_sample_fractions.append(float(selected.double().mean()))
                                mode_shares.append(float(torch.exp(
                                    torch.logsumexp(log_values[selected], dim=0) - log_total
                                )) if bool(selected.any()) else 0.0)
                            mean = math.exp(summary.log_mean)
                            methods.append({
                                "method": key,
                                "basis": basis,
                                "rank": rank,
                                "objective": objective,
                                "fit_loss": fit_loss,
                                "fit_wall_seconds": fit_wall,
                                "inference_wall_seconds": final_wall,
                                "proposal_digest": canonical_digest(proposal_parameters(proposal)),
                                "log_mean": summary.log_mean,
                                "log_second_moment": summary.log_second_moment,
                                "relative_se": summary.relative_se,
                                "contribution_ess": summary.contribution_ess,
                                "top_one_percent_contribution_share": top_share,
                                "mode_contribution_shares": mode_shares,
                                "mode_sample_fractions": mode_sample_fractions,
                                "maximum_contribution_fraction": summary.maximum_fraction,
                                "normalization_log_mean": normalization.log_mean,
                                "normalization_relative_se": normalization.relative_se,
                                "qualification": _qualification(
                                    mean, summary.relative_se, reference,
                                    config["qualification"],
                                ),
                            })
                repetitions.append({
                    "training_rep": rep,
                    "bank_parent": parent_name,
                    "parent_proposal_digest": canonical_digest(proposal_parameters(parent)),
                    "bank_count": config["bank_count"],
                    "bank_target_ess": bank_ess,
                    "bank_top_weight_fraction": float(weights.max()),
                    "bank_wall_seconds": bank_wall,
                    "methods": methods,
                })
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "reference_mean": reference["mean"],
            "reference_relative_se": reference["relative_se"],
            "smc_training_bank_ancestry": mixture_cell["fresh_bank_diagnostics"],
            "repetitions": repetitions,
        })
        print(json.dumps({"finished_cell": cell["id"]}, allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r15-fresh-ablation.v1",
        "role": "fresh_development_ablation_not_confirmation",
        "source": source,
        "config": config,
        "input_artifact_sha256": hashes,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "CE and SMC-trained parent proposals were selected on earlier development data.",
            "Each new IID bank and final sample is disjoint; fitting within a bank shares training data across ablations.",
            "Single-run relative SE is not a fit-level uncertainty or a fixed-precision speed comparison.",
            "SMC bank ancestry reflects correlated particles and must not be interpreted as IID ESS.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_fresh_ablation_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"fresh ablation output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
