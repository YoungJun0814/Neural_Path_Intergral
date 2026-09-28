"""Equal-payoff-budget R1 CE-versus-SMC bank follow-up (development only)."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _dct, _evaluate, _problem, _seed
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.baselines.weighted_conditional_ce import (
    WeightedConditionalCEConfig,
    proposal_parameters,
    train_weighted_conditional_ce,
)
from src.path_integral.finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture
from src.path_integral.r1_bottleneck_diagnostics import fit_projected_mean_shift
from src.path_integral.research_result_contract import canonical_digest, source_manifest
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.tempered_conditional_smc import (
    TemperedSMCConfig,
    estimate_tempered_normalizer,
)
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)

ROOT = Path(__file__).resolve().parents[1]


def _temperatures(levels: int) -> tuple[float, ...]:
    return tuple((index / levels) ** 4 for index in range(levels + 1))


def _log_potential(problem: RBergomiBaselineProblem, samples: torch.Tensor) -> torch.Tensor:
    return evaluate_rbergomi_conditional_terminal(
        problem, samples,
    ).payoffs.log_left_probability


def _evaluate_method(
    problem: RBergomiBaselineProblem, proposal: DefensiveFiniteRankGaussianMixture,
    ledger: SeedLedger, *, name: str, rep: int, count: int, fit_wall: float,
    training_evaluations: int, ancestry: dict[str, Any] | None,
    reference_mean: float, reference_se: float,
) -> dict[str, Any]:
    estimate = _evaluate(
        problem, proposal, count=count,
        path_seed=_seed(ledger, problem.task_id, "heldout", rep, f"{name}-path"),
        label_seed=_seed(ledger, problem.task_id, "heldout", rep, f"{name}-label"),
    )
    log_mu = estimate["conditional"]["log_mean"]
    # Descriptive reference check only: both the reference SE and final IS SE
    # can be too large for an accuracy qualification.
    z = None
    if log_mu is not None:
        mu = math.exp(log_mu)
        rse = estimate["conditional"]["relative_se"]
        if rse is not None:
            se = mu * rse
            denominator = math.hypot(se, reference_se)
            z = abs(mu - reference_mean) / denominator if denominator > 0 else None
    return {
        "method": name,
        "fit_wall_seconds": fit_wall,
        "training_payoff_evaluations": training_evaluations,
        "heldout": estimate,
        "descriptive_reference_z": z,
        "ancestry": ancestry,
        "proposal_parameters": proposal_parameters(proposal),
        "proposal_digest": canonical_digest(proposal_parameters(proposal)),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    source = source_manifest(ROOT, config=config)
    ledger = SeedLedger()
    spec = config["experiment"]
    count = int(spec["evaluation_count"])
    particles = int(spec["smc_particles"])
    levels = int(spec["smc_levels"])
    temperatures = _temperatures(levels)
    ce_config = WeightedConditionalCEConfig(**config["ce"])
    ce_evaluations = ce_config.iterations * ce_config.samples_per_iteration
    smc_evaluations = particles * (levels + 1)
    if abs(smc_evaluations - ce_evaluations) / ce_evaluations > 0.1:
        raise ValueError("CE and SMC training payoff budgets differ by more than 10%")
    cells = []
    for cell in config["cells"]:
        problem = _problem(config["model"], cell, int(spec["steps"]))
        directions = _dct(problem.steps, int(spec["rank"]))
        log_potential = partial(_log_potential, problem)
        reference_start = time.perf_counter()
        reference = estimate_tempered_normalizer(
            log_potential, dimension=problem.local_dimension,
            config=TemperedSMCConfig(
                particles=particles, temperatures=temperatures,
                mutation_steps=1, pcn_scale=float(spec["reference_pcn_scale"]),
                replicates=int(spec["reference_replicates"]),
                seed=_seed(ledger, problem.task_id, "reference", 0, "independent-smc"),
                retain_final_particles=False,
            ),
        )
        reference_record = {
            "mean": reference.mean,
            "standard_error": reference.standard_error,
            "relative_se": reference.standard_error / reference.mean if reference.mean > 0 else None,
            "replicate_estimates": reference.replicate_estimates.tolist(),
            "log_replicate_estimates": reference.log_replicate_estimates.tolist(),
            "potential_evaluations": reference.potential_evaluations,
            "wall_seconds": time.perf_counter() - reference_start,
            "minimum_incremental_ess_fraction": reference.minimum_incremental_ess_fraction,
        }
        records = []
        for rep in range(int(spec["independent_training_seeds"])):
            ce = train_weighted_conditional_ce(
                problem, training_seed=_seed(ledger, problem.task_id, "training", rep, "ce"),
                config=ce_config,
            )
            methods: list[dict[str, Any]] = []
            methods.append(_evaluate_method(
                problem, ce.proposal, ledger, name="ce", rep=rep, count=count,
                fit_wall=ce.cost.wall_seconds, training_evaluations=ce_evaluations,
                ancestry=None, reference_mean=reference.mean,
                reference_se=reference.standard_error,
            ))
            for scale in spec["smc_pcn_scales"]:
                name = f"smc_pcn_{scale}"
                start = time.perf_counter()
                smc = estimate_tempered_normalizer(
                    log_potential, dimension=problem.local_dimension,
                    config=TemperedSMCConfig(
                        particles=particles, temperatures=temperatures,
                        mutation_steps=1, pcn_scale=float(scale), replicates=1,
                        seed=_seed(ledger, problem.task_id, "training", rep, name),
                        retain_final_particles=True,
                    ),
                )
                if smc.final_particles is None:
                    raise RuntimeError("SMC final bank missing")
                bank = smc.final_particles
                # Correlated resampled particles approximate the final target;
                # equal empirical weights are *fitting* weights only. This is
                # neither IID CE data nor an ordinary IS final estimator.
                zeros = torch.zeros(bank.shape[0], dtype=torch.float64)
                proposal, _ = fit_projected_mean_shift(
                    bank, zeros, zeros, directions, objective="kl",
                    defensive_mass=float(spec["defensive_mass"]),
                    steps=int(spec["optimizer_steps"]),
                )
                fit_wall = time.perf_counter() - start
                ancestry = {
                    **smc.replicate_diagnostics[0],
                    "log_training_normalizer": float(smc.log_replicate_estimates[0]),
                    "minimum_incremental_ess_fraction": smc.minimum_incremental_ess_fraction,
                    "mutation_acceptance_rate": smc.mutation_acceptance_rate,
                }
                methods.append(_evaluate_method(
                    problem, proposal, ledger, name=name, rep=rep, count=count,
                    fit_wall=fit_wall, training_evaluations=smc.potential_evaluations,
                    ancestry=ancestry, reference_mean=reference.mean,
                    reference_se=reference.standard_error,
                ))
            records.append({
                "replicate": rep,
                "ce_target_ess_history": ce.target_ess_history,
                "ce_target_reached": ce.target_reached,
                "methods": methods,
            })
        cells.append({"cell": cell, "reference": reference_record, "replicates": records})
        print(json.dumps({"finished_cell": cell["id"]}, allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r1-bank-followup.v1",
        "role": "development_not_confirmation",
        "source": source,
        "config": config,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "SMC retained particles are correlated approximations to the final target.",
            "Independent SMC reference is unbiased under the fixed temperature schedule but may be imprecise.",
            "Payoff-evaluation counts are matched within 10%; wall costs and method overhead are reported separately.",
            "The fitted family is the same defensive DCT mean-shift mixture, not full V16 or full iCEred.",
            "All results are development data and cannot validate final superiority.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r1_bank_followup_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"R1 follow-up output exists: {output}")
    temporary = output.with_suffix(output.suffix + ".pending")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "cells": len(payload["cells"])},
                     allow_nan=False))


if __name__ == "__main__":
    main()
