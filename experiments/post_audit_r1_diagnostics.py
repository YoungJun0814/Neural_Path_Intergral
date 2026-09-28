"""Reproducible R1 development diagnostics; never a confirmation benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.baselines.weighted_conditional_ce import (
    WeightedConditionalCEConfig,
    proposal_parameters,
    train_weighted_conditional_ce,
)
from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
    build_isotropic_small_noise_safety_component,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.r1_bottleneck_diagnostics import (
    conditional_log_payoff_gradients,
    fit_projected_mean_shift,
    nested_reference_complement,
    summarize_log_contributions,
    weighted_target_directions,
)
from src.path_integral.research_result_contract import canonical_digest, source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.tempered_conditional_smc import (
    TemperedSMCConfig,
    estimate_tempered_normalizer,
)
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)

ROOT = Path(__file__).resolve().parents[1]


def _problem(model: dict[str, Any], cell: dict[str, Any], steps: int) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id=f"{cell['id']}-n{steps}",
        task=TerminalThresholdTask(level=float(cell["threshold"])),
        spot=float(model["spot"]), maturity=float(model["maturity"]),
        steps=steps, hurst=float(cell.get("hurst", model["hurst"])),
        eta=float(cell.get("eta", model["eta"])), xi=float(model["xi"]),
        rho=float(cell.get("rho", model["rho"])),
    )


def _seed(ledger: SeedLedger, task: str, role: str, rep: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r1", role, "development", task, 0, rep, stream,
    ))


def _dct(steps: int, rank: int) -> torch.Tensor:
    if rank % 2:
        raise ValueError("two-channel DCT rank must be even")
    return build_blp_cameron_martin_basis(
        steps=steps, modes_per_driver=rank // 2,
    ).matrix


def _new_bank(
    problem: RBergomiBaselineProblem, proposal: DefensiveFiniteRankGaussianMixture,
    *, count: int, path_seed: int, label_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    draw = proposal.sample(count, path_seed=path_seed, label_seed=label_seed)
    log_g = evaluate_rbergomi_conditional_terminal(
        problem, draw.samples,
    ).payoffs.log_left_probability
    return draw.samples, log_g, draw.log_p_over_q


def _evaluate(
    problem: RBergomiBaselineProblem, proposal: DefensiveFiniteRankGaussianMixture,
    *, count: int, path_seed: int, label_seed: int, raw_seed: int | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    samples, log_g, log_p_over_q = _new_bank(
        problem, proposal, count=count, path_seed=path_seed,
        label_seed=label_seed,
    )
    conditional = summarize_log_contributions(log_g + log_p_over_q)
    normalization = summarize_log_contributions(log_p_over_q)
    record: dict[str, Any] = {
        "conditional": asdict(conditional),
        "normalization": asdict(normalization),
        "wall_seconds": time.perf_counter() - start,
        "raw_sample_count": count,
    }
    if raw_seed is not None:
        raw_start = time.perf_counter()
        generator = torch.Generator().manual_seed(raw_seed)
        price = torch.randn(
            (count, problem.steps), dtype=torch.float64, generator=generator,
        )
        full = torch.cat((samples, price), dim=1)
        hit = problem.hard_event(problem.simulate_latent(full))
        raw_logs = torch.where(
            hit, log_p_over_q, torch.full_like(log_p_over_q, -math.inf),
        )
        record["raw"] = asdict(summarize_log_contributions(raw_logs))
        record["raw_hits"] = int(torch.sum(hit))
        record["raw_extra_wall_seconds"] = time.perf_counter() - raw_start
    return record


def _ablation_proposals(
    learned: DefensiveFiniteRankGaussianMixture,
    *, safety_variance: float = 4.0, defensive_mass: float = 0.1,
    safety_mass: float = 0.1,
) -> dict[str, DefensiveFiniteRankGaussianMixture | FiniteRankGaussianComponent]:
    d = learned.dimension
    natural = FiniteRankGaussianComponent.natural(d)
    best = learned.components[-1]
    safety = build_isotropic_small_noise_safety_component(d, epsilon=1.0 / safety_variance)
    # Learned-only Gaussian has full finite-grid support and finite second
    # moment for this identity-covariance shift, but no uniform p/q cap.
    return {
        "defensive_only": DefensiveFiniteRankGaussianMixture(
            (natural,), torch.ones(1, dtype=torch.float64),
        ),
        "defensive_learned": learned,
        "defensive_safety_learned": DefensiveFiniteRankGaussianMixture(
            (natural, safety, best),
            torch.tensor((defensive_mass, safety_mass, 1.0 - defensive_mass - safety_mass),
                         dtype=torch.float64),
        ),
        "learned_only": best,
    }


def _evaluate_learned_only(
    problem: RBergomiBaselineProblem, component: FiniteRankGaussianComponent,
    *, count: int, path_seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    generator = torch.Generator().manual_seed(path_seed)
    standard = torch.randn((count, component.dimension), dtype=torch.float64,
                           generator=generator)
    samples = component.transform_standard_normal(standard)
    log_p_over_q = -component.log_q_over_p(samples)
    log_g = evaluate_rbergomi_conditional_terminal(
        problem, samples,
    ).payoffs.log_left_probability
    return {
        "conditional": asdict(summarize_log_contributions(log_g + log_p_over_q)),
        "normalization": asdict(summarize_log_contributions(log_p_over_q)),
        "wall_seconds": time.perf_counter() - start,
        "raw_sample_count": count,
        "uniform_defensive_bound": False,
    }


def _smc_diagnostic(
    problem: RBergomiBaselineProblem, *, particles: int, levels: int,
    seed: int, pcn_scale: float,
) -> dict[str, Any]:
    temperatures = tuple((index / levels) ** 4 for index in range(levels + 1))
    start = time.perf_counter()
    result = estimate_tempered_normalizer(
        lambda x: evaluate_rbergomi_conditional_terminal(
            problem, x,
        ).payoffs.log_left_probability,
        dimension=problem.local_dimension,
        config=TemperedSMCConfig(
            particles=particles, temperatures=temperatures,
            mutation_steps=1, pcn_scale=pcn_scale, replicates=1, seed=seed,
            retain_final_particles=True,
        ),
    )
    particles_final = result.final_particles
    assert particles_final is not None
    # Only a rough, declared mode-concentration diagnostic; clusters are not
    # a certificate that all important failure modes have been found.
    scores = evaluate_rbergomi_conditional_terminal(
        problem, particles_final,
    ).payoffs.standardized_left_threshold
    return {
        "log_estimate": float(result.log_replicate_estimates[0]),
        "minimum_incremental_ess_fraction": result.minimum_incremental_ess_fraction,
        "mutation_acceptance_rate": result.mutation_acceptance_rate,
        "potential_evaluations": result.potential_evaluations + particles,
        "wall_seconds": time.perf_counter() - start,
        "ancestry": result.replicate_diagnostics[0],
        "final_score_quantiles": [float(x) for x in torch.quantile(
            scores, torch.tensor((0.05, 0.5, 0.95), dtype=torch.float64),
        )],
        "final_score_positive_fraction": float(torch.mean((scores >= 0).double())),
    }


def _evaluate_method(
    record: dict[str, Any], ledger: SeedLedger, problem: RBergomiBaselineProblem,
    proposal: DefensiveFiniteRankGaussianMixture, *, name: str,
    rep: int, count: int, bank_wall: float, parent_ce_fit_wall: float,
    fit_wall: float, fit_loss: float | None, raw: bool = False,
) -> None:
    evaluation = _evaluate(
        problem, proposal, count=count,
        path_seed=_seed(ledger, problem.task_id, "heldout", rep, f"{name}-path"),
        label_seed=_seed(ledger, problem.task_id, "heldout", rep, f"{name}-label"),
        raw_seed=(_seed(ledger, problem.task_id, "heldout", rep, f"{name}-price")
                  if raw else None),
    )
    record["fits"].append({
        "method": name, "fit_wall_seconds": fit_wall,
        "bank_wall_seconds": bank_wall,
        "parent_ce_fit_wall_seconds": parent_ce_fit_wall,
        "fit_loss": fit_loss, "evaluation": evaluation,
        "proposal_parameters": proposal_parameters(proposal),
        "proposal_digest": canonical_digest(proposal_parameters(proposal)),
        "defensive_mass": proposal.defensive_mass,
    })


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    ledger = SeedLedger()
    source = source_manifest(ROOT, config=config)
    model = config["model"]
    experiment = config["experiment"]
    seeds = int(experiment["independent_training_seeds"])
    bank_count = int(experiment["bank_count"])
    eval_count = int(experiment["evaluation_count"])
    defensive_mass = float(experiment["defensive_mass"])
    fixed_rank = int(experiment["diagnostic_rank"])
    cells: list[dict[str, Any]] = []
    for cell in config["cells"]:
        problem = _problem(model, cell, int(experiment["steps"]))
        rank_sweep = bool(cell["rank_sweep"])
        rep_records: list[dict[str, Any]] = []
        for rep in range(seeds):
            training_seed = _seed(ledger, problem.task_id, "training", rep, "weighted_ce")
            fit = train_weighted_conditional_ce(
                problem, training_seed=training_seed,
                config=WeightedConditionalCEConfig(**config["ce"]),
            )
            bank_start = time.perf_counter()
            bank, log_g, log_p_over_bank = _new_bank(
                problem, fit.proposal, count=bank_count,
                path_seed=_seed(ledger, problem.task_id, "bank", rep, "path"),
                label_seed=_seed(ledger, problem.task_id, "bank", rep, "label"),
            )
            bank_wall = time.perf_counter() - bank_start
            bank_ess = 1.0 / float(torch.sum(torch.softmax(
                log_g + log_p_over_bank, dim=0,
            ).square()))
            record: dict[str, Any] = {
                "replicate": rep, "ce_fit_wall_seconds": fit.cost.wall_seconds,
                "ce_training_samples": config["ce"]["iterations"] * config["ce"]["samples_per_iteration"],
                "ce_target_reached": fit.target_reached,
                "ce_target_ess_history": fit.target_ess_history,
                "ce_proposal_digest": canonical_digest(proposal_parameters(fit.proposal)),
                "bank_wall_seconds": bank_wall,
                "bank_target_ess": bank_ess,
                "fits": [],
            }
            def evaluate_method(name: str, proposal: DefensiveFiniteRankGaussianMixture,
                                fit_wall: float, fit_loss: float | None,
                                *, raw: bool = False,
                                problem: RBergomiBaselineProblem = problem,
                                rep: int = rep, record: dict[str, Any] = record,
                                bank_wall: float = bank_wall,
                                parent_ce_fit_wall: float = fit.cost.wall_seconds) -> None:
                _evaluate_method(
                    record, ledger, problem, proposal, name=name, rep=rep,
                    count=eval_count, bank_wall=bank_wall,
                    parent_ce_fit_wall=parent_ce_fit_wall,
                    fit_wall=fit_wall, fit_loss=fit_loss, raw=raw,
                )

            evaluate_method("conditional_ce", fit.proposal, 0.0, None, raw=True)
            natural = FiniteRankGaussianComponent.natural(problem.local_dimension)
            evaluate_method("conditional_natural", DefensiveFiniteRankGaussianMixture(
                (natural,), torch.ones(1, dtype=torch.float64),
            ), 0.0, None, raw=False)
            directions = _dct(problem.steps, fixed_rank)
            start = time.perf_counter()
            dct_kl, loss = fit_projected_mean_shift(
                bank, log_g, log_p_over_bank, directions,
                objective="kl", defensive_mass=defensive_mass,
                steps=int(experiment["optimizer_steps"]),
            )
            evaluate_method("dct_kl", dct_kl, time.perf_counter() - start, loss, raw=True)
            start = time.perf_counter()
            dct_m2, loss = fit_projected_mean_shift(
                bank, log_g, log_p_over_bank, directions,
                objective="m2", defensive_mass=defensive_mass,
                steps=int(experiment["optimizer_steps"]),
            )
            evaluate_method("dct_m2", dct_m2, time.perf_counter() - start, loss)
            if rank_sweep:
                gradient_start = time.perf_counter()
                gradients = conditional_log_payoff_gradients(problem, bank)
                gradient_wall = time.perf_counter() - gradient_start
                record["gradient_wall_seconds"] = gradient_wall
                for rank in experiment["rank_sweep"]:
                    for kind in ("dct", "target_pca", "fis_matrix", "risk_pca"):
                        if kind == "dct":
                            subspace = _dct(problem.steps, int(rank))
                            construction_wall = 0.0
                        else:
                            start = time.perf_counter()
                            subspace = weighted_target_directions(
                                bank, log_g, log_p_over_bank, rank=int(rank), kind=kind,
                                gradients=gradients if kind == "fis_matrix" else None,
                            )
                            construction_wall = time.perf_counter() - start
                            if kind == "fis_matrix":
                                construction_wall += gradient_wall
                        start = time.perf_counter()
                        proposal, loss = fit_projected_mean_shift(
                            bank, log_g, log_p_over_bank, subspace,
                            objective="kl", defensive_mass=defensive_mass,
                            steps=int(experiment["optimizer_steps"]),
                        )
                        evaluate_method(
                            f"{kind}_r{rank}_kl", proposal,
                            construction_wall + time.perf_counter() - start, loss,
                        )
            if rep == 0 and rank_sweep:
                record["nested"] = {}
                for kind, basis in (("dct", directions), ("target_pca", weighted_target_directions(
                    bank, log_g, log_p_over_bank, rank=fixed_rank, kind="target_pca",
                ))):
                    nested_runs = []
                    for outer_rep in range(int(experiment["nested_outer_replicates"])):
                        nested_runs.append(nested_reference_complement(
                            problem, basis,
                            outer_count=int(experiment["nested_outer_count"]),
                            inner_counts=tuple(int(x) for x in experiment["nested_inner_counts"]),
                            shift=basis.T @ fit.proposal.components[-1].mean,
                            outer_seed=_seed(ledger, problem.task_id, "nested", outer_rep, f"{kind}-outer"),
                            inner_seed=_seed(ledger, problem.task_id, "nested", outer_rep, f"{kind}-inner"),
                        ))
                    record["nested"][kind] = nested_runs
                ablations = _ablation_proposals(dct_kl, defensive_mass=defensive_mass)
                record["mixture_ablations"] = {}
                for name, proposal in ablations.items():
                    path_seed = _seed(ledger, problem.task_id, "ablation", rep, f"{name}-path")
                    if isinstance(proposal, FiniteRankGaussianComponent):
                        record["mixture_ablations"][name] = _evaluate_learned_only(
                            problem, proposal, count=eval_count, path_seed=path_seed,
                        )
                    else:
                        record["mixture_ablations"][name] = _evaluate(
                            problem, proposal, count=eval_count, path_seed=path_seed,
                            label_seed=_seed(ledger, problem.task_id, "ablation", rep, f"{name}-label"),
                        )
            rep_records.append(record)
        cell_result: dict[str, Any] = {
            "cell": cell, "problem": asdict(problem),
            "replicates": rep_records,
        }
        if rank_sweep:
            cell_result["independent_smc_banks"] = [
                _smc_diagnostic(
                    problem, particles=int(experiment["smc_particles"]),
                    levels=int(experiment["smc_levels"]),
                    seed=_seed(ledger, problem.task_id, "smc", bank_rep, "particle-bank"),
                    pcn_scale=float(experiment["smc_pcn_scale"]),
                ) for bank_rep in range(int(experiment["smc_independent_banks"]))
            ]
        cells.append(cell_result)
        print(json.dumps({"finished_cell": cell["id"], "replicates": len(rep_records)},
                         allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r1-diagnostic.v1",
        "role": "development_not_confirmation",
        "source": source, "config": config,
        "seed_ledger": ledger.to_dict(), "cells": cells,
        "limitations": [
            "All cells were inspected for model selection; no independent confirmation is claimed.",
            "FIS matrix uses final conditional CDF and not the complete iCEred annealing algorithm.",
            "Risk PCA is an exploratory contribution-weighted heuristic, not an optimal projector.",
            "Nested floors are biased plug-ins and apply only to reference-complement subspace families.",
            "Same-bank rank fitting is a geometry diagnostic, not independent end-to-end fitting.",
            "SMC ancestry and score quantiles do not certify discovery of all modes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/post_audit/r1_diagnostics_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"R1 result already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".pending")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "cells": len(payload["cells"])},
                     allow_nan=False))


if __name__ == "__main__":
    main()
