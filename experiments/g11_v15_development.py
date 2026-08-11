"""Run a frozen V15 matrix with exact estimands and training-inclusive work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines import (
    CEMTrainingConfig,
    LargeDeviationTrainingConfig,
    evaluate_conditional_terminal_units,
    evaluate_latent_is_units,
    evaluate_smoothing_rqmc_units,
    freeze_conditional_rbergomi_proposal,
    freeze_smoothing_rqmc_proposal,
    train_cem_proposal,
    train_large_deviation_proposal,
)
from src.path_integral.baselines.rbergomi_common import BaselineUnitBatch, RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_drift_basis,
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.cameron_martin_modes import ActionSolverConfig, ModeSearchConfig
from src.path_integral.conditional_transport_adaptation import (
    ConditionalTransportAdaptationConfig,
)
from src.path_integral.finite_rank_gaussian_transport import CurvatureTransportConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import (
    assert_transport_unchanged,
    evaluate_rbergomi_cm_transport,
    train_rbergomi_cm_transport,
)
from src.path_integral.rbergomi_local_volterra_transport import (
    LocalVolterraTransportTrainingConfig,
    evaluate_local_volterra_transport,
    train_local_volterra_transport,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig
from src.path_integral.v15_baseline_protocol import (
    V15BaselineProtocol,
    summarize_v15_method,
    work_efficiency_ratio,
)
from src.path_integral.v15_result_audit import (
    audit_v15_result,
    file_sha256,
    git_source_provenance,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed(root_seed: int, role: str, used: list[int]) -> int:
    digest = hashlib.sha256(f"NPI-G11-V15\0{root_seed}\0{role}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1
    if value in used:
        raise AssertionError("V15 seed derivation collided")
    used.append(value)
    return value


def _summary(values: torch.Tensor) -> tuple[float, float, float]:
    variance = float(torch.var(values, unbiased=True))
    return float(torch.mean(values)), variance, math.sqrt(variance / values.numel())


def _measured_baseline(
    problem: RBergomiBaselineProblem,
    proposal: Any,
    *,
    method: str,
    units: int,
    seed: int,
    rqmc_points: int,
) -> tuple[BaselineUnitBatch, BaselineCostLedger]:
    wall = time.perf_counter()
    cpu = time.process_time()
    if method == "smoothing_rqmc":
        batch = evaluate_smoothing_rqmc_units(
            problem,
            proposal,
            randomizations=units,
            points_per_randomization=rqmc_points,
            seed=seed,
        )
    elif method == "conditional_rbergomi":
        batch = evaluate_conditional_terminal_units(
            problem,
            proposal,
            sample_count=units,
            seed=seed,
        )
    else:
        batch = evaluate_latent_is_units(problem, proposal, sample_count=units, seed=seed)
    gaussian_dimension = (
        problem.local_dimension if method == "conditional_rbergomi" else problem.latent_dimension
    )
    work = batch.raw_sample_count * (gaussian_dimension + problem.steps)
    work += batch.likelihood_evaluations * proposal.dimension
    work += batch.cdf_calls + 32 * batch.quadrature_calls
    cost = BaselineCostLedger(
        final_samples=batch.raw_sample_count,
        likelihood_evaluations=batch.likelihood_evaluations,
        cdf_calls=batch.cdf_calls,
        quadrature_calls=batch.quadrature_calls,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu,
        measurement_mode="standardized_hardware_wall",
    )
    return batch, cost


def _record(
    method: str,
    values: torch.Tensor,
    *,
    training_cost: BaselineCostLedger,
    evaluation_cost: BaselineCostLedger,
    reference_mean: float,
    reference_se: float,
    query_count: int,
) -> tuple[dict[str, Any], Any]:
    summary = summarize_v15_method(
        method,
        values,
        training_cost=training_cost,
        evaluation_cost=evaluation_cost,
    )
    denominator = math.sqrt(summary.standard_error**2 + reference_se**2)
    accuracy_z = abs(summary.estimate - reference_mean) / max(
        denominator,
        torch.finfo(torch.float64).tiny,
    )
    return (
        {
            "method": method,
            "estimate": summary.estimate,
            "sample_variance": summary.sample_variance,
            "standard_error": summary.standard_error,
            "inferential_units": summary.samples,
            "accuracy_z": accuracy_z,
            "training_cost": asdict(training_cost),
            "evaluation_cost": asdict(evaluation_cost),
            "total_work_at_primary_query_count": summary.total_work(query_count),
        },
        summary,
    )


def run(config_path: Path) -> tuple[dict[str, Any], Path]:
    source_provenance = git_source_provenance(ROOT)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = V15BaselineProtocol(
        primary_comparators=tuple(config["protocol"]["primary_comparators"]),
        query_counts=tuple(config["protocol"]["query_counts"]),
    )
    used_seeds: list[int] = []
    root_seed = int(config["root_seed"])
    model = config["model"]
    evaluation = config["evaluation"]
    candidate_config = config["candidate"]
    baseline_config = config["baselines"]
    query_count = int(config["protocol"]["primary_query_count"])
    result_cells = []
    for cell in config["cells"]:
        cell_id = str(cell["cell_id"])
        problem = RBergomiBaselineProblem(
            task_id=cell_id,
            task=TerminalThresholdTask(level=float(cell["threshold"])),
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            steps=int(model["steps"]),
            hurst=float(cell["hurst"]),
            eta=float(cell.get("eta", model["eta"])),
            xi=float(cell.get("xi", model["xi"])),
            rho=float(cell.get("rho", model["rho"])),
        )
        reference_proposal = freeze_conditional_rbergomi_proposal(
            problem,
            training_seed=_seed(root_seed, f"{cell_id}-reference-freeze", used_seeds),
        )
        reference_batch, reference_cost = _measured_baseline(
            problem,
            reference_proposal,
            method="conditional_rbergomi",
            units=int(evaluation["reference_iid"]),
            seed=_seed(root_seed, f"{cell_id}-reference-evaluation", used_seeds),
            rqmc_points=1,
        )
        reference_mean, reference_variance, reference_se = _summary(
            reference_batch.unit_contributions
        )

        solver = ActionSolverConfig(
            maximum_iterations=int(candidate_config["maximum_iterations"]),
            gradient_tolerance=float(candidate_config["gradient_tolerance"]),
        )
        basis_kind = str(candidate_config.get("mode_basis", "channel_dct"))
        if basis_kind == "mesh_compatible_hybrid":
            candidate_basis = build_mesh_compatible_blp_hybrid_basis(
                steps=problem.steps,
                maturity=problem.maturity,
                hurst=problem.hurst,
                drift_modes=int(candidate_config["modes"]),
                bridge_modes=int(candidate_config["bridge_modes"]),
            )
        elif basis_kind == "mesh_compatible_drift":
            candidate_basis = build_mesh_compatible_blp_drift_basis(
                steps=problem.steps,
                maturity=problem.maturity,
                hurst=problem.hurst,
                modes=int(candidate_config["modes"]),
            )
        elif basis_kind == "channel_dct":
            candidate_basis = None
        else:
            raise ValueError("unsupported candidate mode_basis")
        adaptation_values = candidate_config.get("conditional_adaptation")
        adaptation_config = (
            ConditionalTransportAdaptationConfig(
                iterations=int(adaptation_values["iterations"]),
                samples_per_iteration=int(adaptation_values["samples_per_iteration"]),
                smoothing=float(adaptation_values["smoothing"]),
                minimum_ess_fraction=float(
                    adaptation_values["minimum_ess_fraction"]
                ),
                minimum_variance=float(adaptation_values["minimum_variance"]),
                maximum_variance=float(adaptation_values["maximum_variance"]),
                adapt_covariance=bool(adaptation_values.get("adapt_covariance", True)),
            )
            if adaptation_values is not None
            else None
        )
        trained = train_rbergomi_cm_transport(
            problem,
            modes_per_driver=int(candidate_config["modes_per_driver"]),
            basis=candidate_basis,
            adaptation_config=adaptation_config,
            adaptation_seed=(
                _seed(root_seed, f"{cell_id}-candidate-adaptation", used_seeds)
                if adaptation_config is not None
                else None
            ),
            mode_search=ModeSearchConfig(
                methods=("lbfgs", "trust-ncg"),
                random_starts=int(candidate_config["random_starts"]),
                random_seed=_seed(root_seed, f"{cell_id}-candidate-mode", used_seeds),
                start_scale=float(candidate_config["start_scale"]),
                solver=solver,
            ),
            transport_config=CurvatureTransportConfig(
                defensive_mass=float(candidate_config["defensive_mass"]),
                asymptotic_safety_mass=float(
                    candidate_config.get("asymptotic_safety_mass", 0.0)
                ),
                safety_spectrum_decay=(
                    float(candidate_config["safety_spectrum_decay"])
                    if "safety_spectrum_decay" in candidate_config
                    else None
                ),
                safety_spectrum_scale=float(
                    candidate_config.get("safety_spectrum_scale", 1.0)
                ),
                safety_complement_decay=float(
                    candidate_config.get("safety_complement_decay", 2.0)
                ),
            ),
        )
        candidate_eval = evaluate_rbergomi_cm_transport(
            problem,
            trained.proposal,
            sample_count=int(evaluation["iid_units"]),
            path_seed=_seed(root_seed, f"{cell_id}-candidate-path", used_seeds),
            label_seed=_seed(root_seed, f"{cell_id}-candidate-label", used_seeds),
        )
        assert_transport_unchanged(trained.proposal, trained.proposal_sha256)
        candidate_record, candidate_summary = _record(
            "v15_cm_transport",
            candidate_eval.contribution,
            training_cost=trained.training_cost,
            evaluation_cost=candidate_eval.evaluation_cost,
            reference_mean=reference_mean,
            reference_se=reference_se,
            query_count=query_count,
        )
        likelihood_se = math.sqrt(
            float(torch.var(candidate_eval.likelihood, unbiased=True))
            / candidate_eval.likelihood.numel()
        )
        candidate_record["exactness"] = {
            "proposal_sha256": trained.proposal_sha256,
            "proposal_hash_unchanged": True,
            "maximum_likelihood_bound_violation": (
                candidate_eval.maximum_likelihood_bound_violation
            ),
            "likelihood_normalization_mean": candidate_eval.likelihood_normalization_mean,
            "likelihood_normalization_z": abs(
                candidate_eval.likelihood_normalization_mean - 1.0
            )
            / max(likelihood_se, torch.finfo(torch.float64).tiny),
        }
        candidate_record["mode_count"] = len(trained.modes.modes)
        candidate_record["best_action"] = trained.modes.modes[0].action_value
        candidate_record["architecture"] = {
            "version": "v16_hybrid_trace",
            "mode_basis": basis_kind,
            "basis_rank": trained.basis.rank,
            "asymptotic_safety_mass": float(
                candidate_config.get("asymptotic_safety_mass", 0.0)
            ),
            "conditional_adaptation": adaptation_config is not None,
            "adapt_covariance": (
                adaptation_config.adapt_covariance
                if adaptation_config is not None
                else None
            ),
            "adaptation_effective_sample_sizes": list(
                trained.adaptation_effective_sample_sizes
            ),
            "adaptation_tempering_powers": list(trained.adaptation_tempering_powers),
        }

        methods = [candidate_record]
        summaries: dict[str, Any] = {"v15_cm_transport": candidate_summary}

        natural_proposal = freeze_conditional_rbergomi_proposal(
            problem,
            training_seed=_seed(root_seed, f"{cell_id}-natural-freeze", used_seeds),
        )
        natural_batch, natural_cost = _measured_baseline(
            problem,
            natural_proposal,
            method="conditional_rbergomi",
            units=int(evaluation["iid_units"]),
            seed=_seed(root_seed, f"{cell_id}-natural-evaluation", used_seeds),
            rqmc_points=1,
        )
        record, summary = _record(
            "conditional_rbergomi",
            natural_batch.unit_contributions,
            training_cost=natural_proposal.training_cost,
            evaluation_cost=natural_cost,
            reference_mean=reference_mean,
            reference_se=reference_se,
            query_count=query_count,
        )
        methods.append(record)
        summaries["conditional_rbergomi"] = summary

        local_training = train_local_volterra_transport(
            problem,
            training_seed=_seed(root_seed, f"{cell_id}-v14-training", used_seeds),
            config=LocalVolterraTransportTrainingConfig(
                target_powers=tuple(float(x) for x in baseline_config["v14_target_powers"]),
                shifted_weights=tuple(float(x) for x in baseline_config["v14_shifted_weights"]),
                defensive_weight=float(baseline_config["defensive_mass"]),
                smc=AdaptiveResidualSMCConfig(
                    particles=int(baseline_config["v14_particles"]),
                    target_ess_fraction=0.7,
                    pcn_scale=0.25,
                    pcn_sweeps_per_stage=int(baseline_config["v14_pcn_sweeps"]),
                    maximum_stages=int(baseline_config["v14_maximum_stages"]),
                ),
            ),
        )
        local_eval = evaluate_local_volterra_transport(
            problem,
            local_training.proposal,
            sample_count=int(evaluation["iid_units"]),
            proposal_seed=_seed(root_seed, f"{cell_id}-v14-proposal", used_seeds),
            coordinate_seed=_seed(root_seed, f"{cell_id}-v14-coordinate", used_seeds),
        )
        record, summary = _record(
            "v14_local_volterra",
            local_eval.contribution,
            training_cost=local_training.proposal.training_cost,
            evaluation_cost=local_eval.evaluation_cost,
            reference_mean=reference_mean,
            reference_se=reference_se,
            query_count=query_count,
        )
        methods.append(record)
        summaries["v14_local_volterra"] = summary

        cem = train_cem_proposal(
            problem,
            method="defensive_cem",
            training_seed=_seed(root_seed, f"{cell_id}-cem-training", used_seeds),
            config=CEMTrainingConfig(
                iterations=int(baseline_config["cem_iterations"]),
                samples_per_iteration=int(baseline_config["cem_samples"]),
                defensive_weight=float(baseline_config["defensive_mass"]),
                time_bins=int(baseline_config["cem_time_bins"]),
            ),
        )
        cem_batch, cem_cost = _measured_baseline(
            problem,
            cem,
            method="defensive_cem",
            units=int(evaluation["iid_units"]),
            seed=_seed(root_seed, f"{cell_id}-cem-evaluation", used_seeds),
            rqmc_points=1,
        )
        record, summary = _record(
            "defensive_cem",
            cem_batch.unit_contributions,
            training_cost=cem.training_cost,
            evaluation_cost=cem_cost,
            reference_mean=reference_mean,
            reference_se=reference_se,
            query_count=query_count,
        )
        methods.append(record)
        summaries["defensive_cem"] = summary

        ld = train_large_deviation_proposal(
            problem,
            training_seed=_seed(root_seed, f"{cell_id}-ld-training", used_seeds),
            config=LargeDeviationTrainingConfig(
                optimizer_steps=int(baseline_config["ld_optimizer_steps"]),
                restarts=int(baseline_config["ld_restarts"]),
                defensive_weight=float(baseline_config["defensive_mass"]),
            ),
        )
        ld_batch, ld_cost = _measured_baseline(
            problem,
            ld,
            method="ld_subspace_is",
            units=int(evaluation["iid_units"]),
            seed=_seed(root_seed, f"{cell_id}-ld-evaluation", used_seeds),
            rqmc_points=1,
        )
        record, summary = _record(
            "ld_subspace_is",
            ld_batch.unit_contributions,
            training_cost=ld.training_cost,
            evaluation_cost=ld_cost,
            reference_mean=reference_mean,
            reference_se=reference_se,
            query_count=query_count,
        )
        methods.append(record)
        summaries["ld_subspace_is"] = summary

        rqmc = freeze_smoothing_rqmc_proposal(
            problem,
            training_seed=_seed(root_seed, f"{cell_id}-rqmc-freeze", used_seeds),
        )
        rqmc_batch, rqmc_cost = _measured_baseline(
            problem,
            rqmc,
            method="smoothing_rqmc",
            units=int(evaluation["rqmc_randomizations"]),
            seed=_seed(root_seed, f"{cell_id}-rqmc-evaluation", used_seeds),
            rqmc_points=int(evaluation["rqmc_points"]),
        )
        record, summary = _record(
            "smoothing_rqmc",
            rqmc_batch.unit_contributions,
            training_cost=rqmc.training_cost,
            evaluation_cost=rqmc_cost,
            reference_mean=reference_mean,
            reference_se=reference_se,
            query_count=query_count,
        )
        methods.append(record)
        summaries["smoothing_rqmc"] = summary

        accurate_primary = [
            item
            for item in methods
            if item["method"] in protocol.primary_comparators
            and item["accuracy_z"] <= float(config["gates"]["maximum_accuracy_z"])
            and item["sample_variance"] > 0.0
        ]
        ratios = {
            item["method"]: work_efficiency_ratio(
                summaries[item["method"]],
                candidate_summary,
                query_count=query_count,
            )
            for item in accurate_primary
        }
        best_ratio = min(ratios.values()) if ratios else 0.0
        result_cells.append(
            {
                "cell_id": cell_id,
                "reference": {
                    "estimate": reference_mean,
                    "sample_variance": reference_variance,
                    "standard_error": reference_se,
                    "inferential_units": reference_batch.unit_contributions.numel(),
                    "evaluation_cost": asdict(reference_cost),
                },
                "methods": methods,
                "accuracy_qualified_primary": [item["method"] for item in accurate_primary],
                "primary_over_v15_work_ratios": ratios,
                "best_primary_over_v15_work_ratio": best_ratio,
            }
        )
    relative_config_path = config_path.resolve().relative_to(ROOT)
    payload: dict[str, Any] = {
        "schema": "npi.g11.v15-experiment-result.v1",
        "stage": config["stage"],
        "protocol_id": config["protocol_id"],
        "config_binding": {
            "path": relative_config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": source_provenance,
        "claim_boundary": {
            "finite_grid_exact": True,
            "continuous_time_efficiency_proved_under_T16_5_scope": True,
            "joint_mesh_noise_efficiency_proved_under_T16_9_scope": True,
            "end_to_end_complexity_proved": False,
            "top_journal_claim_authorized": False,
        },
        "gates": config["gates"],
        "used_seeds": used_seeds,
        "cells": result_cells,
    }
    audit = audit_v15_result(payload, root=ROOT)
    payload["audit"] = asdict(audit)
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    payload, output = run(args.config.resolve())
    print(
        json.dumps(
            {
                "output": str(output),
                "audit": payload["audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
