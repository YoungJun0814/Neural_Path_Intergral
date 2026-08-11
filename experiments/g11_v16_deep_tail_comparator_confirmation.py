"""Large-cluster, reference-qualified comparator measurements for V16 deep tails."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g11_v15_development import _measured_baseline
from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines import (
    CEMTrainingConfig,
    LargeDeviationTrainingConfig,
    freeze_conditional_rbergomi_proposal,
    freeze_smoothing_rqmc_proposal,
    train_cem_proposal,
    train_large_deviation_proposal,
)
from src.path_integral.baselines.rbergomi_common import (
    BaselineUnitBatch,
    RBergomiBaselineProblem,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_local_volterra_transport import (
    LocalVolterraTransportTrainingConfig,
    evaluate_local_volterra_transport,
    train_local_volterra_transport,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance

ROOT = Path(__file__).resolve().parents[1]


def _load_bound(binding: dict[str, str]) -> dict[str, Any]:
    path = ROOT / binding["path"]
    if not path.is_file() or file_sha256(path) != binding["sha256"]:
        raise RuntimeError(f"artifact binding failed: {binding['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_clusters(
    *,
    method: str,
    training_cost: BaselineCostLedger,
    clusters: int,
    evaluate: Callable[[int], tuple[torch.Tensor, BaselineCostLedger]],
    reference_estimate: float,
    reference_standard_error: float,
    query_count: int,
) -> dict[str, Any]:
    count = 0
    total = 0.0
    total_square = 0.0
    evaluation_cost = BaselineCostLedger()
    cluster_means = []
    for cluster in range(clusters):
        values, cost = evaluate(cluster)
        if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
            raise RuntimeError(f"{method} returned invalid inferential units")
        count += values.numel()
        total += float(torch.sum(values))
        total_square += float(torch.sum(values.square()))
        cluster_means.append(float(torch.mean(values)))
        evaluation_cost = evaluation_cost.plus(cost)
    estimate = total / count
    variance = (total_square - total**2 / count) / (count - 1)
    standard_error = math.sqrt(max(variance, 0.0) / count)
    between_cluster_standard_error = math.sqrt(
        float(
            torch.var(
                torch.tensor(cluster_means, dtype=torch.float64),
                unbiased=True,
            )
        )
        / clusters
    )
    robust_standard_error = max(standard_error, between_cluster_standard_error)
    accuracy_z = abs(estimate - reference_estimate) / math.sqrt(
        robust_standard_error**2 + reference_standard_error**2
    )
    total_work = (
        training_cost.algorithmic_work_units
        + query_count * evaluation_cost.algorithmic_work_units
    )
    return {
        "method": method,
        "estimate": estimate,
        "sample_variance": variance,
        "standard_error": standard_error,
        "between_cluster_standard_error": between_cluster_standard_error,
        "robust_standard_error": robust_standard_error,
        "robust_relative_standard_error": robust_standard_error / abs(estimate),
        "accuracy_z": accuracy_z,
        "inferential_units": count,
        "training_cost": asdict(training_cost),
        "evaluation_cost": asdict(evaluation_cost),
        "total_work_at_primary_query_count": total_work,
        "work_normalized_variance": variance * total_work / count,
        "clusters": cluster_means,
    }


def _baseline_values(
    problem: RBergomiBaselineProblem,
    proposal: Any,
    *,
    method: str,
    units: int,
    seed: int,
    rqmc_points: int,
) -> tuple[torch.Tensor, BaselineCostLedger]:
    batch, cost = _measured_baseline(
        problem,
        proposal,
        method=method,
        units=units,
        seed=seed,
        rqmc_points=rqmc_points,
    )
    if not isinstance(batch, BaselineUnitBatch):
        raise RuntimeError("baseline evaluation returned the wrong batch type")
    return batch.unit_contributions, cost


def _baseline_cluster_evaluator(
    problem: RBergomiBaselineProblem,
    proposal: Any,
    *,
    method: str,
    units: int,
    seed: int,
    rqmc_points: int,
) -> Callable[[int], tuple[torch.Tensor, BaselineCostLedger]]:
    def evaluate(cluster: int) -> tuple[torch.Tensor, BaselineCostLedger]:
        return _baseline_values(
            problem,
            proposal,
            method=method,
            units=units,
            seed=seed + 1009 * cluster,
            rqmc_points=rqmc_points,
        )

    return evaluate


def _local_cluster_evaluator(
    problem: RBergomiBaselineProblem,
    proposal: Any,
    *,
    units: int,
    proposal_seed: int,
    coordinate_seed: int,
) -> Callable[[int], tuple[torch.Tensor, BaselineCostLedger]]:
    def evaluate(cluster: int) -> tuple[torch.Tensor, BaselineCostLedger]:
        result = evaluate_local_volterra_transport(
            problem,
            proposal,
            sample_count=units,
            proposal_seed=proposal_seed + 1009 * cluster,
            coordinate_seed=coordinate_seed + 1009 * cluster,
        )
        return result.contribution, result.evaluation_cost

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reference = _load_bound(config["artifacts"]["independent_reference"])
    references = {item["cell_id"]: item for item in reference["records"]}
    model = config["model"]
    sampling = config["sampling"]
    baseline = config["baselines"]
    gates = config["gates"]
    cells = []
    failures = []
    for cell_index, cell in enumerate(config["cells"]):
        cell_id = str(cell["cell_id"])
        external = references[cell_id]
        problem = RBergomiBaselineProblem(
            task_id=cell_id,
            task=TerminalThresholdTask(level=float(cell["threshold"])),
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            steps=int(model["steps"]),
            hurst=float(cell.get("hurst", model["hurst"])),
            eta=float(cell.get("eta", model["eta"])),
            xi=float(cell.get("xi", model["xi"])),
            rho=float(cell.get("rho", model["rho"])),
        )
        seed = int(config["seed"]) + 1_000_003 * cell_index
        clusters = int(sampling["clusters"])
        iid = int(sampling["iid_per_cluster"])
        query_count = int(sampling["primary_query_count"])
        reference_estimate = float(external["estimate"])
        reference_se = float(external["standard_error"])
        natural = freeze_conditional_rbergomi_proposal(
            problem,
            training_seed=seed + 1,
        )
        cem = train_cem_proposal(
            problem,
            method="defensive_cem",
            training_seed=seed + 2,
            config=CEMTrainingConfig(
                iterations=int(baseline["cem_iterations"]),
                samples_per_iteration=int(baseline["cem_samples"]),
                defensive_weight=float(baseline["defensive_mass"]),
                time_bins=int(baseline["cem_time_bins"]),
            ),
        )
        ld = train_large_deviation_proposal(
            problem,
            training_seed=seed + 3,
            config=LargeDeviationTrainingConfig(
                optimizer_steps=int(baseline["ld_optimizer_steps"]),
                restarts=int(baseline["ld_restarts"]),
                defensive_weight=float(baseline["defensive_mass"]),
            ),
        )
        rqmc = freeze_smoothing_rqmc_proposal(problem, training_seed=seed + 4)
        local = train_local_volterra_transport(
            problem,
            training_seed=seed + 5,
            config=LocalVolterraTransportTrainingConfig(
                target_powers=tuple(float(x) for x in baseline["v14_target_powers"]),
                shifted_weights=tuple(float(x) for x in baseline["v14_shifted_weights"]),
                defensive_weight=float(baseline["defensive_mass"]),
                smc=AdaptiveResidualSMCConfig(
                    particles=int(baseline["v14_particles"]),
                    target_ess_fraction=0.7,
                    pcn_scale=0.25,
                    pcn_sweeps_per_stage=int(baseline["v14_pcn_sweeps"]),
                    maximum_stages=int(baseline["v14_maximum_stages"]),
                ),
            ),
        )
        methods = []
        methods.append(
            _aggregate_clusters(
                method="conditional_rbergomi",
                training_cost=natural.training_cost,
                clusters=clusters,
                evaluate=_baseline_cluster_evaluator(
                    problem,
                    natural,
                    method="conditional_rbergomi",
                    units=iid,
                    seed=seed + 100_003,
                    rqmc_points=1,
                ),
                reference_estimate=reference_estimate,
                reference_standard_error=reference_se,
                query_count=query_count,
            )
        )
        methods.append(
            _aggregate_clusters(
                method="v14_local_volterra",
                training_cost=local.proposal.training_cost,
                clusters=clusters,
                evaluate=_local_cluster_evaluator(
                    problem,
                    local.proposal,
                    units=iid,
                    proposal_seed=seed + 200_003,
                    coordinate_seed=seed + 300_003,
                ),
                reference_estimate=reference_estimate,
                reference_standard_error=reference_se,
                query_count=query_count,
            )
        )
        for offset, name, proposal in (
            (400_003, "defensive_cem", cem),
            (500_003, "ld_subspace_is", ld),
        ):
            methods.append(
                _aggregate_clusters(
                    method=name,
                    training_cost=proposal.training_cost,
                    clusters=clusters,
                    evaluate=_baseline_cluster_evaluator(
                        problem,
                        proposal,
                        method=name,
                        units=iid,
                        seed=seed + offset,
                        rqmc_points=1,
                    ),
                    reference_estimate=reference_estimate,
                    reference_standard_error=reference_se,
                    query_count=query_count,
                )
            )
        methods.append(
            _aggregate_clusters(
                method="smoothing_rqmc",
                training_cost=rqmc.training_cost,
                clusters=clusters,
                evaluate=_baseline_cluster_evaluator(
                    problem,
                    rqmc,
                    method="smoothing_rqmc",
                    units=int(sampling["rqmc_randomizations_per_cluster"]),
                    seed=seed + 600_003,
                    rqmc_points=int(sampling["rqmc_points"]),
                ),
                reference_estimate=reference_estimate,
                reference_standard_error=reference_se,
                query_count=query_count,
            )
        )
        qualified = [
            item
            for item in methods
            if item["sample_variance"] > 0.0
            and item["accuracy_z"] <= float(gates["maximum_accuracy_z"])
            and item["robust_relative_standard_error"]
            <= float(gates["maximum_robust_relative_standard_error"])
        ]
        if len(qualified) < int(gates["minimum_qualified_comparators"]):
            failures.append(f"{cell_id}: too few accuracy-qualified comparators")
        cells.append(
            {
                "cell_id": cell_id,
                "reference": external,
                "methods": methods,
                "accuracy_qualified_comparators": [
                    item["method"] for item in qualified
                ],
            }
        )
    payload = {
        "schema": "npi.g11.v16-deep-tail-comparator-confirmation.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "artifact_bindings": config["artifacts"],
        "source_provenance": git_source_provenance(ROOT),
        "passed": not failures,
        "failures": failures,
        "cells": cells,
        "claim_boundary": {
            "large_cluster_variance_measurement": True,
            "proposal_frozen_across_clusters": True,
            "independent_reference": True,
            "candidate_transport_used": False,
        },
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": payload["passed"]}, indent=2))


if __name__ == "__main__":
    main()
