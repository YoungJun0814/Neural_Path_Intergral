"""Large-cluster development of V16 transports in failed deep-tail regimes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.cameron_martin_modes import ActionSolverConfig, ModeSearchConfig
from src.path_integral.conditional_transport_adaptation import (
    ConditionalTransportAdaptationConfig,
)
from src.path_integral.finite_rank_gaussian_transport import CurvatureTransportConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import (
    evaluate_rbergomi_cm_transport,
    train_rbergomi_cm_transport,
)
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance

ROOT = Path(__file__).resolve().parents[1]


def _load_bound(binding: dict[str, str]) -> dict[str, Any]:
    path = ROOT / binding["path"]
    if not path.is_file() or file_sha256(path) != binding["sha256"]:
        raise RuntimeError(f"artifact binding failed: {binding['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _method_wnv(method: dict[str, Any]) -> float:
    return (
        float(method["sample_variance"])
        * float(method["total_work_at_primary_query_count"])
        / int(method["inferential_units"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline = _load_bound(config["artifacts"]["baseline"])
    reference = _load_bound(config["artifacts"]["independent_reference"])
    baseline_cells = {item["cell_id"]: item for item in baseline["cells"]}
    reference_cells = {item["cell_id"]: item for item in reference["records"]}
    solver_values = config["mode_solver"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    model = config["model"]
    records = []
    failures = []
    for variant in config["variants"]:
        variant_id = str(variant["id"])
        reference_cell = str(variant["reference_cell"])
        external = reference_cells[reference_cell]
        problem = RBergomiBaselineProblem(
            task_id=variant_id,
            task=TerminalThresholdTask(level=float(variant["threshold"])),
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            steps=int(model["steps"]),
            hurst=float(variant["hurst"]),
            eta=float(variant.get("eta", model["eta"])),
            xi=float(variant.get("xi", model["xi"])),
            rho=float(variant.get("rho", model["rho"])),
        )
        basis = build_mesh_compatible_blp_hybrid_basis(
            steps=problem.steps,
            maturity=problem.maturity,
            hurst=problem.hurst,
            drift_modes=int(variant["drift_modes"]),
            bridge_modes=int(variant["bridge_modes"]),
        )
        adaptation = variant["adaptation"]
        seed = int(variant["seed"])
        trained = train_rbergomi_cm_transport(
            problem,
            basis=basis,
            adaptation_config=ConditionalTransportAdaptationConfig(
                iterations=int(adaptation["iterations"]),
                samples_per_iteration=int(adaptation["samples"]),
                smoothing=0.7,
                minimum_ess_fraction=float(adaptation["minimum_ess_fraction"]),
                minimum_variance=0.05,
                maximum_variance=20.0,
                adapt_covariance=bool(adaptation["adapt_covariance"]),
                final_components=int(adaptation["final_components"]),
            ),
            adaptation_seed=seed + 1,
            mode_search=ModeSearchConfig(
                methods=("lbfgs", "trust-ncg"),
                random_starts=int(solver_values["random_starts"]),
                random_seed=seed,
                start_scale=float(solver_values["start_scale"]),
                solver=ActionSolverConfig(
                    maximum_iterations=int(solver_values["maximum_iterations"]),
                    gradient_tolerance=float(solver_values["gradient_tolerance"]),
                ),
            ),
            transport_config=CurvatureTransportConfig(
                defensive_mass=float(variant["defensive_mass"]),
                asymptotic_safety_mass=float(variant["safety_mass"]),
                safety_spectrum_decay=2.0,
                safety_spectrum_scale=4.0,
                safety_complement_decay=2.0,
            ),
        )
        total_count = 0
        contribution_sum = 0.0
        contribution_square_sum = 0.0
        likelihood_sum = 0.0
        likelihood_square_sum = 0.0
        evaluation_work = 0.0
        maximum_bound_violation = 0.0
        cluster_records = []
        for cluster in range(int(evaluation["clusters"])):
            evaluated = evaluate_rbergomi_cm_transport(
                problem,
                trained.proposal,
                sample_count=int(evaluation["samples_per_cluster"]),
                path_seed=seed + 1_000_003 + 1009 * cluster,
                label_seed=seed + 2_000_003 + 1009 * cluster,
            )
            values = evaluated.contribution
            likelihood = evaluated.likelihood
            count = values.numel()
            cluster_mean = float(torch.mean(values))
            cluster_variance = float(torch.var(values, unbiased=True))
            cluster_se = math.sqrt(cluster_variance / count)
            cluster_z = abs(cluster_mean - float(external["estimate"])) / math.sqrt(
                cluster_se**2 + float(external["standard_error"]) ** 2
            )
            cluster_records.append(
                {
                    "cluster": cluster,
                    "estimate": cluster_mean,
                    "standard_error": cluster_se,
                    "external_accuracy_z": cluster_z,
                }
            )
            total_count += count
            contribution_sum += float(torch.sum(values))
            contribution_square_sum += float(torch.sum(values.square()))
            likelihood_sum += float(torch.sum(likelihood))
            likelihood_square_sum += float(torch.sum(likelihood.square()))
            evaluation_work += evaluated.evaluation_cost.algorithmic_work_units
            maximum_bound_violation = max(
                maximum_bound_violation,
                evaluated.maximum_likelihood_bound_violation,
            )
        estimate = contribution_sum / total_count
        variance = (
            contribution_square_sum - contribution_sum**2 / total_count
        ) / (total_count - 1)
        standard_error = math.sqrt(variance / total_count)
        likelihood_mean = likelihood_sum / total_count
        likelihood_variance = (
            likelihood_square_sum - likelihood_sum**2 / total_count
        ) / (total_count - 1)
        likelihood_z = abs(likelihood_mean - 1.0) / math.sqrt(
            likelihood_variance / total_count
        )
        accuracy_z = abs(estimate - float(external["estimate"])) / math.sqrt(
            standard_error**2 + float(external["standard_error"]) ** 2
        )
        total_work = (
            trained.training_cost.algorithmic_work_units
            + int(evaluation["primary_query_count"]) * evaluation_work
        )
        candidate_wnv = variance * total_work / total_count
        baseline_methods = {
            item["method"]: item for item in baseline_cells[reference_cell]["methods"]
        }
        v14_ratio = _method_wnv(baseline_methods["v14_local_volterra"]) / candidate_wnv
        variant_failures = []
        if accuracy_z > float(gates["maximum_accuracy_z"]):
            variant_failures.append("external accuracy")
        if max(item["external_accuracy_z"] for item in cluster_records) > float(
            gates["maximum_cluster_accuracy_z"]
        ):
            variant_failures.append("cluster accuracy")
        if standard_error / abs(estimate) > float(
            gates["maximum_relative_standard_error"]
        ):
            variant_failures.append("relative standard error")
        if likelihood_z > float(gates["maximum_likelihood_normalization_z"]):
            variant_failures.append("likelihood normalization")
        if maximum_bound_violation > float(
            gates["maximum_likelihood_bound_violation"]
        ):
            variant_failures.append("likelihood bound")
        if v14_ratio < float(gates["minimum_v14_over_candidate_work_ratio"]):
            variant_failures.append("V14 work ratio")
        records.append(
            {
                "variant_id": variant_id,
                "reference_cell": reference_cell,
                "passed": not variant_failures,
                "failures": variant_failures,
                "estimate": estimate,
                "sample_variance": variance,
                "standard_error": standard_error,
                "relative_standard_error": standard_error / abs(estimate),
                "external_reference_estimate": float(external["estimate"]),
                "external_reference_standard_error": float(external["standard_error"]),
                "external_accuracy_z": accuracy_z,
                "maximum_cluster_accuracy_z": max(
                    item["external_accuracy_z"] for item in cluster_records
                ),
                "likelihood_normalization_mean": likelihood_mean,
                "likelihood_normalization_z": likelihood_z,
                "maximum_likelihood_bound_violation": maximum_bound_violation,
                "training_work": trained.training_cost.algorithmic_work_units,
                "evaluation_work": evaluation_work,
                "total_work_at_primary_query_count": total_work,
                "work_normalized_variance": candidate_wnv,
                "v14_over_candidate_work_ratio": v14_ratio,
                "adaptation_effective_sample_sizes": list(
                    trained.adaptation_effective_sample_sizes
                ),
                "adaptation_tempering_powers": list(
                    trained.adaptation_tempering_powers
                ),
                "clusters": cluster_records,
            }
        )
    passing_by_cell: dict[str, list[str]] = {}
    for record in records:
        if record["passed"]:
            passing_by_cell.setdefault(str(record["reference_cell"]), []).append(
                str(record["variant_id"])
            )
    for target in {str(item["reference_cell"]) for item in config["variants"]}:
        if target not in passing_by_cell:
            failures.append(f"{target}: no variant passed all gates")
    payload = {
        "schema": "npi.g11.v16-deep-tail-transport-development.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "artifact_bindings": config["artifacts"],
        "source_provenance": git_source_provenance(ROOT),
        "passed": not failures,
        "failures": failures,
        "passing_variants_by_cell": passing_by_cell,
        "variants": records,
        "claim_boundary": {
            "development_selection_only": True,
            "independent_evaluation_clusters": True,
            "proposal_frozen_across_clusters": True,
            "fresh_confirmation_required_after_selection": True,
        },
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": payload["passed"]}, indent=2))


if __name__ == "__main__":
    main()
