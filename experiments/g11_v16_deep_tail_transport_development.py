"""Large-cluster development of V16 transports in failed deep-tail regimes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_hybrid_basis,
    build_mesh_compatible_blp_trace_safety_geometry,
)
from src.path_integral.cameron_martin_basis import CameronMartinBasis
from src.path_integral.cameron_martin_modes import ActionSolverConfig, ModeSearchConfig
from src.path_integral.conditional_transport_adaptation import (
    ConditionalTransportAdaptationConfig,
    adapt_conditional_transport,
)
from src.path_integral.finite_rank_gaussian_transport import (
    CurvatureTransportConfig,
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
    build_trace_class_small_noise_safety_component,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import (
    evaluate_rbergomi_cm_transport,
    train_rbergomi_cm_transport,
)
from src.path_integral.rbergomi_local_volterra_transport import (
    LocalVolterraTransportTrainingConfig,
    train_local_volterra_transport,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig
from src.path_integral.tempered_conditional_smc import TemperedSMCConfig
from src.path_integral.tempered_target_transport import (
    TemperedTargetTransportConfig,
    fit_tempered_target_transport,
)
from src.path_integral.v14_tempered_hybrid_transport import (
    V14TemperedHybridConfig,
    convert_local_proposal,
    fit_v14_tempered_hybrid_transport,
)
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance
from src.path_integral.v16_transport_policy import (
    route_v16_hybrid_v2,
    route_v16_hybrid_v3,
    route_v16_hybrid_v4,
)

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


def _qualify_comparator(
    method: dict[str, Any],
    *,
    reference_estimate: float,
    reference_standard_error: float,
    maximum_accuracy_z: float,
) -> dict[str, Any]:
    variance = float(method["sample_variance"])
    units = int(method["inferential_units"])
    method_standard_error = (
        math.sqrt(variance / units) if variance >= 0.0 and units > 0 else math.inf
    )
    accuracy_z = abs(float(method["estimate"]) - reference_estimate) / math.sqrt(
        method_standard_error**2 + reference_standard_error**2
    )
    qualified = (
        variance > 0.0
        and units > 1
        and math.isfinite(accuracy_z)
        and accuracy_z <= maximum_accuracy_z
    )
    return {
        "method": str(method["method"]),
        "qualified": qualified,
        "estimate": float(method["estimate"]),
        "standard_error": method_standard_error,
        "external_accuracy_z": accuracy_z,
        "work_normalized_variance": _method_wnv(method) if qualified else None,
    }


def _v14_seeded_transport(
    problem: RBergomiBaselineProblem,
    basis: CameronMartinBasis,
    variant: dict[str, Any],
    *,
    seed: int,
    adaptation_config: ConditionalTransportAdaptationConfig,
) -> tuple[DefensiveFiniteRankGaussianMixture, float, tuple[float, ...], tuple[float, ...]]:
    initializer = variant["v14_initializer"]
    safety_mass = float(variant["safety_mass"])
    local_defensive_mass = float(variant["defensive_mass"]) / (1.0 - safety_mass)
    if local_defensive_mass >= 1.0:
        raise ValueError("V14-seeded defensive and safety masses leave no shifted mass")
    local = train_local_volterra_transport(
        problem,
        training_seed=seed,
        config=LocalVolterraTransportTrainingConfig(
            target_powers=tuple(float(value) for value in initializer["target_powers"]),
            shifted_weights=tuple(
                float(value) for value in initializer["shifted_weights"]
            ),
            defensive_weight=local_defensive_mass,
            smc=AdaptiveResidualSMCConfig(
                particles=int(initializer["particles"]),
                target_ess_fraction=0.7,
                pcn_scale=0.25,
                pcn_sweeps_per_stage=int(initializer["pcn_sweeps"]),
                maximum_stages=int(initializer["maximum_stages"]),
            ),
        ),
    )
    components = [
        FiniteRankGaussianComponent(
            mean=torch.tensor(mean, dtype=torch.float64),
            directions=torch.empty((problem.local_dimension, 0), dtype=torch.float64),
            variance_eigenvalues=torch.empty(0, dtype=torch.float64),
        )
        for mean in local.proposal.component_means
    ]
    directions, spectrum = build_mesh_compatible_blp_trace_safety_geometry(
        steps=problem.steps,
        maturity=problem.maturity,
        hurst=problem.hurst,
        spectrum_decay=2.0,
        spectrum_scale=4.0,
        complement_decay=2.0,
    )
    safety = build_trace_class_small_noise_safety_component(
        directions,
        spectrum,
        epsilon=1.0,
    )
    original_weights = torch.tensor(local.proposal.component_weights, dtype=torch.float64)
    weights = torch.cat(
        (
            (1.0 - safety_mass) * original_weights[:1],
            torch.tensor([safety_mass], dtype=torch.float64),
            (1.0 - safety_mass) * original_weights[1:],
        )
    )
    initial = DefensiveFiniteRankGaussianMixture(
        components=(components[0], safety, *components[1:]),
        weights=weights,
    )
    adapted = adapt_conditional_transport(
        problem,
        basis,
        initial,
        epsilon=1.0,
        seed=seed + 1,
        config=adaptation_config,
    )
    training_work = (
        local.proposal.training_cost.algorithmic_work_units
        + adapted.training_cost.algorithmic_work_units
    )
    return (
        adapted.proposal,
        training_work,
        adapted.effective_sample_sizes,
        adapted.tempering_powers,
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
        variant = dict(variant)
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
        routing_policy = variant.get("routing_policy")
        routing_regime = None
        if routing_policy is not None:
            if routing_policy not in {
                "v16_hybrid_routing_v2",
                "v16_hybrid_routing_v3",
                "v16_hybrid_routing_v4",
            }:
                raise ValueError(f"unsupported routing policy: {routing_policy}")
            if routing_policy == "v16_hybrid_routing_v2":
                route = route_v16_hybrid_v2(problem)
            elif routing_policy == "v16_hybrid_routing_v3":
                route = route_v16_hybrid_v3(problem)
            else:
                route = route_v16_hybrid_v4(problem)
            routing_regime = route.regime
            variant.update(
                {
                    "initializer": route.initializer,
                    "drift_modes": route.drift_modes,
                    "bridge_modes": route.bridge_modes,
                    "defensive_mass": route.defensive_mass,
                    "safety_mass": route.safety_mass,
                }
            )
            if route.initializer == "tempered_smc":
                variant["tempered_smc"] = {
                    "particles": route.tempered_particles,
                    "temperature_stages": route.tempered_temperature_stages,
                    "temperature_power": route.tempered_temperature_power,
                    "mutation_steps": route.tempered_mutation_steps,
                    "pcn_scale": route.tempered_pcn_scale,
                    "replicates": route.tempered_replicates,
                    "components": route.final_components,
                    "clustering": route.tempered_clustering,
                    "kmeans_iterations": route.tempered_kmeans_iterations,
                }
            else:
                variant["adaptation"] = {
                    "iterations": route.adaptation_iterations,
                    "samples": route.adaptation_samples,
                    "minimum_ess_fraction": 0.05,
                    "adapt_covariance": route.adapt_covariance,
                    "final_components": route.final_components,
                }
        basis = build_mesh_compatible_blp_hybrid_basis(
            steps=problem.steps,
            maturity=problem.maturity,
            hurst=problem.hurst,
            drift_modes=int(variant["drift_modes"]),
            bridge_modes=int(variant["bridge_modes"]),
        )
        seed = int(variant["seed"])
        initializer = str(variant.get("initializer", "cm_action"))
        initializer_diagnostics: dict[str, Any] = {}
        adaptation_config = None
        if initializer in {"cm_action", "v14_local"}:
            adaptation = variant["adaptation"]
            adaptation_config = ConditionalTransportAdaptationConfig(
                iterations=int(adaptation["iterations"]),
                samples_per_iteration=int(adaptation["samples"]),
                smoothing=0.7,
                minimum_ess_fraction=float(adaptation["minimum_ess_fraction"]),
                minimum_variance=0.05,
                maximum_variance=20.0,
                adapt_covariance=bool(adaptation["adapt_covariance"]),
                final_components=int(adaptation["final_components"]),
            )
        if initializer == "cm_action" and adaptation_config is not None:
            trained = train_rbergomi_cm_transport(
                problem,
                basis=basis,
                adaptation_config=adaptation_config,
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
            proposal = trained.proposal
            training_work = trained.training_cost.algorithmic_work_units
            effective_sample_sizes = trained.adaptation_effective_sample_sizes
            tempering_powers = trained.adaptation_tempering_powers
        elif initializer == "v14_local" and adaptation_config is not None:
            (
                proposal,
                training_work,
                effective_sample_sizes,
                tempering_powers,
            ) = _v14_seeded_transport(
                problem,
                basis,
                variant,
                seed=seed,
                adaptation_config=adaptation_config,
            )
        elif initializer == "tempered_smc":
            tempered = variant["tempered_smc"]
            stages = int(tempered["temperature_stages"])
            power = float(tempered["temperature_power"])
            fitted = fit_tempered_target_transport(
                problem,
                basis,
                config=TemperedTargetTransportConfig(
                    smc=TemperedSMCConfig(
                        particles=int(tempered["particles"]),
                        temperatures=tuple(
                            (index / stages) ** power
                            for index in range(stages + 1)
                        ),
                        mutation_steps=int(tempered["mutation_steps"]),
                        pcn_scale=float(tempered["pcn_scale"]),
                        replicates=int(tempered["replicates"]),
                        seed=seed,
                        retain_final_particles=True,
                    ),
                    defensive_mass=float(variant["defensive_mass"]),
                    safety_mass=float(variant["safety_mass"]),
                    components=int(tempered["components"]),
                    clustering=cast(
                        Literal["pca_quantile", "kmeans", "replicate_kmeans"],
                        str(tempered.get("clustering", "pca_quantile")),
                    ),
                    kmeans_iterations=int(tempered.get("kmeans_iterations", 25)),
                    covariance_scales=tuple(
                        float(value)
                        for value in tempered.get("covariance_scales", [1.0])
                    ),
                ),
            )
            proposal = fitted.proposal
            training_work = fitted.training_cost.algorithmic_work_units
            effective_sample_sizes = (
                fitted.minimum_incremental_ess_fraction
                * int(tempered["particles"]),
            )
            tempering_powers = (1.0,)
            initializer_diagnostics = {
                "tempered_normalizer_estimate": fitted.normalizer_estimate,
                "tempered_normalizer_standard_error": (
                    fitted.normalizer_standard_error
                ),
                "tempered_minimum_incremental_ess_fraction": (
                    fitted.minimum_incremental_ess_fraction
                ),
                "tempered_mutation_acceptance_rate": (
                    fitted.mutation_acceptance_rate
                ),
                "tempered_fitted_particle_count": fitted.fitted_particle_count,
            }
        elif initializer == "v14_tempered_hybrid":
            tempered = variant["tempered_smc"]
            stages = int(tempered["temperature_stages"])
            power = float(tempered["temperature_power"])
            local_values = variant["v14_initializer"]
            hybrid = fit_v14_tempered_hybrid_transport(
                problem,
                basis,
                target_seed=seed,
                local_seed=seed + 1,
                config=V14TemperedHybridConfig(
                    target=TemperedTargetTransportConfig(
                        smc=TemperedSMCConfig(
                            particles=int(tempered["particles"]),
                            temperatures=tuple(
                                (index / stages) ** power
                                for index in range(stages + 1)
                            ),
                            mutation_steps=int(tempered["mutation_steps"]),
                            pcn_scale=float(tempered["pcn_scale"]),
                            replicates=int(tempered["replicates"]),
                            seed=seed,
                            retain_final_particles=True,
                        ),
                        defensive_mass=float(variant["defensive_mass"]),
                        safety_mass=float(variant["safety_mass"]),
                        components=int(tempered["components"]),
                        clustering=cast(
                            Literal[
                                "pca_quantile",
                                "kmeans",
                                "replicate_kmeans",
                            ],
                            str(tempered.get("clustering", "pca_quantile")),
                        ),
                        kmeans_iterations=int(
                            tempered.get("kmeans_iterations", 25)
                        ),
                        covariance_scales=tuple(
                            float(value)
                            for value in tempered.get(
                                "covariance_scales",
                                [1.0],
                            )
                        ),
                    ),
                    local=LocalVolterraTransportTrainingConfig(
                        target_powers=tuple(
                            float(value) for value in local_values["target_powers"]
                        ),
                        shifted_weights=tuple(
                            float(value)
                            for value in local_values["shifted_weights"]
                        ),
                        defensive_weight=float(local_values["defensive_mass"]),
                        replicates_per_power=int(
                            local_values.get("replicates_per_power", 1)
                        ),
                        replicate_aggregation=cast(
                            Literal["average", "mixture"],
                            str(
                                local_values.get(
                                    "replicate_aggregation", "average"
                                )
                            ),
                        ),
                        smc=AdaptiveResidualSMCConfig(
                            particles=int(local_values["particles"]),
                            target_ess_fraction=0.7,
                            pcn_scale=0.25,
                            pcn_sweeps_per_stage=int(local_values["pcn_sweeps"]),
                            maximum_stages=int(local_values["maximum_stages"]),
                        ),
                    ),
                    target_mass=float(variant["target_mass"]),
                ),
            )
            proposal = hybrid.proposal
            training_work = hybrid.training_cost.algorithmic_work_units
            effective_sample_sizes = (
                hybrid.target.minimum_incremental_ess_fraction
                * int(tempered["particles"]),
            )
            tempering_powers = (1.0,)
            initializer_diagnostics = {
                "target_mass": hybrid.target_mass,
                "local_component_count": hybrid.local_component_count,
                "tempered_normalizer_estimate": (
                    hybrid.target.normalizer_estimate
                ),
                "tempered_normalizer_standard_error": (
                    hybrid.target.normalizer_standard_error
                ),
                "tempered_minimum_incremental_ess_fraction": (
                    hybrid.target.minimum_incremental_ess_fraction
                ),
                "tempered_mutation_acceptance_rate": (
                    hybrid.target.mutation_acceptance_rate
                ),
                "tempered_fitted_particle_count": (
                    hybrid.target.fitted_particle_count
                ),
            }
        elif initializer == "v14_only":
            local_values = variant["v14_initializer"]
            local = train_local_volterra_transport(
                problem,
                training_seed=seed,
                config=LocalVolterraTransportTrainingConfig(
                    target_powers=tuple(
                        float(value) for value in local_values["target_powers"]
                    ),
                    shifted_weights=tuple(
                        float(value) for value in local_values["shifted_weights"]
                    ),
                    defensive_weight=float(local_values["defensive_mass"]),
                    replicates_per_power=int(
                        local_values.get("replicates_per_power", 1)
                    ),
                    replicate_aggregation=cast(
                        Literal["average", "mixture"],
                        str(
                            local_values.get(
                                "replicate_aggregation", "average"
                            )
                        ),
                    ),
                    smc=AdaptiveResidualSMCConfig(
                        particles=int(local_values["particles"]),
                        target_ess_fraction=0.7,
                        pcn_scale=0.25,
                        pcn_sweeps_per_stage=int(local_values["pcn_sweeps"]),
                        maximum_stages=int(local_values["maximum_stages"]),
                    ),
                ),
            )
            proposal = convert_local_proposal(
                local.proposal.component_means,
                local.proposal.component_weights,
            )
            training_work = local.proposal.training_cost.algorithmic_work_units
            effective_sample_sizes = ()
            tempering_powers = tuple(
                float(value) for value in local_values["target_powers"]
            )
            initializer_diagnostics = {
                "local_component_count": len(proposal.components),
                "local_mean_norms": list(local.mean_norms),
                "local_replicate_mean_minimum_cosines": list(
                    local.replicate_mean_minimum_cosines
                ),
            }
        else:
            raise ValueError(f"unsupported initializer: {initializer}")
        total_count = 0
        contribution_sum = 0.0
        contribution_square_sum = 0.0
        likelihood_sum = 0.0
        likelihood_square_sum = 0.0
        evaluation_work = 0.0
        maximum_bound_violation = 0.0
        cluster_records = []
        clusters = int(variant.get("clusters", evaluation["clusters"]))
        for cluster in range(clusters):
            evaluated = evaluate_rbergomi_cm_transport(
                problem,
                proposal,
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
        cluster_estimates = torch.tensor(
            [item["estimate"] for item in cluster_records],
            dtype=torch.float64,
        )
        between_cluster_standard_error = math.sqrt(
            float(torch.var(cluster_estimates, unbiased=True)) / clusters
        )
        robust_standard_error = max(standard_error, between_cluster_standard_error)
        cluster_to_pooled_se_ratio = between_cluster_standard_error / max(
            standard_error,
            float.fromhex("0x1.0p-1022"),
        )
        likelihood_mean = likelihood_sum / total_count
        likelihood_variance = (
            likelihood_square_sum - likelihood_sum**2 / total_count
        ) / (total_count - 1)
        likelihood_z = abs(likelihood_mean - 1.0) / math.sqrt(
            likelihood_variance / total_count
        )
        accuracy_z = abs(estimate - float(external["estimate"])) / math.sqrt(
            robust_standard_error**2 + float(external["standard_error"]) ** 2
        )
        total_work = (
            training_work
            + int(evaluation["primary_query_count"]) * evaluation_work
        )
        candidate_wnv = variance * total_work / total_count
        baseline_methods = {
            item["method"]: item for item in baseline_cells[reference_cell]["methods"]
        }
        maximum_comparator_accuracy_z = float(
            gates.get("maximum_comparator_accuracy_z", gates["maximum_accuracy_z"])
        )
        comparator_records = [
            _qualify_comparator(
                item,
                reference_estimate=float(external["estimate"]),
                reference_standard_error=float(external["standard_error"]),
                maximum_accuracy_z=maximum_comparator_accuracy_z,
            )
            for name, item in baseline_methods.items()
            if name != "v15_cm_transport"
        ]
        qualified_comparators = [
            item for item in comparator_records if item["qualified"]
        ]
        comparator_ratios = {
            str(item["method"]): float(item["work_normalized_variance"])
            / candidate_wnv
            for item in qualified_comparators
        }
        strongest_comparator_ratio = (
            min(comparator_ratios.values()) if comparator_ratios else 0.0
        )
        v14_record = next(
            item
            for item in comparator_records
            if item["method"] == "v14_local_volterra"
        )
        v14_ratio = comparator_ratios.get("v14_local_volterra")
        variant_failures = []
        if accuracy_z > float(gates["maximum_accuracy_z"]):
            variant_failures.append("external accuracy")
        if "maximum_cluster_accuracy_z" in gates and max(
            item["external_accuracy_z"] for item in cluster_records
        ) > float(gates["maximum_cluster_accuracy_z"]):
            variant_failures.append("cluster accuracy")
        if "maximum_cluster_to_pooled_se_ratio" in gates and (
            cluster_to_pooled_se_ratio
            > float(gates["maximum_cluster_to_pooled_se_ratio"])
        ):
            variant_failures.append("cluster/pooled SE consistency")
        maximum_relative_se = float(
            gates.get(
                "maximum_robust_relative_standard_error",
                gates.get("maximum_relative_standard_error", math.inf),
            )
        )
        if robust_standard_error / abs(estimate) > maximum_relative_se:
            variant_failures.append("relative standard error")
        if likelihood_z > float(gates["maximum_likelihood_normalization_z"]):
            variant_failures.append("likelihood normalization")
        if maximum_bound_violation > float(
            gates["maximum_likelihood_bound_violation"]
        ):
            variant_failures.append("likelihood bound")
        minimum_qualified = int(gates.get("minimum_qualified_comparators", 1))
        if len(qualified_comparators) < minimum_qualified:
            variant_failures.append("too few accuracy-qualified comparators")
        if "minimum_strongest_comparator_over_candidate_work_ratio" in gates and (
            strongest_comparator_ratio
            < float(
                gates["minimum_strongest_comparator_over_candidate_work_ratio"]
            )
        ):
            variant_failures.append("strongest comparator work ratio")
        if "minimum_v14_over_candidate_work_ratio" in gates:
            if not v14_record["qualified"]:
                variant_failures.append("V14 comparator accuracy")
            elif v14_ratio is None or v14_ratio < float(
                gates["minimum_v14_over_candidate_work_ratio"]
            ):
                variant_failures.append("V14 work ratio")
        records.append(
            {
                "variant_id": variant_id,
                "selection_family": str(
                    variant.get(
                        "selection_family",
                        (
                            variant_id.rsplit("-", maxsplit=1)[0]
                            if variant_id.rsplit("-", maxsplit=1)[-1]
                            in {"a", "b"}
                            else variant_id
                        ),
                    )
                ),
                "reference_cell": reference_cell,
                "passed": not variant_failures,
                "failures": variant_failures,
                "estimate": estimate,
                "sample_variance": variance,
                "standard_error": standard_error,
                "between_cluster_standard_error": between_cluster_standard_error,
                "robust_standard_error": robust_standard_error,
                "cluster_to_pooled_se_ratio": cluster_to_pooled_se_ratio,
                "relative_standard_error": standard_error / abs(estimate),
                "robust_relative_standard_error": robust_standard_error
                / abs(estimate),
                "external_reference_estimate": float(external["estimate"]),
                "external_reference_standard_error": float(external["standard_error"]),
                "external_accuracy_z": accuracy_z,
                "maximum_cluster_accuracy_z": max(
                    item["external_accuracy_z"] for item in cluster_records
                ),
                "likelihood_normalization_mean": likelihood_mean,
                "likelihood_normalization_z": likelihood_z,
                "maximum_likelihood_bound_violation": maximum_bound_violation,
                "initializer": initializer,
                "routing_policy": routing_policy,
                "routing_regime": routing_regime,
                "initializer_diagnostics": initializer_diagnostics,
                "training_work": training_work,
                "evaluation_work": evaluation_work,
                "total_work_at_primary_query_count": total_work,
                "work_normalized_variance": candidate_wnv,
                "v14_over_candidate_work_ratio": v14_ratio,
                "qualified_comparators": [
                    item["method"] for item in qualified_comparators
                ],
                "comparator_diagnostics": comparator_records,
                "comparator_over_candidate_work_ratios": comparator_ratios,
                "strongest_comparator_over_candidate_work_ratio": (
                    strongest_comparator_ratio
                ),
                "adaptation_effective_sample_sizes": list(
                    effective_sample_sizes
                ),
                "adaptation_tempering_powers": list(
                    tempering_powers
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
    family_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["reference_cell"]), str(record["selection_family"]))
        family_records.setdefault(key, []).append(record)
    passing_families_by_cell: dict[str, list[str]] = {}
    minimum_family_size = int(gates.get("minimum_family_size", 1))
    minimum_family_pass_fraction = float(
        gates.get("minimum_family_pass_fraction", 0.0)
    )
    for (cell, family), members in family_records.items():
        pass_fraction = sum(bool(item["passed"]) for item in members) / len(members)
        if (
            len(members) >= minimum_family_size
            and pass_fraction >= minimum_family_pass_fraction
        ):
            passing_families_by_cell.setdefault(cell, []).append(family)
    for target in {str(item["reference_cell"]) for item in config["variants"]}:
        if target not in passing_by_cell:
            failures.append(f"{target}: no variant passed all gates")
        if (
            "minimum_family_pass_fraction" in gates
            and target not in passing_families_by_cell
        ):
            failures.append(f"{target}: no training-seed-stable family passed")
    config_schema = str(config["schema"])
    version = config_schema.rsplit(".v", maxsplit=1)[-1]
    run_class = "confirmation" if "confirmation" in config_schema else "development"
    result_schema = f"npi.g11.v16-deep-tail-transport-{run_class}.v{version}"
    payload = {
        "schema": result_schema,
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "artifact_bindings": config["artifacts"],
        "source_provenance": git_source_provenance(ROOT),
        "passed": not failures,
        "failures": failures,
        "passing_variants_by_cell": passing_by_cell,
        "passing_families_by_cell": passing_families_by_cell,
        "variants": records,
        "claim_boundary": {
            "development_selection_only": run_class == "development",
            "independent_evaluation_clusters": True,
            "proposal_frozen_across_clusters": True,
            "fresh_confirmation_required_after_selection": run_class == "development",
        },
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": payload["passed"]}, indent=2))


if __name__ == "__main__":
    main()
