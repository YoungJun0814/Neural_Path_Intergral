"""End-to-end V6 rarity routing, capped selection, and achieved-RMSE execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from experiments.g11_v6_baseline_qualification import (
    _design_from_pilot,
    _DirectRBergomiSampler,
    _load_references,
    _smoke_cells,
    _task,
    _work_record,
)
from experiments.g11_v6_reference import _load_manifest
from src.path_integral import (
    HybridProfileOpportunity,
    HybridTarget,
    RarityRouterConfig,
    RBergomiHybridTermSampler,
    RoutingWorkInterval,
    SeedKey,
    SeedLedger,
    SingleTermDesign,
    TimePiecewiseTwoDriverControl,
    V6WorkLedger,
    advance_sequential_crossover,
    audit_v6_policy,
    conservative_bernoulli_variance_upper,
    exact_binomial_probability_interval,
    execute_v6_policy,
    freeze_rarity_route,
    prepare_v6_direct_policy,
    prepare_v6_hybrid_policy,
    rbergomi_hybrid_candidate_profiles,
    rbergomi_hybrid_profile_ids,
    update_profile_intervals,
    v6_policy_preparation_to_dict,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

_SCHEMA = "npi.g11.v6-routed-policy.config.v1"


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 routed-policy config")
    expected = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "hierarchy",
        "proposal",
        "router",
        "selector",
        "sampling",
        "gates",
    }
    if set(payload) != expected:
        raise ValueError("malformed V6 routed-policy config fields")
    if payload["phase"] not in ("development", "qualification", "confirmation"):
        raise ValueError("unsupported V6 routed-policy phase")
    if payload["phase"] != "development" and payload["frozen"] is not True:
        raise ValueError("qualification and confirmation policy configs must be frozen")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("routed policy must declare a fixed-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _router_config(payload: dict[str, Any], *, smoke: bool) -> RarityRouterConfig:
    maximum = int(payload["maximum_screening_trials"])
    initial = int(payload["initial_screening_trials"])
    if smoke:
        maximum = min(maximum, 256)
        initial = min(initial, maximum)
    fallback = str(payload["ambiguous_fallback"])
    if fallback not in ("crude", "dcs_slis"):
        raise ValueError("unsupported ambiguous router fallback")
    return RarityRouterConfig(
        probability_cutoff=float(payload["probability_cutoff"]),
        confidence_level=float(payload["confidence_level"]),
        initial_screening_trials=initial,
        maximum_screening_trials=maximum,
        minimum_certified_relative_saving=float(payload["minimum_certified_relative_saving"]),
        maximum_hybrid_profile_work=(
            1.0 if smoke else float(payload["maximum_hybrid_profile_work"])
        ),
        maximum_profile_fraction=float(payload["maximum_profile_fraction"]),
        ambiguous_fallback=cast(Literal["crude", "dcs_slis"], fallback),
    )


def _append_screening(
    *,
    sampler,
    ledger: SeedLedger,
    protocol: str,
    cell_id: str,
    cluster: int,
    look: int,
    count: int,
) -> tuple[torch.Tensor, float, float, float]:
    seed = ledger.allocate(
        SeedKey(
            protocol,
            "router-screening",
            f"{cell_id}:cluster-{cluster}",
            "crude",
            0,
            look,
            "proposal",
        )
    )
    cpu_started = time.process_time()
    batch = sampler("crude", "pilot", count, {"proposal": seed})
    cpu = time.process_time() - cpu_started
    return batch.values, batch.work_units, batch.wall_seconds, cpu


def _work_interval_from_profile(profile, *, preprocessing_work: float, target: float):
    lower, upper = profile.moments.variance_interval
    point = profile.moments.sample_variance
    cost = profile.cost_per_sample
    return RoutingWorkInterval(
        "dcs_slis",
        preprocessing_work + lower * cost / target,
        preprocessing_work + point * cost / target,
        preprocessing_work + upper * cost / target,
    )


def _crude_work_interval(
    values: torch.Tensor,
    *,
    cost: float,
    preprocessing_work: float,
    target: float,
    confidence_level: float,
) -> RoutingWorkInterval:
    hits = int(torch.count_nonzero(values))
    interval = exact_binomial_probability_interval(
        hits, values.numel(), confidence_level=confidence_level
    )
    endpoint_variances = (
        interval.lower * (1.0 - interval.lower),
        interval.upper * (1.0 - interval.upper),
    )
    lower = min(endpoint_variances)
    upper = conservative_bernoulli_variance_upper(interval)
    point = float(torch.var(values, unbiased=True))
    return RoutingWorkInterval(
        "crude",
        preprocessing_work + lower * cost / target,
        preprocessing_work + point * cost / target,
        preprocessing_work + upper * cost / target,
    )


def _linear_quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def run(config_path: Path, manifest_path: Path, reference_path: Path, *, smoke: bool = False):
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    references, reference_hash = _load_references(reference_path)
    if not smoke and config["phase"] != "development":
        if (
            manifest.phase != config["phase"]
            or not manifest.frozen
            or manifest.smoke
        ):
            raise ValueError("formal routed policy requires a same-phase frozen manifest")
    router_config = _router_config(config["router"], smoke=smoke)
    sampling = config["sampling"]
    hierarchy = config["hierarchy"]
    finest_level = int(hierarchy["finest_level"])
    if int(hierarchy["coarsest_steps"]) * 2**finest_level != manifest.cells[0].finest_steps:
        raise ValueError("routed-policy hierarchy does not match the manifest grid")
    cells = _smoke_cells(manifest.cells) if smoke else manifest.cells
    clusters = 1 if smoke else int(sampling["clusters"])
    relative_rmse = max(0.50, float(sampling["relative_sampling_rmse"])) if smoke else float(
        sampling["relative_sampling_rmse"]
    )
    proposal_weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
    profile_ids = rbergomi_hybrid_profile_ids(finest_level)
    candidate_profiles = rbergomi_hybrid_candidate_profiles(finest_level)
    master_ledger = SeedLedger()
    records = []
    for cell in cells:
        if cell.cell_id not in references:
            raise ValueError(f"reference artifact lacks cell {cell.cell_id}")
        reference_probability, reference_se, reference_cell = references[cell.cell_id]
        if reference_cell != cell.to_dict():
            raise ValueError(f"reference estimand drift for cell {cell.cell_id}")
        task = _task(cell)
        simulator = RBergomiSimulator(
            H=cell.hurst, eta=cell.eta, xi=cell.xi, rho=cell.rho, device="cpu"
        )
        controls = tuple(
            TimePiecewiseTwoDriverControl(
                tuple(tuple(float(value) for value in segment) for segment in schedule),
                maturity=cell.maturity,
            )
            for schedule in config["proposal"]["controls"]
        )
        natural = controls[0]
        for cluster in range(clusters):
            ledger = SeedLedger()
            work = V6WorkLedger()
            crude_interval = None
            dcs_interval = None
            hybrid_opportunity = None
            crude_sampler = _DirectRBergomiSampler(
                method="crude",
                simulator=simulator,
                task=task,
                natural=natural,
                fitted=None,
                defensive_weight=float(proposal_weights[0]),
                cell=cell,
                engine=str(sampling["engine"]),
            )
            screening, units, wall, cpu = _append_screening(
                sampler=crude_sampler,
                ledger=ledger,
                protocol=str(config["protocol_id"]),
                cell_id=cell.cell_id,
                cluster=cluster,
                look=0,
                count=router_config.initial_screening_trials,
            )
            work = work.append(
                _work_record(
                    "screening",
                    method="v6_policy",
                    cell_id=cell.cell_id,
                    samples=screening.numel(),
                    work_units=units,
                    wall_seconds=wall,
                    cpu_seconds=cpu,
                )
            )
            route = freeze_rarity_route(
                successes=int(torch.count_nonzero(screening)),
                trials=int(screening.numel()),
                screening_work=work.category_work("screening"),
                config=router_config,
            )
            if route.action == "continue_screening":
                increment = router_config.maximum_screening_trials - screening.numel()
                extra, units, wall, cpu = _append_screening(
                    sampler=crude_sampler,
                    ledger=ledger,
                    protocol=str(config["protocol_id"]),
                    cell_id=cell.cell_id,
                    cluster=cluster,
                    look=1,
                    count=increment,
                )
                screening = torch.cat((screening, extra))
                work = work.append(
                    _work_record(
                        "screening",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=extra.numel(),
                        work_units=units,
                        wall_seconds=wall,
                        cpu_seconds=cpu,
                    )
                )
                route = freeze_rarity_route(
                    successes=int(torch.count_nonzero(screening)),
                    trials=int(screening.numel()),
                    screening_work=work.category_work("screening"),
                    config=router_config,
                )

            dcs_sampler = RBergomiHybridTermSampler(
                simulator,
                controls,
                proposal_weights,
                task,
                spot=cell.spot,
                maturity=cell.maturity,
                coarsest_steps=int(hierarchy["coarsest_steps"]),
                finest_level=finest_level,
                engine=str(sampling["engine"]),
            )
            dcs_profile = None
            if route.action != "crude":
                dcs_count = 128 if smoke else int(sampling["dcs_pilot_samples"])
                dcs_seeds = {
                    stream: ledger.allocate(
                        SeedKey(
                            str(config["protocol_id"]),
                            "allocation-pilot",
                            f"{cell.cell_id}:cluster-{cluster}",
                            "dcs-slis",
                            finest_level,
                            0,
                            stream,
                        )
                    )
                    for stream in ("proposal", "labels")
                }
                dcs_cpu_started = time.process_time()
                dcs_batch = dcs_sampler(f"single_{finest_level}", "pilot", dcs_count, dcs_seeds)
                dcs_cpu = time.process_time() - dcs_cpu_started
                work = work.append(
                    _work_record(
                        "allocation_pilot",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=dcs_count,
                        work_units=dcs_batch.work_units,
                        wall_seconds=dcs_batch.wall_seconds,
                        cpu_seconds=dcs_cpu,
                    )
                )
                dcs_profile = update_profile_intervals(
                    {f"single_{finest_level}": dcs_batch.values},
                    absolute_bounds={
                        f"single_{finest_level}": dcs_sampler.defensive_absolute_bound
                    },
                    costs_per_sample={
                        f"single_{finest_level}": dcs_sampler.cost_per_sample(
                            f"single_{finest_level}"
                        )
                    },
                    familywise_alpha=float(config["selector"]["familywise_alpha"]),
                    total_predeclared_looks=1,
                )[0]
                target_variance = (cell.nominal_probability * relative_rmse) ** 2
                common_prework = work.total_work_units
                crude_interval = _crude_work_interval(
                    screening,
                    cost=crude_sampler.cost,
                    preprocessing_work=common_prework,
                    target=target_variance,
                    confidence_level=router_config.confidence_level,
                )
                dcs_interval = _work_interval_from_profile(
                    dcs_profile,
                    preprocessing_work=common_prework,
                    target=target_variance,
                )
                first_look = 32 if smoke else int(config["selector"]["looks"][0])
                minimum_profile_work = math.fsum(
                    first_look * dcs_sampler.cost_per_sample(profile_id)
                    for profile_id in profile_ids
                )
                hybrid_opportunity = HybridProfileOpportunity(
                    minimum_profile_work=minimum_profile_work,
                    optimistic_total_work=common_prework + minimum_profile_work,
                    external_profile_work_cap=float(
                        config["router"]["maximum_hybrid_profile_work"]
                    ),
                )
                route = freeze_rarity_route(
                    successes=int(torch.count_nonzero(screening)),
                    trials=int(screening.numel()),
                    screening_work=work.category_work("screening"),
                    config=router_config,
                    crude_work=crude_interval,
                    dcs_work=dcs_interval,
                    hybrid_opportunity=hybrid_opportunity,
                )
            work = work.append(
                _work_record(
                    "routing",
                    method="v6_policy",
                    cell_id=cell.cell_id,
                    samples=0,
                    work_units=0.0,
                    wall_seconds=0.0,
                    cpu_seconds=0.0,
                )
            )

            selection_state = None
            if route.action == "profile_hybrid":
                looks = tuple(int(value) for value in config["selector"]["looks"])
                if smoke:
                    looks = (32, 64)
                observations = {
                    profile_id: torch.empty(0, dtype=torch.float64) for profile_id in profile_ids
                }
                previous = 0
                selector_work = 0.0
                selector_wall = 0.0
                selector_cpu_total = 0.0
                for look_index, cumulative in enumerate(looks):
                    increment = cumulative - previous
                    for profile_index, profile_id in enumerate(profile_ids):
                        seeds = {
                            stream: ledger.allocate(
                                SeedKey(
                                    str(config["protocol_id"]),
                                    "selector-profile",
                                    f"{cell.cell_id}:cluster-{cluster}",
                                    profile_id,
                                    profile_index,
                                    look_index,
                                    stream,
                                )
                            )
                            for stream in ("proposal", "labels")
                        }
                        selector_cpu_started = time.process_time()
                        batch = dcs_sampler(profile_id, "pilot", increment, seeds)
                        selector_cpu = time.process_time() - selector_cpu_started
                        observations[profile_id] = torch.cat(
                            (observations[profile_id], batch.values)
                        )
                        selector_work += batch.work_units
                        selector_wall += batch.wall_seconds
                        selector_cpu_total += selector_cpu
                    selection_state = advance_sequential_crossover(
                        observations,
                        absolute_bounds={
                            profile_id: dcs_sampler.defensive_absolute_bound
                            for profile_id in profile_ids
                        },
                        costs_per_sample={
                            profile_id: dcs_sampler.cost_per_sample(profile_id)
                            for profile_id in profile_ids
                        },
                        candidate_profiles=candidate_profiles,
                        preprocessing_work={
                            candidate: work.total_work_units + selector_work
                            for candidate in candidate_profiles
                        },
                        sampling_variance_target=(cell.nominal_probability * relative_rmse) ** 2,
                        predeclared_looks=looks,
                        look_index=look_index,
                        familywise_alpha=float(config["selector"]["familywise_alpha"]),
                        simpler_candidate=f"start_{finest_level}",
                        previous_state=selection_state,
                        elimination_relative_tolerance=float(
                            config["selector"]["elimination_relative_tolerance"]
                        ),
                        practical_equivalence_relative_tolerance=float(
                            config["selector"]["practical_equivalence_relative_tolerance"]
                        ),
                        maximum_profile_work=route.effective_profile_work_cap,
                        maximum_profile_fraction_of_best_point=float(
                            config["router"]["maximum_profile_fraction"]
                        ),
                    )
                    previous = cumulative
                    if selection_state.stopped:
                        break
                assert selection_state is not None and selection_state.frozen_decision is not None
                work = work.append(
                    _work_record(
                        "selector_profile",
                        method="v6_policy",
                        cell_id=cell.cell_id,
                        samples=previous * len(profile_ids),
                        work_units=selector_work,
                        wall_seconds=selector_wall,
                        cpu_seconds=selector_cpu_total,
                    )
                )
                selected = selection_state.frozen_decision.selected_candidate
                prepared = prepare_v6_hybrid_policy(
                    HybridTarget(
                        f"{cell.cell_id}:v6-policy",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    selection_state.profiles,
                    policy_name="v6_policy",
                    cell_id=cell.cell_id,
                    route=route,
                    selection=selection_state.frozen_decision,
                    selected_profile_ids=candidate_profiles[selected],
                    protocol=f"{config['protocol_id']}:cluster-{cluster}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=(128 if smoke else int(sampling["minimum_final_samples"])),
                    allocation_safety_factor=float(sampling["allocation_safety_factor"]),
                    preparation_seed_ledger=ledger,
                )
                final_sampler = dcs_sampler
            elif route.action == "dcs_slis":
                assert dcs_profile is not None
                design = SingleTermDesign(
                    profile_id=dcs_profile.profile_id,
                    pilot_count=dcs_profile.moments.sample_count,
                    pilot_mean=dcs_profile.moments.sample_mean,
                    pilot_variance=dcs_profile.moments.sample_variance,
                    design_variance=(
                        max(
                            2.0 * dcs_profile.moments.sample_variance,
                            cell.nominal_probability**2,
                        )
                        if smoke
                        else float(sampling["allocation_safety_factor"])
                        * dcs_profile.moments.variance_interval[1]
                    ),
                    cost_per_sample=dcs_profile.cost_per_sample,
                    absolute_bound=dcs_sampler.defensive_absolute_bound,
                )
                prepared = prepare_v6_direct_policy(
                    HybridTarget(
                        f"{cell.cell_id}:v6-policy",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    design,
                    policy_name="v6_policy",
                    cell_id=cell.cell_id,
                    execution_method="dcs_slis",
                    protocol=f"{config['protocol_id']}:cluster-{cluster}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    route=route,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=(128 if smoke else int(sampling["minimum_final_samples"])),
                    preparation_seed_ledger=ledger,
                )
                final_sampler = dcs_sampler
            elif route.action == "crude":
                design = _design_from_pilot(
                    "crude",
                    screening,
                    cost_per_sample=crude_sampler.cost,
                    nominal_probability=cell.nominal_probability,
                    confidence_level=router_config.confidence_level,
                    pure_safety=1.0,
                    defensive_bound=1.0,
                    bounded_alpha=float(config["selector"]["familywise_alpha"]),
                )
                prepared = prepare_v6_direct_policy(
                    HybridTarget(
                        f"{cell.cell_id}:v6-policy",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    design,
                    policy_name="v6_policy",
                    cell_id=cell.cell_id,
                    execution_method="crude",
                    protocol=f"{config['protocol_id']}:cluster-{cluster}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    route=route,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=(128 if smoke else int(sampling["minimum_final_samples"])),
                    streams=("proposal",),
                    preparation_seed_ledger=ledger,
                )
                final_sampler = crude_sampler
            else:
                raise AssertionError("router remained unresolved")

            result = execute_v6_policy(
                prepared,
                final_sampler,
                reference_probability=reference_probability,
                reference_standard_error=reference_se,
                final_peak_memory_bytes=0,
            )
            audit = audit_v6_policy(prepared, result)
            selection_fraction = (
                result.total_work.category_work("selector_profile")
                / result.total_work.total_work_units
                if result.total_work.total_work_units > 0.0
                else 0.0
            )
            records.append(
                {
                    "cell_id": cell.cell_id,
                    "cluster": cluster,
                    "nominal_probability": cell.nominal_probability,
                    "reference_probability": reference_probability,
                    "reference_standard_error": reference_se,
                    "route": asdict(route),
                    "router_inputs": {
                        "successes": int(torch.count_nonzero(screening)),
                        "trials": int(screening.numel()),
                        "screening_work": work.category_work("screening"),
                        "config": asdict(router_config),
                        "crude_work": (
                            None if crude_interval is None else asdict(crude_interval)
                        ),
                        "dcs_work": None if dcs_interval is None else asdict(dcs_interval),
                        "hybrid_opportunity": (
                            None
                            if hybrid_opportunity is None
                            else asdict(hybrid_opportunity)
                        ),
                    },
                    "selection": None if selection_state is None else asdict(selection_state),
                    "selection_work_fraction": selection_fraction,
                    "preparation": v6_policy_preparation_to_dict(prepared),
                    "result": asdict(result),
                    "audit": asdict(audit),
                }
            )
            final_ledger = SeedLedger.from_dict(result.core.seed_ledger_payload)
            for seed_record in final_ledger.records:
                master_ledger.allocate(seed_record.key)
    selection_fractions = [float(record["selection_work_fraction"]) for record in records]
    median_fraction = _linear_quantile(selection_fractions, 0.5)
    p90_fraction = _linear_quantile(selection_fractions, 0.9)
    gates = {
        "complete_matrix": len(records) == len(cells) * clusters,
        "all_routes_resolved": all(
            record["route"]["action"] != "continue_screening" for record in records
        ),
        "all_runs_complete": all(record["result"]["core"]["complete"] for record in records),
        "no_resource_censoring": all(
            not record["result"]["core"]["resource_censored"] for record in records
        ),
        "all_independent_audits": all(record["audit"]["passed"] for record in records),
        "median_selection_fraction": median_fraction is not None
        and median_fraction <= float(config["gates"]["maximum_median_selection_fraction"]),
        "p90_selection_fraction": p90_fraction is not None
        and p90_fraction <= float(config["gates"]["maximum_p90_selection_fraction"]),
        "reference_not_used_by_router_schema": all(
            all("reference" not in key for key in record["route"])
            for record in records
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "frozen_manifest": manifest.frozen,
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
    }
    return {
        "schema": "npi.g11.v6-routed-policy.v1",
        "protocol_id": config["protocol_id"],
        "phase": config["phase"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_artifact_sha256": reference_hash,
        "smoke": smoke,
        "records": records,
        "selection_fraction_summary": {"median": median_fraction, "p90": p90_fraction},
        "gates": gates,
        "formal_readiness": formal,
        "policy_qualified": all(gates.values()) and all(formal.values()),
        "seed_ledger": master_ledger.to_dict(),
        "seed_ledger_sha256": master_ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/routed_policy_development.yaml"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.manifest, arguments.reference, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"qualified": result["policy_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
