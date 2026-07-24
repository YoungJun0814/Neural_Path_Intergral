"""Fixed DCS-SLIS and smoothing-off secondary baselines for V6."""

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
    _load_references,
    _smoke_cells,
    _task,
    _work_record,
)
from experiments.g11_v6_planning import (
    PlanningVarianceStatistic,
    collect_replicated_direct_pilot,
    plugin_design_variance,
    replicated_direct_certificate,
    validate_replicated_planning,
)
from experiments.g11_v6_reference import _load_manifest
from experiments.g11_v6_routed_policy import (
    _apportion_shared_training,
    _task_conditioned_training_source_audit,
)
from src.path_integral import (
    HybridTarget,
    RBergomiHybridTermSampler,
    SeedKey,
    SeedLedger,
    SingleTermDesign,
    TimePiecewiseTwoDriverControl,
    V6ProgressJournal,
    V6WorkLedger,
    audit_v6_policy,
    execute_v6_policy,
    execute_v6_policy_durable,
    load_v6_progress,
    prepare_v6_direct_policy,
    save_v6_progress,
    update_profile_intervals,
    v6_policy_preparation_to_dict,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

_SCHEMA_V1 = "npi.g11.v6-secondary-baselines.config.v1"
_SCHEMA_V2 = "npi.g11.v6-secondary-baselines.config.v2"
_AGGREGATE_ACCURACY_RULE = (
    "deferred_to_prespecified_method_cell_attainment_and_bootstrap_rmse_co_gates"
)
_OPERATIONAL_QUALIFICATION_GATE_NAMES = (
    "complete_matrix",
    "all_runs_complete",
    "no_resource_censoring",
    "all_design_targets_attained",
    "all_independent_audits",
    "smoothing_pair_present",
    "all_planning_certificates_valid",
    "all_final_seed_roles_separate",
)
SecondaryMethod = Literal["fixed_dcs_slis", "fixed_raw_defensive"]


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") not in {
        _SCHEMA_V1,
        _SCHEMA_V2,
    }:
        raise ValueError("unsupported V6 secondary-baseline config")
    expected = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "methods",
        "hierarchy",
        "proposal",
        "sampling",
    }
    if payload["schema"] == _SCHEMA_V2:
        expected.update(
            {"planning", "rarity_band_design", "qualification_decision"}
        )
    if set(payload) != expected:
        raise ValueError("malformed V6 secondary-baseline config fields")
    if payload["phase"] not in ("development", "qualification", "confirmation"):
        raise ValueError("unsupported V6 secondary-baseline phase")
    if payload["phase"] != "development" and payload["frozen"] is not True:
        raise ValueError("formal secondary-baseline configs must be frozen")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("secondary baselines require the fixed-finest-grid estimand")
    methods = payload["methods"]
    allowed = {"fixed_dcs_slis", "fixed_raw_defensive"}
    if (
        not isinstance(methods, list)
        or not methods
        or len(methods) != len(set(methods))
        or any(method not in allowed for method in methods)
    ):
        raise ValueError("secondary-baseline methods are invalid")
    proposal = payload["proposal"]
    if (
        not isinstance(proposal, dict)
        or proposal.get("training_derivation")
        != "componentwise_median_pure_cem_then_zero_half_full_bank"
    ):
        raise ValueError("secondary baselines require the verified V3 proposal bank")
    if payload["schema"] == _SCHEMA_V2:
        sampling = payload["sampling"]
        if not isinstance(sampling, dict) or set(sampling) != {
            "clusters",
            "relative_sampling_rmse",
            "confidence_level",
            "minimum_final_samples",
            "chunk_size",
            "allocation_safety_factor",
            "familywise_alpha",
            "operation_work_cap",
            "engine",
        }:
            raise ValueError("malformed V2 secondary sampling fields")
        if (
            isinstance(sampling["clusters"], bool)
            or int(sampling["clusters"]) < 1
            or not math.isfinite(
                float(sampling["relative_sampling_rmse"])
            )
            or not 0.0 < float(sampling["relative_sampling_rmse"]) < 1.0
            or not math.isfinite(float(sampling["confidence_level"]))
            or not 0.5 < float(sampling["confidence_level"]) < 1.0
            or int(sampling["minimum_final_samples"]) < 2
            or int(sampling["chunk_size"]) < 1
            or not math.isfinite(
                float(sampling["allocation_safety_factor"])
            )
            or float(sampling["allocation_safety_factor"]) < 1.0
            or not math.isfinite(float(sampling["familywise_alpha"]))
            or not 0.0 < float(sampling["familywise_alpha"]) < 1.0
            or not math.isfinite(float(sampling["operation_work_cap"]))
            or float(sampling["operation_work_cap"]) <= 0.0
            or sampling["engine"] not in {"fft", "reference"}
        ):
            raise ValueError("invalid V2 secondary sampling contract")
        planning = payload["planning"]
        if not isinstance(planning, dict) or set(planning) != {
            "decision_mode",
            "replicates",
            "samples_per_replicate",
            "variance_statistic",
            "zero_variance_fallback",
        }:
            raise ValueError("malformed V2 secondary replicated-planning fields")
        if planning["decision_mode"] != "replicated_point_variance":
            raise ValueError("unsupported V2 secondary planning decision mode")
        validate_replicated_planning(
            replicates=planning["replicates"],
            samples_per_replicate=planning["samples_per_replicate"],
            variance_statistic=planning["variance_statistic"],
        )
        if planning["zero_variance_fallback"] != "nominal_probability_squared":
            raise ValueError("unsupported V2 secondary zero-variance fallback")
        rarity = payload["rarity_band_design"]
        if not isinstance(rarity, dict) or set(rarity) != {
            "nominal_probability_upper_multiplier",
            "reference_certificate_z",
        }:
            raise ValueError("malformed V2 secondary rarity-band certificate")
        multiplier = rarity["nominal_probability_upper_multiplier"]
        certificate_z = rarity["reference_certificate_z"]
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not math.isfinite(float(multiplier))
            or float(multiplier) < 1.0
            or isinstance(certificate_z, bool)
            or not isinstance(certificate_z, (int, float))
            or not math.isfinite(float(certificate_z))
            or float(certificate_z) <= 0.0
        ):
            raise ValueError("invalid V2 secondary rarity-band certificate")
        decision = payload["qualification_decision"]
        if not isinstance(decision, dict) or set(decision) != {
            "per_record_empirical_target_role",
            "aggregate_accuracy_protocol_id",
        }:
            raise ValueError("malformed V2 secondary qualification decision")
        if decision["per_record_empirical_target_role"] != _AGGREGATE_ACCURACY_RULE:
            raise ValueError("V2 secondary accuracy must use aggregate co-gates")
        if (
            not isinstance(decision["aggregate_accuracy_protocol_id"], str)
            or not decision["aggregate_accuracy_protocol_id"]
        ):
            raise ValueError("V2 secondary aggregate accuracy protocol must be named")
    return payload, hashlib.sha256(raw).hexdigest()


def _checkpoint_path(
    directory: Path, *, cell_id: str, cluster: int, method: str
) -> Path:
    identity = json.dumps(
        {"cell_id": cell_id, "cluster": cluster, "method": method},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return directory / "records" / f"{hashlib.sha256(identity).hexdigest()}.json"


def _clear_checkpoint(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".v6.json").unlink(missing_ok=True)


def run(
    config_path: Path,
    manifest_path: Path,
    reference_path: Path,
    *,
    proposal_training_source_path: Path | None = None,
    smoke: bool = False,
    checkpoint_directory: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    references, reference_hash = _load_references(reference_path)
    if resume and checkpoint_directory is None:
        raise ValueError("secondary-baseline resume requires a checkpoint directory")
    if not smoke and config["phase"] != "development":
        if (
            manifest.phase != config["phase"]
            or not manifest.frozen
            or manifest.smoke
        ):
            raise ValueError("formal secondary baselines require a same-phase manifest")
    cells = _smoke_cells(manifest.cells) if smoke else manifest.cells
    sampling = config["sampling"]
    clusters = 1 if smoke else int(sampling["clusters"])
    relative_rmse = (
        max(0.50, float(sampling["relative_sampling_rmse"]))
        if smoke
        else float(sampling["relative_sampling_rmse"])
    )
    pilot_count = (
        128
        if smoke
        else int(sampling["pilot_samples"])
        if config["schema"] == _SCHEMA_V1
        else 0
    )
    minimum_final = 128 if smoke else int(sampling["minimum_final_samples"])
    proposal = config["proposal"]
    training_allocations, training_contract = _apportion_shared_training(
        proposal,
        len(cells) * clusters,
        enforce_declared_count=not smoke,
    )
    if proposal_training_source_path is not None:
        training_audit = _task_conditioned_training_source_audit(
            proposal, proposal_training_source_path
        )
    elif config["phase"] != "development":
        raise ValueError("formal secondary baselines require the proposal training source")
    else:
        training_audit = {
            "verified": False,
            "formal_training_source_readiness": False,
            "source_artifact_sha256": proposal["training_source_artifact_sha256"],
            "reason": "development execution did not receive the source artifact",
        }

    progress_identities: dict[str, object] = {
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_sha256": reference_hash,
        "smoke": smoke,
    }
    progress_path = (
        None if checkpoint_directory is None else checkpoint_directory / "progress.json"
    )
    if progress_path is not None and resume:
        records = list(
            load_v6_progress(
                progress_path,
                experiment="g11_v6_secondary_baselines",
                identities=progress_identities,
            ).records
        )
    else:
        if progress_path is not None and progress_path.exists():
            raise FileExistsError("fresh secondary execution refuses existing progress")
        records = []
    completed = {
        (str(record["cell_id"]), int(record["cluster"]), str(record["method"]))
        for record in records
    }
    master_ledger = SeedLedger()
    for record in records:
        restored = SeedLedger.from_dict(record["result"]["core"]["seed_ledger_payload"])
        for seed_record in restored.records:
            master_ledger.allocate(seed_record.key)
    if progress_path is not None and not progress_path.exists():
        save_v6_progress(
            progress_path,
            V6ProgressJournal(
                "g11_v6_secondary_baselines", progress_identities, tuple(records)
            ),
        )

    hierarchy = config["hierarchy"]
    finest_level = int(hierarchy["finest_level"])
    if (
        int(hierarchy["coarsest_steps"]) * 2**finest_level
        != manifest.cells[0].finest_steps
    ):
        raise ValueError("secondary-baseline hierarchy does not match the manifest")
    weights = torch.tensor(proposal["weights"], dtype=torch.float64)
    for cell_index, cell in enumerate(cells):
        if cell.cell_id not in references:
            raise ValueError(f"reference artifact lacks cell {cell.cell_id}")
        reference_probability, reference_se, reference_cell = references[cell.cell_id]
        if reference_cell != cell.to_dict():
            raise ValueError(f"reference estimand drift for cell {cell.cell_id}")
        simulator = RBergomiSimulator(
            H=cell.hurst, eta=cell.eta, xi=cell.xi, rho=cell.rho, device="cpu"
        )
        controls = tuple(
            TimePiecewiseTwoDriverControl(
                tuple(
                    (float(segment[0]), float(segment[1]))
                    for segment in schedule
                ),
                maturity=cell.maturity,
            )
            for schedule in proposal["task_controls"][cell.task]
        )
        for cluster in range(clusters):
            shared_training = training_allocations[cell_index * clusters + cluster]
            for method in config["methods"]:
                if (cell.cell_id, cluster, method) in completed:
                    continue
                correction_method = (
                    "dcs_mgi" if method == "fixed_dcs_slis" else "raw_defensive"
                )
                execution_method = (
                    "dcs_slis" if method == "fixed_dcs_slis" else "raw_defensive"
                )
                sampler = RBergomiHybridTermSampler(
                    simulator,
                    controls,
                    weights,
                    _task(cell),
                    spot=cell.spot,
                    maturity=cell.maturity,
                    coarsest_steps=int(hierarchy["coarsest_steps"]),
                    finest_level=finest_level,
                    engine=cast(Any, str(sampling["engine"])),
                    correction_method=cast(Any, correction_method),
                )
                ledger = SeedLedger()
                work = V6WorkLedger(
                    (
                        _work_record(
                            "proposal_training",
                            method=method,
                            cell_id=cell.cell_id,
                            samples=int(shared_training["samples"]),
                            work_units=float(shared_training["work_units"]),
                            wall_seconds=float(shared_training["wall_seconds"]),
                            cpu_seconds=float(shared_training["cpu_seconds"]),
                        ),
                    )
                )
                planning_certificate = None
                if config["schema"] == _SCHEMA_V2:
                    planning = config["planning"]
                    replicated = collect_replicated_direct_pilot(
                        sampler=sampler,
                        ledger=ledger,
                        protocol=str(config["protocol_id"]),
                        cell_id=cell.cell_id,
                        cluster=cluster,
                        method=method,
                        profile_id=f"single_{finest_level}",
                        level=finest_level,
                        streams=("proposal", "labels"),
                        replicates=(3 if smoke else int(planning["replicates"])),
                        samples_per_replicate=(
                            64
                            if smoke
                            else int(planning["samples_per_replicate"])
                        ),
                        variance_statistic=cast(
                            PlanningVarianceStatistic,
                            str(planning["variance_statistic"]),
                        ),
                    )
                    pilot_values = replicated.values
                    pilot_work_units = replicated.work_units
                    pilot_wall_seconds = replicated.wall_seconds
                    pilot_cpu = replicated.cpu_seconds
                    pilot_samples = int(pilot_values.numel())
                else:
                    seeds = {
                        stream: ledger.allocate(
                            SeedKey(
                                str(config["protocol_id"]),
                                "allocation-pilot",
                                f"{cell.cell_id}:cluster-{cluster}",
                                method,
                                finest_level,
                                0,
                                stream,
                            )
                        )
                        for stream in ("proposal", "labels")
                    }
                    cpu_started = time.process_time()
                    batch = sampler(
                        f"single_{finest_level}", "pilot", pilot_count, seeds
                    )
                    pilot_cpu = time.process_time() - cpu_started
                    pilot_values = batch.values
                    pilot_work_units = batch.work_units
                    pilot_wall_seconds = batch.wall_seconds
                    pilot_samples = pilot_count
                work = work.append(
                    _work_record(
                        "allocation_pilot",
                        method=method,
                        cell_id=cell.cell_id,
                        samples=pilot_samples,
                        work_units=pilot_work_units,
                        wall_seconds=pilot_wall_seconds,
                        cpu_seconds=pilot_cpu,
                    )
                )
                profile = update_profile_intervals(
                    {f"single_{finest_level}": pilot_values},
                    absolute_bounds={
                        f"single_{finest_level}": sampler.defensive_absolute_bound
                    },
                    costs_per_sample={
                        f"single_{finest_level}": sampler.cost_per_sample(
                            f"single_{finest_level}"
                        )
                    },
                    familywise_alpha=float(sampling["familywise_alpha"]),
                    total_predeclared_looks=1,
                )[0]
                if config["schema"] == _SCHEMA_V2:
                    zero_fallback = cell.nominal_probability**2
                    design_variance = plugin_design_variance(
                        replicated.planning_variance,
                        safety_factor=(
                            2.0
                            if smoke
                            else float(sampling["allocation_safety_factor"])
                        ),
                        zero_variance_fallback=zero_fallback,
                    )
                    rarity = config["rarity_band_design"]
                    planning_certificate = replicated_direct_certificate(
                        replicated,
                        method=method,
                        variance_statistic=cast(
                            PlanningVarianceStatistic,
                            str(planning["variance_statistic"]),
                        ),
                        safety_factor=(
                            2.0
                            if smoke
                            else float(sampling["allocation_safety_factor"])
                        ),
                        zero_variance_fallback=zero_fallback,
                        selected_design_variance=design_variance,
                        absolute_bound=sampler.defensive_absolute_bound,
                        nominal_probability=cell.nominal_probability,
                        nominal_probability_upper_multiplier=float(
                            rarity["nominal_probability_upper_multiplier"]
                        ),
                        reference_probability=reference_probability,
                        reference_standard_error=reference_se,
                        reference_certificate_z=float(
                            rarity["reference_certificate_z"]
                        ),
                    )
                    if not planning_certificate["certified"]:
                        raise ValueError(
                            "V2 secondary direct-planning certificate failed for "
                            f"{cell.cell_id}:{cluster}:{method}"
                        )
                else:
                    design_variance = (
                        max(
                            2.0 * profile.moments.sample_variance,
                            cell.nominal_probability**2,
                        )
                        if smoke
                        else float(sampling["allocation_safety_factor"])
                        * profile.moments.variance_interval[1]
                    )
                design = SingleTermDesign(
                    profile_id=profile.profile_id,
                    pilot_count=profile.moments.sample_count,
                    pilot_mean=profile.moments.sample_mean,
                    pilot_variance=(
                        float(torch.var(pilot_values, unbiased=True))
                        if config["schema"] == _SCHEMA_V2
                        else profile.moments.sample_variance
                    ),
                    design_variance=design_variance,
                    cost_per_sample=profile.cost_per_sample,
                    absolute_bound=sampler.defensive_absolute_bound,
                )
                prepared = prepare_v6_direct_policy(
                    HybridTarget(
                        f"{cell.cell_id}:{method}",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    design,
                    policy_name=method,
                    cell_id=cell.cell_id,
                    execution_method=cast(Any, execution_method),
                    protocol=f"{config['protocol_id']}:cluster-{cluster}:{method}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=minimum_final,
                    streams=("proposal", "labels"),
                    preparation_seed_ledger=ledger,
                )
                checkpoint = (
                    None
                    if checkpoint_directory is None
                    else _checkpoint_path(
                        checkpoint_directory,
                        cell_id=cell.cell_id,
                        cluster=cluster,
                        method=method,
                    )
                )
                if checkpoint is None:
                    result = execute_v6_policy(
                        prepared,
                        sampler,
                        reference_probability=reference_probability,
                        reference_standard_error=reference_se,
                        final_peak_memory_bytes=0,
                    )
                else:
                    state = checkpoint.with_suffix(checkpoint.suffix + ".v6.json")
                    result = execute_v6_policy_durable(
                        prepared,
                        sampler,
                        checkpoint_path=checkpoint,
                        resume=checkpoint.exists() or state.exists(),
                        chunks_per_checkpoint=1,
                        reference_probability=reference_probability,
                        reference_standard_error=reference_se,
                        final_peak_memory_bytes=0,
                    )
                audit = audit_v6_policy(prepared, result)
                records.append(
                    {
                        "cell_id": cell.cell_id,
                        "cluster": cluster,
                        "method": method,
                        "dcs_smoothing": method == "fixed_dcs_slis",
                        "nominal_probability": cell.nominal_probability,
                        "reference_probability": reference_probability,
                        "reference_standard_error": reference_se,
                        "design": asdict(design),
                        "planning_certificate": planning_certificate,
                        "preparation": v6_policy_preparation_to_dict(prepared),
                        "result": asdict(result),
                        "audit": asdict(audit),
                    }
                )
                final_ledger = SeedLedger.from_dict(result.core.seed_ledger_payload)
                for seed_record in final_ledger.records:
                    master_ledger.allocate(seed_record.key)
                if progress_path is not None:
                    save_v6_progress(
                        progress_path,
                        V6ProgressJournal(
                            "g11_v6_secondary_baselines",
                            progress_identities,
                            tuple(records),
                        ),
                    )
                    if checkpoint is not None:
                        _clear_checkpoint(checkpoint)

    gates = {
        "complete_matrix": len(records)
        == len(cells) * clusters * len(config["methods"]),
        "all_runs_complete": all(record["result"]["core"]["complete"] for record in records),
        "no_resource_censoring": all(
            not record["result"]["core"]["resource_censored"] for record in records
        ),
        "all_design_targets_attained": all(
            record["result"]["core"]["design_target_attained"] for record in records
        ),
        "all_empirical_targets_attained": all(
            record["result"]["core"]["empirical_target_attained"] is True
            for record in records
        ),
        "all_independent_audits": all(record["audit"]["passed"] for record in records),
        "smoothing_pair_present": set(config["methods"])
        == {"fixed_dcs_slis", "fixed_raw_defensive"},
        "all_planning_certificates_valid": all(
            isinstance(record.get("planning_certificate"), dict)
            and record["planning_certificate"].get("certified") is True
            and record["planning_certificate"].get(
                "structural_bound_used_for_allocation"
            )
            is False
            for record in records
        )
        if config["schema"] == _SCHEMA_V2
        else True,
        "all_final_seed_roles_separate": all(
            all(
                seed["key"]["role"] == "final"
                for seed in record["result"]["core"]["seed_ledger_payload"][
                    "records"
                ]
                if seed["key"]["role"]
                not in {"proposal-training", "allocation-pilot"}
            )
            for record in records
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "frozen_manifest": manifest.frozen,
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
        "proposal_training_source_verified": bool(training_audit["verified"]),
        "proposal_training_source_formal": bool(
            training_audit["formal_training_source_readiness"]
        ),
    }
    qualification_gates = (
        {
            name: gates[name]
            for name in _OPERATIONAL_QUALIFICATION_GATE_NAMES
        }
        if config["schema"] == _SCHEMA_V2
        else dict(gates)
    )
    qualification_contract = None
    if config["schema"] == _SCHEMA_V2:
        qualification_contract = {
            "schema": "npi.g11.v6-secondary-qualification-contract.v1",
            "expected_cell_ids": [cell.cell_id for cell in cells],
            "expected_clusters": clusters,
            "methods": list(config["methods"]),
            "operational_gate_names": list(
                _OPERATIONAL_QUALIFICATION_GATE_NAMES
            ),
            "per_record_empirical_target_role": _AGGREGATE_ACCURACY_RULE,
            "aggregate_accuracy_protocol_id": config[
                "qualification_decision"
            ]["aggregate_accuracy_protocol_id"],
        }
    return {
        "schema": "npi.g11.v6-secondary-baselines.v1",
        "protocol_id": config["protocol_id"],
        "config_schema": config["schema"],
        "phase": config["phase"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_artifact_sha256": reference_hash,
        "proposal_training_audit": training_audit,
        "proposal_training_allocation": training_contract,
        "methods": list(config["methods"]),
        "smoke": smoke,
        "records": records,
        "gates": gates,
        "qualification_gates": qualification_gates,
        "qualification_contract": qualification_contract,
        "formal_readiness": formal,
        "secondary_baselines_qualified": all(
            qualification_gates.values()
        )
        and all(formal.values()),
        "seed_ledger": master_ledger.to_dict(),
        "seed_ledger_sha256": master_ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--proposal-training-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.manifest,
        arguments.reference,
        proposal_training_source_path=arguments.proposal_training_source,
        smoke=arguments.smoke,
        checkpoint_directory=arguments.checkpoint_directory,
        resume=arguments.resume,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"qualified": result["secondary_baselines_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
