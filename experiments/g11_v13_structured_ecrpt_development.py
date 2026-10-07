"""Run the frozen V13 structured ECRPT development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g11_v12_ecrpt_microstudy import (
    _allocate,
    _bound_cells,
    _combined_reference_z,
    _moments,
    _plugin_work_to_target,
    _problem,
    _sha256,
    _z_difference,
)
from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.conditional_rbergomi import (
    evaluate_conditional_terminal_units,
    freeze_conditional_rbergomi_proposal,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.baselines.smoothing_rqmc import (
    evaluate_smoothing_rqmc_units,
    freeze_smoothing_rqmc_proposal,
)
from src.path_integral.ecrpt_protocol import aggregate_structured_ecrpt_development
from src.path_integral.low_rank_residual_flow import LowRankResidualFlowTrainingConfig
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.rbergomi_structured_ecrpt import (
    StructuredECRPTTrainingConfig,
    evaluate_structured_ecrpt,
    train_structured_ecrpt,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig
from src.path_integral.residual_stability import rao_blackwell_variance_gap_integrand
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.tail_safe_allocation import (
    BoundedRange,
    StreamingMoments,
    TailSafeAllocationPolicy,
    plan_empirical_bernstein_tail_safe_allocation,
)
from src.path_integral.v10r1_full_latent_dcs import evaluate_full_latent_dcs
from src.path_integral.v10r1_proposal_bank import proposal_from_dict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v13-structured-ecrpt-development.v1"


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V13 development schema")
    config["config_path"] = str(path)
    return config, hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any], *, root: Path = ROOT) -> None:
    if config.get("stage") != "development":
        raise ValueError("V13 runner accepts development evidence only")
    for name, binding in config.get("bindings", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid binding: {name}")
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"binding mismatch: {name}")
    if int(config["clusters"]) < 2:
        raise ValueError("cluster inference requires at least two clusters")
    if len({str(cell["cell_id"]) for cell in config["cells"]}) != len(config["cells"]):
        raise ValueError("cell IDs must be unique")
    points = int(config["final_budget"]["rqmc_points_per_randomization"])
    if points < 1 or points & (points - 1):
        raise ValueError("RQMC points must be a power of two")


def _training_config(config: dict[str, Any]) -> StructuredECRPTTrainingConfig:
    frozen = config["candidate_training"]
    return StructuredECRPTTrainingConfig(
        direction_samples=int(frozen["direction_samples"]),
        direction_rates=tuple(float(x) for x in frozen["direction_rates"]),
        smc=AdaptiveResidualSMCConfig(
            particles=int(frozen["smc_particles"]),
            target_ess_fraction=float(frozen["target_ess_fraction"]),
            pcn_scale=float(frozen["pcn_scale"]),
            pcn_sweeps_per_stage=int(frozen["pcn_sweeps_per_stage"]),
            maximum_stages=int(frozen["maximum_stages"]),
        ),
        flow=LowRankResidualFlowTrainingConfig(
            layers=int(frozen["flow_layers"]),
            rank=int(frozen["flow_rank"]),
            epochs=int(frozen["flow_epochs"]),
            learning_rate=float(frozen["learning_rate"]),
            defensive_weight=float(frozen["defensive_weight"]),
            maximum_log_scale=float(frozen["maximum_log_scale"]),
            l2_penalty=float(frozen["l2_penalty"]),
            partition_style=str(frozen["partition_style"]),  # type: ignore[arg-type]
        ),
    )


def _moments_v13(values: torch.Tensor) -> dict[str, Any]:
    """Store the V12 arithmetic summary plus the fourth sufficient statistic."""

    summary = _moments(values)
    moments = StreamingMoments()
    moments.update(values)
    summary["sum_fourth_powers"] = moments.sum_fourth_powers
    return summary


def _tail_forecast(
    values: torch.Tensor,
    *,
    bounds: BoundedRange,
    target_variance: float,
    unit_kind: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    moments = StreamingMoments()
    moments.update(values)
    frozen = config["tail_safe_policy"]
    maximum = int(
        frozen[
            "maximum_rqmc_randomizations"
            if unit_kind == "rqmc_randomization"
            else "maximum_iid_units"
        ]
    )
    plan = plan_empirical_bernstein_tail_safe_allocation(
        moments,
        bounds=bounds,
        target_estimator_variance=target_variance,
        unit_kind=unit_kind,  # type: ignore[arg-type]
        policy=TailSafeAllocationPolicy(
            confidence_level=float(frozen["confidence_level"]),
            minimum_iid_units=int(frozen["minimum_iid_units"]),
            minimum_rqmc_randomizations=int(frozen["minimum_rqmc_randomizations"]),
            maximum_units=maximum,
        ),
    )
    payload = asdict(plan)
    payload["status"] = "v13_development_forecast_not_executed"
    return payload


def _candidate_record(
    problem: RBergomiBaselineProblem,
    cell: dict[str, Any],
    cluster: int,
    config: dict[str, Any],
    ledger: SeedLedger,
) -> tuple[dict[str, Any], float]:
    training = train_structured_ecrpt(
        problem,
        training_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-training",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="structured-ecrpt",
        ),
        config=_training_config(config),
    )
    budget = config["final_budget"]
    final = evaluate_structured_ecrpt(
        problem,
        training.proposal,
        sample_count=int(budget["candidate_iid_paths"]),
        gaussian_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="gaussian",
        ),
        label_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="label",
        ),
        coordinate_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="coordinate",
        ),
        reconstruction_paths=int(budget["reconstruction_paths"]),
    )
    estimate = _moments_v13(final.ecrpt_contribution)
    raw = _moments_v13(final.raw_contribution)
    difference = _moments_v13(final.raw_contribution - final.ecrpt_contribution)
    likelihood = _moments_v13(final.likelihood_normalization)
    conditional_probability = final.ecrpt_contribution / final.likelihood_normalization
    gap = rao_blackwell_variance_gap_integrand(
        final.likelihood_normalization, conditional_probability
    )
    target_variance = (
        float(config["relative_rmse_target"]) * float(cell["reference_estimate"])
    ) ** 2
    one_time = training.proposal.training_cost.algorithmic_work_units
    unit_work = final.ecrpt_cost.algorithmic_work_units / int(estimate["count"])
    return {
        "proposal": asdict(training.proposal),
        "direction_selection": asdict(training.direction_selection),
        "smc": {
            "root_seed": training.smc.root_seed,
            "used_seeds": list(training.smc.used_seeds),
            "final_beta": training.smc.final_beta,
            "final_particles_equally_weighted": training.smc.final_particles_equally_weighted,
            "particles_are_final_inferential_units": training.smc.particles_are_final_inferential_units,
            "stages": [asdict(stage) for stage in training.smc.stages],
        },
        "all_training_seeds": list(training.all_training_seeds),
        "charged_one_time_algorithmic_work_units": one_time,
        "estimate": estimate,
        "raw_estimate": raw,
        "difference": difference,
        "likelihood": likelihood,
        "paired_difference_z": _z_difference(
            float(difference["estimate"]), float(difference["standard_error"])
        ),
        "likelihood_normalization_z": _z_difference(
            float(likelihood["estimate"]) - 1.0, float(likelihood["standard_error"])
        ),
        "combined_reference_z": _combined_reference_z(estimate, cell),
        "raw_over_ecrpt_variance_ratio": float(raw["variance"]) / float(estimate["variance"])
        if float(estimate["variance"]) > 0.0
        else 0.0,
        "rao_blackwell_gap_estimate": float(torch.mean(gap)),
        "rao_blackwell_gap_minimum": float(torch.amin(gap)),
        "exactness": {
            "maximum_residual_projection_error": final.maximum_residual_projection_error,
            "maximum_path_reconstruction_error": final.maximum_path_reconstruction_error,
            "maximum_full_path_reconstruction_error": final.maximum_full_path_reconstruction_error,
            "maximum_likelihood_bound_violation": final.maximum_likelihood_bound_violation,
            "hard_threshold_mismatch_count": float(final.hard_threshold_mismatch_count),
        },
        "evaluation_cost": asdict(final.ecrpt_cost),
        "plugin_work_to_target": _plugin_work_to_target(
            unit_variance=float(estimate["variance"]),
            unit_work=unit_work,
            one_time_work=one_time,
            target_variance=target_variance,
            query_count=int(config["primary_query_count"]),
            minimum_units=int(config["tail_safe_policy"]["minimum_iid_units"]),
        ),
        "tail_safe_forecast": _tail_forecast(
            final.ecrpt_contribution,
            bounds=BoundedRange(0.0, 1.0 / training.proposal.defensive_weight),
            target_variance=target_variance,
            unit_kind="iid_path",
            config=config,
        ),
    }, target_variance


def _comparator_record(
    name: str,
    values: torch.Tensor,
    cost: BaselineCostLedger,
    one_time: float,
    bounds: BoundedRange,
    unit_kind: str,
    cell: dict[str, Any],
    config: dict[str, Any],
    target_variance: float,
) -> dict[str, Any]:
    summary = _moments_v13(values)
    minimum = int(
        config["tail_safe_policy"][
            "minimum_rqmc_randomizations"
            if unit_kind == "rqmc_randomization"
            else "minimum_iid_units"
        ]
    )
    return {
        "method": name,
        "estimate": summary,
        "combined_reference_z": _combined_reference_z(summary, cell),
        "evaluation_cost": asdict(cost),
        "plugin_work_to_target": _plugin_work_to_target(
            unit_variance=float(summary["variance"]),
            unit_work=cost.algorithmic_work_units / int(summary["count"]),
            one_time_work=one_time,
            target_variance=target_variance,
            query_count=int(config["primary_query_count"]),
            minimum_units=minimum,
        ),
        "tail_safe_forecast": _tail_forecast(
            values,
            bounds=bounds,
            target_variance=target_variance,
            unit_kind=unit_kind,
            config=config,
        ),
    }


def _record(
    problem: RBergomiBaselineProblem,
    cell: dict[str, Any],
    cluster: int,
    config: dict[str, Any],
    ledger: SeedLedger,
    bank_entry: dict[str, Any],
) -> dict[str, Any]:
    candidate, target_variance = _candidate_record(problem, cell, cluster, config, ledger)
    budget = config["final_budget"]
    conditional_proposal = freeze_conditional_rbergomi_proposal(
        problem,
        training_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-freeze",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="conditional",
        ),
    )
    conditional = evaluate_conditional_terminal_units(
        problem,
        conditional_proposal,
        sample_count=int(budget["conditional_iid_paths"]),
        seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="conditional",
        ),
    )
    conditional_cost = BaselineCostLedger(
        final_samples=conditional.raw_sample_count,
        cdf_calls=conditional.cdf_calls,
        algorithmic_work_units=float(
            conditional.raw_sample_count * (problem.local_dimension + problem.steps + 1)
        ),
    )
    rqmc_proposal = freeze_smoothing_rqmc_proposal(
        problem,
        training_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-freeze",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="rqmc",
        ),
    )
    rqmc = evaluate_smoothing_rqmc_units(
        problem,
        rqmc_proposal,
        randomizations=int(budget["rqmc_randomizations"]),
        points_per_randomization=int(budget["rqmc_points_per_randomization"]),
        seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="rqmc",
        ),
    )
    rqmc_cost = BaselineCostLedger(
        final_samples=rqmc.raw_sample_count,
        cdf_calls=rqmc.cdf_calls,
        algorithmic_work_units=float(
            rqmc.raw_sample_count * (problem.latent_dimension + problem.steps + 1)
        ),
    )
    v10_proposal = proposal_from_dict(bank_entry["proposal"])
    v10 = evaluate_full_latent_dcs(
        problem,
        v10_proposal,
        sample_count=int(budget["v10r1_iid_paths"]),
        gaussian_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="v10-gaussian",
        ),
        label_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="v10-label",
        ),
    )
    comparators = {
        "conditional_rbergomi": _comparator_record(
            "conditional_rbergomi",
            conditional.unit_contributions,
            conditional_cost,
            0.0,
            BoundedRange(0.0, 1.0),
            "iid_path",
            cell,
            config,
            target_variance,
        ),
        "smoothing_rqmc": _comparator_record(
            "smoothing_rqmc",
            rqmc.unit_contributions,
            rqmc_cost,
            0.0,
            BoundedRange(0.0, 1.0),
            "rqmc_randomization",
            cell,
            config,
            target_variance,
        ),
        "v10r1_full_cem_dcs": _comparator_record(
            "v10r1_full_cem_dcs",
            v10.dcs_contribution,
            v10.dcs_cost,
            v10_proposal.training_cost.algorithmic_work_units,
            BoundedRange(0.0, 1.0 / float(v10_proposal.component_weights[0])),
            "iid_path",
            cell,
            config,
            target_variance,
        ),
    }
    return {
        "cell_id": problem.task_id,
        "cluster": cluster,
        "hurst": float(cell["hurst"]),
        "nominal_probability": float(cell["nominal_probability"]),
        "threshold": float(cell["threshold"]),
        "reference_estimate": float(cell["reference_estimate"]),
        "reference_standard_error": float(cell["reference_standard_error"]),
        "target_estimator_variance": target_variance,
        "candidate": candidate,
        "comparators": comparators,
    }


def run(config: dict[str, Any], config_sha256: str, *, root: Path = ROOT) -> dict[str, Any]:
    validate_config(config, root=root)
    cells = _bound_cells(config, root=root)
    bank = json.loads(
        (root / config["bindings"]["v10r1_proposal_bank"]["path"]).read_text(encoding="utf-8")
    )
    bank_index = {
        (str(entry["cell_id"]), int(entry["replicate"])): entry for entry in bank["entries"]
    }
    ledger = SeedLedger()
    records = []
    for cell in cells:
        problem = _problem(config, cell)
        for cluster in range(int(config["clusters"])):
            key = (problem.task_id, cluster)
            if key not in bank_index:
                raise ValueError(f"V10R1 bank lacks {key}")
            records.append(_record(problem, cell, cluster, config, ledger, bank_index[key]))
    aggregate = aggregate_structured_ecrpt_development(config=config, records=records)
    authorized = bool(aggregate["stage_pass"])
    return {
        "schema": "npi.g11.v13-structured-ecrpt-development-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "stage": "development",
        "config_path": str(config["config_path"]),
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "bindings": config["bindings"],
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "records": records,
        "aggregate": aggregate,
        "decision": {
            "software_correctness_evidence_pass": bool(aggregate["correctness_pass"]),
            "development_performance_gate_pass": bool(aggregate["performance_pass"]),
            "qualification_authorized": authorized,
            "qualification_executed": False,
            "broad_performance_claim_authorized": False,
            "top_journal_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, digest = load_config(args.config)
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
