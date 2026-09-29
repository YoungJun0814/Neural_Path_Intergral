"""Run the frozen V12 ECRPT development micro-study.

This is development evidence, not qualification evidence.  Every estimator is an
ordinary mean under an exact finite-dimensional law.  Candidate screening uses
training-only streams and all candidate-training/screening work is charged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

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
from src.path_integral.ecrpt_protocol import aggregate_ecrpt_microstudy
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.rbergomi_residual_transport import (
    RBergomiResidualTrainingConfig,
    evaluate_rbergomi_residual_transport,
    train_rbergomi_residual_transport,
)
from src.path_integral.residual_transport import ResidualTransportTrainingConfig
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.tail_safe_allocation import (
    BoundedRange,
    StreamingMoments,
    TailSafeAllocationPolicy,
    plan_tail_safe_allocation,
)
from src.path_integral.v10r1_full_latent_dcs import evaluate_full_latent_dcs
from src.path_integral.v10r1_proposal_bank import proposal_from_dict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v12-ecrpt-microstudy.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected ECRPT micro-study schema")
    config["config_path"] = str(path)
    return config, hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any], *, root: Path = ROOT) -> None:
    if config.get("stage") != "development":
        raise ValueError("this runner is frozen to development evidence")
    bindings = config.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("exact artifact bindings are required")
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid binding: {name}")
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"binding mismatch: {name}")
    if int(config["clusters"]) < 2:
        raise ValueError("cluster inference requires at least two clusters")
    if len({str(cell["cell_id"]) for cell in config["cells"]}) != len(config["cells"]):
        raise ValueError("cell IDs must be unique")
    budget = config["final_budget"]
    rqmc_points = int(budget["rqmc_points_per_randomization"])
    if rqmc_points < 1 or rqmc_points & (rqmc_points - 1):
        raise ValueError("RQMC points per randomization must be a power of two")


def _bound_cells(config: dict[str, Any], *, root: Path) -> list[dict[str, Any]]:
    reference = json.loads(
        (root / config["bindings"]["reference"]["path"]).read_text(encoding="utf-8")
    )
    indexed = {str(cell["cell_id"]): cell for cell in reference["cells"]}
    cells: list[dict[str, Any]] = []
    for design in config["cells"]:
        cell_id = str(design["cell_id"])
        if cell_id not in indexed:
            raise ValueError(f"reference lacks cell {cell_id}")
        ref = indexed[cell_id]
        if not math.isclose(float(design["threshold"]), float(ref["threshold"]), rel_tol=0.0, abs_tol=1e-14):
            raise ValueError(f"threshold differs from bound reference for {cell_id}")
        cells.append(
            {
                **design,
                "reference_estimate": float(ref["estimate"]),
                "reference_standard_error": float(ref["standard_error"]),
            }
        )
    return cells


def _problem(config: dict[str, Any], cell: dict[str, Any]) -> RBergomiBaselineProblem:
    model = config["model"]
    return RBergomiBaselineProblem(
        task_id=str(cell["cell_id"]),
        task=TerminalThresholdTask(float(cell["threshold"])),
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        steps=int(model["steps"]),
        hurst=float(cell["hurst"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
    )


def _moments(values: torch.Tensor) -> dict[str, Any]:
    stats = StreamingMoments()
    stats.update(values)
    variance = stats.sample_variance
    return {
        "count": stats.count,
        "sum": stats.mean * stats.count,
        "sum_squares": stats.sum_squares,
        "minimum": stats.minimum,
        "maximum": stats.maximum,
        "nonzero_count": stats.nonzero_count,
        "estimate": stats.mean,
        "variance": variance,
        "standard_error": math.sqrt(variance / stats.count),
    }


def _z_difference(mean: float, standard_error: float) -> float:
    if standard_error > 0.0:
        return abs(mean) / standard_error
    return 0.0 if mean == 0.0 else math.inf


def _combined_reference_z(summary: dict[str, Any], cell: dict[str, Any]) -> float:
    denominator = math.sqrt(
        float(summary["standard_error"]) ** 2
        + float(cell["reference_standard_error"]) ** 2
    )
    return abs(float(summary["estimate"]) - float(cell["reference_estimate"])) / denominator


def _plugin_work_to_target(
    *,
    unit_variance: float,
    unit_work: float,
    one_time_work: float,
    target_variance: float,
    query_count: int,
    minimum_units: int,
) -> dict[str, Any]:
    if unit_variance < 0.0 or unit_work <= 0.0 or one_time_work < 0.0:
        raise ValueError("invalid work inputs")
    required = max(minimum_units, math.ceil(unit_variance / target_variance))
    evaluation = required * unit_work
    amortized = one_time_work / query_count
    return {
        "required_units": required,
        "unit_variance": unit_variance,
        "unit_work": unit_work,
        "one_time_work": one_time_work,
        "amortized_one_time_work": amortized,
        "evaluation_work": evaluation,
        "total_work": evaluation + amortized,
        "status": "descriptive_plugin_only",
    }


def _tail_forecast(
    values: torch.Tensor,
    *,
    bounds: BoundedRange,
    target_variance: float,
    unit_kind: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    stats = StreamingMoments()
    stats.update(values)
    frozen = config["tail_safe_policy"]
    maximum = (
        int(frozen["maximum_rqmc_randomizations"])
        if unit_kind == "rqmc_randomization"
        else int(frozen["maximum_iid_units"])
    )
    policy = TailSafeAllocationPolicy(
        confidence_level=float(frozen["confidence_level"]),
        minimum_iid_units=int(frozen["minimum_iid_units"]),
        minimum_rqmc_randomizations=int(frozen["minimum_rqmc_randomizations"]),
        maximum_units=maximum,
    )
    plan = plan_tail_safe_allocation(
        stats,
        bounds=bounds,
        target_estimator_variance=target_variance,
        unit_kind=unit_kind,  # type: ignore[arg-type]
        policy=policy,
    )
    result = asdict(plan)
    result["status"] = "forecast_from_development_final_units_not_executed"
    return result


def _allocate(
    ledger: SeedLedger,
    *,
    protocol_id: str,
    role: str,
    cell_id: str,
    cluster: int,
    stream: str,
) -> int:
    return ledger.allocate(
        SeedKey(
            protocol=protocol_id,
            role=role,
            regime="terminal-left-tail",
            task=cell_id,
            level=0,
            replicate=cluster,
            stream=stream,
        )
    )


def _training_config(config: dict[str, Any], candidate: dict[str, Any]) -> RBergomiResidualTrainingConfig:
    common = config["candidate_training"]
    return RBergomiResidualTrainingConfig(
        direction_samples=int(common["direction_samples"]),
        transport_samples=int(common["transport_samples"]),
        direction_rates=tuple(float(x) for x in common["direction_rates"]),
        adaptive_rounds=int(common.get("adaptive_rounds", 0)),
        adaptive_samples_per_round=int(common.get("adaptive_samples_per_round", 2048)),
        tempering_powers=(
            tuple(float(x) for x in common["tempering_powers"])
            if "tempering_powers" in common
            else None
        ),
        transport=ResidualTransportTrainingConfig(
            components=int(candidate["components"]),
            defensive_weight=float(common["defensive_weight"]),
            epochs=int(candidate["epochs"]),
            learning_rate=float(candidate["learning_rate"]),
            maximum_mean_norm=float(common["maximum_mean_norm"]),
            initialization_scale=float(common["initialization_scale"]),
            l2_penalty=float(common["l2_penalty"]),
        ),
    )


def _candidate_record(
    *,
    problem: RBergomiBaselineProblem,
    cell: dict[str, Any],
    cluster: int,
    config: dict[str, Any],
    ledger: SeedLedger,
) -> tuple[dict[str, Any], float]:
    trained: list[tuple[float, Any, float]] = []
    total_one_time = 0.0
    for index, candidate in enumerate(config["candidate_training"]["candidates"]):
        training_seed = _allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-training",
            cell_id=problem.task_id,
            cluster=cluster,
            stream=f"components-{candidate['components']}",
        )
        fitted = train_rbergomi_residual_transport(
            problem,
            training_seed=training_seed,
            config=_training_config(config, candidate),
        )
        screening = evaluate_rbergomi_residual_transport(
            problem,
            fitted.proposal,
            sample_count=int(config["candidate_training"]["screening_samples"]),
            gaussian_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="candidate-screening", cell_id=problem.task_id, cluster=cluster, stream=f"gaussian-{index}"),
            label_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="candidate-screening", cell_id=problem.task_id, cluster=cluster, stream=f"label-{index}"),
            coordinate_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="candidate-screening", cell_id=problem.task_id, cluster=cluster, stream=f"coordinate-{index}"),
            reconstruction_paths=min(16, int(config["candidate_training"]["screening_samples"])),
        )
        screening_variance = float(torch.var(screening.ecrpt_contribution, unbiased=True))
        unit_work = screening.ecrpt_cost.algorithmic_work_units / screening.ecrpt_contribution.numel()
        score = screening_variance * unit_work
        total_one_time += (
            fitted.proposal.training_cost.algorithmic_work_units
            + screening.ecrpt_cost.algorithmic_work_units
        )
        trained.append((score, fitted, screening_variance))
    selected_index = min(range(len(trained)), key=lambda i: (trained[i][0], i))
    _, selected, selected_screen_variance = trained[selected_index]
    budget = config["final_budget"]
    final = evaluate_rbergomi_residual_transport(
        problem,
        selected.proposal,
        sample_count=int(budget["candidate_iid_paths"]),
        gaussian_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="candidate-final", cell_id=problem.task_id, cluster=cluster, stream="gaussian"),
        label_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="candidate-final", cell_id=problem.task_id, cluster=cluster, stream="label"),
        coordinate_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="candidate-final", cell_id=problem.task_id, cluster=cluster, stream="coordinate"),
        reconstruction_paths=int(budget["reconstruction_paths"]),
    )
    ecrpt = _moments(final.ecrpt_contribution)
    raw = _moments(final.raw_contribution)
    difference = _moments(final.raw_contribution - final.ecrpt_contribution)
    likelihood = _moments(final.likelihood_normalization)
    target_variance = (float(config["relative_rmse_target"]) * float(cell["reference_estimate"])) ** 2
    delta = selected.proposal.spec().defensive_weight
    unit_work = final.ecrpt_cost.algorithmic_work_units / int(ecrpt["count"])
    paired_z = _z_difference(float(difference["estimate"]), float(difference["standard_error"]))
    norm_z = _z_difference(float(likelihood["estimate"]) - 1.0, float(likelihood["standard_error"]))
    return {
        "proposal": asdict(selected.proposal),
        "selected_candidate_index": selected_index,
        "selected_components": selected.proposal.components,
        "candidate_scores_variance_times_unit_work": [float(item[0]) for item in trained],
        "candidate_screening_variances": [float(item[2]) for item in trained],
        "selected_screening_variance": selected_screen_variance,
        "charged_one_time_algorithmic_work_units": total_one_time,
        "estimate": ecrpt,
        "raw_estimate": raw,
        "difference": difference,
        "likelihood": likelihood,
        "paired_difference_z": paired_z,
        "likelihood_normalization_z": norm_z,
        "combined_reference_z": _combined_reference_z(ecrpt, cell),
        # A zero observed denominator is not evidence of infinite improvement.
        # The bounded tail certificate remains valid, while the empirical
        # mechanism gate fails closed until positive variance is resolved.
        "raw_over_ecrpt_variance_ratio": (
            float(raw["variance"]) / float(ecrpt["variance"])
            if float(ecrpt["variance"]) > 0.0
            else 0.0
        ),
        "exactness": {
            "maximum_residual_projection_error": final.maximum_residual_projection_error,
            "maximum_path_reconstruction_error": final.maximum_path_reconstruction_error,
            "maximum_full_path_reconstruction_error": final.maximum_full_path_reconstruction_error,
            "maximum_likelihood_bound_violation": final.maximum_likelihood_bound_violation,
            "hard_threshold_mismatch_count": float(final.hard_threshold_mismatch_count),
        },
        "evaluation_cost": asdict(final.ecrpt_cost),
        "plugin_work_to_target": _plugin_work_to_target(
            unit_variance=float(ecrpt["variance"]),
            unit_work=unit_work,
            one_time_work=total_one_time,
            target_variance=target_variance,
            query_count=int(config["primary_query_count"]),
            minimum_units=int(config["tail_safe_policy"]["minimum_iid_units"]),
        ),
        "tail_safe_forecast": _tail_forecast(
            final.ecrpt_contribution,
            bounds=BoundedRange(0.0, 1.0 / delta),
            target_variance=target_variance,
            unit_kind="iid_path",
            config=config,
        ),
    }, target_variance


def _comparator_record(
    *,
    name: str,
    values: torch.Tensor,
    cost: BaselineCostLedger,
    one_time_work: float,
    bounds: BoundedRange,
    unit_kind: str,
    cell: dict[str, Any],
    config: dict[str, Any],
    target_variance: float,
) -> dict[str, Any]:
    summary = _moments(values)
    unit_work = cost.algorithmic_work_units / int(summary["count"])
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
            unit_work=unit_work,
            one_time_work=one_time_work,
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
    *,
    problem: RBergomiBaselineProblem,
    cell: dict[str, Any],
    cluster: int,
    config: dict[str, Any],
    ledger: SeedLedger,
    bank_entry: dict[str, Any],
) -> dict[str, Any]:
    candidate, target_variance = _candidate_record(
        problem=problem, cell=cell, cluster=cluster, config=config, ledger=ledger
    )
    budget = config["final_budget"]
    conditional_proposal = freeze_conditional_rbergomi_proposal(
        problem,
        training_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="comparator-freeze", cell_id=problem.task_id, cluster=cluster, stream="conditional"),
    )
    conditional = evaluate_conditional_terminal_units(
        problem,
        conditional_proposal,
        sample_count=int(budget["conditional_iid_paths"]),
        seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="comparator-final", cell_id=problem.task_id, cluster=cluster, stream="conditional"),
    )
    conditional_work = conditional.raw_sample_count * (
        problem.local_dimension + problem.steps + 1
    )
    conditional_cost = BaselineCostLedger(
        final_samples=conditional.raw_sample_count,
        cdf_calls=conditional.cdf_calls,
        algorithmic_work_units=float(conditional_work),
    )
    rqmc_proposal = freeze_smoothing_rqmc_proposal(
        problem,
        training_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="comparator-freeze", cell_id=problem.task_id, cluster=cluster, stream="rqmc"),
    )
    rqmc = evaluate_smoothing_rqmc_units(
        problem,
        rqmc_proposal,
        randomizations=int(budget["rqmc_randomizations"]),
        points_per_randomization=int(budget["rqmc_points_per_randomization"]),
        seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="comparator-final", cell_id=problem.task_id, cluster=cluster, stream="rqmc"),
    )
    rqmc_work = rqmc.raw_sample_count * (problem.latent_dimension + problem.steps + 1)
    rqmc_cost = BaselineCostLedger(
        final_samples=rqmc.raw_sample_count,
        cdf_calls=rqmc.cdf_calls,
        algorithmic_work_units=float(rqmc_work),
    )
    v10_proposal = proposal_from_dict(bank_entry["proposal"])
    v10 = evaluate_full_latent_dcs(
        problem,
        v10_proposal,
        sample_count=int(budget["v10r1_iid_paths"]),
        gaussian_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="comparator-final", cell_id=problem.task_id, cluster=cluster, stream="v10-gaussian"),
        label_seed=_allocate(ledger, protocol_id=str(config["protocol_id"]), role="comparator-final", cell_id=problem.task_id, cluster=cluster, stream="v10-label"),
    )
    comparators = {
        "conditional_rbergomi": _comparator_record(
            name="conditional_rbergomi", values=conditional.unit_contributions,
            cost=conditional_cost, one_time_work=0.0, bounds=BoundedRange(0.0, 1.0),
            unit_kind="iid_path", cell=cell, config=config, target_variance=target_variance,
        ),
        "smoothing_rqmc": _comparator_record(
            name="smoothing_rqmc", values=rqmc.unit_contributions,
            cost=rqmc_cost, one_time_work=0.0, bounds=BoundedRange(0.0, 1.0),
            unit_kind="rqmc_randomization", cell=cell, config=config, target_variance=target_variance,
        ),
        "v10r1_full_cem_dcs": _comparator_record(
            name="v10r1_full_cem_dcs", values=v10.dcs_contribution,
            cost=v10.dcs_cost,
            one_time_work=v10_proposal.training_cost.algorithmic_work_units,
            bounds=BoundedRange(0.0, 1.0 / float(v10_proposal.component_weights[0])),
            unit_kind="iid_path", cell=cell, config=config, target_variance=target_variance,
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
        (str(entry["cell_id"]), int(entry["replicate"])): entry
        for entry in bank["entries"]
    }
    ledger = SeedLedger()
    records: list[dict[str, Any]] = []
    for cell in cells:
        problem = _problem(config, cell)
        for cluster in range(int(config["clusters"])):
            key = (problem.task_id, cluster)
            if key not in bank_index:
                raise ValueError(f"V10R1 bank lacks {key}")
            records.append(
                _record(
                    problem=problem,
                    cell=cell,
                    cluster=cluster,
                    config=config,
                    ledger=ledger,
                    bank_entry=bank_index[key],
                )
            )
    aggregate = aggregate_ecrpt_microstudy(config=config, records=records)
    correctness = bool(aggregate["correctness_pass"])
    performance = bool(aggregate["performance_pass"])
    return {
        "schema": "npi.g11.v12-ecrpt-microstudy-result.v1",
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
            "software_correctness_evidence_pass": correctness,
            "development_performance_gate_pass": performance,
            "qualification_authorized": correctness and performance,
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
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
