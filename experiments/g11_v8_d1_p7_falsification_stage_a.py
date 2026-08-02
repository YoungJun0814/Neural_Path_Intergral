"""Run D1 Stage A falsification on moderate and rare terminal/barrier cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from src.path_integral.baseline_diagnostics import (
    evaluate_baseline_likelihood_diagnostics,
    evaluate_coupling_flow_roundtrip,
)
from src.path_integral.baselines import (
    CEMTrainingConfig,
    FlowTrainingConfig,
    LargeDeviationTrainingConfig,
    RBergomiBaselineProblem,
    freeze_conditional_rbergomi_proposal,
    freeze_crude_or_antithetic_proposal,
    freeze_smoothing_rqmc_proposal,
    train_cem_proposal,
    train_coupling_flow_proposal,
    train_large_deviation_proposal,
)
from src.path_integral.benchmark_executor import (
    BaselineExecutionRequest,
    execute_baseline_lifecycle,
)
from src.path_integral.controllers.markov import TimePiecewiseTwoDriverControl
from src.path_integral.dcs_benchmark import evaluate_paired_dcs_benchmark
from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    TerminalThresholdTask,
)
from src.path_integral.provenance import runtime_provenance, source_provenance

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-d1-p7-falsification-stage-a.v1"
SCHEMA_V2 = "npi.g11.v8-d1-p7-falsification-stage-a.v2"
RESULT_SCHEMA = "npi.g11.v8-d1-p7-falsification-stage-a-result.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") not in {SCHEMA, SCHEMA_V2}:
        raise ValueError("unexpected D1 Stage A config schema")
    if config.get("schema") == SCHEMA_V2:
        parent = config.get("parent_config")
        if not isinstance(parent, dict) or set(parent) != {"path", "sha256"}:
            raise ValueError("D1 V2 requires an exact parent-config binding")
        parent_path = ROOT / str(parent["path"])
        if not parent_path.is_file() or parent["sha256"] != _sha256(parent_path):
            raise ValueError("D1 V2 parent-config binding mismatch")
        inherited = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
        if not isinstance(inherited, dict) or inherited.get("schema") != SCHEMA:
            raise ValueError("D1 V2 parent has the wrong schema")
        inherited["schema"] = SCHEMA_V2
        inherited["protocol_id"] = config.get("protocol_id")
        inherited["namespace"] = config.get("namespace")
        inherited["base_seed"] = config.get("base_seed")
        inherited["outcome_data_used_before_freeze"] = config.get("outcome_data_used_before_freeze")
        inherited["bindings"] = dict(inherited["bindings"])
        inherited["bindings"]["v1_failure_receipt"] = config.get("failure_receipt")
        config = inherited
    if config.get("outcome_data_used_before_freeze") is not False:
        raise ValueError("D1 Stage A config must be outcome-blind for its namespace")
    return config, hashlib.sha256(raw).hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    for name, binding in config.get("bindings", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid binding: {name}")
        relative = binding["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or "\\" in relative:
            raise ValueError(f"nonportable binding: {name}")
        path = ROOT / relative
        if not path.is_file() or binding["sha256"] != _sha256(path):
            raise ValueError(f"binding hash mismatch: {name}")
    parent = config.get("source_parent_commit")
    current = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    if parent != current:
        raise ValueError("D1 Stage A must start from the frozen B1 parent commit")
    b1 = json.loads((ROOT / config["bindings"]["b1_audit"]["path"]).read_text())
    reference_audit = json.loads((ROOT / config["bindings"]["reference_audit"]["path"]).read_text())
    if b1.get("passed") is not True or reference_audit.get("passed") is not True:
        raise ValueError("B1 and R2 audits must pass before D1")
    if config.get("performance_claim_authorized") is not False:
        raise ValueError("D1 Stage A cannot authorize performance claims")
    calibration = json.loads(
        (ROOT / config["bindings"]["threshold_calibration"]["path"]).read_text()
    )
    calibrated = {
        item["cell_id"]: float(item["calibrated_threshold"]) for item in calibration["cells"]
    }
    reference = json.loads((ROOT / config["bindings"]["reference"]["path"]).read_text())
    references = {
        item["cell_id"]: (float(item["estimate"]), float(item["standard_error"]))
        for item in reference["cells"]
        if item["method"] == "dcs_reference"
    }
    cells = config.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("D1 Stage A requires exactly four representative cells")
    for cell in cells:
        cell_id = cell.get("cell_id")
        if (
            cell_id not in calibrated
            or float(cell.get("threshold")) != calibrated[cell_id]
            or cell_id not in references
            or (
                float(cell.get("reference_estimate")),
                float(cell.get("reference_standard_error")),
            )
            != references[cell_id]
        ):
            raise ValueError(f"D1 cell is not threshold/reference bound: {cell_id}")
    primary = config.get("external_methods", {}).get("primary")
    secondary = config.get("external_methods", {}).get("secondary")
    if primary != ["pure_cem", "smoothing_rqmc"]:
        raise ValueError("D1 primary external comparator roster changed")
    if not isinstance(secondary, list) or set(primary) & set(secondary):
        raise ValueError("D1 primary and secondary comparator rosters overlap")
    budgets = config.get("budgets")
    if not isinstance(budgets, list) or [item.get("id") for item in budgets] != ["low", "high"]:
        raise ValueError("D1 requires the frozen low/high budget ladder")
    if int(config.get("clusters", 0)) < 2:
        raise ValueError("D1 requires at least two independent clusters")


def _problem(cell: dict[str, Any], model: dict[str, Any]) -> RBergomiBaselineProblem:
    threshold = float(cell["threshold"])
    task: TerminalThresholdTask | DiscreteBarrierHitTask
    if cell["task"] == "terminal_left_tail":
        task = TerminalThresholdTask(threshold)
    elif cell["task"] == "discrete_lower_barrier":
        task = DiscreteBarrierHitTask(threshold)
    else:
        raise ValueError("unsupported D1 task")
    return RBergomiBaselineProblem(
        task_id=str(cell["cell_id"]),
        task=task,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        steps=int(model["steps"]),
        hurst=float(model["hurst"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
    )


def _proposal_entry(config: dict[str, Any], cell_id: str) -> dict[str, Any]:
    manifest = json.loads((ROOT / config["bindings"]["proposal_manifest"]["path"]).read_text())
    matches = [
        item
        for item in manifest["entries"]
        if item["cell_id"] == cell_id and item["method"] == "dcs_reference"
    ]
    if len(matches) != 1:
        raise ValueError(f"missing unique DCS proposal for {cell_id}")
    return matches[0]


def _controls(entry: dict[str, Any], maturity: float):
    return tuple(
        TimePiecewiseTwoDriverControl(
            tuple((float(segment[0]), float(segment[1])) for segment in schedule),
            maturity=maturity,
        )
        for schedule in entry["schedules"]
    )


def _train_proposal(
    problem: RBergomiBaselineProblem,
    method: str,
    seed: int,
    budget: dict[str, Any],
):
    if method in {"crude_mc", "antithetic_mc"}:
        return freeze_crude_or_antithetic_proposal(
            problem,
            method=cast(Literal["crude_mc", "antithetic_mc"], method),
            training_seed=seed,
        )
    if method == "conditional_rbergomi":
        return freeze_conditional_rbergomi_proposal(problem, training_seed=seed)
    if method == "smoothing_rqmc":
        return freeze_smoothing_rqmc_proposal(problem, training_seed=seed)
    if method in {"pure_cem", "defensive_cem"}:
        return train_cem_proposal(
            problem,
            method=cast(Literal["pure_cem", "defensive_cem"], method),
            training_seed=seed,
            config=CEMTrainingConfig(**budget["cem"]),
        )
    if method == "ld_subspace_is":
        return train_large_deviation_proposal(
            problem,
            training_seed=seed,
            config=LargeDeviationTrainingConfig(**budget["large_deviation"]),
        )
    if method == "flow_is":
        return train_coupling_flow_proposal(
            problem,
            training_seed=seed,
            config=FlowTrainingConfig(**budget["flow"]),
        )
    raise ValueError(f"unsupported D1 method: {method}")


def _moments(values: torch.Tensor) -> tuple[float, float, float]:
    mean = float(torch.mean(values))
    variance = float(torch.var(values, unbiased=True))
    return mean, variance, math.sqrt(variance / values.numel())


def _paired_record(
    config: dict[str, Any],
    cell: dict[str, Any],
    budget: dict[str, Any],
    cluster: int,
    seeds: tuple[int, int],
) -> dict[str, Any]:
    problem = _problem(cell, config["model"])
    entry = _proposal_entry(config, problem.task_id)
    batch = evaluate_paired_dcs_benchmark(
        task=problem.task,
        task_id=problem.task_id,
        spot=problem.spot,
        maturity=problem.maturity,
        steps=problem.steps,
        hurst=problem.hurst,
        eta=problem.eta,
        xi=problem.xi,
        rho=problem.rho,
        controls=_controls(entry, problem.maturity),
        weights=torch.tensor(entry["weights"], dtype=torch.float64),
        sample_count=int(budget["paired_dcs_paths"]),
        path_seed=seeds[0],
        label_seed=seeds[1],
    )
    raw_mean, raw_variance, raw_se = _moments(batch.raw_contribution)
    dcs_mean, dcs_variance, dcs_se = _moments(batch.dcs_contribution)
    difference = batch.raw_contribution - batch.dcs_contribution
    difference_mean, difference_variance, difference_se = _moments(difference)
    normalization_mean, normalization_variance, normalization_se = _moments(
        batch.likelihood_normalization
    )
    normalization_z = (
        (normalization_mean - 1.0) / normalization_se
        if normalization_se > 0.0
        else (0.0 if normalization_mean == 1.0 else math.inf)
    )
    reference = float(cell["reference_estimate"])
    reference_se = float(cell["reference_standard_error"])
    return {
        "cell_id": problem.task_id,
        "budget_id": budget["id"],
        "cluster": cluster,
        "path_seed": seeds[0],
        "label_seed": seeds[1],
        "sample_count": batch.raw_contribution.numel(),
        "proposal_source": entry["entry_id"],
        "component_counts": batch.component_counts,
        "raw": {
            "estimate": raw_mean,
            "variance": raw_variance,
            "standard_error": raw_se,
            "combined_reference_z": abs(raw_mean - reference)
            / math.sqrt(raw_se**2 + reference_se**2),
            "cost": asdict(batch.raw_cost),
        },
        "dcs": {
            "estimate": dcs_mean,
            "variance": dcs_variance,
            "standard_error": dcs_se,
            "combined_reference_z": abs(dcs_mean - reference)
            / math.sqrt(dcs_se**2 + reference_se**2),
            "cost": asdict(batch.dcs_cost),
        },
        "mechanism": {
            "variance_ratio_raw_over_dcs": raw_variance / dcs_variance
            if dcs_variance > 0.0
            else math.inf,
            "difference_mean": difference_mean,
            "difference_variance": difference_variance,
            "difference_standard_error": difference_se,
            "difference_z": abs(difference_mean) / difference_se
            if difference_se > 0.0
            else (0.0 if difference_mean == 0.0 else math.inf),
        },
        "likelihood": {
            "normalization_mean": normalization_mean,
            "normalization_variance": normalization_variance,
            "normalization_standard_error": normalization_se,
            "normalization_z": normalization_z,
            "log_weight_minimum": float(torch.amin(batch.log_likelihood)),
            "log_weight_median": float(torch.quantile(batch.log_likelihood, 0.5)),
            "log_weight_q99": float(torch.quantile(batch.log_likelihood, 0.99)),
            "log_weight_maximum": float(torch.amax(batch.log_likelihood)),
        },
        "exactness": {
            "maximum_path_reconstruction_error": batch.maximum_path_reconstruction_error,
            "maximum_component_density_error": batch.maximum_component_density_error,
            "maximum_mixture_density_error": batch.maximum_mixture_density_error,
            "maximum_full_likelihood_error": batch.maximum_full_likelihood_error,
        },
    }


def _external_record(
    config: dict[str, Any],
    cell: dict[str, Any],
    budget: dict[str, Any],
    method: str,
    cluster: int,
    seeds: tuple[int, int, int, int],
) -> dict[str, Any]:
    problem = _problem(cell, config["model"])
    proposal = _train_proposal(problem, method, seeds[0], budget)
    rqmc = method == "smoothing_rqmc"
    maximum_units = int(
        budget["rqmc_maximum_final_units"] if rqmc else budget["iid_maximum_final_units"]
    )
    points = int(budget["rqmc_points_per_randomization"]) if rqmc else 1
    target_variance = (
        float(config["relative_rmse_target"]) * float(cell["reference_estimate"])
    ) ** 2
    artifact = execute_baseline_lifecycle(
        problem,
        proposal,
        BaselineExecutionRequest(
            pilot_units=int(budget["pilot_units"]),
            target_estimator_variance=target_variance,
            pilot_seed=seeds[1],
            final_seed=seeds[2],
            minimum_final_units=4,
            maximum_final_units=maximum_units,
            rqmc_points_per_randomization=points,
        ),
    )
    diagnostics = evaluate_baseline_likelihood_diagnostics(
        proposal,
        sample_count=int(config["diagnostic_samples"]),
        seed=seeds[3],
    )
    required_units = math.ceil(artifact.pilot_variance / target_variance)
    estimate = artifact.estimate
    estimate_se = math.sqrt(estimate.estimator_variance)
    reference = float(cell["reference_estimate"])
    reference_se = float(cell["reference_standard_error"])
    total_cost = artifact.audit.total_cost
    diagnostic_work = int(config["diagnostic_samples"]) * 2 * proposal.dimension
    flow_roundtrip = (
        asdict(
            evaluate_coupling_flow_roundtrip(
                proposal,
                sample_count=min(256, int(config["diagnostic_samples"])),
                seed=seeds[3],
            )
        )
        if method == "flow_is"
        else None
    )
    return {
        "cell_id": problem.task_id,
        "budget_id": budget["id"],
        "method": method,
        "cluster": cluster,
        "seeds": {
            "training": seeds[0],
            "pilot": seeds[1],
            "final": seeds[2],
            "diagnostic": seeds[3],
        },
        "proposal": {
            "sha256": proposal.sha256,
            "family": proposal.family,
            "exact_likelihood": proposal.exact_likelihood,
            "self_normalized": proposal.self_normalized,
            "conditional_integral": proposal.conditional_integral,
            "training_cost": asdict(proposal.training_cost),
            "fixed_identity_covariance": method in {"pure_cem", "defensive_cem"},
            "defensive_target_component": method in {"defensive_cem", "ld_subspace_is"},
        },
        "pilot": {
            "unit_count": artifact.pilot_unit_count,
            "mean": artifact.pilot_mean,
            "variance": artifact.pilot_variance,
            "cost": asdict(artifact.pilot_cost),
        },
        "allocation": {
            "target_estimator_variance": target_variance,
            "required_unclipped_units": required_units,
            "planned_units": artifact.plan.planned_units,
            "points_per_unit": artifact.plan.points_per_unit,
            "planned_final_samples": artifact.plan.planned_final_samples,
            "floor_binding": required_units < 4,
            "resource_censored": required_units > maximum_units,
        },
        "estimate": {
            "value": estimate.estimate,
            "standard_error": estimate_se,
            "variance": estimate.estimator_variance,
            "combined_reference_z": abs(estimate.estimate - reference)
            / math.sqrt(estimate_se**2 + reference_se**2),
            "final_cost": asdict(estimate.final_cost),
        },
        "likelihood_diagnostics": asdict(diagnostics),
        "flow_roundtrip": flow_roundtrip,
        "lifecycle_audit_passed": artifact.audit.passed,
        "total_algorithmic_work_units_including_diagnostic": (
            total_cost.algorithmic_work_units + diagnostic_work
        ),
        "total_wall_seconds_excluding_diagnostic": total_cost.wall_seconds,
    }


def _aggregate_stage_a(
    config: dict[str, Any],
    paired: list[dict[str, Any]],
    external: list[dict[str, Any]],
) -> dict[str, Any]:
    maximum_error = float(config["maximum_exactness_error"])
    normalization_limit = float(config["maximum_likelihood_normalization_absolute_z"])
    exactness_pass = all(
        max(record["exactness"].values()) <= maximum_error for record in paired
    ) and all(record["lifecycle_audit_passed"] for record in external)
    finite_likelihoods = all(
        record["likelihood_diagnostics"]["nonfinite_weight_count"] == 0 for record in external
    )
    representable_likelihood_moments = all(
        record["likelihood_diagnostics"]["normalization_moments_representable"] is True
        for record in external
    )
    normalization_fraction = sum(
        record["likelihood_diagnostics"]["normalization_moments_representable"] is True
        and abs(record["likelihood_diagnostics"]["normalization_z"]) <= normalization_limit
        for record in external
    ) / len(external)
    ratios = [record["mechanism"]["variance_ratio_raw_over_dcs"] for record in paired]
    finite_ratios = [value for value in ratios if math.isfinite(value) and value > 0.0]
    geometric_ratio = (
        math.exp(sum(math.log(value) for value in finite_ratios) / len(finite_ratios))
        if len(finite_ratios) == len(ratios) and ratios
        else math.inf
    )
    mechanism_pass = geometric_ratio >= float(config["gate"]["minimum_mechanism_variance_ratio"])
    primary = set(config["external_methods"]["primary"])
    primary_records = [record for record in external if record["method"] in primary]
    expected_primary = (
        len(config["cells"]) * len(config["budgets"]) * int(config["clusters"]) * len(primary)
    )
    primary_evaluable = len(primary_records) == expected_primary and all(
        math.isfinite(record["estimate"]["value"])
        and math.isfinite(record["estimate"]["standard_error"])
        for record in primary_records
    )
    primary_censoring = sum(record["allocation"]["resource_censored"] for record in primary_records)
    flow_pass = all(
        record["flow_roundtrip"] is None
        or (
            record["flow_roundtrip"]["maximum_reconstruction_error"] <= maximum_error
            and record["flow_roundtrip"]["maximum_log_jacobian_cancellation_error"] <= maximum_error
        )
        for record in external
    )
    stage_b_authorized = (
        exactness_pass and finite_likelihoods and mechanism_pass and primary_evaluable and flow_pass
    )
    proposal_training_cost_closed = False
    return {
        "paired_record_count": len(paired),
        "external_record_count": len(external),
        "maximum_exactness_error": max(max(record["exactness"].values()) for record in paired),
        "exactness_pass": exactness_pass,
        "all_external_weights_finite": finite_likelihoods,
        "all_likelihood_moments_representable": representable_likelihood_moments,
        "likelihood_normalization_pass_fraction": normalization_fraction,
        "mechanism_geometric_variance_ratio": geometric_ratio,
        "mechanism_pass": mechanism_pass,
        "primary_external_evaluable": primary_evaluable,
        "primary_resource_censoring_count": primary_censoring,
        "flow_roundtrip_pass": flow_pass,
        "dcs_proposal_training_cost_closed": proposal_training_cost_closed,
        "stage_b_authorized": stage_b_authorized,
        "p8_blockers": [
            *([] if exactness_pass else ["exactness_failure"]),
            *([] if finite_likelihoods else ["nonfinite_likelihood_weight"]),
            *([] if representable_likelihood_moments else ["nonrepresentable_likelihood_moment"]),
            *([] if mechanism_pass else ["mechanism_gate_failure"]),
            *([] if primary_evaluable else ["primary_external_method_not_evaluable"]),
            *([] if primary_censoring == 0 else ["primary_resource_censoring"]),
            "dcs_proposal_training_cost_not_closed",
            "stage_b_and_stage_c_not_complete",
            "t1_theorem_and_novelty_not_closed",
        ],
    }


def run(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    _validate_config(config)
    paired: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    cursor = int(config["base_seed"])

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal cursor
        values = tuple(range(cursor, cursor + count))
        cursor += count
        if used_seeds & set(values):
            raise AssertionError("D1 seed collision")
        used_seeds.update(values)
        return values

    primary = list(config["external_methods"]["primary"])
    secondary = list(config["external_methods"]["secondary"])
    secondary_budget = str(config["external_methods"]["secondary_budget"])
    for cell in config["cells"]:
        barrier = cell["task"] == "discrete_lower_barrier"
        for budget in config["budgets"]:
            methods = primary + (secondary if budget["id"] == secondary_budget else [])
            for cluster in range(int(config["clusters"])):
                paired.append(
                    _paired_record(
                        config,
                        cell,
                        budget,
                        cluster,
                        cast(tuple[int, int], allocate(2)),
                    )
                )
                for method in methods:
                    if barrier and method == "conditional_rbergomi":
                        continue
                    external.append(
                        _external_record(
                            config,
                            cell,
                            budget,
                            method,
                            cluster,
                            cast(tuple[int, int, int, int], allocate(4)),
                        )
                    )
    aggregate = _aggregate_stage_a(config, paired, external)
    seed_payload = json.dumps(sorted(used_seeds), separators=(",", ":")).encode()
    return {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "seed_count": len(used_seeds),
        "seed_set_sha256": hashlib.sha256(seed_payload).hexdigest(),
        "paired_records": paired,
        "external_records": external,
        "aggregate": aggregate,
        "decision": {
            "stage_a_complete": True,
            "stage_b_authorized": aggregate["stage_b_authorized"],
            "p8_qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config, digest = load_config(args.config)
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
