"""Train and independently falsify cell-tuned rank-one CEM proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from experiments.g11_v8_p5_reference import _cell_task, _method_values
from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from experiments.g11_v8_p7_calibration import _draw
from src.path_integral import (
    REFERENCE_METHODS,
    SeedKey,
    SufficientStatistics,
    TimePiecewiseTwoDriverControl,
    derive_seed,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator
from src.training import fit_rbergomi_piecewise_cem

SCHEMA = "npi.g11.v8-p5-cell-tuned-cem-proposal.v1"
RESULT_SCHEMA = "npi.g11.v8-p5-cell-tuned-cem-proposal-result.v1"
EXPECTED_TARGETS = {
    "h0.20-discrete_lower_barrier-p1e-05": "raw_crosscheck",
    "h0.05-discrete_lower_barrier-p1e-05": "dcs_reference",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("cell-tuned proposal artifact binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("cell-tuned proposal artifact path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("cell-tuned proposal artifact hash mismatch")
    return path


def load_cell_tuned_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected cell-tuned CEM proposal schema")
    for field in (
        "barrier_proposal_result",
        "barrier_proposal_failure_audit",
        "threshold_binding",
    ):
        _bound_path(config.get(field))
    if (
        config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or config.get("training_namespace") != "v8-r2-cell-tuned-cem-training-v1"
        or config.get("validation_namespace")
        != "v8-r2-cell-tuned-cem-validation-v1"
    ):
        raise ValueError("cell-tuned proposal provenance contract is invalid")
    cells = config.get("cells")
    if (
        not isinstance(cells, list)
        or {cell.get("cell_id"): cell.get("target_method") for cell in cells}
        != EXPECTED_TARGETS
    ):
        raise ValueError("cell-tuned proposal must target the two audited failures")
    for cell in cells:
        initial = cell.get("initial_control")
        if (
            not isinstance(initial, list)
            or len(initial) != 4
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(math.isfinite(float(value)) for value in pair)
                for pair in initial
            )
            or any(float(pair[1]) >= 0.0 for pair in initial)
        ):
            raise ValueError("cell-tuned initial control is malformed")
    training = config.get("training")
    families = config.get("proposal_families")
    validation = config.get("validation")
    decision = config.get("decision")
    if (
        not isinstance(training, dict)
        or int(training.get("training_seed_replicates", 0)) != 3
        or int(training.get("paths_per_iteration", 0)) != 16384
        or int(training.get("maximum_iterations", 0)) != 8
        or float(training.get("elite_quantile", 0.0)) != 0.90
        or float(training.get("smoothing", 0.0)) != 0.50
        or int(training.get("minimum_elite_paths", 0)) != 256
        or float(training.get("control_bound", 0.0)) != 12.0
        or int(training.get("target_level_repetitions", 0)) != 2
        or training.get("price_driver_sign") != "negative"
        or float(training.get("minimum_price_driver_magnitude", 0.0)) != 0.05
    ):
        raise ValueError("cell-tuned CEM training contract is invalid")
    if (
        not isinstance(families, list)
        or [family.get("id") for family in families]
        != ["low_amplitude", "centered_amplitude", "wide_amplitude"]
    ):
        raise ValueError("cell-tuned proposal-family roster changed")
    for family in families:
        scales = family.get("scales")
        weights = family.get("weights")
        if (
            not isinstance(scales, list)
            or not isinstance(weights, list)
            or len(scales) != len(weights)
            or len(scales) < 2
            or float(scales[0]) != 0.0
            or any(float(scale) <= 0.0 for scale in scales[1:])
            or any(float(weight) <= 0.0 for weight in weights)
            or not math.isclose(sum(float(weight) for weight in weights), 1.0)
        ):
            raise ValueError("cell-tuned defensive mixture is invalid")
    if (
        not isinstance(validation, dict)
        or int(validation.get("replicates", 0)) != 6
        or int(validation.get("paths_per_replicate", 0)) != 8192
        or validation.get("engine") != "fft"
        or float(validation.get("allocation_safety_factor", 0.0)) != 6.0
        or int(validation.get("maximum_final_samples", 0)) != 8388608
        or float(
            validation.get(
                "selected_method_maximum_requested_to_cap_ratio", 0.0
            )
        )
        != 0.75
        or int(
            validation.get("minimum_raw_nonzero_contributions_per_replicate", 0)
        )
        != 20
        or float(
            validation.get("maximum_likelihood_normalization_absolute_z", 0.0)
        )
        != 4.0
        or not isinstance(decision, dict)
        or decision.get("new_full_pilot_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
    ):
        raise ValueError("cell-tuned validation or decision contract is invalid")
    return config, hashlib.sha256(raw).hexdigest()


def _training_seed(
    config: dict[str, Any],
    cell_id: str,
    replicate: int,
) -> tuple[SeedKey, int]:
    key = SeedKey(
        config["protocol_id"],
        "cell-tuned-cem-training",
        cell_id,
        "fixed-grid",
        0,
        replicate,
        config["training_namespace"],
    )
    return key, derive_seed(key)


def _validation_seeds(
    config: dict[str, Any],
    cell_id: str,
    candidate_id: str,
    replicate: int,
) -> tuple[tuple[SeedKey, int], tuple[SeedKey, int]]:
    role = f"cell-tuned-cem-validation/{candidate_id}"
    common = (
        config["protocol_id"],
        role,
        cell_id,
        "paired-dcs-raw",
        0,
        replicate,
    )
    proposal = SeedKey(*common, f"{config['validation_namespace']}/proposal")
    labels = SeedKey(*common, f"{config['validation_namespace']}/labels")
    return (proposal, derive_seed(proposal)), (labels, derive_seed(labels))


def _scaled_schedules(
    profile: tuple[tuple[float, float], ...],
    scales: list[float],
) -> list[list[list[float]]]:
    schedules = [
        [[float(scale) * first, float(scale) * second] for first, second in profile]
        for scale in scales
    ]
    nonzero = schedules[1:]
    if any(pair[1] >= 0.0 for schedule in nonzero for pair in schedule):
        raise ValueError("rank-one price schedules must remain strictly negative")
    return schedules


def _controls(
    schedules: list[list[list[float]]],
    maturity: float,
) -> tuple[TimePiecewiseTwoDriverControl, ...]:
    return tuple(
        TimePiecewiseTwoDriverControl(
            tuple((float(pair[0]), float(pair[1])) for pair in schedule),
            maturity=maturity,
        )
        for schedule in schedules
    )


def _fit_profiles(
    config: dict[str, Any],
    context: Any,
    cells: list[dict[str, Any]],
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training = config["training"]
    seed_replicates = (
        int(training["smoke_training_seed_replicates"])
        if smoke
        else int(training["training_seed_replicates"])
    )
    paths = (
        int(training["smoke_paths_per_iteration"])
        if smoke
        else int(training["paths_per_iteration"])
    )
    iterations = (
        int(training["smoke_maximum_iterations"])
        if smoke
        else int(training["maximum_iterations"])
    )
    fits: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    for specification in cells:
        cell_id = str(specification["cell_id"])
        cell = context.cells_by_id[cell_id]
        simulator = RBergomiSimulator(
            H=float(cell["hurst"]),
            eta=float(cell["eta"]),
            xi=float(cell["xi"]),
            rho=float(cell["rho"]),
            device="cpu",
        )
        for replicate in range(seed_replicates):
            key, seed = _training_seed(config, cell_id, replicate)
            seed_records.append({"key": asdict(key), "seed": seed})
            fit = fit_rbergomi_piecewise_cem(
                simulator,
                _cell_task(cell),
                spot=float(cell["spot"]),
                maturity=float(cell["maturity"]),
                dt=float(cell["maturity"]) / int(cell["finest_steps"]),
                initial_control=tuple(
                    (float(pair[0]), float(pair[1]))
                    for pair in specification["initial_control"]
                ),
                num_paths=paths,
                seed=seed,
                max_iterations=iterations,
                elite_quantile=float(training["elite_quantile"]),
                smoothing=float(training["smoothing"]),
                min_elite_paths=min(int(training["minimum_elite_paths"]), paths),
                control_bound=float(training["control_bound"]),
                target_level_repetitions=int(training["target_level_repetitions"]),
                price_driver_sign=cast(
                    Literal["negative"], training["price_driver_sign"]
                ),
                minimum_price_driver_magnitude=float(
                    training["minimum_price_driver_magnitude"]
                ),
            )
            if any(
                not math.isfinite(value)
                for pair in fit.control
                for value in pair
            ) or any(second >= 0.0 for _first, second in fit.control):
                raise FloatingPointError("CEM produced an inadmissible rank-one profile")
            fits.append(
                {
                    "cell_id": cell_id,
                    "training_replicate": replicate,
                    "training_seed": seed,
                    "control": fit.control,
                    "converged": fit.converged,
                    "history": [asdict(item) for item in fit.history],
                }
            )
    return fits, seed_records


def _evaluate_candidate(
    config: dict[str, Any],
    context: Any,
    *,
    cell: dict[str, Any],
    target_method: str,
    candidate_id: str,
    schedules: list[list[list[float]]],
    weights: list[float],
    replicates: int,
    paths: int,
    seed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = config["validation"]
    simulator = RBergomiSimulator(
        H=float(cell["hurst"]),
        eta=float(cell["eta"]),
        xi=float(cell["xi"]),
        rho=float(cell["rho"]),
        device="cpu",
    )
    method_variances: dict[str, list[float]] = {
        method: [] for method in REFERENCE_METHODS
    }
    method_means: dict[str, list[float]] = {
        method: [] for method in REFERENCE_METHODS
    }
    method_maxima: dict[str, list[float]] = {
        method: [] for method in REFERENCE_METHODS
    }
    raw_nonzero_counts: list[int] = []
    normalizations: list[torch.Tensor] = []
    for replicate in range(replicates):
        proposal_record, label_record = _validation_seeds(
            config, str(cell["cell_id"]), candidate_id, replicate
        )
        seed_records.extend(
            (
                {"key": asdict(proposal_record[0]), "seed": proposal_record[1]},
                {"key": asdict(label_record[0]), "seed": label_record[1]},
            )
        )
        model = {
            "spot": float(cell["spot"]),
            "maturity": float(cell["maturity"]),
            "xi": float(cell["xi"]),
            "eta": float(cell["eta"]),
            "rho": float(cell["rho"]),
            "H": float(cell["hurst"]),
        }
        sample = _draw(
            simulator=simulator,
            controls=_controls(schedules, float(cell["maturity"])),
            weights=torch.tensor(weights, dtype=torch.float64),
            model=model,
            steps=int(cell["finest_steps"]),
            count=paths,
            proposal_seed=proposal_record[1],
            label_seed=label_record[1],
            engine=cast(Literal["fft", "reference"], validation["engine"]),
        )
        if (
            not bool(torch.isfinite(sample.paths.spot).all())
            or not bool(torch.isfinite(sample.paths.variance).all())
            or bool((sample.paths.spot <= 0.0).any())
            or bool((sample.paths.variance <= 0.0).any())
        ):
            raise FloatingPointError("cell-tuned proposal produced invalid paths")
        normalization = torch.exp(sample.mixture_log_likelihood)
        if not bool(torch.isfinite(normalization).all()):
            raise FloatingPointError("cell-tuned likelihood is nonfinite")
        normalizations.append(normalization.detach().cpu())
        task = _cell_task(cell)
        for method in REFERENCE_METHODS:
            values = _method_values(
                sample,
                task=task,
                rho=float(cell["rho"]),
                method=method,
            ).detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("cell-tuned contribution is nonfinite")
            method_variances[method].append(float(torch.var(values, unbiased=True)))
            method_means[method].append(float(torch.mean(values)))
            method_maxima[method].append(float(torch.max(torch.abs(values))))
            if method == "raw_crosscheck":
                raw_nonzero_counts.append(int(torch.count_nonzero(values)))
    target_standard_error = 0.10 * 0.20 * float(cell["nominal_probability"])
    entries = []
    for method in REFERENCE_METHODS:
        design_variance = max(method_variances[method])
        requested = max(
            8192,
            math.ceil(
                float(validation["allocation_safety_factor"])
                * design_variance
                / target_standard_error**2
            ),
        )
        entries.append(
            {
                "method": method,
                "target_standard_error": target_standard_error,
                "replicate_variances": method_variances[method],
                "replicate_means": method_means[method],
                "replicate_max_absolute_contributions": method_maxima[method],
                "allocation_design_variance": design_variance,
                "projected_final_samples": requested,
                "requested_to_cap_ratio": requested
                / int(validation["maximum_final_samples"]),
                "projected_cap_pass": requested
                <= int(validation["maximum_final_samples"]),
                "raw_nonzero_counts": (
                    raw_nonzero_counts if method == "raw_crosscheck" else None
                ),
                "raw_coverage_pass": (
                    min(raw_nonzero_counts)
                    >= int(
                        validation[
                            "minimum_raw_nonzero_contributions_per_replicate"
                        ]
                    )
                    if method == "raw_crosscheck"
                    else True
                ),
            }
        )
    normalization = SufficientStatistics.from_tensor(torch.cat(normalizations))
    normalization_z = (
        (normalization.mean - 1.0) / normalization.standard_error
        if normalization.standard_error > 0.0
        else (0.0 if normalization.mean == 1.0 else math.inf)
    )
    target_entry = next(entry for entry in entries if entry["method"] == target_method)
    gates = {
        "target_method_margin_pass": float(target_entry["requested_to_cap_ratio"])
        <= float(
            validation["selected_method_maximum_requested_to_cap_ratio"]
        ),
        "raw_coverage_pass": next(
            entry for entry in entries if entry["method"] == "raw_crosscheck"
        )["raw_coverage_pass"],
        "likelihood_normalization_pass": abs(normalization_z)
        <= float(validation["maximum_likelihood_normalization_absolute_z"]),
    }
    return {
        "candidate_id": candidate_id,
        "cell_id": cell["cell_id"],
        "target_method": target_method,
        "weights": weights,
        "schedules": schedules,
        "entries": entries,
        "normalization_mean": normalization.mean,
        "normalization_standard_error": normalization.standard_error,
        "normalization_z": normalization_z,
        "gates": gates,
        "passes": all(gates.values()),
    }


def run_cell_tuned_proposal(
    config_path: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    config, config_sha256 = load_cell_tuned_config(config_path)
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
    )
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("cell-tuned config binds a different threshold manifest")
    cells = (
        config["cells"][: int(config["validation"]["smoke_cells"])]
        if smoke
        else config["cells"]
    )
    families = (
        config["proposal_families"][
            : int(config["validation"]["smoke_proposal_families"])
        ]
        if smoke
        else config["proposal_families"]
    )
    replicates = (
        int(config["validation"]["smoke_replicates"])
        if smoke
        else int(config["validation"]["replicates"])
    )
    paths = (
        int(config["validation"]["smoke_paths_per_replicate"])
        if smoke
        else int(config["validation"]["paths_per_replicate"])
    )
    fits, seed_records = _fit_profiles(
        config, context, cells, smoke=smoke
    )
    candidates: list[dict[str, Any]] = []
    for fit in fits:
        specification = next(
            cell for cell in cells if cell["cell_id"] == fit["cell_id"]
        )
        cell = context.cells_by_id[fit["cell_id"]]
        profile = tuple(
            (float(pair[0]), float(pair[1])) for pair in fit["control"]
        )
        for family in families:
            candidate_id = (
                f"{fit['cell_id']}/train-{fit['training_replicate']}/"
                f"{family['id']}"
            )
            candidates.append(
                _evaluate_candidate(
                    config,
                    context,
                    cell=cell,
                    target_method=str(specification["target_method"]),
                    candidate_id=candidate_id,
                    schedules=_scaled_schedules(profile, family["scales"]),
                    weights=[float(weight) for weight in family["weights"]],
                    replicates=replicates,
                    paths=paths,
                    seed_records=seed_records,
                )
            )
    selected: dict[str, dict[str, Any]] = {}
    for specification in cells:
        passing = [
            candidate
            for candidate in candidates
            if candidate["cell_id"] == specification["cell_id"]
            and candidate["passes"]
        ]
        if not passing:
            continue
        target_method = specification["target_method"]
        selected_candidate = min(
            passing,
            key=lambda candidate: next(
                entry["requested_to_cap_ratio"]
                for entry in candidate["entries"]
                if entry["method"] == target_method
            ),
        )
        selected[str(specification["cell_id"])] = {
            "target_method": target_method,
            "candidate_id": selected_candidate["candidate_id"],
            "weights": selected_candidate["weights"],
            "schedules": selected_candidate["schedules"],
            "target_method_entry": next(
                entry
                for entry in selected_candidate["entries"]
                if entry["method"] == target_method
            ),
        }
    passed = len(selected) == len(cells)
    all_seeds = [record["seed"] for record in seed_records]
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("cell-tuned training and validation seeds overlap")
    provenance = source_provenance()
    return {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "training_namespace": config["training_namespace"],
        "validation_namespace": config["validation_namespace"],
        "smoke": smoke,
        "design_informed_by_prior_development_outcomes": True,
        "current_namespace_outcomes_inspected_before_freeze": False,
        "cells": [cell["cell_id"] for cell in cells],
        "training_fits": fits,
        "validation_replicates": replicates,
        "validation_paths_per_replicate": paths,
        "candidates": candidates,
        "selected_proposals": selected,
        "seed_records": seed_records,
        "seed_count": len(seed_records),
        "passed": passed,
        "decision": {
            "status": (
                "cell_tuned_cem_proposal_falsification_pass"
                if passed
                else "cell_tuned_cem_proposal_falsification_fail"
            ),
            "selected_proposal_frozen": False,
            "new_full_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"refusing to overwrite cell-tuned result: {arguments.output}"
        )
    result = run_cell_tuned_proposal(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
