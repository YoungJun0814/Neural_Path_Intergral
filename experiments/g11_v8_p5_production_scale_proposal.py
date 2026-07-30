"""Develop method-specific proposals at production-relevant validation scale.

Raw importance sampling admits arbitrary deterministic finite mixtures when the
exact balance likelihood is used.  The implemented DCS identity is narrower:
its price-driver controls must share one deterministic direction.  This module
keeps those two admissibility classes separate and fails closed.
"""

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
    SeedKey,
    SufficientStatistics,
    TimePiecewiseTwoDriverControl,
    derive_seed,
    rank_one_price_control_span,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator
from src.training import fit_rbergomi_piecewise_cem

SCHEMA_V1 = "npi.g11.v8-p5-production-scale-proposal.v1"
SCHEMA_V2 = "npi.g11.v8-p5-production-scale-proposal.v2"
SCHEMA_V3 = "npi.g11.v8-p5-production-scale-proposal.v3"
SUPPORTED_SCHEMAS = {SCHEMA_V1, SCHEMA_V2, SCHEMA_V3}
RESULT_SCHEMAS = {
    SCHEMA_V1: "npi.g11.v8-p5-production-scale-proposal-result.v1",
    SCHEMA_V2: "npi.g11.v8-p5-production-scale-proposal-result.v2",
    SCHEMA_V3: "npi.g11.v8-p5-production-scale-proposal-result.v3",
}
RAW_METHOD = "raw_crosscheck"
DCS_METHOD = "dcs_reference"
EXPECTED_REQUIREMENTS = {
    ("h0.05-terminal_left_tail-p1e-04", RAW_METHOD),
    ("h0.20-discrete_lower_barrier-p1e-04", RAW_METHOD),
    ("h0.12-discrete_lower_barrier-p1e-03", DCS_METHOD),
    ("h0.05-terminal_left_tail-p1e-05", RAW_METHOD),
    ("h0.05-terminal_left_tail-p1e-05", DCS_METHOD),
    ("h0.12-terminal_left_tail-p1e-05", DCS_METHOD),
    ("h0.12-discrete_lower_barrier-p1e-05", RAW_METHOD),
    ("h0.20-terminal_left_tail-p1e-05", RAW_METHOD),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("production-scale proposal binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("production-scale proposal bound path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("production-scale proposal artifact hash mismatch")
    return path


def _failure_requirements(path: Path) -> set[tuple[str, str]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("allocation failure receipt must be a mapping")
    entries = receipt.get("entries")
    if not isinstance(entries, list):
        raise ValueError("allocation failure receipt entries are malformed")
    return {
        (str(entry["cell_id"]), str(entry["method"]))
        for entry in entries
        if isinstance(entry, dict) and entry.get("resource_feasible") is False
    }


def load_production_scale_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("unexpected production-scale proposal schema")
    version = {
        SCHEMA_V1: 1,
        SCHEMA_V2: 2,
        SCHEMA_V3: 3,
    }[config["schema"]]
    failure_path = _bound_path(config.get("allocation_failure"))
    audit_path = _bound_path(config.get("allocation_failure_audit"))
    _bound_path(config.get("threshold_binding"))
    if version == 2:
        prior_failure_path = _bound_path(config.get("prior_execution_failure"))
        prior_failure = json.loads(prior_failure_path.read_text(encoding="utf-8"))
        if (
            not isinstance(prior_failure, dict)
            or prior_failure.get("decision", {}).get(
                "new_protocol_namespace_required"
            )
            is not True
            or prior_failure.get("decision", {}).get(
                "training_namespace_burned"
            )
            is not True
            or prior_failure.get("decision", {}).get(
                "validation_namespace_burned"
            )
            is not True
        ):
            raise ValueError("V1 execution failure does not authorize V2 recovery")
    if version == 3:
        prior_result_path = _bound_path(config.get("prior_result"))
        prior_audit_path = _bound_path(config.get("prior_audit"))
        prior_result = json.loads(prior_result_path.read_text(encoding="utf-8"))
        prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
        if (
            not isinstance(prior_result, dict)
            or prior_result.get("passed") is not False
            or not isinstance(prior_audit, dict)
            or prior_audit.get("passed") is not True
            or prior_audit.get("decision", {}).get(
                "permuted_block_protocol_required"
            )
            is not True
            or prior_audit.get("decision", {}).get(
                "v2_training_namespace_burned"
            )
            is not True
            or prior_audit.get("decision", {}).get(
                "v2_validation_namespace_burned"
            )
            is not True
        ):
            raise ValueError("V2 falsification audit does not authorize V3")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("passed") is not True
        or audit.get("decision", {}).get("new_reference_design_required") is not True
        or _failure_requirements(failure_path) != EXPECTED_REQUIREMENTS
    ):
        raise ValueError("allocation failure evidence does not authorize this redesign")
    cells = config.get("cells")
    if not isinstance(cells, list):
        raise ValueError("production-scale proposal cells must be a list")
    actual_requirements = {
        (str(cell.get("cell_id")), str(method))
        for cell in cells
        for method in cell.get("target_methods", [])
    }
    if (
        config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or config.get("training_namespace")
        != f"v8-r2-production-scale-proposal-training-v{version}"
        or config.get("validation_namespace")
        != f"v8-r2-production-scale-proposal-validation-v{version}"
        or actual_requirements != EXPECTED_REQUIREMENTS
        or len(cells) != 7
    ):
        raise ValueError("production-scale proposal provenance or target roster changed")
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
            raise ValueError("production-scale initial control is malformed")
    training = config.get("training")
    validation = config.get("validation")
    raw_families = config.get("raw_full_rank_families")
    dcs_families = config.get("dcs_rank_one_families")
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
        raise ValueError("production-scale CEM training contract changed")
    expected_raw_family_ids = (
        [
            "profile_compact",
            "profile_broad",
            "profile_dense",
            "profile_refined",
        ]
        if version == 3
        else ["profile_compact", "profile_broad", "profile_dense"]
    )
    expected_dcs_family_ids = (
        ["rank_one_dense", "rank_one_low", "rank_one_focused"]
        if version == 3
        else ["rank_one_dense"]
    )
    if (
        not isinstance(raw_families, list)
        or [family.get("id") for family in raw_families]
        != expected_raw_family_ids
        or not isinstance(dcs_families, list)
        or [family.get("id") for family in dcs_families]
        != expected_dcs_family_ids
    ):
        raise ValueError("production-scale proposal family roster changed")
    for family in raw_families:
        scales = family.get("scales")
        if (
            not isinstance(scales, list)
            or len(scales) < 3
            or any(float(scale) <= 0.0 for scale in scales)
            or not 0.0 < float(family.get("natural_weight", 0.0)) < 1.0
        ):
            raise ValueError("raw full-rank family is malformed")
    for family in dcs_families:
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
            raise ValueError("DCS rank-one family is malformed")
        target_cells = family.get("target_cells")
        if version == 3 and (
            not isinstance(target_cells, list)
            or not target_cells
            or not set(str(cell) for cell in target_cells).issubset(
                {
                    cell_id
                    for cell_id, method in EXPECTED_REQUIREMENTS
                    if method == DCS_METHOD
                }
            )
        ):
            raise ValueError("V3 DCS family target cells are malformed")
    if (
        not isinstance(validation, dict)
        or int(validation.get("replicates", 0)) != 8
        or int(validation.get("paths_per_replicate", 0)) != 32768
        or int(validation.get("blocks_per_replicate", 0)) != 8
        or validation.get("engine") != "fft"
        or float(validation.get("allocation_safety_factor", 0.0)) != 6.0
        or int(validation.get("maximum_final_samples", 0)) != 8388608
        or float(validation.get("maximum_requested_to_cap_ratio", 0.0)) != 0.50
        or int(validation.get("minimum_raw_nonzero_per_replicate", 0)) != 1024
        or int(validation.get("minimum_raw_nonzero_per_block", 0)) != 64
        or float(validation.get("maximum_likelihood_normalization_absolute_z", 0.0))
        != 4.0
        or float(validation.get("maximum_block_variance_to_median_ratio", 0.0))
        != 30.0
        or float(validation.get("maximum_full_contribution_share", 0.0)) != 0.05
        or float(validation.get("maximum_block_contribution_share", 0.0)) != 0.25
        or not isinstance(decision, dict)
        or decision.get("new_formal_pilot_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
    ):
        raise ValueError("production-scale validation or decision contract changed")
    return config, hashlib.sha256(raw).hexdigest()


def _seed(
    config: dict[str, Any],
    *,
    stage: str,
    cell_id: str,
    candidate_id: str,
    replicate: int,
    namespace: str,
) -> tuple[SeedKey, int]:
    key = SeedKey(
        config["protocol_id"],
        f"production-scale-proposal/{stage}/{candidate_id}",
        cell_id,
        "fixed-grid",
        0,
        replicate,
        namespace,
    )
    return key, derive_seed(key)


def _controls(
    schedules: list[list[list[float]]], maturity: float
) -> tuple[TimePiecewiseTwoDriverControl, ...]:
    return tuple(
        TimePiecewiseTwoDriverControl(
            tuple((float(pair[0]), float(pair[1])) for pair in schedule),
            maturity=maturity,
        )
        for schedule in schedules
    )


def _rank_one_diagnostic(
    schedules: list[list[list[float]]], *, maturity: float, steps: int
) -> dict[str, Any]:
    times = torch.arange(steps, dtype=torch.float64) * (maturity / steps)
    expanded = torch.stack(
        [control.deterministic_schedule(times) for control in _controls(schedules, maturity)]
    )
    try:
        span = rank_one_price_control_span(expanded, step_dt=maturity / steps)
    except ValueError as error:
        return {
            "maximum_span_residual": None,
            "rank_one_price_span": False,
            "diagnostic": str(error),
        }
    return {
        "maximum_span_residual": span.maximum_span_residual,
        "rank_one_price_span": span.maximum_span_residual <= 1e-10,
        "diagnostic": None,
    }


def _raw_full_rank_mixture(
    profiles: list[tuple[tuple[float, float], ...]],
    *,
    scales: list[float],
    natural_weight: float,
) -> tuple[list[list[list[float]]], list[float]]:
    if not profiles or not 0.0 < natural_weight < 1.0:
        raise ValueError("raw profile mixture inputs are invalid")
    segment_count = len(profiles[0])
    if segment_count == 0 or any(len(profile) != segment_count for profile in profiles):
        raise ValueError("raw profiles must have the same nonzero segment count")
    schedules: list[list[list[float]]] = [
        [[0.0, 0.0] for _ in range(segment_count)]
    ]
    for profile in profiles:
        for scale in scales:
            schedules.append(
                [
                    [float(scale) * float(first), float(scale) * float(second)]
                    for first, second in profile
                ]
            )
    nonnatural_weight = (1.0 - natural_weight) / (len(schedules) - 1)
    weights = [natural_weight] + [nonnatural_weight] * (len(schedules) - 1)
    return schedules, weights


def _dcs_rank_one_mixture(
    profile: tuple[tuple[float, float], ...],
    *,
    scales: list[float],
    weights: list[float],
) -> tuple[list[list[list[float]]], list[float]]:
    schedules = [
        [
            [float(scale) * float(first), float(scale) * float(second)]
            for first, second in profile
        ]
        for scale in scales
    ]
    if any(pair[1] >= 0.0 for schedule in schedules[1:] for pair in schedule):
        raise ValueError("DCS nonnatural price controls must remain strictly negative")
    return schedules, [float(weight) for weight in weights]


def _fit_profiles(
    config: dict[str, Any],
    context: Any,
    cells: list[dict[str, Any]],
    *,
    smoke: bool,
) -> tuple[dict[str, list[tuple[tuple[float, float], ...]]], list[dict[str, Any]], list[dict[str, Any]]]:
    training = config["training"]
    replicates = (
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
    profiles: dict[str, list[tuple[tuple[float, float], ...]]] = {}
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
        profiles[cell_id] = []
        for replicate in range(replicates):
            key, seed = _seed(
                config,
                stage="training",
                cell_id=cell_id,
                candidate_id="cem",
                replicate=replicate,
                namespace=config["training_namespace"],
            )
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
            profile = tuple(
                (float(pair[0]), float(pair[1])) for pair in fit.control
            )
            if any(
                not math.isfinite(value) for pair in profile for value in pair
            ) or any(second >= 0.0 for _first, second in profile):
                raise FloatingPointError("CEM produced an inadmissible profile")
            profiles[cell_id].append(profile)
            fits.append(
                {
                    "cell_id": cell_id,
                    "training_replicate": replicate,
                    "training_seed": seed,
                    "control": profile,
                    "converged": fit.converged,
                    "history": [asdict(item) for item in fit.history],
                }
            )
    return profiles, fits, seed_records


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        0.5 * (ordered[middle - 1] + ordered[middle])
        if len(ordered) % 2 == 0
        else ordered[middle]
    )


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _assert_json_finite(value: Any, *, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{path} contains a non-finite JSON float")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_finite(item, path=f"{path}[{index}]")


def _candidate_failure(
    *,
    cell_id: str,
    method: str,
    candidate_id: str,
    schedules: list[list[list[float]]],
    weights: list[float],
    error: FloatingPointError,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "cell_id": cell_id,
        "method": method,
        "weights": weights,
        "schedules": schedules,
        "entries": [],
        "gates": {
            "finite_paths_and_contributions": False,
            "exact_likelihood_bound_pass": False,
            "likelihood_normalization_pass": False,
            "projected_allocation_margin_pass": False,
            "block_variance_stability_pass": False,
            "contribution_concentration_pass": False,
            "raw_coverage_pass": False,
            "method_structure_pass": False,
        },
        "numerical_failure": {
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        },
        "passes": False,
    }


def _evaluate_candidate(
    config: dict[str, Any],
    *,
    cell: dict[str, Any],
    method: str,
    candidate_id: str,
    schedules: list[list[list[float]]],
    weights: list[float],
    replicates: int,
    paths: int,
    blocks: int,
    seed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = config["validation"]
    if paths % blocks != 0:
        raise ValueError("validation paths must divide exactly into blocks")
    if len(schedules) != len(weights) or any(weight <= 0.0 for weight in weights):
        raise ValueError("proposal weights and schedules are invalid")
    if not math.isclose(sum(weights), 1.0):
        raise ValueError("proposal weights do not sum to one")
    if any(abs(value) > 0.0 for pair in schedules[0] for value in pair):
        raise ValueError("proposal must begin with the natural expert")
    structure = _rank_one_diagnostic(
        schedules,
        maturity=float(cell["maturity"]),
        steps=int(cell["finest_steps"]),
    )
    if method == DCS_METHOD and not structure["rank_one_price_span"]:
        raise ValueError("DCS proposal violates the required rank-one price span")
    simulator = RBergomiSimulator(
        H=float(cell["hurst"]),
        eta=float(cell["eta"]),
        xi=float(cell["xi"]),
        rho=float(cell["rho"]),
        device="cpu",
    )
    model = {
        "spot": float(cell["spot"]),
        "maturity": float(cell["maturity"]),
        "xi": float(cell["xi"]),
        "eta": float(cell["eta"]),
        "rho": float(cell["rho"]),
        "H": float(cell["hurst"]),
    }
    task = _cell_task(cell)
    full_variances: list[float] = []
    block_variances: list[float] = []
    full_shares: list[float] = []
    block_shares: list[float] = []
    raw_nonzero_full: list[int] = []
    raw_nonzero_blocks: list[int] = []
    normalization_batches: list[torch.Tensor] = []
    maximum_likelihoods: list[float] = []
    entries: list[dict[str, Any]] = []
    block_size = paths // blocks
    for replicate in range(replicates):
        proposal_key, proposal_seed = _seed(
            config,
            stage="validation-proposal",
            cell_id=str(cell["cell_id"]),
            candidate_id=candidate_id,
            replicate=replicate,
            namespace=config["validation_namespace"],
        )
        label_key, label_seed = _seed(
            config,
            stage="validation-labels",
            cell_id=str(cell["cell_id"]),
            candidate_id=candidate_id,
            replicate=replicate,
            namespace=config["validation_namespace"],
        )
        seed_records.extend(
            (
                {"key": asdict(proposal_key), "seed": proposal_seed},
                {"key": asdict(label_key), "seed": label_seed},
            )
        )
        permutation_seed: int | None = None
        if (
            config["schema"] == SCHEMA_V3
            or validation.get("block_partition")
            == "independent_seeded_uniform_permutation_before_equal_slicing"
        ):
            permutation_key, permutation_seed = _seed(
                config,
                stage="validation-block-permutation",
                cell_id=str(cell["cell_id"]),
                candidate_id=candidate_id,
                replicate=replicate,
                namespace=config["validation_namespace"],
            )
            seed_records.append(
                {
                    "key": asdict(permutation_key),
                    "seed": permutation_seed,
                }
            )
        sample = _draw(
            simulator=simulator,
            controls=_controls(schedules, float(cell["maturity"])),
            weights=torch.tensor(weights, dtype=torch.float64),
            model=model,
            steps=int(cell["finest_steps"]),
            count=paths,
            proposal_seed=proposal_seed,
            label_seed=label_seed,
            engine=cast(Literal["fft", "reference"], validation["engine"]),
        )
        if (
            not bool(torch.isfinite(sample.paths.spot).all())
            or not bool(torch.isfinite(sample.paths.variance).all())
            or bool((sample.paths.spot <= 0.0).any())
            or bool((sample.paths.variance <= 0.0).any())
        ):
            raise FloatingPointError("production-scale proposal produced invalid paths")
        likelihood = torch.exp(sample.mixture_log_likelihood).detach().cpu()
        values = _method_values(
            sample,
            task=task,
            rho=float(cell["rho"]),
            method=cast(Any, method),
        ).detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(likelihood).all()) or not bool(
            torch.isfinite(values).all()
        ):
            raise FloatingPointError("production-scale likelihood or value is nonfinite")
        normalization_batches.append(likelihood)
        maximum_likelihoods.append(float(torch.max(likelihood)))
        full_variance = float(torch.var(values, unbiased=True))
        full_abs_sum = float(torch.sum(torch.abs(values)))
        full_share = (
            float(torch.max(torch.abs(values))) / full_abs_sum
            if full_abs_sum > 0.0
            else math.inf
        )
        full_variances.append(full_variance)
        full_shares.append(full_share)
        if method == RAW_METHOD:
            raw_nonzero_full.append(int(torch.count_nonzero(values)))
        if permutation_seed is None:
            permuted_values = values
        else:
            permutation_generator = torch.Generator(device="cpu")
            permutation_generator.manual_seed(permutation_seed)
            block_permutation = torch.randperm(
                paths,
                generator=permutation_generator,
                device="cpu",
            )
            permuted_values = values[block_permutation]
        replicate_block_variances = []
        replicate_block_shares = []
        replicate_block_nonzero = []
        for block in range(blocks):
            block_values = permuted_values[
                block * block_size : (block + 1) * block_size
            ]
            variance = float(torch.var(block_values, unbiased=True))
            absolute_sum = float(torch.sum(torch.abs(block_values)))
            share = (
                float(torch.max(torch.abs(block_values))) / absolute_sum
                if absolute_sum > 0.0
                else math.inf
            )
            replicate_block_variances.append(variance)
            replicate_block_shares.append(share)
            block_variances.append(variance)
            block_shares.append(share)
            if method == RAW_METHOD:
                nonzero = int(torch.count_nonzero(block_values))
                replicate_block_nonzero.append(nonzero)
                raw_nonzero_blocks.append(nonzero)
        entries.append(
            {
                "replicate": replicate,
                "mean": float(torch.mean(values)),
                "variance": full_variance,
                "maximum_absolute_contribution": float(torch.max(torch.abs(values))),
                "maximum_contribution_share": _finite_or_none(full_share),
                "block_variances": replicate_block_variances,
                "block_maximum_contribution_shares": [
                    _finite_or_none(value) for value in replicate_block_shares
                ],
                "raw_nonzero_count": (
                    raw_nonzero_full[-1] if method == RAW_METHOD else None
                ),
                "raw_block_nonzero_counts": (
                    replicate_block_nonzero if method == RAW_METHOD else None
                ),
                "maximum_likelihood": maximum_likelihoods[-1],
                "block_permutation_seed": permutation_seed,
            }
        )
    design_variance = max(full_variances + block_variances)
    median_block_variance = _median(block_variances)
    variance_ratio = (
        max(block_variances) / median_block_variance
        if median_block_variance > 0.0
        else (1.0 if max(block_variances) == 0.0 else math.inf)
    )
    target_standard_error = 0.10 * 0.20 * float(cell["nominal_probability"])
    projected = max(
        8192,
        math.ceil(
            float(validation["allocation_safety_factor"])
            * design_variance
            / target_standard_error**2
        ),
    )
    normalization = SufficientStatistics.from_tensor(
        torch.cat(normalization_batches)
    )
    normalization_z = (
        (normalization.mean - 1.0) / normalization.standard_error
        if normalization.standard_error > 0.0
        else (0.0 if normalization.mean == 1.0 else math.inf)
    )
    likelihood_upper_bound = 1.0 / weights[0]
    likelihood_bound_pass = max(maximum_likelihoods) <= likelihood_upper_bound * (
        1.0 + 1e-10
    )
    gates = {
        "finite_paths_and_contributions": True,
        "exact_likelihood_bound_pass": likelihood_bound_pass,
        "likelihood_normalization_pass": abs(normalization_z)
        <= float(validation["maximum_likelihood_normalization_absolute_z"]),
        "projected_allocation_margin_pass": projected
        / int(validation["maximum_final_samples"])
        <= float(validation["maximum_requested_to_cap_ratio"]),
        "block_variance_stability_pass": variance_ratio
        <= float(validation["maximum_block_variance_to_median_ratio"]),
        "contribution_concentration_pass": max(full_shares)
        <= float(validation["maximum_full_contribution_share"])
        and max(block_shares)
        <= float(validation["maximum_block_contribution_share"]),
        "raw_coverage_pass": (
            min(raw_nonzero_full)
            >= int(validation["minimum_raw_nonzero_per_replicate"])
            and min(raw_nonzero_blocks)
            >= int(validation["minimum_raw_nonzero_per_block"])
            if method == RAW_METHOD
            else True
        ),
        "method_structure_pass": (
            structure["rank_one_price_span"] if method == DCS_METHOD else True
        ),
    }
    return {
        "candidate_id": candidate_id,
        "cell_id": cell["cell_id"],
        "method": method,
        "weights": weights,
        "schedules": schedules,
        "structure": structure,
        "natural_component_likelihood_upper_bound": likelihood_upper_bound,
        "replicates": entries,
        "full_replicate_variances": full_variances,
        "block_variances": block_variances,
        "block_variance_to_median_ratio": _finite_or_none(variance_ratio),
        "allocation_design_variance": design_variance,
        "target_standard_error": target_standard_error,
        "projected_final_samples": projected,
        "requested_to_cap_ratio": projected
        / int(validation["maximum_final_samples"]),
        "normalization_mean": normalization.mean,
        "normalization_standard_error": normalization.standard_error,
        "normalization_z": _finite_or_none(normalization_z),
        "gates": gates,
        "passes": all(gates.values()),
    }


def run_production_scale_proposal(
    config_path: Path, *, smoke: bool = False
) -> dict[str, Any]:
    config, config_sha256 = load_production_scale_config(config_path)
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v4.yaml"
    )
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("production-scale proposal binds a different threshold manifest")
    cells = (
        config["cells"][: int(config["validation"]["smoke_cells"])]
        if smoke
        else config["cells"]
    )
    profiles, fits, seed_records = _fit_profiles(
        config, context, cells, smoke=smoke
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
    blocks = (
        int(config["validation"]["smoke_blocks_per_replicate"])
        if smoke
        else int(config["validation"]["blocks_per_replicate"])
    )
    raw_families = (
        config["raw_full_rank_families"][
            : int(config["validation"]["smoke_raw_families"])
        ]
        if smoke
        else config["raw_full_rank_families"]
    )
    dcs_families = (
        config["dcs_rank_one_families"][
            : int(config["validation"]["smoke_dcs_families"])
        ]
        if smoke
        else config["dcs_rank_one_families"]
    )
    candidates: list[dict[str, Any]] = []
    for specification in cells:
        cell_id = str(specification["cell_id"])
        cell = context.cells_by_id[cell_id]
        target_methods = list(specification["target_methods"])
        if RAW_METHOD in target_methods:
            for family in raw_families:
                schedules, weights = _raw_full_rank_mixture(
                    profiles[cell_id],
                    scales=[float(scale) for scale in family["scales"]],
                    natural_weight=float(family["natural_weight"]),
                )
                candidate_id = f"{cell_id}/raw/{family['id']}"
                try:
                    candidate = _evaluate_candidate(
                        config,
                        cell=cell,
                        method=RAW_METHOD,
                        candidate_id=candidate_id,
                        schedules=schedules,
                        weights=weights,
                        replicates=replicates,
                        paths=paths,
                        blocks=blocks,
                        seed_records=seed_records,
                    )
                except FloatingPointError as error:
                    candidate = _candidate_failure(
                        cell_id=cell_id,
                        method=RAW_METHOD,
                        candidate_id=candidate_id,
                        schedules=schedules,
                        weights=weights,
                        error=error,
                    )
                candidates.append(candidate)
        if DCS_METHOD in target_methods:
            for profile_index, profile in enumerate(profiles[cell_id]):
                for family in dcs_families:
                    if (
                        config["schema"] == SCHEMA_V3
                        and cell_id not in family["target_cells"]
                    ):
                        continue
                    schedules, weights = _dcs_rank_one_mixture(
                        profile,
                        scales=[float(scale) for scale in family["scales"]],
                        weights=[float(weight) for weight in family["weights"]],
                    )
                    candidate_id = (
                        f"{cell_id}/dcs/train-{profile_index}/{family['id']}"
                    )
                    try:
                        candidate = _evaluate_candidate(
                            config,
                            cell=cell,
                            method=DCS_METHOD,
                            candidate_id=candidate_id,
                            schedules=schedules,
                            weights=weights,
                            replicates=replicates,
                            paths=paths,
                            blocks=blocks,
                            seed_records=seed_records,
                        )
                    except FloatingPointError as error:
                        candidate = _candidate_failure(
                            cell_id=cell_id,
                            method=DCS_METHOD,
                            candidate_id=candidate_id,
                            schedules=schedules,
                            weights=weights,
                            error=error,
                        )
                    candidates.append(candidate)
    selected: dict[str, dict[str, Any]] = {}
    for cell_id, method in sorted(EXPECTED_REQUIREMENTS):
        if cell_id not in {str(cell["cell_id"]) for cell in cells}:
            continue
        passing = [
            candidate
            for candidate in candidates
            if candidate["cell_id"] == cell_id
            and candidate["method"] == method
            and candidate["passes"]
        ]
        if not passing:
            continue
        chosen = min(
            passing, key=lambda candidate: float(candidate["requested_to_cap_ratio"])
        )
        selected.setdefault(cell_id, {})[method] = {
            key: chosen[key]
            for key in (
                "candidate_id",
                "weights",
                "schedules",
                "structure",
                "natural_component_likelihood_upper_bound",
                "allocation_design_variance",
                "projected_final_samples",
                "requested_to_cap_ratio",
                "normalization_mean",
                "normalization_standard_error",
                "normalization_z",
                "gates",
            )
        }
    selected_count = sum(len(methods) for methods in selected.values())
    required_count = sum(
        1
        for cell_id, _method in EXPECTED_REQUIREMENTS
        if cell_id in {str(cell["cell_id"]) for cell in cells}
    )
    all_seeds = [int(record["seed"]) for record in seed_records]
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("production-scale training and validation seeds overlap")
    passed = selected_count == required_count
    provenance = source_provenance()
    result = {
        "schema": RESULT_SCHEMAS[config["schema"]],
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "training_namespace": config["training_namespace"],
        "validation_namespace": config["validation_namespace"],
        "smoke": smoke,
        "method_admissibility": {
            RAW_METHOD: (
                "arbitrary deterministic finite mixture with exact balance likelihood"
            ),
            DCS_METHOD: "rank-one price-control span required",
        },
        "validation_design": {
            "replicates": replicates,
            "paths_per_replicate": paths,
            "blocks_per_replicate": blocks,
            "allocation_variance_statistic": (
                "maximum of full-replicate and within-replicate block variances"
            ),
            "block_partition": (
                "independent_seeded_uniform_permutation_before_equal_slicing"
                if config["schema"] == SCHEMA_V3
                else "stored_component_grouped_order"
            ),
        },
        "fits": fits,
        "candidates": candidates,
        "selected_proposals": selected,
        "required_selection_count": required_count,
        "selected_count": selected_count,
        "seed_records": seed_records,
        "seed_count": len(seed_records),
        "passed": passed,
        "decision": {
            "status": (
                "production_scale_proposal_falsification_pass"
                if passed
                else "production_scale_proposal_falsification_fail"
            ),
            "selected_proposals_frozen": False,
            "proposal_manifest_build_authorized": False,
            "new_formal_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }
    _assert_json_finite(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"refusing to overwrite production-scale result: {arguments.output}"
        )
    result = run_production_scale_proposal(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
