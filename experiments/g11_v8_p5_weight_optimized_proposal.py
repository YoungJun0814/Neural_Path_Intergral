"""Optimize raw mixture weights exactly and refine unresolved rank-one DCS cells."""

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

from experiments.g11_v8_p5_production_scale_proposal import (
    DCS_METHOD,
    RAW_METHOD,
    _assert_json_finite,
    _controls,
    _dcs_rank_one_mixture,
    _evaluate_candidate,
    _seed,
)
from experiments.g11_v8_p5_reference import _cell_task
from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from experiments.g11_v8_p7_calibration import _draw
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

SCHEMA = "npi.g11.v8-p5-weight-optimized-proposal.v1"
RESULT_SCHEMA = "npi.g11.v8-p5-weight-optimized-proposal-result.v1"
RETAINED_REQUIREMENTS = {
    ("h0.12-terminal_left_tail-p1e-05", DCS_METHOD),
    ("h0.20-discrete_lower_barrier-p1e-04", RAW_METHOD),
    ("h0.20-terminal_left_tail-p1e-05", RAW_METHOD),
}
UNRESOLVED_REQUIREMENTS = {
    ("h0.05-terminal_left_tail-p1e-04", RAW_METHOD),
    ("h0.12-discrete_lower_barrier-p1e-03", DCS_METHOD),
    ("h0.05-terminal_left_tail-p1e-05", DCS_METHOD),
    ("h0.05-terminal_left_tail-p1e-05", RAW_METHOD),
    ("h0.12-discrete_lower_barrier-p1e-05", RAW_METHOD),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("weight-optimized proposal binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("weight-optimized proposal path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("weight-optimized proposal artifact hash mismatch")
    return path


def load_weight_optimized_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected weight-optimized proposal schema")
    prior_result_path = _bound_path(config.get("prior_result"))
    prior_audit_path = _bound_path(config.get("prior_audit"))
    _bound_path(config.get("threshold_binding"))
    prior_result = json.loads(prior_result_path.read_text(encoding="utf-8"))
    prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
    retained = config.get("retained_requirements")
    raw_specs = config.get("raw_weight_optimization")
    dcs_specs = config.get("dcs_candidate_grids")
    if (
        not isinstance(prior_result, dict)
        or prior_result.get("passed") is not False
        or int(prior_result.get("selected_count", -1)) != 3
        or not isinstance(prior_audit, dict)
        or prior_audit.get("passed") is not True
        or prior_audit.get("decision", {}).get(
            "proposal_weight_optimization_required"
        )
        is not True
        or config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or config.get("weight_training_namespace")
        != "v8-r2-weight-optimized-proposal-training-v1"
        or config.get("validation_namespace")
        != "v8-r2-weight-optimized-proposal-validation-v1"
        or not isinstance(retained, list)
        or {
            (str(item.get("cell_id")), str(item.get("method")))
            for item in retained
        }
        != RETAINED_REQUIREMENTS
        or not isinstance(raw_specs, list)
        or {
            (str(item.get("cell_id")), RAW_METHOD) for item in raw_specs
        }
        != {
            requirement
            for requirement in UNRESOLVED_REQUIREMENTS
            if requirement[1] == RAW_METHOD
        }
        or not isinstance(dcs_specs, list)
        or {
            (str(item.get("cell_id")), DCS_METHOD) for item in dcs_specs
        }
        != {
            requirement
            for requirement in UNRESOLVED_REQUIREMENTS
            if requirement[1] == DCS_METHOD
        }
    ):
        raise ValueError("weight-optimized provenance or requirement roster changed")
    for specification in raw_specs:
        if (
            not isinstance(specification.get("source_candidate_id"), str)
            or not specification["source_candidate_id"].startswith(
                f"{specification['cell_id']}/raw/"
            )
        ):
            raise ValueError("raw source-candidate binding is malformed")
    for specification in dcs_specs:
        profiles = specification.get("source_training_replicates")
        families = specification.get("families")
        if (
            not isinstance(profiles, list)
            or not profiles
            or len(profiles) != len(set(int(value) for value in profiles))
            or any(int(value) not in {0, 1, 2} for value in profiles)
            or not isinstance(families, list)
            or not families
        ):
            raise ValueError("DCS source-profile or family roster is malformed")
        for family in families:
            scales = family.get("scales")
            weights = family.get("weights")
            if (
                not isinstance(family.get("id"), str)
                or not isinstance(scales, list)
                or not isinstance(weights, list)
                or len(scales) != len(weights)
                or len(scales) < 2
                or float(scales[0]) != 0.0
                or any(float(scale) <= 0.0 for scale in scales[1:])
                or any(float(weight) <= 0.0 for weight in weights)
                or not math.isclose(sum(float(weight) for weight in weights), 1.0)
            ):
                raise ValueError("DCS candidate family is malformed")
    training = config.get("weight_training")
    validation = config.get("validation")
    decision = config.get("decision")
    if (
        not isinstance(training, dict)
        or not isinstance(validation, dict)
        or not isinstance(decision, dict)
    ):
        raise ValueError("weight training, validation, and decision must be mappings")
    if (
        int(training.get("replicates", 0)) != 4
        or int(training.get("paths_per_replicate", 0)) != 32768
        or int(training.get("iterations", 0)) != 500
        or float(training.get("learning_rate", 0.0)) != 0.05
        or float(training.get("natural_weight", 0.0)) != 0.08
        or float(training.get("minimum_nonnatural_weight", 0.0)) != 0.005
        or validation.get("block_partition")
        != "independent_seeded_uniform_permutation_before_equal_slicing"
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
        or decision.get("proposal_manifest_build_authorized") is not False
        or decision.get("new_formal_pilot_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
        or decision.get("submission_authorized") is not False
    ):
        raise ValueError("weight training, validation, or decision contract changed")
    return config, hashlib.sha256(raw).hexdigest()


def _candidate_by_id(
    prior_result: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in prior_result["candidates"]
        if candidate.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"source candidate is not unique: {candidate_id}")
    return matches[0]


def _profile(
    prior_result: dict[str, Any], cell_id: str, replicate: int
) -> tuple[tuple[float, float], ...]:
    matches = [
        fit
        for fit in prior_result["fits"]
        if fit.get("cell_id") == cell_id
        and int(fit.get("training_replicate", -1)) == replicate
    ]
    if len(matches) != 1:
        raise ValueError("source CEM profile is not unique")
    return tuple(
        (float(pair[0]), float(pair[1])) for pair in matches[0]["control"]
    )


def _floored_weights(
    logits: torch.Tensor,
    *,
    natural_weight: float,
    minimum_nonnatural_weight: float,
) -> torch.Tensor:
    if logits.ndim != 1 or logits.numel() < 1:
        raise ValueError("weight logits must be a nonempty vector")
    remaining = (
        1.0
        - natural_weight
        - minimum_nonnatural_weight * int(logits.numel())
    )
    if remaining <= 0.0:
        raise ValueError("weight floor leaves no optimizable simplex mass")
    nonnatural = minimum_nonnatural_weight + remaining * torch.softmax(logits, dim=0)
    natural = torch.tensor(
        [natural_weight], device=logits.device, dtype=logits.dtype
    )
    return torch.cat((natural, nonnatural))


def _raw_second_moment_objective(
    logits: torch.Tensor,
    *,
    component_log_q_over_p: torch.Tensor,
    base_log_q_over_p: torch.Tensor,
    hard_event: torch.Tensor,
    natural_weight: float,
    minimum_nonnatural_weight: float,
) -> torch.Tensor:
    weights = _floored_weights(
        logits,
        natural_weight=natural_weight,
        minimum_nonnatural_weight=minimum_nonnatural_weight,
    )
    candidate_log_q_over_p = torch.logsumexp(
        component_log_q_over_p + torch.log(weights)[None, :],
        dim=1,
    )
    return torch.mean(
        hard_event
        * torch.exp(-candidate_log_q_over_p - base_log_q_over_p)
    )


def _optimize_raw_weights(
    config: dict[str, Any],
    *,
    context: Any,
    cell: dict[str, Any],
    schedules: list[list[list[float]]],
    base_weights: list[float],
    candidate_id: str,
    seed_records: list[dict[str, Any]],
    smoke: bool,
) -> dict[str, Any]:
    training = config["weight_training"]
    replicates = (
        int(training["smoke_replicates"])
        if smoke
        else int(training["replicates"])
    )
    paths = (
        int(training["smoke_paths_per_replicate"])
        if smoke
        else int(training["paths_per_replicate"])
    )
    iterations = (
        int(training["smoke_iterations"])
        if smoke
        else int(training["iterations"])
    )
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
    component_batches = []
    base_batches = []
    event_batches = []
    for replicate in range(replicates):
        proposal_key, proposal_seed = _seed(
            config,
            stage="raw-weight-training-proposal",
            cell_id=str(cell["cell_id"]),
            candidate_id=candidate_id,
            replicate=replicate,
            namespace=config["weight_training_namespace"],
        )
        label_key, label_seed = _seed(
            config,
            stage="raw-weight-training-labels",
            cell_id=str(cell["cell_id"]),
            candidate_id=candidate_id,
            replicate=replicate,
            namespace=config["weight_training_namespace"],
        )
        seed_records.extend(
            (
                {"key": asdict(proposal_key), "seed": proposal_seed},
                {"key": asdict(label_key), "seed": label_seed},
            )
        )
        sample = _draw(
            simulator=simulator,
            controls=_controls(schedules, float(cell["maturity"])),
            weights=torch.tensor(base_weights, dtype=torch.float64),
            model=model,
            steps=int(cell["finest_steps"]),
            count=paths,
            proposal_seed=proposal_seed,
            label_seed=label_seed,
            engine=cast(Literal["fft", "reference"], training["engine"]),
        )
        hard_event = task.hard_event(sample.paths.spot, sample.paths.step_dt)
        component_batches.append(sample.component_log_q_over_p.detach().cpu())
        base_batches.append(sample.log_mixture_q_over_p.detach().cpu())
        event_batches.append(hard_event.detach().to(device="cpu", dtype=torch.float64))
    component_log = torch.cat(component_batches)
    base_log = torch.cat(base_batches)
    event = torch.cat(event_batches)
    natural_weight = float(training["natural_weight"])
    floor = float(training["minimum_nonnatural_weight"])
    base_nonnatural = torch.tensor(base_weights[1:], dtype=torch.float64)
    residual = 1.0 - natural_weight - floor * len(base_weights[1:])
    initialization = torch.clamp(
        (base_nonnatural - floor) / residual, min=1e-12
    )
    logits = torch.log(initialization)
    logits = (logits - torch.mean(logits)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=float(training["learning_rate"]))
    history = []
    with torch.no_grad():
        base_objective = float(
            _raw_second_moment_objective(
                logits,
                component_log_q_over_p=component_log,
                base_log_q_over_p=base_log,
                hard_event=event,
                natural_weight=natural_weight,
                minimum_nonnatural_weight=floor,
            )
        )
    best_objective = base_objective
    best_weights = torch.tensor(base_weights, dtype=torch.float64)
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        objective = _raw_second_moment_objective(
            logits,
            component_log_q_over_p=component_log,
            base_log_q_over_p=base_log,
            hard_event=event,
            natural_weight=natural_weight,
            minimum_nonnatural_weight=floor,
        )
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("raw mixture-weight objective became nonfinite")
        objective.backward()
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
            raise FloatingPointError("raw mixture-weight gradient became nonfinite")
        optimizer.step()
        with torch.no_grad():
            current_weights = _floored_weights(
                logits,
                natural_weight=natural_weight,
                minimum_nonnatural_weight=floor,
            )
            current = float(
                _raw_second_moment_objective(
                    logits,
                    component_log_q_over_p=component_log,
                    base_log_q_over_p=base_log,
                    hard_event=event,
                    natural_weight=natural_weight,
                    minimum_nonnatural_weight=floor,
                )
            )
            if current < best_objective:
                best_objective = current
                best_weights = current_weights.detach().clone()
            if iteration % 50 == 0 or iteration == iterations - 1:
                history.append(
                    {
                        "iteration": iteration,
                        "objective": current,
                        "minimum_weight": float(torch.min(current_weights)),
                        "maximum_weight": float(torch.max(current_weights)),
                    }
                )
    optimized = [float(value) for value in best_weights]
    if (
        not math.isclose(sum(optimized), 1.0)
        or not math.isclose(optimized[0], natural_weight)
        or any(weight <= 0.0 for weight in optimized)
        or best_objective > base_objective * (1.0 + 1e-12)
    ):
        raise RuntimeError("optimized raw weights violate the defensive contract")
    return {
        "cell_id": cell["cell_id"],
        "candidate_id": candidate_id,
        "training_replicates": replicates,
        "training_paths_per_replicate": paths,
        "event_count": int(torch.count_nonzero(event)),
        "base_weights": base_weights,
        "optimized_weights": optimized,
        "base_empirical_second_moment": base_objective,
        "optimized_empirical_second_moment": best_objective,
        "objective_ratio": (
            best_objective / base_objective
            if base_objective > 0.0
            else 1.0
        ),
        "history": history,
    }


def run_weight_optimized_proposal(
    config_path: Path, *, smoke: bool = False
) -> dict[str, Any]:
    config, config_sha256 = load_weight_optimized_config(config_path)
    prior_result = json.loads(
        _bound_path(config["prior_result"]).read_text(encoding="utf-8")
    )
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v4.yaml"
    )
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("weight-optimized config binds a different threshold manifest")
    raw_specs = (
        config["raw_weight_optimization"][
            : int(config["validation"]["smoke_raw_cells"])
        ]
        if smoke
        else config["raw_weight_optimization"]
    )
    dcs_specs = (
        config["dcs_candidate_grids"][
            : int(config["validation"]["smoke_dcs_cells"])
        ]
        if smoke
        else config["dcs_candidate_grids"]
    )
    validation_replicates = (
        int(config["validation"]["smoke_replicates"])
        if smoke
        else int(config["validation"]["replicates"])
    )
    validation_paths = (
        int(config["validation"]["smoke_paths_per_replicate"])
        if smoke
        else int(config["validation"]["paths_per_replicate"])
    )
    validation_blocks = (
        int(config["validation"]["smoke_blocks_per_replicate"])
        if smoke
        else int(config["validation"]["blocks_per_replicate"])
    )
    seed_records: list[dict[str, Any]] = []
    weight_fits = []
    candidates = []
    for specification in raw_specs:
        cell_id = str(specification["cell_id"])
        cell = context.cells_by_id[cell_id]
        source = _candidate_by_id(
            prior_result, str(specification["source_candidate_id"])
        )
        schedules = source["schedules"]
        base_weights = [float(weight) for weight in source["weights"]]
        candidate_id = f"{cell_id}/raw/second-moment-optimized"
        fit = _optimize_raw_weights(
            config,
            context=context,
            cell=cell,
            schedules=schedules,
            base_weights=base_weights,
            candidate_id=candidate_id,
            seed_records=seed_records,
            smoke=smoke,
        )
        weight_fits.append(fit)
        candidates.append(
            _evaluate_candidate(
                config,
                cell=cell,
                method=RAW_METHOD,
                candidate_id=candidate_id,
                schedules=schedules,
                weights=fit["optimized_weights"],
                replicates=validation_replicates,
                paths=validation_paths,
                blocks=validation_blocks,
                seed_records=seed_records,
            )
        )
    dcs_family_limit = (
        int(config["validation"]["smoke_dcs_families"])
        if smoke
        else None
    )
    for specification in dcs_specs:
        cell_id = str(specification["cell_id"])
        cell = context.cells_by_id[cell_id]
        families = (
            specification["families"][:dcs_family_limit]
            if dcs_family_limit is not None
            else specification["families"]
        )
        for replicate in specification["source_training_replicates"]:
            profile = _profile(prior_result, cell_id, int(replicate))
            for family in families:
                schedules, weights = _dcs_rank_one_mixture(
                    profile,
                    scales=[float(value) for value in family["scales"]],
                    weights=[float(value) for value in family["weights"]],
                )
                candidate_id = (
                    f"{cell_id}/dcs/train-{replicate}/{family['id']}"
                )
                candidates.append(
                    _evaluate_candidate(
                        config,
                        cell=cell,
                        method=DCS_METHOD,
                        candidate_id=candidate_id,
                        schedules=schedules,
                        weights=weights,
                        replicates=validation_replicates,
                        paths=validation_paths,
                        blocks=validation_blocks,
                        seed_records=seed_records,
                    )
                )
    new_selected: dict[str, dict[str, Any]] = {}
    active_requirements = {
        (str(specification["cell_id"]), RAW_METHOD)
        for specification in raw_specs
    } | {
        (str(specification["cell_id"]), DCS_METHOD)
        for specification in dcs_specs
    }
    for cell_id, method in sorted(active_requirements):
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
            passing,
            key=lambda candidate: float(candidate["requested_to_cap_ratio"]),
        )
        new_selected.setdefault(cell_id, {})[method] = {
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
    retained = {
        str(cell_id): {
            str(method): selected
            for method, selected in methods.items()
            if (str(cell_id), str(method)) in RETAINED_REQUIREMENTS
        }
        for cell_id, methods in prior_result["selected_proposals"].items()
        if any(
            (str(cell_id), str(method)) in RETAINED_REQUIREMENTS
            for method in methods
        )
    }
    complete_selected = {
        cell_id: dict(methods) for cell_id, methods in retained.items()
    }
    for cell_id, methods in new_selected.items():
        complete_selected.setdefault(cell_id, {}).update(methods)
    selected_count = sum(len(methods) for methods in complete_selected.values())
    required_count = (
        len(RETAINED_REQUIREMENTS)
        + len(active_requirements)
    )
    seeds = [int(record["seed"]) for record in seed_records]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("weight-training and validation seeds overlap")
    passed = selected_count == required_count
    provenance = source_provenance()
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "weight_training_namespace": config["weight_training_namespace"],
        "validation_namespace": config["validation_namespace"],
        "smoke": smoke,
        "raw_weight_theory": {
            "objective": (
                "E_Qb[event * (p/q_candidate) * (p/q_base)]"
            ),
            "identity": "candidate raw importance-sampling second moment",
            "natural_weight_fixed": config["weight_training"]["natural_weight"],
            "self_normalized": False,
        },
        "dcs_weight_theory": {
            "raw_off_policy_optimizer_applied": False,
            "selection": "frozen finite rank-one candidate grid",
        },
        "validation_design": {
            "replicates": validation_replicates,
            "paths_per_replicate": validation_paths,
            "blocks_per_replicate": validation_blocks,
            "block_partition": config["validation"]["block_partition"],
            "allocation_variance_statistic": (
                "maximum of full-replicate and permuted-block variances"
            ),
        },
        "weight_fits": weight_fits,
        "candidates": candidates,
        "retained_v3_proposals": retained,
        "new_selected_proposals": new_selected,
        "complete_selected_proposals": complete_selected,
        "selected_count": selected_count,
        "required_selection_count": required_count,
        "seed_records": seed_records,
        "seed_count": len(seed_records),
        "passed": passed,
        "decision": {
            "status": (
                "weight_optimized_proposal_falsification_pass"
                if passed
                else "weight_optimized_proposal_falsification_fail"
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
            f"refusing to overwrite weight-optimized result: {arguments.output}"
        )
    result = run_weight_optimized_proposal(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
