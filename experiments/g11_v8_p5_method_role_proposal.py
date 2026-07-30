"""Validate proposals under distinct primary-reference and crosscheck precision roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v8_p5_production_scale_proposal import (
    DCS_METHOD,
    RAW_METHOD,
    _assert_json_finite,
    _dcs_rank_one_mixture,
    _evaluate_candidate,
)
from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from src.path_integral.provenance import runtime_provenance, source_provenance

SCHEMA = "npi.g11.v8-p5-method-role-proposal.v1"
RESULT_SCHEMA = "npi.g11.v8-p5-method-role-proposal-result.v1"
RETAINED_REQUIREMENTS = {
    ("h0.12-terminal_left_tail-p1e-05", DCS_METHOD),
    ("h0.12-discrete_lower_barrier-p1e-03", DCS_METHOD),
    ("h0.20-discrete_lower_barrier-p1e-04", RAW_METHOD),
    ("h0.20-terminal_left_tail-p1e-05", RAW_METHOD),
}
UNRESOLVED_REQUIREMENTS = {
    ("h0.05-terminal_left_tail-p1e-04", RAW_METHOD),
    ("h0.05-terminal_left_tail-p1e-05", DCS_METHOD),
    ("h0.05-terminal_left_tail-p1e-05", RAW_METHOD),
    ("h0.12-discrete_lower_barrier-p1e-05", RAW_METHOD),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("method-role proposal binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("method-role proposal path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("method-role proposal artifact hash mismatch")
    return path


def load_method_role_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected method-role proposal schema")
    weight_result_path = _bound_path(config.get("weight_result"))
    weight_audit_path = _bound_path(config.get("weight_audit"))
    _bound_path(config.get("profile_result"))
    _bound_path(config.get("threshold_binding"))
    weight_result = json.loads(weight_result_path.read_text(encoding="utf-8"))
    weight_audit = json.loads(weight_audit_path.read_text(encoding="utf-8"))
    retained = config.get("retained_requirements")
    raw_specs = config.get("raw_candidates")
    dcs_specs = config.get("dcs_candidates")
    precision = config.get("method_role_precision")
    validation = config.get("validation")
    decision = config.get("decision")
    if (
        not isinstance(weight_result, dict)
        or weight_result.get("passed") is not False
        or int(weight_result.get("selected_count", -1)) != 4
        or not isinstance(weight_audit, dict)
        or weight_audit.get("passed") is not True
        or weight_audit.get("decision", {}).get(
            "method_role_precision_redesign_required"
        )
        is not True
        or config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or config.get("validation_namespace")
        != "v8-r2-method-role-proposal-validation-v1"
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
        raise ValueError("method-role provenance or requirement roster changed")
    for specification in raw_specs:
        natural_weights = specification.get("natural_weights")
        if (
            not isinstance(specification.get("source_candidate_id"), str)
            or not specification["source_candidate_id"].startswith(
                f"{specification['cell_id']}/raw/"
            )
            or natural_weights != [0.08, 0.12, 0.20]
        ):
            raise ValueError("raw defensive candidate roster is malformed")
    for specification in dcs_specs:
        profiles = specification.get("source_training_replicates")
        families = specification.get("families")
        if (
            profiles != [1, 0]
            or not isinstance(families, list)
            or [family.get("id") for family in families]
            != ["focused08", "focused12", "focused20", "focused_shift"]
        ):
            raise ValueError("DCS method-role candidate roster is malformed")
        for family in families:
            scales = family.get("scales")
            weights = family.get("weights")
            if (
                not isinstance(scales, list)
                or not isinstance(weights, list)
                or len(scales) != len(weights)
                or float(scales[0]) != 0.0
                or any(float(scale) <= 0.0 for scale in scales[1:])
                or any(float(weight) <= 0.0 for weight in weights)
                or not math.isclose(sum(float(weight) for weight in weights), 1.0)
            ):
                raise ValueError("DCS method-role family is malformed")
    if (
        not isinstance(precision, dict)
        or precision.get("primary_method") != DCS_METHOD
        or float(precision.get("dcs_relative_standard_error_target", 0.0))
        != 0.02
        or float(precision.get("raw_crosscheck_relative_standard_error_target", 0.0))
        != 0.05
        or float(precision.get("maximum_combined_agreement_z", 0.0)) != 4.0
        or precision.get("raw_may_replace_primary_reference") is not False
        or precision.get("self_normalized") is not False
        or not isinstance(validation, dict)
        or validation.get("method_relative_standard_error_targets")
        != {DCS_METHOD: 0.02, RAW_METHOD: 0.05}
        or validation.get("block_partition")
        != "independent_seeded_uniform_permutation_before_equal_slicing"
        or int(validation.get("replicates", 0)) != 8
        or int(validation.get("paths_per_replicate", 0)) != 32768
        or int(validation.get("blocks_per_replicate", 0)) != 8
        or validation.get("engine") != "fft"
        or float(validation.get("allocation_safety_factor", 0.0)) != 6.0
        or int(validation.get("maximum_final_samples", 0)) != 8388608
        or float(validation.get("maximum_requested_to_cap_ratio", 0.0)) != 0.75
        or int(validation.get("minimum_raw_nonzero_per_replicate", 0)) != 1024
        or int(validation.get("minimum_raw_nonzero_per_block", 0)) != 64
        or float(validation.get("maximum_likelihood_normalization_absolute_z", 0.0))
        != 4.0
        or float(validation.get("maximum_block_variance_to_median_ratio", 0.0))
        != 30.0
        or float(validation.get("maximum_full_contribution_share", 0.0)) != 0.05
        or float(validation.get("maximum_block_contribution_share", 0.0)) != 0.25
        or not isinstance(decision, dict)
        or decision.get("proposal_manifest_build_authorized") is not False
        or decision.get("new_formal_pilot_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
        or decision.get("submission_authorized") is not False
    ):
        raise ValueError("method-role precision, validation, or decision changed")
    return config, hashlib.sha256(raw).hexdigest()


def _candidate_by_id(result: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in result["candidates"]
        if candidate.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"source candidate is not unique: {candidate_id}")
    return matches[0]


def _profile(
    result: dict[str, Any], cell_id: str, replicate: int
) -> tuple[tuple[float, float], ...]:
    matches = [
        fit
        for fit in result["fits"]
        if fit.get("cell_id") == cell_id
        and int(fit.get("training_replicate", -1)) == replicate
    ]
    if len(matches) != 1:
        raise ValueError("source DCS profile is not unique")
    return tuple(
        (float(pair[0]), float(pair[1])) for pair in matches[0]["control"]
    )


def _defensive_weights(
    base_weights: list[float], natural_weight: float
) -> list[float]:
    if (
        len(base_weights) < 2
        or not 0.0 < natural_weight < 1.0
        or any(weight <= 0.0 for weight in base_weights)
    ):
        raise ValueError("defensive weight inputs are invalid")
    nonnatural_sum = sum(base_weights[1:])
    result = [natural_weight] + [
        (1.0 - natural_weight) * weight / nonnatural_sum
        for weight in base_weights[1:]
    ]
    if any(weight <= 0.0 for weight in result) or not math.isclose(
        sum(result), 1.0
    ):
        raise RuntimeError("defensive weights violate the simplex")
    return result


def run_method_role_proposal(
    config_path: Path, *, smoke: bool = False
) -> dict[str, Any]:
    config, config_sha256 = load_method_role_config(config_path)
    weight_result = json.loads(
        _bound_path(config["weight_result"]).read_text(encoding="utf-8")
    )
    profile_result = json.loads(
        _bound_path(config["profile_result"]).read_text(encoding="utf-8")
    )
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v4.yaml"
    )
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("method-role config binds a different threshold manifest")
    raw_specs = (
        config["raw_candidates"][: int(config["validation"]["smoke_raw_cells"])]
        if smoke
        else config["raw_candidates"]
    )
    dcs_specs = (
        config["dcs_candidates"][: int(config["validation"]["smoke_dcs_cells"])]
        if smoke
        else config["dcs_candidates"]
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
    seed_records: list[dict[str, Any]] = []
    candidates = []
    raw_weight_limit = (
        int(config["validation"]["smoke_raw_natural_weights"])
        if smoke
        else None
    )
    for specification in raw_specs:
        cell_id = str(specification["cell_id"])
        source = _candidate_by_id(
            profile_result, str(specification["source_candidate_id"])
        )
        schedules = source["schedules"]
        base_weights = [float(value) for value in source["weights"]]
        natural_weights = (
            specification["natural_weights"][:raw_weight_limit]
            if raw_weight_limit is not None
            else specification["natural_weights"]
        )
        for natural_weight in natural_weights:
            weights = _defensive_weights(base_weights, float(natural_weight))
            candidate_id = (
                f"{cell_id}/raw/defensive-natural-{float(natural_weight):.2f}"
            )
            candidates.append(
                _evaluate_candidate(
                    config,
                    cell=context.cells_by_id[cell_id],
                    method=RAW_METHOD,
                    candidate_id=candidate_id,
                    schedules=schedules,
                    weights=weights,
                    replicates=replicates,
                    paths=paths,
                    blocks=blocks,
                    seed_records=seed_records,
                )
            )
    dcs_family_limit = (
        int(config["validation"]["smoke_dcs_families"])
        if smoke
        else None
    )
    dcs_profile_limit = (
        int(config["validation"]["smoke_dcs_profiles"])
        if smoke
        else None
    )
    for specification in dcs_specs:
        cell_id = str(specification["cell_id"])
        profile_replicates = (
            specification["source_training_replicates"][:dcs_profile_limit]
            if dcs_profile_limit is not None
            else specification["source_training_replicates"]
        )
        families = (
            specification["families"][:dcs_family_limit]
            if dcs_family_limit is not None
            else specification["families"]
        )
        for profile_replicate in profile_replicates:
            profile = _profile(
                profile_result, cell_id, int(profile_replicate)
            )
            for family in families:
                schedules, weights = _dcs_rank_one_mixture(
                    profile,
                    scales=[float(value) for value in family["scales"]],
                    weights=[float(value) for value in family["weights"]],
                )
                candidate_id = (
                    f"{cell_id}/dcs/train-{profile_replicate}/{family['id']}"
                )
                candidates.append(
                    _evaluate_candidate(
                        config,
                        cell=context.cells_by_id[cell_id],
                        method=DCS_METHOD,
                        candidate_id=candidate_id,
                        schedules=schedules,
                        weights=weights,
                        replicates=replicates,
                        paths=paths,
                        blocks=blocks,
                        seed_records=seed_records,
                    )
                )
    active_requirements = {
        (str(specification["cell_id"]), RAW_METHOD)
        for specification in raw_specs
    } | {
        (str(specification["cell_id"]), DCS_METHOD)
        for specification in dcs_specs
    }
    new_selected: dict[str, dict[str, Any]] = {}
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
        for cell_id, methods in weight_result["complete_selected_proposals"].items()
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
    required_count = len(RETAINED_REQUIREMENTS) + len(active_requirements)
    seeds = [int(record["seed"]) for record in seed_records]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("method-role validation seeds overlap")
    passed = selected_count == required_count
    provenance = source_provenance()
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "validation_namespace": config["validation_namespace"],
        "smoke": smoke,
        "method_role_precision": config["method_role_precision"],
        "validation_design": {
            "replicates": replicates,
            "paths_per_replicate": paths,
            "blocks_per_replicate": blocks,
            "block_partition": config["validation"]["block_partition"],
            "method_relative_standard_error_targets": config["validation"][
                "method_relative_standard_error_targets"
            ],
        },
        "candidates": candidates,
        "retained_proposals": retained,
        "new_selected_proposals": new_selected,
        "complete_selected_proposals": complete_selected,
        "selected_count": selected_count,
        "required_selection_count": required_count,
        "seed_records": seed_records,
        "seed_count": len(seed_records),
        "passed": passed,
        "decision": {
            "status": (
                "method_role_proposal_falsification_pass"
                if passed
                else "method_role_proposal_falsification_fail"
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
            f"refusing to overwrite method-role result: {arguments.output}"
        )
    result = run_method_role_proposal(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
