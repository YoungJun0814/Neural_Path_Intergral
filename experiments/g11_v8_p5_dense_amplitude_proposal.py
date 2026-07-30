"""Falsify dense amplitude mixtures on the two remaining raw bottlenecks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v8_p5_cell_tuned_cem_proposal import (
    _evaluate_candidate,
    _scaled_schedules,
)
from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from src.path_integral.provenance import runtime_provenance, source_provenance

SCHEMA_V1 = "npi.g11.v8-p5-dense-amplitude-proposal.v1"
SCHEMA_V2 = "npi.g11.v8-p5-dense-amplitude-proposal.v2"
RESULT_SCHEMAS = {
    SCHEMA_V1: "npi.g11.v8-p5-dense-amplitude-proposal-result.v1",
    SCHEMA_V2: "npi.g11.v8-p5-dense-amplitude-proposal-result.v2",
}
EXPECTED_CELLS_V1 = [
    "h0.05-terminal_left_tail-p1e-05",
    "h0.05-discrete_lower_barrier-p1e-05",
]
EXPECTED_CELLS_V2 = ["h0.05-terminal_left_tail-p1e-05"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("dense-amplitude artifact binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("dense-amplitude artifact path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("dense-amplitude artifact hash mismatch")
    return path


def load_dense_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") not in {
        SCHEMA_V1,
        SCHEMA_V2,
    }:
        raise ValueError("unexpected dense-amplitude proposal schema")
    for field in (
        "all_failed_cells_result",
        "all_failed_cells_audit",
        "threshold_binding",
    ):
        _bound_path(config.get(field))
    version = 1 if config["schema"] == SCHEMA_V1 else 2
    if version == 2:
        _bound_path(config.get("prior_dense_result"))
        _bound_path(config.get("prior_dense_audit"))
    expected_cells = EXPECTED_CELLS_V1 if version == 1 else EXPECTED_CELLS_V2
    if (
        config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or config.get("validation_namespace")
        != f"v8-r2-dense-amplitude-validation-v{version}"
        or config.get("cells") != expected_cells
        or config.get("target_method") != "raw_crosscheck"
    ):
        raise ValueError("dense-amplitude provenance or target contract is invalid")
    families = config.get("proposal_families")
    expected_families = (
        ["dense_low", "dense_center", "dense_broad", "dense_guarded"]
        if version == 1
        else [
            "focused_mid",
            "focused_high",
            "focused_smooth",
            "broad_refined",
            "balanced_refined",
        ]
    )
    if not isinstance(families, list) or [
        family.get("id") for family in families
    ] != expected_families:
        raise ValueError("dense-amplitude family roster changed")
    expected_experts = 7 if version == 1 else 8
    for family in families:
        scales = family.get("scales")
        weights = family.get("weights")
        if (
            not isinstance(scales, list)
            or not isinstance(weights, list)
            or len(scales) != expected_experts
            or len(weights) != expected_experts
            or float(scales[0]) != 0.0
            or any(float(scale) <= 0.0 for scale in scales[1:])
            or any(float(weight) <= 0.0 for weight in weights)
            or not math.isclose(sum(float(weight) for weight in weights), 1.0)
        ):
            raise ValueError("dense-amplitude family is invalid")
    validation = config.get("validation")
    decision = config.get("decision")
    if (
        not isinstance(validation, dict)
        or int(validation.get("replicates", 0)) != 8
        or int(validation.get("paths_per_replicate", 0)) != 8192
        or validation.get("engine") != "fft"
        or float(validation.get("allocation_safety_factor", 0.0)) != 6.0
        or int(validation.get("maximum_final_samples", 0)) != 8388608
        or float(
            validation.get(
                "selected_method_maximum_requested_to_cap_ratio", 0.0
            )
        )
        != 0.50
        or int(
            validation.get("minimum_raw_nonzero_contributions_per_replicate", 0)
        )
        != 256
        or float(
            validation.get("maximum_likelihood_normalization_absolute_z", 0.0)
        )
        != 4.0
        or float(
            validation.get("maximum_replicate_variance_to_median_ratio", 0.0)
        )
        != 20.0
        or float(validation.get("maximum_single_contribution_share", 0.0))
        != 0.10
        or not isinstance(decision, dict)
        or decision.get("proposal_manifest_freeze_authorized") is not False
        or decision.get("new_full_pilot_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
    ):
        raise ValueError("dense-amplitude validation contract is invalid")
    return config, hashlib.sha256(raw).hexdigest()


def _load_v3_result(config: dict[str, Any]) -> dict[str, Any]:
    path = _bound_path(config["all_failed_cells_result"])
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "npi.g11.v8-p5-cell-tuned-cem-proposal-result.v3"
        or value.get("dirty_worktree") is not False
    ):
        raise ValueError("bound V3 CEM result is invalid")
    return value


def run_dense_proposal(
    config_path: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    config, config_sha256 = load_dense_config(config_path)
    v3_result = _load_v3_result(config)
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
    )
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("dense-amplitude config binds a different threshold manifest")
    validation = config["validation"]
    cells = (
        config["cells"][: int(validation["smoke_cells"])]
        if smoke
        else config["cells"]
    )
    profiles_per_cell = (
        int(validation["smoke_profiles_per_cell"]) if smoke else 3
    )
    families = (
        config["proposal_families"][
            : int(validation["smoke_proposal_families"])
        ]
        if smoke
        else config["proposal_families"]
    )
    replicates = (
        int(validation["smoke_replicates"])
        if smoke
        else int(validation["replicates"])
    )
    paths = (
        int(validation["smoke_paths_per_replicate"])
        if smoke
        else int(validation["paths_per_replicate"])
    )
    candidates: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    for cell_id in cells:
        fits = [
            fit
            for fit in v3_result["training_fits"]
            if fit["cell_id"] == cell_id
        ][:profiles_per_cell]
        if len(fits) != profiles_per_cell:
            raise ValueError("bound V3 result is missing a required fitted profile")
        for fit in fits:
            profile = tuple(
                (float(pair[0]), float(pair[1])) for pair in fit["control"]
            )
            for family in families:
                candidate_id = (
                    f"{cell_id}/v3-train-{fit['training_replicate']}/"
                    f"{family['id']}"
                )
                try:
                    candidate = _evaluate_candidate(
                        config,
                        context,
                        cell=context.cells_by_id[cell_id],
                        target_methods=["raw_crosscheck"],
                        candidate_id=candidate_id,
                        schedules=_scaled_schedules(profile, family["scales"]),
                        weights=[float(weight) for weight in family["weights"]],
                        replicates=replicates,
                        paths=paths,
                        seed_records=seed_records,
                    )
                except FloatingPointError as error:
                    candidate = {
                        "candidate_id": candidate_id,
                        "cell_id": cell_id,
                        "target_methods": ["raw_crosscheck"],
                        "weights": family["weights"],
                        "schedules": _scaled_schedules(
                            profile, family["scales"]
                        ),
                        "entries": [],
                        "method_gates": {
                            "raw_crosscheck": {
                                "target_method_margin_pass": False,
                                "replicate_variance_stability_pass": False,
                                "single_contribution_concentration_pass": False,
                            }
                        },
                        "gates": {
                            "finite_paths_and_contributions": False,
                            "raw_coverage_pass": False,
                            "likelihood_normalization_pass": False,
                            "all_target_methods_pass": False,
                        },
                        "numerical_failure": {
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                        },
                        "passes": False,
                    }
                candidates.append(candidate)
    selected: dict[str, dict[str, Any]] = {}
    for cell_id in cells:
        passing = [
            candidate
            for candidate in candidates
            if candidate["cell_id"] == cell_id
            and candidate.get("numerical_failure") is None
            and candidate["passes"]
        ]
        if not passing:
            continue
        chosen = min(
            passing,
            key=lambda candidate: next(
                entry["requested_to_cap_ratio"]
                for entry in candidate["entries"]
                if entry["method"] == "raw_crosscheck"
            ),
        )
        selected[cell_id] = {
            "candidate_id": chosen["candidate_id"],
            "weights": chosen["weights"],
            "schedules": chosen["schedules"],
            "target_method_entry": next(
                entry
                for entry in chosen["entries"]
                if entry["method"] == "raw_crosscheck"
            ),
            "method_gates": chosen["method_gates"]["raw_crosscheck"],
        }
    all_seeds = [record["seed"] for record in seed_records]
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("dense-amplitude validation seeds overlap")
    passed = len(selected) == len(cells)
    provenance = source_provenance()
    return {
        "schema": RESULT_SCHEMAS[config["schema"]],
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "validation_namespace": config["validation_namespace"],
        "smoke": smoke,
        "design_informed_by_prior_development_outcomes": True,
        "current_namespace_outcomes_inspected_before_freeze": False,
        "cells": cells,
        "source_profile_result_sha256": config[
            "all_failed_cells_result"
        ]["sha256"],
        "validation_replicates": replicates,
        "validation_paths_per_replicate": paths,
        "candidates": candidates,
        "selected_proposals": selected,
        "seed_records": seed_records,
        "seed_count": len(seed_records),
        "passed": passed,
        "decision": {
            "status": (
                "dense_amplitude_proposal_falsification_pass"
                if passed
                else "dense_amplitude_proposal_falsification_fail"
            ),
            "selected_proposal_frozen": False,
            "proposal_manifest_freeze_authorized": False,
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
            f"refusing to overwrite dense-amplitude result: {arguments.output}"
        )
    result = run_dense_proposal(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
