"""Independent structural and arithmetic audit for V8 D1 development results."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class D1Audit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    recomputed_aggregate: dict[str, Any]
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_standard_json(path: Path) -> dict[str, Any]:
    """Reject the non-standard NaN/Infinity tokens accepted by Python by default."""

    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON numeric token: {token}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("D1 result must be a JSON object")
    return value


def _effective_config(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    child = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(child, dict):
        raise ValueError("D1 config must be a mapping")
    if child.get("schema") == "npi.g11.v8-d1-p7-falsification-stage-a.v1":
        return child, hashlib.sha256(raw).hexdigest()
    if child.get("schema") != "npi.g11.v8-d1-p7-falsification-stage-a.v2":
        raise ValueError("unsupported D1 audit config schema")
    parent = child.get("parent_config")
    if not isinstance(parent, dict) or set(parent) != {"path", "sha256"}:
        raise ValueError("invalid D1 parent binding")
    parent_path = root / str(parent["path"])
    if not parent_path.is_file() or _sha256(parent_path) != parent["sha256"]:
        raise ValueError("D1 parent binding mismatch")
    effective = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if not isinstance(effective, dict):
        raise ValueError("D1 parent must be a mapping")
    effective["schema"] = child["schema"]
    for key in ("protocol_id", "namespace", "base_seed", "outcome_data_used_before_freeze"):
        effective[key] = child.get(key)
    effective["bindings"] = dict(effective["bindings"])
    effective["bindings"]["v1_failure_receipt"] = child.get("failure_receipt")
    return effective, hashlib.sha256(raw).hexdigest()


def _geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        return math.inf
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _recompute_aggregate(
    config: dict[str, Any], paired: list[dict[str, Any]], external: list[dict[str, Any]]
) -> dict[str, Any]:
    maximum_error = float(config["maximum_exactness_error"])
    normalization_limit = float(config["maximum_likelihood_normalization_absolute_z"])
    exactness_pass = all(
        max(float(value) for value in record["exactness"].values()) <= maximum_error
        for record in paired
    ) and all(record["lifecycle_audit_passed"] is True for record in external)
    finite_likelihoods = all(
        record["likelihood_diagnostics"]["nonfinite_weight_count"] == 0
        for record in external
    )
    representable = all(
        record["likelihood_diagnostics"]["normalization_moments_representable"] is True
        for record in external
    )
    normalization_passes = sum(
        record["likelihood_diagnostics"]["normalization_moments_representable"] is True
        and abs(float(record["likelihood_diagnostics"]["normalization_z"]))
        <= normalization_limit
        for record in external
    )
    ratios = [float(record["mechanism"]["variance_ratio_raw_over_dcs"]) for record in paired]
    geometric_ratio = _geometric_mean(ratios)
    mechanism_pass = geometric_ratio >= float(
        config["gate"]["minimum_mechanism_variance_ratio"]
    )
    primary = set(config["external_methods"]["primary"])
    primary_records = [record for record in external if record["method"] in primary]
    expected_primary = (
        len(config["cells"]) * len(config["budgets"]) * int(config["clusters"]) * len(primary)
    )
    primary_evaluable = len(primary_records) == expected_primary and all(
        math.isfinite(float(record["estimate"]["value"]))
        and math.isfinite(float(record["estimate"]["standard_error"]))
        for record in primary_records
    )
    primary_censoring = sum(
        bool(record["allocation"]["resource_censored"]) for record in primary_records
    )
    flow_pass = all(
        record["flow_roundtrip"] is None
        or (
            float(record["flow_roundtrip"]["maximum_reconstruction_error"])
            <= maximum_error
            and float(
                record["flow_roundtrip"]["maximum_log_jacobian_cancellation_error"]
            )
            <= maximum_error
        )
        for record in external
    )
    stage_b = exactness_pass and finite_likelihoods and mechanism_pass and primary_evaluable
    stage_b = stage_b and flow_pass
    return {
        "paired_record_count": len(paired),
        "external_record_count": len(external),
        "maximum_exactness_error": max(
            max(float(value) for value in record["exactness"].values()) for record in paired
        ),
        "exactness_pass": exactness_pass,
        "all_external_weights_finite": finite_likelihoods,
        "all_likelihood_moments_representable": representable,
        "likelihood_normalization_pass_fraction": normalization_passes / len(external),
        "mechanism_geometric_variance_ratio": geometric_ratio,
        "mechanism_pass": mechanism_pass,
        "primary_external_evaluable": primary_evaluable,
        "primary_resource_censoring_count": primary_censoring,
        "flow_roundtrip_pass": flow_pass,
        "dcs_proposal_training_cost_closed": False,
        "stage_b_authorized": stage_b,
        "p8_blockers": [
            *([] if exactness_pass else ["exactness_failure"]),
            *([] if finite_likelihoods else ["nonfinite_likelihood_weight"]),
            *([] if representable else ["nonrepresentable_likelihood_moment"]),
            *([] if mechanism_pass else ["mechanism_gate_failure"]),
            *([] if primary_evaluable else ["primary_external_method_not_evaluable"]),
            *([] if primary_censoring == 0 else ["primary_resource_censoring"]),
            "dcs_proposal_training_cost_not_closed",
            "stage_b_and_stage_c_not_complete",
            "t1_theorem_and_novelty_not_closed",
        ],
    }


def audit_d1_stage_a(
    *, config_path: Path, result: dict[str, Any], root: Path
) -> D1Audit:
    """Recompute D1 Stage A structure, streams, costs, gates, and claim locks."""

    config, config_sha256 = _effective_config(config_path, root)
    paired = result.get("paired_records")
    external = result.get("external_records")
    if not isinstance(paired, list) or not isinstance(external, list):
        raise ValueError("D1 result records must be lists")
    recomputed = _recompute_aggregate(config, paired, external)

    expected_paired = len(config["cells"]) * len(config["budgets"]) * int(config["clusters"])
    barrier_cells = sum(cell["task"] == "discrete_lower_barrier" for cell in config["cells"])
    terminal_cells = len(config["cells"]) - barrier_cells
    primary_count = expected_paired * len(config["external_methods"]["primary"])
    secondary_methods = config["external_methods"]["secondary"]
    secondary_per_cluster = len(config["cells"]) * len(secondary_methods) - barrier_cells
    expected_external = primary_count + (
        secondary_per_cluster * int(config["clusters"])
    )

    seeds: list[int] = []
    for record in paired:
        seeds.extend((int(record["path_seed"]), int(record["label_seed"])))
    for record in external:
        seeds.extend(int(value) for value in record["seeds"].values())
    ordered_seeds = sorted(seeds)
    seed_payload = json.dumps(ordered_seeds, separators=(",", ":")).encode()
    seed_sha256 = hashlib.sha256(seed_payload).hexdigest()
    expected_seeds = list(range(int(config["base_seed"]), int(config["base_seed"]) + len(seeds)))

    bindings_valid = True
    for binding in config.get("bindings", {}).values():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            bindings_valid = False
            break
        path = root / str(binding["path"])
        bindings_valid = path.is_file() and _sha256(path) == binding["sha256"]
        if not bindings_valid:
            break

    costs_close = True
    for record in external:
        training = float(record["proposal"]["training_cost"]["algorithmic_work_units"])
        planning = float(record["pilot"]["cost"]["algorithmic_work_units"])
        final = float(record["estimate"]["final_cost"]["algorithmic_work_units"])
        method_dimension = (
            2 if record["method"] == "conditional_rbergomi" else 3
        ) * int(config["model"]["steps"])
        diagnostic = int(config["diagnostic_samples"]) * 2 * method_dimension
        recorded = float(record["total_algorithmic_work_units_including_diagnostic"])
        if not math.isclose(recorded, training + planning + final + diagnostic, rel_tol=1e-12):
            costs_close = False
            break

    claim_locks = result.get("decision") == {
        "stage_a_complete": True,
        "stage_b_authorized": recomputed["stage_b_authorized"],
        "p8_qualification_authorized": False,
        "performance_claim_authorized": False,
        "submission_authorized": False,
    }
    checks = (
        ("config_hash", result.get("config_sha256") == config_sha256),
        ("protocol", result.get("protocol_id") == config.get("protocol_id")),
        ("namespace", result.get("namespace") == config.get("namespace")),
        ("bindings", bindings_valid),
        ("paired_record_count", len(paired) == expected_paired),
        ("external_record_count", len(external) == expected_external),
        ("seed_uniqueness", len(seeds) == len(set(seeds))),
        ("seed_contiguity", ordered_seeds == expected_seeds),
        ("seed_count", result.get("seed_count") == len(seeds)),
        ("seed_hash", result.get("seed_set_sha256") == seed_sha256),
        ("aggregate_recomputation", result.get("aggregate") == recomputed),
        ("cost_conservation", costs_close),
        ("claim_locks", claim_locks),
        ("terminal_cells_present", terminal_cells > 0),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return D1Audit(
        checks=checks,
        failures=failures,
        recomputed_aggregate=recomputed,
        passed=not failures,
    )
