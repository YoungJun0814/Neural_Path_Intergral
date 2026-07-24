"""Independent aggregate audit for the complete V7 qualification package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import scipy.stats
import yaml

from src.path_integral.provenance import runtime_provenance, source_provenance


def _load(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _config_hash(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected config {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-13)


def _quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid audit quantile")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_rmse(
    errors: list[float],
    *,
    repetitions: int,
    confidence: float,
    seed: int,
) -> float:
    generator = random.Random(seed)
    count = len(errors)
    values = []
    for _ in range(repetitions):
        squared_sum = math.fsum(
            errors[generator.randrange(count)] ** 2 for _ in range(count)
        )
        values.append(math.sqrt(squared_sum / count))
    return _quantile(values, confidence)


def _one_sided_effect(values: list[float], confidence: float) -> dict[str, float]:
    count = len(values)
    mean = math.fsum(values) / count
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        count - 1
    )
    standard_error = math.sqrt(variance / count)
    if standard_error == 0.0:
        lower = mean
        p_value = 0.0 if mean > 0.0 else 1.0
    else:
        critical = float(scipy.stats.t.ppf(confidence, df=count - 1))
        lower = mean - critical * standard_error
        p_value = float(scipy.stats.t.sf(mean / standard_error, df=count - 1))
    return {
        "mean_log_ratio": mean,
        "standard_error": standard_error,
        "geometric_mean_ratio": math.exp(mean),
        "one_sided_lower_geometric_mean_ratio": math.exp(lower),
        "p_value_against_ratio_one": p_value,
    }


def _work(record: dict[str, Any], roles: set[str] | None = None) -> float:
    return math.fsum(
        float(entry["work_units"])
        for entry in record["result"]["core"]["work"]["entries"]
        if roles is None or str(entry["role"]) in roles
    )


def _wall(record: dict[str, Any]) -> float:
    return math.fsum(
        float(entry["wall_seconds"])
        for entry in record["result"]["core"]["work"]["entries"]
        if entry["role"] == "final"
    )


def _seed_values(payload: dict[str, Any]) -> set[int]:
    records = payload["seed_ledger"]["records"]
    values = [int(record["seed"]) for record in records]
    if len(values) != len(set(values)):
        raise ValueError("qualification package contains duplicate seed values")
    return set(values)


def _recompute_joint_effects(
    fixed: dict[str, Any],
    *,
    confidence: float,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    records = {
        (
            str(record["cell_id"]),
            int(record["cluster"]),
            str(record["method"]),
        ): record
        for record in fixed["records"]
    }
    cells = sorted({key[0] for key in records})
    clusters = sorted({key[1] for key in records})
    metrics: dict[str, list[float]] = {
        "execution_variance": [],
        "final_work": [],
        "nontraining_work": [],
        "training_inclusive_work": [],
        "final_wall": [],
    }
    floors = {"fixed_dcs_slis": 0, "fixed_raw_defensive": 0}
    for cluster in clusters:
        within: dict[str, list[float]] = {
            metric: [] for metric in metrics
        }
        for cell in cells:
            dcs = records[(cell, cluster, "fixed_dcs_slis")]
            raw = records[(cell, cluster, "fixed_raw_defensive")]
            ratios = {
                "execution_variance": float(
                    raw["result"]["core"]["terms"][0]["variance"]
                )
                / float(dcs["result"]["core"]["terms"][0]["variance"]),
                "final_work": _work(raw, {"final"})
                / _work(dcs, {"final"}),
                "nontraining_work": _work(
                    raw, {"allocation_pilot", "final"}
                )
                / _work(dcs, {"allocation_pilot", "final"}),
                "training_inclusive_work": _work(raw) / _work(dcs),
                "final_wall": _wall(raw) / _wall(dcs),
            }
            for metric, ratio in ratios.items():
                within[metric].append(math.log(ratio))
            for method, record in (
                ("fixed_dcs_slis", dcs),
                ("fixed_raw_defensive", raw),
            ):
                if (
                    int(record["result"]["core"]["allocations"][0]["final_count"])
                    == int(record["preparation"]["minimum_final_samples"])
                ):
                    floors[method] += 1
        for metric in metrics:
            metrics[metric].append(math.fsum(within[metric]) / len(cells))
    return (
        {
            metric: _one_sided_effect(values, confidence)
            for metric, values in metrics.items()
        },
        floors,
    )


def _recompute_accuracy(
    fixed: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    records = list(fixed["records"])
    methods = list(config["matrix"]["methods"])
    claims = int(config["requirements"]["expected_claims"])
    confidence = 1.0 - float(config["statistics"]["familywise_alpha"]) / claims
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    multiplier = float(config["accuracy"]["rmse_engineering_multiplier"])
    minimum_attainment = float(
        config["accuracy"]["minimum_target_attainment_rate"]
    )
    output = []
    for method_index, method in enumerate(methods):
        method_records = [
            record for record in records if record["method"] == method
        ]
        cells = sorted({str(record["cell_id"]) for record in method_records})
        for cell_index, cell in enumerate(cells):
            group = [
                record
                for record in method_records
                if record["cell_id"] == cell
            ]
            first = group[0]
            reference = float(first["reference_probability"])
            reference_se = float(first["reference_standard_error"])
            nominal = float(first["nominal_probability"])
            requested = float(
                first["result"]["core"]["requested_relative_sampling_rmse"]
            )
            errors = [
                float(record["result"]["core"]["estimate"]) - reference
                for record in group
            ]
            successes = sum(
                bool(record["result"]["core"]["empirical_target_attained"])
                for record in group
            )
            count = len(group)
            lower = (
                0.0
                if successes == 0
                else float(
                    scipy.stats.beta.ppf(
                        1.0 - confidence,
                        successes,
                        count - successes + 1,
                    )
                )
            )
            empirical_rmse = math.sqrt(
                math.fsum(error * error for error in errors) / count
            )
            upper = _bootstrap_rmse(
                errors,
                repetitions=repetitions,
                confidence=confidence,
                seed=base_seed + method_index * 100_000 + cell_index,
            )
            tolerance = math.hypot(
                multiplier * nominal * requested,
                reference_se,
            )
            output.append(
                {
                    "method": method,
                    "cell_id": cell,
                    "cluster_count": count,
                    "target_attainment_count": successes,
                    "target_attainment_rate": successes / count,
                    "one_sided_exact_attainment_lower": lower,
                    "minimum_target_attainment_rate": minimum_attainment,
                    "empirical_rmse": empirical_rmse,
                    "bootstrap_rmse_upper": upper,
                    "rmse_tolerance_including_reference": tolerance,
                    "attainment_gate": lower >= minimum_attainment,
                    "rmse_gate": upper <= tolerance,
                }
            )
    return output


def audit(
    freeze_path: Path,
    probe_path: Path,
    probe_audit_path: Path,
    fixed_path: Path,
    fixed_audit_path: Path,
    resources_path: Path,
    joint_path: Path,
    accuracy_path: Path,
    probe_config_path: Path,
    fixed_config_path: Path,
    joint_config_path: Path,
    accuracy_config_path: Path,
) -> dict[str, Any]:
    freeze, freeze_hash = _load(
        freeze_path, "npi.g11.v7-qualification-freeze.v1"
    )
    probe, probe_hash = _load(
        probe_path, "npi.g11.v7-mechanism-probe.v1"
    )
    probe_audit, probe_audit_hash = _load(
        probe_audit_path, "npi.g11.v7-mechanism-probe-audit.v1"
    )
    fixed, fixed_hash = _load(
        fixed_path, "npi.g11.v6-secondary-baselines.v1"
    )
    fixed_audit, fixed_audit_hash = _load(
        fixed_audit_path, "npi.g11.v6-independent-audit.v1"
    )
    resources, resources_hash = _load(
        resources_path, "npi.g11.v6-resource-supplement.v1"
    )
    joint, joint_hash = _load(
        joint_path, "npi.g11.v7-mechanism-analysis.v1"
    )
    accuracy, accuracy_hash = _load(
        accuracy_path, "npi.g11.v7-accuracy.v1"
    )
    probe_config, probe_config_hash = _config_hash(
        probe_config_path, "npi.g11.v7-mechanism-probe.config.v2"
    )
    fixed_config, fixed_config_hash = _config_hash(
        fixed_config_path, "npi.g11.v6-secondary-baselines.config.v2"
    )
    joint_config, joint_config_hash = _config_hash(
        joint_config_path, "npi.g11.v7-mechanism-analysis.config.v1"
    )
    accuracy_config, accuracy_config_hash = _config_hash(
        accuracy_config_path, "npi.g11.v7-accuracy.config.v1"
    )
    failures: list[str] = []
    expected_config_hashes = {
        "mechanism_probe": probe_config_hash,
        "fixed_estimators": fixed_config_hash,
        "joint_analysis": joint_config_hash,
        "simultaneous_accuracy": accuracy_config_hash,
    }
    if freeze.get("qualification_config_sha256") != expected_config_hashes:
        failures.append("qualification config hashes differ from freeze")
    source_commit = freeze.get("source_commit")
    formal_artifacts = (
        probe,
        probe_audit,
        fixed,
        fixed_audit,
        resources,
        joint,
        accuracy,
    )
    if any(
        artifact.get("source_commit") != source_commit
        or bool(artifact.get("dirty_worktree"))
        for artifact in formal_artifacts
    ):
        failures.append("formal source commit or cleanliness drift")
    links = {
        "probe audit": probe_audit.get("source_artifact_sha256")
        == probe_hash,
        "fixed audit": fixed_audit.get("source_artifact_sha256")
        == fixed_hash,
        "resource supplement": resources.get("source_artifact_sha256")
        == fixed_hash,
        "joint probe": joint.get("source_artifact_sha256", {}).get(
            "mechanism_probe"
        )
        == probe_hash,
        "joint fixed": joint.get("source_artifact_sha256", {}).get(
            "fixed_estimators"
        )
        == fixed_hash,
        "accuracy fixed": accuracy.get("source_artifact_sha256")
        == fixed_hash,
    }
    if not all(links.values()):
        failures.append("qualification artifact hash link failed")
    decisions = {
        "freeze": bool(freeze.get("qualification_authorized")),
        "probe": bool(probe.get("formal_mechanism_passed")),
        "probe_audit": bool(probe_audit.get("passed")),
        "fixed": bool(fixed.get("secondary_baselines_qualified")),
        "fixed_audit": bool(fixed_audit.get("qualification_audit_passed")),
        "resources": bool(resources.get("passed")),
        "joint": bool(joint.get("formal_mechanism_qualified")),
        "accuracy": bool(accuracy.get("accuracy_qualified")),
    }
    if not all(decisions.values()):
        failures.append("one or more qualification decisions failed")
    if not _seed_values(probe).isdisjoint(_seed_values(fixed)):
        failures.append("mechanism and fixed-estimator seeds overlap")
    recomputed_effects, recomputed_floors = _recompute_joint_effects(
        fixed,
        confidence=float(joint_config["statistics"]["confidence_level"]),
    )
    for metric, recomputed in recomputed_effects.items():
        recorded = joint["effects"][metric]
        if any(
            not _close(float(recorded[field]), value)
            for field, value in recomputed.items()
        ):
            failures.append(f"joint {metric} effect mismatch")
    if joint.get("floor_counts") != recomputed_floors:
        failures.append("joint floor counts mismatch")
    recomputed_accuracy = _recompute_accuracy(fixed, accuracy_config)
    recorded_accuracy = accuracy.get("accuracy")
    if (
        not isinstance(recorded_accuracy, list)
        or len(recorded_accuracy) != len(recomputed_accuracy)
    ):
        failures.append("accuracy group count mismatch")
    else:
        for index, (recorded, recomputed) in enumerate(
            zip(recorded_accuracy, recomputed_accuracy, strict=True)
        ):
            for field, expected in recomputed.items():
                observed = recorded.get(field)
                if isinstance(expected, float):
                    if (
                        isinstance(observed, bool)
                        or not isinstance(observed, (int, float))
                        or not _close(float(observed), expected)
                    ):
                        failures.append(
                            f"accuracy group {index} field {field} mismatch"
                        )
                elif observed != expected:
                    failures.append(
                        f"accuracy group {index} field {field} mismatch"
                    )
    provenance = source_provenance()
    return {
        "schema": "npi.g11.v7-qualification-audit.v1",
        "freeze_sha256": freeze_hash,
        "artifact_sha256": {
            "mechanism_probe": probe_hash,
            "mechanism_probe_audit": probe_audit_hash,
            "fixed_estimators": fixed_hash,
            "fixed_estimators_audit": fixed_audit_hash,
            "resource_supplement": resources_hash,
            "joint_analysis": joint_hash,
            "simultaneous_accuracy": accuracy_hash,
        },
        "links": links,
        "decisions": decisions,
        "recomputed_effects": recomputed_effects,
        "recomputed_floor_counts": recomputed_floors,
        "recomputed_accuracy_groups": len(recomputed_accuracy),
        "failures": failures,
        "qualification_audit_passed": not failures,
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-audit", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--fixed-audit", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--fixed-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--accuracy-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(
        arguments.freeze,
        arguments.probe,
        arguments.probe_audit,
        arguments.fixed,
        arguments.fixed_audit,
        arguments.resources,
        arguments.joint,
        arguments.accuracy,
        arguments.probe_config,
        arguments.fixed_config,
        arguments.joint_config,
        arguments.accuracy_config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "qualification_audit_passed": result[
                    "qualification_audit_passed"
                ],
                "failure_count": len(result["failures"]),
            }
        )
    )


if __name__ == "__main__":
    main()
