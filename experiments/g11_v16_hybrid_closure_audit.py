"""Fail-closed audit of the V16 mesh-compatible hybrid closure artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.v15_result_audit import file_sha256, git_source_provenance

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_by_epsilon(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(record["epsilon"]): record for record in payload["records"]}


def _strictly_decreasing(values: list[float]) -> bool:
    return all(right < left for left, right in zip(values[:-1], values[1:], strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = {
        key: ROOT / str(value) for key, value in config["artifacts"].items()
    }
    payloads = {key: _load(path) for key, path in artifacts.items()}
    candidate = _records_by_epsilon(payloads["candidate"])
    isotropic = _records_by_epsilon(payloads["isotropic_reference"])
    unrelated = _records_by_epsilon(payloads["unrelated_reference"])
    common_epsilons = sorted(candidate.keys() & isotropic.keys() & unrelated.keys(), reverse=True)
    if not common_epsilons:
        raise RuntimeError("closure audit has no common epsilon cells")

    cell_records = []
    failures: list[str] = []
    maximum_z = 0.0
    for epsilon in common_epsilons:
        current = candidate[epsilon]
        old = isotropic[epsilon]
        reference = unrelated[epsilon]
        combined_se = math.sqrt(
            float(current["candidate_standard_error"]) ** 2
            + float(reference["standard_error"]) ** 2
        )
        z_score = abs(
            float(current["candidate_estimate"]) - float(reference["estimate"])
        ) / combined_se
        maximum_z = max(maximum_z, z_score)
        old_rv = float(old["relative_variance"])
        new_rv = float(current["relative_variance"])
        improvement = 1.0 - new_rv / old_rv
        if bool(config["gates"]["require_relative_variance_improvement_at_every_epsilon"]):
            if improvement <= 0.0:
                failures.append(f"epsilon={epsilon}: candidate did not improve relative variance")
        violation = float(current["maximum_likelihood_bound_violation"])
        if violation > float(config["gates"]["maximum_likelihood_bound_violation"]):
            failures.append(f"epsilon={epsilon}: likelihood bound failed")
        cell_records.append(
            {
                "epsilon": epsilon,
                "candidate_estimate": float(current["candidate_estimate"]),
                "candidate_standard_error": float(current["candidate_standard_error"]),
                "unrelated_reference_estimate": float(reference["estimate"]),
                "unrelated_reference_standard_error": float(reference["standard_error"]),
                "unrelated_reference_z": z_score,
                "candidate_relative_variance": new_rv,
                "isotropic_v3_relative_variance": old_rv,
                "relative_variance_improvement_fraction": improvement,
                "second_moment_exponent_ratio": float(
                    current["second_moment_exponent_ratio"]
                ),
            }
        )
    if maximum_z > float(config["gates"]["maximum_unrelated_reference_z"]):
        failures.append("maximum unrelated-reference z gate failed")

    deepest = min(common_epsilons)
    deepest_ratio = float(candidate[deepest]["second_moment_exponent_ratio"])
    if deepest_ratio < float(config["gates"]["minimum_deepest_second_moment_exponent_ratio"]):
        failures.append("deepest second-moment exponent-ratio gate failed")

    continuum_mesh = {
        int(record["steps"]): record
        for record in payloads["continuum_mesh"]["fixed_rank_mesh"]
    }
    hybrid_mesh = {
        int(record["steps"]): record for record in payloads["hybrid_mesh"]["fixed_rank_mesh"]
    }
    common_steps = sorted(continuum_mesh.keys() & hybrid_mesh.keys())
    bridge_norms = [float(hybrid_mesh[steps]["bridge_coefficient_norm"]) for steps in common_steps]
    action_gaps = [
        float(continuum_mesh[steps]["action"]) - float(hybrid_mesh[steps]["action"])
        for steps in common_steps
    ]
    if bool(config["gates"]["require_monotone_bridge_norm"]) and not _strictly_decreasing(
        bridge_norms
    ):
        failures.append("bridge norm is not strictly decreasing")
    if bool(config["gates"]["require_monotone_hybrid_action_gap"]) and not _strictly_decreasing(
        action_gaps
    ):
        failures.append("hybrid action gap is not strictly decreasing")
    if bridge_norms[-1] > float(config["gates"]["maximum_final_bridge_norm"]):
        failures.append("finest-grid bridge norm gate failed")
    bridge_decay_rate = -math.log2(bridge_norms[-1] / bridge_norms[-2])
    action_gap_decay_rate = -math.log2(action_gaps[-1] / action_gaps[-2])

    provenance_failures = []
    for name, payload in payloads.items():
        provenance = payload.get("source_provenance", {})
        if provenance.get("source_dirty_before_run") is not False:
            provenance_failures.append(f"{name}: dirty source provenance")
    failures.extend(provenance_failures)
    output = {
        "schema": "npi.g11.v16-hybrid-closure-audit.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": git_source_provenance(ROOT),
        "artifact_bindings": {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path)}
            for key, path in artifacts.items()
        },
        "passed": not failures,
        "failures": failures,
        "cells": cell_records,
        "maximum_unrelated_reference_z": maximum_z,
        "deepest_epsilon": deepest,
        "deepest_second_moment_exponent_ratio": deepest_ratio,
        "mesh": {
            "steps": common_steps,
            "bridge_coefficient_norms": bridge_norms,
            "hybrid_action_gaps": action_gaps,
            "last_bridge_decay_rate": bridge_decay_rate,
            "last_action_gap_decay_rate": action_gap_decay_rate,
        },
        "claim_boundary": {
            "confirms_frozen_matrix_only": True,
            "proves_new_asymptotic_rate": False,
            "authorizes_joint_mesh_noise_limit": False,
        },
    }
    output_path = ROOT / str(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "passed": not failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
