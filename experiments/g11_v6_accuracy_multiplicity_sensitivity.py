"""Post-hoc simultaneous-accuracy sensitivity for completed V6 confirmations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_confirmatory import _accuracy_groups, _load
from src.path_integral.provenance import runtime_provenance, source_provenance

_SCHEMA = "npi.g11.v6-accuracy-multiplicity.config.v1"


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported accuracy-multiplicity config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "post_hoc",
        "statistics",
        "accuracy",
        "requirements",
    }:
        raise ValueError("malformed accuracy-multiplicity config fields")
    if (
        payload["phase"] != "post_confirmation_sensitivity"
        or payload["frozen"] is not True
        or payload["post_hoc"] is not True
    ):
        raise ValueError("multiplicity analysis must remain a frozen post-hoc sensitivity")
    statistics = payload["statistics"]
    if set(statistics) != {
        "familywise_alpha",
        "bootstrap_repetitions",
        "bootstrap_seed",
    }:
        raise ValueError("malformed multiplicity statistics")
    alpha = float(statistics["familywise_alpha"])
    repetitions = statistics["bootstrap_repetitions"]
    seed = statistics["bootstrap_seed"]
    if (
        not 0.0 < alpha < 0.5
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("invalid multiplicity statistics")
    accuracy = payload["accuracy"]
    if set(accuracy) != {
        "rmse_engineering_multiplier",
        "minimum_target_attainment_rate",
    }:
        raise ValueError("malformed multiplicity accuracy contract")
    multiplier = float(accuracy["rmse_engineering_multiplier"])
    attainment = float(accuracy["minimum_target_attainment_rate"])
    if multiplier <= 0.0 or not 0.0 < attainment < 1.0:
        raise ValueError("invalid multiplicity accuracy contract")
    requirements = payload["requirements"]
    if set(requirements) != {
        "expected_cells",
        "expected_methods",
        "expected_accuracy_gates_per_group",
        "expected_claims",
        "minimum_clusters",
    }:
        raise ValueError("malformed multiplicity requirements")
    integer_requirements = [
        requirements[field]
        for field in (
            "expected_cells",
            "expected_methods",
            "expected_accuracy_gates_per_group",
            "expected_claims",
            "minimum_clusters",
        )
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in integer_requirements
    ):
        raise ValueError("invalid multiplicity requirements")
    implied_claims = (
        requirements["expected_cells"]
        * requirements["expected_methods"]
        * requirements["expected_accuracy_gates_per_group"]
    )
    if implied_claims != requirements["expected_claims"]:
        raise ValueError("expected claim count is internally inconsistent")
    return payload, hashlib.sha256(raw).hexdigest()


def run(config_path: Path, baseline_path: Path, policy_path: Path) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    baseline, baseline_hash = _load(
        baseline_path, "npi.g11.v6-baseline-qualification.v1"
    )
    policy, policy_hash = _load(policy_path, "npi.g11.v6-routed-policy.v1")
    baseline_records = [
        record for record in baseline["records"] if record["method"] == "pure_cem"
    ]
    policy_records = list(policy["records"])
    baseline_key_list = [
        (str(record["cell_id"]), int(record["cluster"]))
        for record in baseline_records
    ]
    policy_key_list = [
        (str(record["cell_id"]), int(record["cluster"]))
        for record in policy_records
    ]
    baseline_keys = set(baseline_key_list)
    policy_keys = set(policy_key_list)
    if (
        len(baseline_keys) != len(baseline_key_list)
        or len(policy_keys) != len(policy_key_list)
    ):
        raise ValueError("multiplicity sensitivity rejects duplicate cell-cluster records")
    if not baseline_keys or baseline_keys != policy_keys:
        raise ValueError("multiplicity sensitivity requires identical complete pairs")
    cells = sorted({key[0] for key in baseline_keys})
    clusters = sorted({key[1] for key in baseline_keys})
    expected_keys = {(cell, cluster) for cell in cells for cluster in clusters}
    if baseline_keys != expected_keys:
        raise ValueError("every cell must contain every paired cluster")
    requirements = config["requirements"]
    method_count = int(requirements["expected_methods"])
    gates_per_group = int(requirements["expected_accuracy_gates_per_group"])
    claim_count = len(cells) * method_count * gates_per_group
    if (
        len(cells) != int(requirements["expected_cells"])
        or method_count != 2
        or gates_per_group != 2
        or claim_count != int(requirements["expected_claims"])
    ):
        raise ValueError("completed matrix does not match the frozen claim family")
    familywise_alpha = float(config["statistics"]["familywise_alpha"])
    per_claim_alpha = familywise_alpha / claim_count
    simultaneous_confidence = 1.0 - per_claim_alpha
    accuracy = config["accuracy"]
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    seed = int(config["statistics"]["bootstrap_seed"])
    groups = _accuracy_groups(
        baseline_records,
        method_label="pure_cem",
        multiplier=float(accuracy["rmse_engineering_multiplier"]),
        minimum_attainment=float(accuracy["minimum_target_attainment_rate"]),
        confidence_level=simultaneous_confidence,
        repetitions=repetitions,
        seed=seed,
    ) + _accuracy_groups(
        policy_records,
        method_label="v6_policy",
        multiplier=float(accuracy["rmse_engineering_multiplier"]),
        minimum_attainment=float(accuracy["minimum_target_attainment_rate"]),
        confidence_level=simultaneous_confidence,
        repetitions=repetitions,
        seed=seed + 100_000,
    )
    gates = {
        "complete_claim_family": len(groups) * gates_per_group == claim_count,
        "minimum_cluster_count": len(clusters)
        >= int(requirements["minimum_clusters"]),
        "all_simultaneous_attainment_gates": all(
            bool(group["attainment_gate"]) for group in groups
        ),
        "all_simultaneous_rmse_gates": all(
            bool(group["rmse_gate"]) for group in groups
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "post_hoc_label_retained": bool(config["post_hoc"]),
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke_sources": not bool(baseline.get("smoke"))
        and not bool(policy.get("smoke")),
    }
    rmse_ratios = [
        float(group["bootstrap_rmse_upper"])
        / float(group["rmse_tolerance_including_reference"])
        for group in groups
    ]
    attainment_lowers = [
        float(group["one_sided_exact_attainment_lower"]) for group in groups
    ]
    return {
        "schema": "npi.g11.v6-accuracy-multiplicity.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "source_artifact_sha256": {
            "baseline": baseline_hash,
            "policy": policy_hash,
        },
        "cell_count": len(cells),
        "cluster_count": len(clusters),
        "method_count": method_count,
        "claim_count": claim_count,
        "familywise_alpha": familywise_alpha,
        "per_claim_alpha": per_claim_alpha,
        "simultaneous_confidence": simultaneous_confidence,
        "bootstrap_repetitions": repetitions,
        "accuracy": groups,
        "minimum_simultaneous_attainment_lower": min(attainment_lowers),
        "maximum_simultaneous_rmse_ratio": max(rmse_ratios),
        "gates": gates,
        "formal_readiness": formal,
        "sensitivity_gates_passed": all(gates.values()),
        "post_hoc_sensitivity_passed": all(gates.values()) and all(formal.values()),
        "interpretation": (
            "post-hoc Bonferroni robustness only; this artifact does not convert "
            "the original confirmation into a preregistered simultaneous analysis"
        ),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.baseline, arguments.policy)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["post_hoc_sensitivity_passed"], **result["gates"]}))


if __name__ == "__main__":
    main()
