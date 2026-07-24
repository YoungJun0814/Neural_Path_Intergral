"""Materialize all proposal-dependent V6 qualification configs atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_materialize_proposal_bank import (
    materialize_proposal_policy,
)


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


def materialize_qualification_suite(
    template_path: Path,
    training_source_path: Path,
    *,
    manifest_cell_count: int,
    clusters: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build full-policy, selector-off, and fixed-secondary frozen configs."""

    policy, bank_receipt = materialize_proposal_policy(
        template_path,
        training_source_path,
        protocol_id="g11-v6-routed-policy-qualification-v2",
        phase="qualification",
        manifest_cell_count=manifest_cell_count,
        clusters=clusters,
    )
    policy["schema"] = "npi.g11.v6-routed-policy.config.v4"
    sampling_section = policy.pop("sampling")
    sampling_section.pop("dcs_pilot_samples", None)
    gate_section = policy.pop("gates")
    policy["direct_planning"] = {
        "decision_mode": "replicated_point_variance",
        "replicates": int(policy["selector"]["planning_replicates"]),
        "samples_per_replicate": int(
            policy["selector"]["samples_per_replicate"]
        ),
        "variance_statistic": str(
            policy["selector"]["planning_variance_statistic"]
        ),
        "dcs_zero_variance_fallback": "nominal_probability_squared",
        "crude_zero_variance_fallback": "nominal_bernoulli_variance",
    }
    policy["rarity_band_design"] = {
        "nominal_probability_upper_multiplier": 2.0,
        "reference_certificate_z": 4.0,
    }
    policy["sampling"] = sampling_section
    policy["gates"] = gate_section
    policy["qualification_decision"] = {
        "per_record_empirical_target_role": (
            "deferred_to_prespecified_method_cell_attainment_and_bootstrap_rmse_co_gates"
        ),
        "aggregate_accuracy_protocol_id": (
            "g11-v6-power-analysis-qualification-v1"
        ),
    }
    selector_off = json.loads(json.dumps(policy))
    selector_off["protocol_id"] = (
        "g11-v6-routed-policy-selector-off-qualification-v2"
    )
    selector_off["router"]["maximum_hybrid_profile_work"] = 1.0
    minimum_profile_batch_work = (
        int(selector_off["selector"]["planning_replicates"])
        * int(selector_off["selector"]["samples_per_replicate"])
        * int(selector_off["hierarchy"]["coarsest_steps"])
    )
    if (
        float(selector_off["router"]["maximum_hybrid_profile_work"])
        >= minimum_profile_batch_work
    ):
        raise AssertionError("selector-off cap does not preclude one profile batch")

    sampling = policy["sampling"]
    secondary = {
        "schema": "npi.g11.v6-secondary-baselines.config.v2",
        "protocol_id": "g11-v6-secondary-baselines-qualification-v2",
        "phase": "qualification",
        "frozen": True,
        "estimand": "fixed_finest_grid",
        "methods": ["fixed_dcs_slis", "fixed_raw_defensive"],
        "hierarchy": json.loads(json.dumps(policy["hierarchy"])),
        "proposal": json.loads(json.dumps(policy["proposal"])),
        "planning": {
            "decision_mode": "replicated_point_variance",
            "replicates": int(policy["selector"]["planning_replicates"]),
            "samples_per_replicate": max(
                512, int(policy["selector"]["samples_per_replicate"])
            ),
            "variance_statistic": str(
                policy["selector"]["planning_variance_statistic"]
            ),
            "zero_variance_fallback": "nominal_probability_squared",
        },
        "rarity_band_design": json.loads(
            json.dumps(policy["rarity_band_design"])
        ),
        "sampling": {
            "clusters": clusters,
            "relative_sampling_rmse": float(
                sampling["relative_sampling_rmse"]
            ),
            "confidence_level": float(sampling["confidence_level"]),
            "minimum_final_samples": int(sampling["minimum_final_samples"]),
            "chunk_size": int(sampling["chunk_size"]),
            "allocation_safety_factor": float(
                sampling["allocation_safety_factor"]
            ),
            "familywise_alpha": float(
                policy["selector"]["familywise_alpha"]
            ),
            "operation_work_cap": float(sampling["operation_work_cap"]),
            "engine": str(sampling["engine"]),
        },
        "qualification_decision": {
            "per_record_empirical_target_role": (
                "deferred_to_prespecified_method_cell_attainment_and_bootstrap_rmse_co_gates"
            ),
            "aggregate_accuracy_protocol_id": (
                "g11-v6-secondary-accuracy-qualification-v1"
            ),
        },
    }
    payloads = {
        "routed_policy_qualification_v2.yaml": policy,
        "routed_policy_selector_off_qualification_v2.yaml": selector_off,
        "secondary_baselines_qualification_v2.yaml": secondary,
    }
    hashes = {
        name: hashlib.sha256(_yaml_bytes(payload)).hexdigest()
        for name, payload in payloads.items()
    }
    receipt = {
        "schema": "npi.g11.v6-qualification-suite-materialization-receipt.v2",
        "manifest_cell_count": manifest_cell_count,
        "clusters": clusters,
        "minimum_selector_profile_batch_work": minimum_profile_batch_work,
        "selector_off_profile_cap": float(
            selector_off["router"]["maximum_hybrid_profile_work"]
        ),
        "config_sha256": hashes,
        "proposal_bank_receipt": bank_receipt,
    }
    return payloads, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--proposal-training-source", type=Path, required=True)
    parser.add_argument("--manifest-cell-count", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    payloads, receipt = materialize_qualification_suite(
        arguments.template,
        arguments.proposal_training_source,
        manifest_cell_count=arguments.manifest_cell_count,
        clusters=arguments.clusters,
    )
    targets = {
        name: arguments.output_directory / name for name in payloads
    }
    targets["receipt"] = (
        arguments.output_directory / "qualification_suite_receipt.json"
    )
    if any(path.exists() for path in targets.values()):
        raise FileExistsError(
            "qualification-suite materialization refuses to overwrite outputs"
        )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        targets[name].write_bytes(_yaml_bytes(payload))
    targets["receipt"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
