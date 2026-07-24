"""Independent serialized-artifact audit tests."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from experiments.g11_v6_result_audit import (
    _audit_crude_design_certificate,
    _audit_defensive_design_certificate,
    _audit_record,
    _audit_router,
    _audit_v6_baseline_summary,
    _load_config,
    run,
)
from src.path_integral import (
    HybridTarget,
    LevelBatch,
    SingleTermDesign,
    V6WorkLedger,
    V6WorkRecord,
    execute_v6_policy,
    prepare_v6_direct_policy,
    update_profile_intervals,
    v6_policy_preparation_to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "result_audit_development.yaml"


class _Sampler:
    def __call__(self, profile_id, role, count, seeds):
        del profile_id, role
        generator = torch.Generator().manual_seed(seeds["proposal"])
        values = (torch.rand(count, generator=generator) < 0.1).to(torch.float64)
        return LevelBatch(values, float(count), wall_seconds=0.001)


def _work(category: str) -> V6WorkRecord:
    return V6WorkRecord(
        category=category,
        method="pure_cem",
        cell_id="audit-cell",
        attempt=0,
        samples=32,
        work_units=32.0,
        wall_seconds=0.001,
        cpu_seconds=0.001,
        peak_memory_bytes=0,
        successful=True,
    )


def _record() -> dict[str, object]:
    prepared = prepare_v6_direct_policy(
        HybridTarget("audit-target", 0.1, 0.5),
        SingleTermDesign("cem", 64, 0.1, 0.02, 0.025, 1.0, None),
        policy_name="pure_cem",
        cell_id="audit-cell",
        execution_method="pure_cem",
        protocol="g11-v6-independent-audit-test",
        regime="gaussian",
        task="digital",
        operation_work_cap=1e9,
        preprocessing_work=V6WorkLedger(
            (_work("proposal_training"), _work("allocation_pilot"))
        ),
        minimum_final_samples=32,
        streams=("proposal",),
    )
    result = execute_v6_policy(prepared, _Sampler(), final_peak_memory_bytes=0)
    return {
        "cell_id": "audit-cell",
        "cluster": 0,
        "method": "pure_cem",
        "preparation": v6_policy_preparation_to_dict(prepared),
        "result": asdict(result),
    }


def test_v6_independent_audit_config_is_strict() -> None:
    config, digest = _load_config(CONFIG)
    assert "npi.g11.v6-routed-policy.v1" in config["accepted_source_schemas"]
    assert len(digest) == 64


def test_v6_router_audit_rejects_unordered_work_interval() -> None:
    record = {
        "route": {},
        "router_inputs": {
            "config": {
                "probability_cutoff": 0.05,
                "confidence_level": 0.99,
                "initial_screening_trials": 256,
                "maximum_screening_trials": 1024,
                "minimum_certified_relative_saving": 0.10,
                "maximum_hybrid_profile_work": 5.0e8,
                "maximum_profile_fraction": 0.40,
                "ambiguous_fallback": "dcs_slis",
            },
            "successes": 2,
            "trials": 256,
            "screening_work": 256.0,
            "crude_work": {
                "method": "crude",
                "lower": 2.0,
                "point": 1.0,
                "upper": 3.0,
            },
            "dcs_work": None,
            "hybrid_opportunity": None,
        },
    }
    assert not _audit_router(record)


def test_v6_baseline_summary_defers_random_per_record_misses() -> None:
    records = []
    for method in ("crude", "pure_cem", "defensive_cem"):
        record = {
            "cell_id": "cell-a",
            "cluster": 0,
            "method": method,
            "cem_fit": (
                None
                if method == "crude"
                else {"converged": True, "control": [[1.0, -1.0]]}
            ),
            "crude_design_certificate": (
                {"certified": True} if method == "crude" else None
            ),
            "defensive_design_certificate": (
                {"certified": True} if method == "defensive_cem" else None
            ),
            "result": {
                "core": {
                    "complete": True,
                    "resource_censored": False,
                    "design_target_attained": True,
                    "empirical_target_attained": method != "pure_cem",
                    "seed_ledger_payload": {
                        "records": [{"key": {"role": "final"}}]
                    },
                },
                "total_work": {
                    "records": [
                        {
                            "category": (
                                "final" if method == "crude" else "proposal_training"
                            )
                        }
                    ]
                },
            },
        }
        records.append(record)
    operational_names = [
        "complete_matrix",
        "all_runs_complete",
        "no_resource_censoring",
        "all_design_targets_attained",
        "all_cem_training_charged",
        "all_cem_fits_converged",
        "all_cem_controls_finite_and_bounded",
        "all_defensive_designs_certified",
        "all_crude_designs_certified",
        "all_final_seed_roles_separate",
    ]
    qualification_gates = {name: True for name in operational_names}
    gates = {
        "complete_matrix": True,
        "all_runs_complete": True,
        "no_resource_censoring": True,
        "all_design_targets_attained": True,
        "all_empirical_targets_attained": False,
        "all_cem_training_charged": True,
        "all_cem_fits_converged": True,
        "all_cem_controls_finite_and_bounded": True,
        "all_defensive_designs_certified": True,
        "all_crude_designs_certified": True,
        "all_final_seed_roles_separate": True,
    }
    source = {
        "protocol_id": "g11-v6-baseline-qualification-v6",
        "methods": ["crude", "pure_cem", "defensive_cem"],
        "records": records,
        "gates": gates,
        "qualification_gates": qualification_gates,
        "qualification_contract": {
            "schema": "npi.g11.v6-baseline-qualification-contract.v1",
            "expected_cell_ids": ["cell-a"],
            "expected_clusters": 1,
            "methods": ["crude", "pure_cem", "defensive_cem"],
            "control_bound": 20.0,
            "operational_gate_names": operational_names,
            "per_record_empirical_target_role": (
                "deferred_to_prespecified_method_cell_attainment_and_bootstrap_rmse_co_gates"
            ),
            "aggregate_accuracy_protocol_id": (
                "g11-v6-power-analysis-qualification-v1"
            ),
        },
        "formal_readiness": {
            "frozen_config": True,
            "frozen_manifest": True,
            "clean_source": True,
            "non_smoke": True,
        },
        "baseline_qualified": True,
    }
    assert _audit_v6_baseline_summary(source)

    tampered = copy.deepcopy(source)
    tampered["qualification_gates"]["all_runs_complete"] = False
    assert not _audit_v6_baseline_summary(tampered)


def test_v6_auditor_replays_defensive_rarity_band_certificate() -> None:
    values = torch.zeros(4096, dtype=torch.float64)
    values[:4] = 1.0
    profile = update_profile_intervals(
        {"defensive": values},
        absolute_bounds={"defensive": 5.0},
        costs_per_sample={"defensive": 1.0},
        familywise_alpha=0.05,
        total_predeclared_looks=1,
    )[0]
    probability_upper = 2.0e-3
    structural_upper = 5.0 * probability_upper
    selected = max(
        profile.moments.sample_variance,
        min(profile.moments.variance_interval[1], structural_upper),
    )
    record = {
        "method": "defensive_cem",
        "nominal_probability": 1.0e-3,
        "reference_probability": 1.0e-3,
        "reference_standard_error": 1.0e-6,
        "pilot_tail_diagnostics": {
            "count": 4096,
            "mean": profile.moments.sample_mean,
            "variance": float(torch.var(values, unbiased=True)),
        },
        "design": {"design_variance": selected},
        "defensive_design_certificate": {
            "schema": "npi.g11.v6-defensive-design-certificate.v1",
            "nominal_probability": 1.0e-3,
            "nominal_probability_upper_multiplier": 2.0,
            "probability_upper_bound": probability_upper,
            "reference_certificate_z": 4.0,
            "reference_upper_bound": 1.004e-3,
            "certified": True,
            "absolute_bound": 5.0,
            "familywise_alpha": 0.05,
            "pilot_count": profile.moments.sample_count,
            "pilot_mean": profile.moments.sample_mean,
            "pilot_variance": profile.moments.sample_variance,
            "rigorous_bounded_variance_upper": (
                profile.moments.variance_interval[1]
            ),
            "structural_variance_upper": structural_upper,
            "selected_design_variance": selected,
        },
    }
    assert _audit_defensive_design_certificate(
        record, relative=1e-13, absolute=1e-12, required=True
    )
    tampered = copy.deepcopy(record)
    tampered["defensive_design_certificate"]["structural_variance_upper"] *= 0.5
    assert not _audit_defensive_design_certificate(
        tampered, relative=1e-13, absolute=1e-12, required=True
    )

    plugin = copy.deepcopy(record)
    unbiased_variance = float(torch.var(values, unbiased=True))
    plugin_selected = max(5.0 * unbiased_variance, 1.0e-6)
    plugin["design"]["design_variance"] = plugin_selected
    plugin["defensive_design_certificate"] = {
        "schema": "npi.g11.v6-defensive-plugin-design-certificate.v1",
        "nominal_probability": 1.0e-3,
        "nominal_probability_upper_multiplier": 2.0,
        "probability_upper_bound": probability_upper,
        "reference_certificate_z": 4.0,
        "reference_upper_bound": 1.004e-3,
        "certified": True,
        "absolute_bound": 5.0,
        "pilot_count": profile.moments.sample_count,
        "pilot_mean": profile.moments.sample_mean,
        "pilot_variance": unbiased_variance,
        "variance_safety_factor": 5.0,
        "zero_variance_fallback": 1.0e-6,
        "structural_variance_upper_diagnostic": structural_upper,
        "selected_design_variance": plugin_selected,
    }
    assert _audit_defensive_design_certificate(
        plugin, relative=1e-13, absolute=1e-12, required=True
    )
    plugin["defensive_design_certificate"]["variance_safety_factor"] = 4.0
    assert not _audit_defensive_design_certificate(
        plugin, relative=1e-13, absolute=1e-12, required=True
    )


def test_v6_auditor_replays_crude_rarity_band_certificate() -> None:
    count = 4096
    mean = 1.0 / count
    variance = mean * (1.0 - mean) * count / (count - 1)
    probability_upper = 2.0e-3
    structural_upper = probability_upper * (1.0 - probability_upper)
    record = {
        "method": "crude",
        "nominal_probability": 1.0e-3,
        "reference_probability": 1.0e-3,
        "reference_standard_error": 1.0e-6,
        "pilot_tail_diagnostics": {
            "count": count,
            "mean": mean,
            "variance": variance,
        },
        "design": {"design_variance": structural_upper},
        "crude_design_certificate": {
            "schema": "npi.g11.v6-crude-design-certificate.v1",
            "nominal_probability": 1.0e-3,
            "nominal_probability_upper_multiplier": 2.0,
            "probability_upper_bound": probability_upper,
            "reference_certificate_z": 4.0,
            "reference_upper_bound": 1.004e-3,
            "certified": True,
            "pilot_count": count,
            "pilot_mean": mean,
            "pilot_variance": variance,
            "structural_variance_upper": structural_upper,
            "selected_design_variance": structural_upper,
        },
    }
    assert _audit_crude_design_certificate(
        record, relative=1e-13, absolute=1e-12, required=True
    )
    tampered = copy.deepcopy(record)
    tampered["crude_design_certificate"]["reference_upper_bound"] += 1.0e-4
    assert not _audit_crude_design_certificate(
        tampered, relative=1e-13, absolute=1e-12, required=True
    )


def test_offline_auditor_recomputes_valid_record_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    record = _record()
    valid = _audit_record(record, relative=1e-13, absolute=1e-12)
    assert valid["passed"]

    allocation_tamper = copy.deepcopy(record)
    allocation_tamper["preparation"]["core"]["allocations"][0]["final_count"] += 1
    assert not _audit_record(
        allocation_tamper, relative=1e-13, absolute=1e-12
    )["passed"]

    work_tamper = copy.deepcopy(record)
    work_tamper["result"]["total_work"]["records"][-1]["work_units"] += 1.0
    assert not _audit_record(work_tamper, relative=1e-13, absolute=1e-12)["passed"]

    result_tamper = copy.deepcopy(record)
    result_tamper["result"]["core"]["estimate"] += 0.01
    assert not _audit_record(result_tamper, relative=1e-13, absolute=1e-12)["passed"]

    source = {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "smoke": True,
        "records": [record],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    artifact = run(CONFIG, source_path)
    assert artifact["gates"]["all_records_pass"]
    assert not artifact["gates"]["non_smoke_if_required"]
    assert not artifact["qualification_audit_passed"]

    seed_tamper = copy.deepcopy(record)
    seed_tamper["result"]["core"]["seed_ledger_payload"]["records"][-1]["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        _audit_record(seed_tamper, relative=1e-13, absolute=1e-12)
