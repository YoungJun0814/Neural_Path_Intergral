from __future__ import annotations

import copy

from src.path_integral.v9_terminal_protocol import aggregate_terminal_benchmark


def _config() -> dict[str, object]:
    threshold = {
        "confidence": 0.95,
        "minimum_passing_hurst_groups": 1,
        "minimum_dcs_vs_raw_geometric_ratio": 1.1,
        "minimum_dcs_vs_raw_lower_bound": 1.0,
        "minimum_dcs_vs_best_geometric_ratio": 0.8,
        "minimum_dcs_vs_best_lower_bound": 0.67,
    }
    return {
        "stage": "development",
        "maximum_exactness_error": 1e-10,
        "maximum_likelihood_normalization_absolute_z": 6.0,
        "primary_query_count": 100,
        "gate": {
            "maximum_combined_reference_z": 4.0,
            "maximum_paired_difference_z": 4.0,
            "minimum_likelihood_normalization_pass_fraction": 0.95,
            "development": threshold,
            "qualification": threshold,
        },
    }


def _records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    paired: list[dict[str, object]] = []
    external: list[dict[str, object]] = []
    for cluster in range(3):
        for rarity in range(2):
            paired.append(
                {
                    "cell_id": f"cell-{rarity}",
                    "hurst": 0.1,
                    "cluster": cluster,
                    "exactness": {"error": 0.0},
                    "likelihood": {"normalization_z": 0.0},
                    "dcs": {"combined_reference_z": 0.1},
                    "mechanism": {"difference_z": 0.1},
                    "comparison": {
                        "100": {
                            "dcs_vs_raw_total_work_ratio": 1.3,
                            "dcs_vs_best_primary_total_work_ratio": 0.9,
                        }
                    },
                }
            )
            external.append(
                {
                    "lifecycle_audit_passed": True,
                    "proposal": {"exact_likelihood": True, "self_normalized": False},
                    "likelihood_diagnostics": {"nonfinite_weight_count": 0},
                    "estimate": {"combined_reference_z": 0.1},
                    "allocation": {"resource_censored": False},
                }
            )
    return paired, external


def test_v9_aggregate_passes_stable_regime_and_fails_resource_mutation() -> None:
    paired, external = _records()
    aggregate = aggregate_terminal_benchmark(
        config=_config(), paired_records=paired, external_records=external
    )
    assert aggregate["stage_pass"] is True
    assert aggregate["selected_hurst_groups"] == [0.1]
    mutated = copy.deepcopy(external)
    mutated[0]["allocation"]["resource_censored"] = True
    failed = aggregate_terminal_benchmark(
        config=_config(), paired_records=paired, external_records=mutated
    )
    assert failed["stage_pass"] is False
    assert "resource_censoring" in failed["blockers"]
