from __future__ import annotations

from src.path_integral.ecrpt_protocol import aggregate_structured_ecrpt_development


def test_v13_mechanism_uses_exact_gap_not_noisy_sample_variance_order() -> None:
    config = {
        "cells": [{"cell_id": "cell"}],
        "clusters": 2,
        "primary_comparators": ["baseline"],
        "gate": {
            "confidence_level": 0.95,
            "maximum_exactness_error": 1e-10,
            "maximum_combined_reference_z": 4.0,
            "maximum_paired_difference_z": 4.0,
            "maximum_likelihood_normalization_z": 6.0,
            "minimum_raw_over_ecrpt_variance_ratio": 1.0,
            "minimum_best_primary_over_ecrpt_geometric_ratio": 0.1,
            "minimum_one_sided_lower_ratio": 0.1,
            "minimum_cells_favoring_ecrpt": 0,
        },
    }
    records = []
    for cluster in range(2):
        forecast = {"resource_censored": False}
        records.append(
            {
                "cell_id": "cell",
                "cluster": cluster,
                "candidate": {
                    "exactness": {"error": 0.0},
                    "paired_difference_z": 0.0,
                    "likelihood_normalization_z": 0.0,
                    "combined_reference_z": 0.0,
                    "raw_over_ecrpt_variance_ratio": 0.5,
                    "rao_blackwell_gap_estimate": 0.1,
                    "rao_blackwell_gap_minimum": 0.0,
                    "plugin_work_to_target": {"total_work": 1.0},
                    "tail_safe_forecast": forecast,
                },
                "comparators": {
                    "baseline": {
                        "combined_reference_z": 0.0,
                        "plugin_work_to_target": {"total_work": 2.0},
                        "tail_safe_forecast": forecast,
                    }
                },
            }
        )
    aggregate = aggregate_structured_ecrpt_development(config=config, records=records)
    assert aggregate["mechanism_pass"]
    assert aggregate["correctness_pass"]
    assert "paired_variance" not in aggregate["blockers"]
