from __future__ import annotations

import copy

from src.path_integral.ecrpt_protocol import aggregate_ecrpt_microstudy


def _config() -> dict:
    return {
        "cells": [{"cell_id": "a"}, {"cell_id": "b"}, {"cell_id": "c"}],
        "clusters": 2,
        "primary_comparators": ["baseline"],
        "gate": {
            "confidence_level": 0.95,
            "maximum_exactness_error": 1e-10,
            "maximum_combined_reference_z": 4.0,
            "maximum_paired_difference_z": 4.0,
            "maximum_likelihood_normalization_z": 6.0,
            "minimum_raw_over_ecrpt_variance_ratio": 1.0,
            "minimum_best_primary_over_ecrpt_geometric_ratio": 1.5,
            "minimum_one_sided_lower_ratio": 1.0,
            "minimum_cells_favoring_ecrpt": 2,
        },
    }


def _records() -> list[dict]:
    records = []
    for cell in ("a", "b", "c"):
        for cluster in range(2):
            records.append(
                {
                    "cell_id": cell,
                    "cluster": cluster,
                    "candidate": {
                        "exactness": {"x": 1e-14},
                        "paired_difference_z": 0.5,
                        "likelihood_normalization_z": 0.2,
                        "combined_reference_z": 0.8,
                        "raw_over_ecrpt_variance_ratio": 2.0,
                        "tail_safe_forecast": {"resource_censored": False},
                        "plugin_work_to_target": {"total_work": 10.0},
                    },
                    "comparators": {
                        "baseline": {
                            "combined_reference_z": 0.7,
                            "tail_safe_forecast": {"resource_censored": False},
                            "plugin_work_to_target": {"total_work": 20.0},
                        }
                    },
                }
            )
    return records


def test_ecrpt_aggregate_passes_only_complete_uncensored_evidence() -> None:
    aggregate = aggregate_ecrpt_microstudy(config=_config(), records=_records())
    assert aggregate["correctness_pass"]
    assert aggregate["performance_pass"]
    assert aggregate["stage_pass"]


def test_ecrpt_aggregate_fails_closed_on_censoring_or_mutation() -> None:
    records = _records()
    records[0]["comparators"]["baseline"]["tail_safe_forecast"]["resource_censored"] = True
    aggregate = aggregate_ecrpt_microstudy(config=_config(), records=records)
    assert aggregate["correctness_pass"]
    assert not aggregate["performance_pass"]
    assert "tail_safe_uncensored" in aggregate["blockers"]

    mutated = copy.deepcopy(_records())
    mutated[0]["candidate"]["paired_difference_z"] = 10.0
    assert not aggregate_ecrpt_microstudy(config=_config(), records=mutated)[
        "correctness_pass"
    ]
