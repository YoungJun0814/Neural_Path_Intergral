"""Independent V7 mechanism-audit arithmetic tests."""

from __future__ import annotations

import math

from experiments.g11_v7_mechanism_probe_audit import _record_failures


def test_independent_auditor_recomputes_paired_moment_identities() -> None:
    count = 100
    dcs_variance = 2.0
    residual_variance = 3.0
    covariance = 0.25
    raw_variance = dcs_variance + residual_variance + 2.0 * covariance
    product_variance = 4.0
    covariance_se = math.sqrt(product_variance / count) * count / (count - 1)
    record = {
        "cell_id": "cell",
        "reference_probability": 1.0,
        "reference_standard_error": 0.1,
        "residual_z_score": 0.5 / math.sqrt(residual_variance / count),
        "raw_reference_z_score": 0.5
        / math.hypot(math.sqrt(raw_variance / count), 0.1),
        "dcs_reference_z_score": 0.0,
        "diagnostics": {
            "count": count,
            "raw_mean": 1.5,
            "dcs_mean": 1.0,
            "residual_mean": 0.5,
            "residual_standard_error": math.sqrt(residual_variance / count),
            "raw_variance": raw_variance,
            "dcs_variance": dcs_variance,
            "residual_variance": residual_variance,
            "dcs_residual_covariance": covariance,
            "dcs_residual_covariance_product_variance": product_variance,
            "dcs_residual_covariance_standard_error": covariance_se,
            "dcs_residual_covariance_z_score": covariance / covariance_se,
            "dcs_residual_correlation": covariance
            / math.sqrt(dcs_variance * residual_variance),
            "raw_over_dcs_variance_ratio": raw_variance / dcs_variance,
            "variance_decomposition_error": 0.0,
        },
        "work_record": {
            "category": "mechanism_probe",
            "method": "paired_raw_dcs",
            "cell_id": "cell",
            "samples": count,
            "successful": True,
        },
    }
    assert not _record_failures(record, expected_samples=count)


def test_independent_auditor_rejects_corrupted_variance_ratio() -> None:
    count = 10
    record = {
        "cell_id": "cell",
        "reference_probability": 0.0,
        "reference_standard_error": 0.1,
        "residual_z_score": 0.0,
        "raw_reference_z_score": 0.0,
        "dcs_reference_z_score": 0.0,
        "diagnostics": {
            "count": count,
            "raw_mean": 0.0,
            "dcs_mean": 0.0,
            "residual_mean": 0.0,
            "residual_standard_error": math.sqrt(0.1),
            "raw_variance": 2.0,
            "dcs_variance": 1.0,
            "residual_variance": 1.0,
            "dcs_residual_covariance": 0.0,
            "dcs_residual_covariance_product_variance": 1.0,
            "dcs_residual_covariance_standard_error": math.sqrt(0.1)
            * count
            / (count - 1),
            "dcs_residual_covariance_z_score": 0.0,
            "dcs_residual_correlation": 0.0,
            "raw_over_dcs_variance_ratio": 99.0,
            "variance_decomposition_error": 0.0,
        },
        "work_record": {
            "category": "mechanism_probe",
            "method": "paired_raw_dcs",
            "cell_id": "cell",
            "samples": count,
            "successful": True,
        },
    }
    failures = _record_failures(record, expected_samples=count)
    assert "variance ratio mismatch" in failures
