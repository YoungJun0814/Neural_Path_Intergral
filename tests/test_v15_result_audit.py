import hashlib
from pathlib import Path

import yaml

from src.path_integral.v15_result_audit import audit_v15_result


def test_result_audit_is_fail_closed_on_open_theory_gate(tmp_path: Path) -> None:
    root = tmp_path
    (root / "configs/g11_v15").mkdir(parents=True)
    config = root / "config.yaml"
    config.write_text("stage: development\n", encoding="utf-8")
    ledger = {
        "gates": {"G5": {"pass": False}},
    }
    (root / "configs/g11_v15/theorem_ledger_v1.yaml").write_text(
        yaml.safe_dump(ledger),
        encoding="utf-8",
    )
    methods = []
    for method in (
        "conditional_rbergomi",
        "v14_local_volterra",
        "defensive_cem",
        "ld_subspace_is",
        "smoothing_rqmc",
    ):
        methods.append(
            {
                "method": method,
                "estimate": 0.01,
                "sample_variance": 0.01,
                "accuracy_z": 0.0,
            }
        )
    methods.append(
        {
            "method": "v15_cm_transport",
            "estimate": 0.01,
            "sample_variance": 0.005,
            "accuracy_z": 0.0,
            "exactness": {
                "maximum_likelihood_bound_violation": 0.0,
                "proposal_hash_unchanged": True,
            },
        }
    )
    payload = {
        "schema": "npi.g11.v15-experiment-result.v1",
        "config_binding": {
            "path": "config.yaml",
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        },
        "source_provenance": {
            "git_commit": "a" * 40,
            "source_dirty_before_run": False,
            "source_changes_before_run": [],
        },
        "used_seeds": [1, 2, 3],
        "gates": {"maximum_accuracy_z": 4.0, "minimum_work_ratio": 1.0},
        "cells": [
            {
                "cell_id": "toy",
                "methods": methods,
                "best_primary_over_v15_work_ratio": 2.0,
            }
        ],
    }
    audit = audit_v15_result(payload, root=root)
    assert audit.passed_integrity
    assert audit.passed_numerical
    assert not audit.passed_theory
    assert not audit.passed_top_journal_gate


def test_result_audit_does_not_confuse_theory_pass_with_submission_readiness(
    tmp_path: Path,
) -> None:
    root = tmp_path
    (root / "configs/g11_v15").mkdir(parents=True)
    config = root / "config.yaml"
    config.write_text("stage: qualification\n", encoding="utf-8")
    (root / "configs/g11_v15/theorem_ledger_v1.yaml").write_text(
        yaml.safe_dump({"gates": {"G5": {"pass": True}}}),
        encoding="utf-8",
    )
    (root / "configs/g11_v15/claim_contract_v1.yaml").write_text(
        yaml.safe_dump(
            {
                "external_novelty_review": {
                    "required": 2,
                    "completed": 0,
                    "submission_lock": True,
                },
                "gates": {"p8_baselines": "pending", "qualification": "locked"},
            }
        ),
        encoding="utf-8",
    )
    methods = [
        {
            "method": method,
            "estimate": 0.01,
            "sample_variance": 0.01,
            "accuracy_z": 0.0,
        }
        for method in (
            "conditional_rbergomi",
            "v14_local_volterra",
            "defensive_cem",
            "ld_subspace_is",
            "smoothing_rqmc",
        )
    ]
    methods.append(
        {
            "method": "v15_cm_transport",
            "estimate": 0.01,
            "sample_variance": 0.005,
            "accuracy_z": 0.0,
            "exactness": {
                "maximum_likelihood_bound_violation": 0.0,
                "proposal_hash_unchanged": True,
            },
        }
    )
    payload = {
        "schema": "npi.g11.v15-experiment-result.v1",
        "config_binding": {
            "path": "config.yaml",
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        },
        "source_provenance": {
            "git_commit": "a" * 40,
            "source_dirty_before_run": False,
            "source_changes_before_run": [],
        },
        "used_seeds": [1, 2, 3],
        "gates": {"maximum_accuracy_z": 4.0, "minimum_work_ratio": 1.0},
        "cells": [
            {
                "cell_id": "toy",
                "methods": methods,
                "best_primary_over_v15_work_ratio": 2.0,
            }
        ],
    }
    audit = audit_v15_result(payload, root=root)
    assert audit.passed_theory
    assert audit.passed_numerical
    assert not audit.passed_top_journal_gate
    assert "top-journal submission locks remain open" in audit.failures
