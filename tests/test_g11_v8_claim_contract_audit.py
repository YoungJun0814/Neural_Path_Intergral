from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_claim_contract_audit import (
    EXPECTED_SCHEMA,
    REPORT_SCHEMA,
    _load_config,
    audit_contract,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g11_v8" / "top_journal_claim_contract_v1.yaml"


def _canonical() -> tuple[dict[str, object], str]:
    raw = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(config, dict)
    return config, hashlib.sha256(raw).hexdigest()


def test_canonical_contract_passes() -> None:
    config, digest = _canonical()
    report = audit_contract(config, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["config_sha256"] == digest
    assert report["failure_count"] == 0
    assert report["failures"] == []
    assert report["passed"] is True
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda config: config["estimand"].__setitem__(
                "continuous_monitoring", True
            ),
            "continuous_monitoring_not_claimed",
        ),
        (
            lambda config: config["estimand"].__setitem__("continuous_time", True),
            "estimand_keys_exact",
        ),
        (
            lambda config: config["paper"]["contributions"].append(
                "post_hoc_fourth_claim"
            ),
            "exactly_three_predeclared_contributions",
        ),
        (
            lambda config: config["comparators"]["roles"][
                "closest_published_method"
            ].__setitem__(
                "id", "crude_antithetic_mc"
            ),
            "primary_comparator_roles_exact",
        ),
        (
            lambda config: config["comparators"]["roles"]["adaptive_work"].__setitem__(
                "training_inclusive", False
            ),
            "primary_comparators_training_inclusive",
        ),
        (
            lambda config: config["comparators"].__setitem__(
                "outcome_selected_primary_comparator", True
            ),
            "comparator_selection_predeclared",
        ),
        (
            lambda config: config["flow_extension"].__setitem__(
                "residual_flow_dcs_requires_tractable_conditional_integral", False
            ),
            "residual_flow_requires_tractable_conditional_integral",
        ),
        (
            lambda config: config["statistics"]["provisional_gates"].__setitem__(
                "minimum_external_training_inclusive_work_ratio_lower", 1.0
            ),
            "gate_minimum_external_training_inclusive_work_ratio_lower",
        ),
    ],
)
def test_contract_corruption_fails_closed(mutation, failed_check: str) -> None:
    config, digest = _canonical()
    corrupted = copy.deepcopy(config)
    mutation(corrupted)

    report = audit_contract(corrupted, digest)

    assert report["passed"] is False
    assert failed_check in report["failures"]
    assert report["checks"][failed_check] is False


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema: npi.g11.unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected contract schema"):
        _load_config(path)


def test_cli_writes_auditable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_claim_contract_audit.py",
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(output),
        ],
    )

    main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["schema"] == REPORT_SCHEMA
    assert report["contract_schema"] == EXPECTED_SCHEMA
    assert report["passed"] is True


def test_cli_refuses_to_overwrite_existing_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_claim_contract_audit.py",
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()


def test_cli_writes_failure_report_before_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _canonical()
    config["estimand"]["continuous_monitoring"] = True
    corrupted_path = tmp_path / "corrupted.yaml"
    corrupted_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "failed-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_claim_contract_audit.py",
            "--config",
            str(corrupted_path),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "continuous_monitoring_not_claimed" in report["failures"]
