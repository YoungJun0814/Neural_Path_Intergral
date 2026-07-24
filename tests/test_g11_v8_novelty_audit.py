from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_novelty_audit import (
    EXPECTED_SCHEMA,
    REPORT_SCHEMA,
    _load_ledger,
    audit_novelty_ledger,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "configs" / "g11_v8" / "novelty_search_ledger_v1.yaml"
CLAIM_CONTRACT_PATH = (
    ROOT / "configs" / "g11_v8" / "top_journal_claim_contract_v1.yaml"
)
DECISION_PATH = (
    ROOT / "docs" / "audits" / "G11_V8_P1_NOVELTY_DECISION_2026-07-25.md"
)


def _canonical() -> tuple[dict[str, object], str]:
    raw = LEDGER_PATH.read_bytes()
    ledger = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(ledger, dict)
    return ledger, hashlib.sha256(raw).hexdigest()


def test_canonical_novelty_ledger_passes() -> None:
    ledger, digest = _canonical()
    report = audit_novelty_ledger(ledger, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["ledger_sha256"] == digest
    assert report["counts"]["families"] == 9
    assert report["counts"]["queries"] == 12
    assert report["counts"]["sources"] == 14
    assert report["failure_count"] == 0
    assert report["failures"] == []
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_decision_binds_canonical_ledger_hash() -> None:
    _, digest = _canonical()

    assert digest in DECISION_PATH.read_text(encoding="utf-8")


def test_p1_source_roles_match_p0_comparator_contract() -> None:
    ledger, _ = _canonical()
    claim_contract = yaml.safe_load(
        CLAIM_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    sources = {source["id"]: source for source in ledger["sources"]}

    assert (
        claim_contract["comparators"]["roles"]["closest_published_method"]["id"]
        == "numerical_smoothing_rqmc"
    )
    assert (
        sources["bayer_benhammouda_tempone_smoothing_qmc_2023"][
            "baseline_role"
        ]
        == "primary_closest_method"
    )
    mandatory = set(claim_contract["comparators"]["mandatory_secondary"])
    assert "large_deviation_adaptive_is" in mandatory
    assert "exact_likelihood_flow_is" in mandatory
    assert sources["tong_stadler_2022"]["baseline_role"] == "secondary_required"
    assert (
        sources["gao_zhang_daniel_boning_2023"]["baseline_role"]
        == "secondary_required"
    )


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda ledger: ledger.__setitem__("outcome_data_used", True),
            "outcome_blind",
        ),
        (
            lambda ledger: ledger["families"].pop(),
            "required_families_exact",
        ),
        (
            lambda ledger: ledger["sources"][0].__setitem__("source_primary", False),
            "all_sources_primary",
        ),
        (
            lambda ledger: ledger["sources"][1].__setitem__(
                "primary_url", ledger["sources"][0]["primary_url"]
            ),
            "primary_urls_unique",
        ),
        (
            lambda ledger: ledger["sources"][2].__setitem__(
                "baseline_role", "secondary"
            ),
            "closest_method_predeclared",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__(
                "candidate_contribution",
                "first conditional smoothing method under rough volatility",
            ),
            "candidate_contribution_narrow",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__(
                "external_expert_review_required", False
            ),
            "external_review_still_required",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__(
                "submission_novelty_authorized", True
            ),
            "submission_novelty_not_authorized",
        ),
        (
            lambda ledger: ledger["queries"][0].__setitem__(
                "executed_on", "2026-07-24"
            ),
            "queries_dated_and_described",
        ),
        (
            lambda ledger: ledger.__setitem__("post_hoc_claim", "shadow field"),
            "root_keys_exact",
        ),
    ],
)
def test_novelty_ledger_corruption_fails_closed(mutation, failed_check: str) -> None:
    ledger, digest = _canonical()
    corrupted = copy.deepcopy(ledger)
    mutation(corrupted)

    report = audit_novelty_ledger(corrupted, digest)

    assert report["passed"] is False
    assert failed_check in report["failures"]
    assert report["checks"][failed_check] is False


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema: npi.g11.unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected novelty schema"):
        _load_ledger(path)


def test_cli_writes_passing_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "novelty-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_novelty_audit.py",
            "--ledger",
            str(LEDGER_PATH),
            "--output",
            str(output),
        ],
    )

    main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["schema"] == REPORT_SCHEMA
    assert report["ledger_schema"] == EXPECTED_SCHEMA
    assert report["passed"] is True


def test_cli_writes_failure_before_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _ = _canonical()
    ledger["outcome_data_used"] = True
    corrupted_path = tmp_path / "corrupted.yaml"
    corrupted_path.write_text(
        yaml.safe_dump(ledger, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "failed-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_novelty_audit.py",
            "--ledger",
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
    assert "outcome_blind" in report["failures"]


def test_cli_refuses_to_overwrite_existing_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "novelty-audit.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_novelty_audit.py",
            "--ledger",
            str(LEDGER_PATH),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()
