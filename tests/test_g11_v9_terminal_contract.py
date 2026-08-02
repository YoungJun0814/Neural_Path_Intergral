from __future__ import annotations

from pathlib import Path

from src.path_integral.v9_terminal_contract import audit_v9_contract

ROOT = Path(__file__).resolve().parents[1]


def test_v9_terminal_contract_is_semantically_closed() -> None:
    audit = audit_v9_contract(ROOT / "configs/g11_v9/terminal_claim_contract_v1.yaml")
    assert audit.passed, audit.failures
