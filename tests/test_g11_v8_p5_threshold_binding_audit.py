from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.g11_v8_p5_threshold_binding_audit import audit_binding, load_binding

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "configs" / "g11_v8" / "p5_threshold_manifest_binding_v1.yaml"


def _canonical() -> tuple[dict[str, Any], str]:
    raw = BINDING.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value, hashlib.sha256(raw).hexdigest()


def test_canonical_p5_threshold_binding_passes() -> None:
    binding, digest = _canonical()
    report = audit_binding(binding, digest)
    assert report["passed"]
    assert len(report["checks"]) >= 18


@pytest.mark.parametrize(
    ("mutate", "check"),
    [
        (lambda x: x.__setitem__("outcome_data_used", True), "outcome_blind"),
        (
            lambda x: x.__setitem__("threshold_manifest_sha256", "0" * 64),
            "manifest_hash_bound",
        ),
        (
            lambda x: x.__setitem__("reference_seed_namespace", "p5-final-method"),
            "reference_and_final_namespaces_fresh",
        ),
        (
            lambda x: x["decision"].__setitem__("performance_claim_authorized", True),
            "performance_refused",
        ),
    ],
)
def test_p5_threshold_binding_corruption_fails_closed(mutate, check: str) -> None:
    binding, digest = _canonical()
    changed = copy.deepcopy(binding)
    mutate(changed)
    report = audit_binding(changed, digest)
    assert not report["passed"]
    assert check in report["failures"]


def test_binding_loader_rejects_wrong_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_binding(bad)
