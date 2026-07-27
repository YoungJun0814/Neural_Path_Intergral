from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.g11_v8_p6_statistical_design_audit import (
    audit_design,
    load_design,
    required_one_sided_clusters,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs" / "g11_v8" / "p6_statistical_design_v1.yaml"


def _canonical() -> tuple[dict[str, Any], str]:
    raw = DESIGN.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value, hashlib.sha256(raw).hexdigest()


def test_canonical_p6_design_passes_and_is_powered_only_by_assumptions() -> None:
    design, digest = _canonical()
    report = audit_design(design, digest)
    assert report["passed"]
    assert report["efficiency_per_endpoint_alpha"] == pytest.approx(0.005)
    assert report["accuracy_per_claim_alpha"] == pytest.approx(0.025 / 192.0)
    assert max(item["required_clusters"] for item in report["power_forecasts"]) <= 32
    assert design["decision"]["actual_power_estimated"] is False


@pytest.mark.parametrize(
    ("mutate", "check"),
    [
        (lambda x: x.__setitem__("outcome_data_used", True), "outcome_blind"),
        (
            lambda x: x["inference"].__setitem__("path_level_pseudoreplication_allowed", True),
            "path_pseudoreplication_refused",
        ),
        (lambda x: x["primary_methods"]["comparators"].pop(), "primary_comparators_exact"),
        (lambda x: x["efficiency_family"].__setitem__("familywise_alpha", 0.05), "efficiency_alpha_exact"),
        (
            lambda x: x["accuracy_family"].__setitem__("expected_claim_count", 96),
            "accuracy_claim_shape_exact",
        ),
        (
            lambda x: x["seed_namespaces"].__setitem__("p10_confirmation", "p5-reference"),
            "p5_p6_seed_sets_disjoint",
        ),
        (
            lambda x: x["failure_rules"].__setitem__("incomplete_record_deletion_allowed", True),
            "no_record_deletion",
        ),
        (
            lambda x: x["decision"].__setitem__("performance_claim_authorized", True),
            "performance_refused",
        ),
    ],
)
def test_p6_corruption_fails_closed(mutate, check: str) -> None:
    design, digest = _canonical()
    changed = copy.deepcopy(design)
    mutate(changed)
    report = audit_design(changed, digest)
    assert not report["passed"]
    assert check in report["failures"]


def test_one_sided_power_requires_a_strict_effect_above_the_gate() -> None:
    required = required_one_sided_clusters(
        true_ratio=1.55,
        minimum_lower_ratio=1.2,
        cluster_log_sd=0.35,
        alpha=0.005,
        power=0.80,
    )
    assert 2 <= required <= 32
    with pytest.raises(ValueError, match="must exceed"):
        required_one_sided_clusters(
            true_ratio=1.2,
            minimum_lower_ratio=1.2,
            cluster_log_sd=0.35,
            alpha=0.005,
            power=0.80,
        )


def test_loader_rejects_wrong_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_design(bad)
