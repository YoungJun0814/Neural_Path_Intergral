from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.g11_v8_p5_matrix_audit import audit_design, load_design

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs" / "g11_v8" / "p5_reference_matrix_design_v1.yaml"


def _canonical() -> tuple[dict[str, Any], str]:
    raw = DESIGN.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value, hashlib.sha256(raw).hexdigest()


def test_canonical_p5_design_passes() -> None:
    design, digest = _canonical()
    report = audit_design(design, digest)
    assert report["passed"]
    assert len(report["checks"]) >= 24


@pytest.mark.parametrize(
    ("mutate", "check"),
    [
        (lambda x: x.__setitem__("outcome_data_used", True), "outcome_blind"),
        (lambda x: x["primary_tasks"].pop(), "primary_tasks_exact"),
        (lambda x: x["nominal_probabilities"].pop(), "primary_probabilities_exact"),
        (lambda x: x["reference_contract"].__setitem__("methods", ["raw_crosscheck"]), "reference_methods_exact"),
        (
            lambda x: x["baseline_matrix"].__setitem__(
                "final_seed_namespace", x["reference_contract"]["final_seed_namespace"]
            ),
            "seed_namespaces_unique",
        ),
        (lambda x: x["baseline_matrix"].__setitem__("fresh_training_per_task_and_cell", False), "fresh_training_required"),
        (lambda x: x["mesh_matrix"]["steps"].pop(), "mesh_steps_exact"),
        (lambda x: x["decision"].__setitem__("performance_claim_authorized", True), "performance_refused"),
    ],
)
def test_p5_corruption_fails_closed(mutate, check: str) -> None:
    design, digest = _canonical()
    changed = copy.deepcopy(design)
    mutate(changed)
    report = audit_design(changed, digest)
    assert not report["passed"]
    assert check in report["failures"]


def test_loader_rejects_wrong_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_design(bad)
