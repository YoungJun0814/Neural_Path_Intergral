"""Cross-phase V7 seed-audit tests."""

from __future__ import annotations

from experiments.g11_v7_phase_seed_audit import audit_payloads


def _payload(phase: str, protocol: str, seeds: list[int]) -> dict:
    return {
        "phase": phase,
        "protocol_id": protocol,
        "dirty_worktree": False,
        "seed_ledger": {"records": [{"seed": seed} for seed in seeds]},
    }


def test_v7_phase_seed_audit_accepts_disjoint_namespaces() -> None:
    counts, intersections, failures = audit_payloads(
        {
            "qualification_probe": _payload("qualification", "qp", [1, 2]),
            "qualification_fixed": _payload("qualification", "qf", [3, 4]),
            "confirmation_probe": _payload("confirmation", "cp", [5, 6]),
            "confirmation_fixed": _payload("confirmation", "cf", [7, 8]),
        }
    )
    assert counts == {
        "qualification_probe": 2,
        "qualification_fixed": 2,
        "confirmation_probe": 2,
        "confirmation_fixed": 2,
    }
    assert set(intersections.values()) == {0}
    assert not failures


def test_v7_phase_seed_audit_rejects_cross_phase_overlap() -> None:
    _, intersections, failures = audit_payloads(
        {
            "qualification_probe": _payload("qualification", "qp", [1, 2]),
            "qualification_fixed": _payload("qualification", "qf", [3, 4]),
            "confirmation_probe": _payload("confirmation", "cp", [2, 5]),
            "confirmation_fixed": _payload("confirmation", "cf", [6, 7]),
        }
    )
    assert intersections["confirmation_probe__qualification_probe"] == 1
    assert any("intersection is nonempty" in failure for failure in failures)
