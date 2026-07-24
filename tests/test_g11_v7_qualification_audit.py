"""Independent V7 qualification-audit primitive tests."""

from __future__ import annotations

from experiments.g11_v7_qualification_audit import _quantile


def test_independent_audit_quantile_uses_frozen_linear_interpolation() -> None:
    assert _quantile([0.0, 10.0], 0.25) == 2.5
    assert _quantile([3.0, 1.0, 2.0], 0.5) == 2.0
