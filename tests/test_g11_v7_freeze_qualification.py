"""V7 qualification-freeze power tests."""

from __future__ import annotations

from experiments.g11_v7_freeze_qualification import _planned_clusters


def test_v7_power_count_increases_when_practical_margin_shrinks() -> None:
    easy = _planned_clusters(
        mean_log_ratio=1.2,
        standard_error=0.04,
        development_clusters=8,
        practical_ratio=1.5,
    )
    hard = _planned_clusters(
        mean_log_ratio=0.5,
        standard_error=0.04,
        development_clusters=8,
        practical_ratio=1.5,
    )
    assert easy >= 2
    assert hard > easy
