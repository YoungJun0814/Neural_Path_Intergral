from __future__ import annotations

from experiments.g11_v8_p5_dense_amplitude_audit import (
    audit_dense_amplitude,
)
from experiments.g11_v8_p5_dense_amplitude_proposal import ROOT

CONFIG = ROOT / "configs/g11_v8/p5_dense_amplitude_proposal_v1.yaml"
RESULT = (
    ROOT / "results/g11_v8_p5_dense_amplitude_proposal_v1_2026-07-31.json"
)


def test_dense_amplitude_partial_result_is_audited_and_fail_closed() -> None:
    report = audit_dense_amplitude(CONFIG, RESULT)

    assert report["passed"] is True
    assert report["decision"]["barrier_raw_development_selection_available"]
    assert report["decision"]["terminal_raw_redesign_required"]
    assert report["decision"]["proposal_manifest_freeze_authorized"] is False
