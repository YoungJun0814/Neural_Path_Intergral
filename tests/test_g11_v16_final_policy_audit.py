from __future__ import annotations

import yaml

from experiments.g11_v16_final_policy_audit import ROOT, build_audit


def test_v16_final_policy_audit_separates_dominance_and_fallback_claims() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/g11_v15/v16_final_policy_audit_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    audit = build_audit(config)
    assert audit["passed"] is False
    assert audit["finite_grid_empirical_claim_authorized"] is False
    assert audit["uniform_ood_dominance_authorized"] is False
    assert audit["top_journal_submission_authorized"] is False
    assert len(audit["dominance_cells"]) == 3
    assert len(audit["correctness_fallback_cells"]) == 3
    assert len(audit["submission_blockers"]) == 3
    assert audit["post_audit_legacy_review"]["semantic_validity"] == "pass"
    assert audit["post_audit_legacy_review"]["statistical_evidence"] == "unresolved"
