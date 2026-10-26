import json

import pytest
import torch

import experiments.post_audit_genealogy_error_calibration as study
from src.path_integral.genealogy_error_calibration import TARGETS
from src.path_integral.reference_protocol import canonical_json_bytes
from src.path_integral.seed_ledger import SeedKey, SeedLedger


def test_complete_stream_design_is_fresh_and_reproducible():
    old = SeedLedger()
    old.allocate(SeedKey("previous", "risk", "development", "cell", 1, 0, "draw"))
    mapping, ledger = study.seal([old.to_dict()])
    assert len(mapping) == len(study.jobs()) == 96
    assert len(ledger["ledger"]["records"]) == 96*11
    assert ledger["previous_seeds_compared"] == 1 and ledger["collision_count"] == 0
    assert study.seal([old.to_dict()]) == (mapping, ledger)
    assert canonical_json_bytes(ledger)


def test_separate_context_and_exact_draw_budgets_cannot_expand():
    b = study.Budget()
    b.charge(contexts=2400000, draws=12288)
    with pytest.raises(TimeoutError):
        b.charge(contexts=1)
    with pytest.raises(TimeoutError):
        b.charge(draws=1)
    assert b.contexts == 2400000 and b.draws == 12288


def test_full_replicate_refresh_preserves_history_and_raw_IS_has_equal_work():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        mapping, _ = study.seal([])
        job = study.jobs()[0]
        b = study.Budget()
        first = study.replicate(TARGETS[0], job, mapping[study.identity(job)], b)
        second = study.replicate(TARGETS[0], job, mapping[study.identity(job)], study.Budget())
        assert first == second
        assert first["contexts"] == 24832 and first["exact_target_draws"] == 128 and first["potential_calls"] == 98
        assert first["refresh"]["flags"] == first["smc"]["flags"]
        assert first["refresh"]["log_normalizer"] == first["smc"]["log_normalizer"]
        assert first["refresh"]["errors"]["relative_normalizer_error"] == first["smc"]["errors"]["relative_normalizer_error"]
        assert first["iid"]["raw_normalizer_not_self_normalized"]
        assert first["Volterra_physical"] == 0
        assert canonical_json_bytes(first)
    finally:
        torch.set_num_threads(old)


def test_partial_calibration_cannot_be_reported_as_complete():
    with pytest.raises(ValueError, match="all96"):
        study.summarize([])


def test_reference_suite_and_constructed_examples_are_json_safe():
    value = study.references()
    assert value["pass"] and len(value["references"]) == len(TARGETS)
    assert len(value["counterexamples"]) == len(TARGETS)
    assert canonical_json_bytes(value)
    assert json.loads(canonical_json_bytes(value)) == value


def test_complete_calibration_never_rewrites_production_gate_or_grants_P2():
    targets = {t.name: t for t in TARGETS}
    records = []
    for job in study.jobs():
        truth = targets[job["target"]].truth()
        method = {"log_normalizer": truth["log_normalizer"],
            "errors": {"target_accurate": True, "normalizer_accurate": True,
                "target_error_score": 0., "relative_normalizer_error": 0.},
            "flags": {"full": False, "ancestor_only": False, "weight_only": True}}
        records.append({"job": job, "smc": method, "iid": method, "refresh": method,
            "SMC_full_result": {"diagnostics": {"weighted_ancestor_ess": 1.}}})
    result = study.summarize(records)
    assert result["decision"] == "CALIBRATION_COMPLETE_NO_RETROACTIVE_ADOPTION"
    assert not result["production_gate_changed"] and not result["P2_qualified"]
    assert not result["event_q_changed"] and result["actual_G_fits"] == 0
    assert result["targets"][0]["diagnostic_confusion"]["smc"]["full"]["target_accurate"]["reject_accurate"] == 32
    assert canonical_json_bytes(result)
