import math

import pytest
import torch

from experiments.post_audit_analytic_residual_batch import Budget
from experiments.post_audit_conditional_marginal_g import decide, evaluate_bank, seed_layout
from src.path_integral.conditional_marginal_g import blend
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_protocol import canonical_json_bytes
from src.path_integral.seed_ledger import SeedKey, SeedLedger


def test_complete_gate_stop_forbids_any_final_artifact():
    training = [{"summary": {"gate": {"pass": False}}}]*4
    references = [{"qualified": False}]*2
    output = decide(training, references, [])
    assert output["decision"] == "STOP_MARGINAL_TRAINING_BANK_UNRELIABLE"
    assert not output["P2_qualified"] and not output["large_training_authorized"]
    assert canonical_json_bytes(output)
    with pytest.raises(ValueError, match="forbids"):
        decide(training, references, [{}])


@pytest.mark.parametrize("failure", ["none", "reference", "precision", "alpha", "numerics", "effect"])
def test_candidate_requires_all_scientific_gates_and_never_grants_P2(failure):
    training = [{"summary": {"gate": {"pass": True}}}]*4
    references = [{"qualified": failure != "reference"}]*2
    final = {"baseline": {"qualified": failure != "precision"}, "selected": {"qualified": True},
        "reference_consistency": True, "selection_numerics_pass": failure != "numerics",
        "alpha": 0 if failure == "alpha" else .25, "block_variance_ratio_proxy": 1. if failure == "effect" else .5}
    output = decide(training, references, [final]*4)
    assert (output["decision"] == "CANDIDATE_FOR_SEPARATE_CONFIRMATION_ONLY") == (failure == "none")
    assert not output["new_G_adopted"] and not output["new_event_q_adopted"] and not output["P2_qualified"]


def test_seed_manifest_has_all_unused_conditional_branches_and_is_repeatable():
    old = SeedLedger()
    old.allocate(SeedKey("old-study", "test", "development", "a", 32, 0, "path"))
    prior = {"seeds": {"ledger": old.to_dict()}}
    mapping, receipt = seed_layout(["a", "b"], prior)
    assert len(mapping) == 132 and len(receipt["ledger"]["records"]) == 264
    assert (mapping, receipt) == seed_layout(["a", "b"], prior)
    assert len({s for streams in mapping.values() for s in streams.values()}) == 264


def test_raw_payoff_is_never_evaluated_before_budget_approval():
    q = GaussianMixtureShiftSpec(torch.zeros((1, 64), dtype=torch.float64), torch.ones(1, dtype=torch.float64))

    class NeverCalled:
        def log_probability(self, z):
            raise AssertionError("physical work was not approved")

    with pytest.raises(TimeoutError, match="physics cap"):
        evaluate_bank(NeverCalled(), q, blend(q, None, 0.), {"path": 791, "label": 792}, 4, False, Budget(1000, 0, 30))


def test_raw_reference_density_targets_the_same_original_q_M2():
    q = GaussianMixtureShiftSpec(torch.zeros((1, 64), dtype=torch.float64), torch.ones(1, dtype=torch.float64))

    class Constant:
        def log_probability(self, z):
            return torch.full((len(z),), math.log(.2), dtype=torch.float64)

    budget = Budget(1000, 4, 30)
    result = evaluate_bank(Constant(), q, blend(q, None, 0.), {"path": 793, "label": 794}, 4, False, budget)
    assert budget.physical == 4 and budget.scalar == 0
    assert all(r["log_score"] == pytest.approx(math.log(.04)) for r in result["rows"])
    assert canonical_json_bytes(result)
