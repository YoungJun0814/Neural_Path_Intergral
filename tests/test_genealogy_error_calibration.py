import math

import pytest
import torch

from src.path_integral.genealogy_error_calibration import (
    TARGETS,
    KnownMixtureTarget,
    confusion,
    constructed_counterexamples,
    independent_reference,
    method_summary,
    rank_correlation,
    truth_errors,
    weighted_moments,
)
from src.path_integral.reference_protocol import canonical_json_bytes


@pytest.mark.parametrize("target", TARGETS)
def test_closed_form_density_normalizer_and_all_independent_scaled_moments(target):
    reference = independent_reference(target)
    assert reference["pass"] and reference["maximum_reference_disagreement"] < 1e-9
    assert reference["scalar_reference_evaluations"] > 0
    x = torch.linspace(-15, 15, 301, dtype=torch.float64)[:, None]
    value = target.evaluate(x)
    value.validate(len(x))
    assert bool((value.log_m2 <= 1e-12).all())
    means = torch.tensor(target.means, dtype=torch.float64)
    log_density = torch.logsumexp(torch.tensor(target.weights, dtype=torch.float64).log()
        -.5*((x-means)/target.sd).square()-math.log(target.sd)-.5*math.log(2*math.pi), 1)
    reconstructed = -.5*x[:, 0].square()-.5*math.log(2*math.pi)+value.log_m2-target.log_z
    torch.testing.assert_close(reconstructed, log_density, rtol=0, atol=1e-12)
    assert canonical_json_bytes(reference)


@pytest.mark.parametrize("target", TARGETS)
def test_constructed_counterexamples_do_not_claim_movement_or_sampling_independence(target):
    data = constructed_counterexamples(target)
    good = data["collapsed_lineage_good_moments"]
    bad = data["many_ancestors_missed_mode"]
    assert good["errors"]["target_accurate"] and not good["ancestor_gate"]
    assert bad["ancestor_gate"] and bad["weight_gate"] and not bad["errors"]["target_accurate"]
    assert good["diagnostics"]["weighted_ancestor_ess"] == pytest.approx(1.)
    assert bad["diagnostics"]["initial_ancestors"] == 128
    assert data["movement_trace_not_fabricated_or_full_gate_claimed"]


def test_exact_target_refresh_can_preserve_wrong_normalizer_and_good_moments():
    target = TARGETS[0]
    q = constructed_counterexamples(target)["collapsed_lineage_good_moments"]
    errors = truth_errors(q["moments"], target.log_z+math.log(2), target.truth())
    assert errors["target_accurate"] and not errors["normalizer_accurate"]
    assert errors["relative_normalizer_error"] == pytest.approx(1.)
    first = target.sample_exact(128, label_seed=13001, noise_seed=13002)
    second = target.sample_exact(128, label_seed=13001, noise_seed=13002)
    assert torch.equal(first, second)


def test_confusion_denominators_are_explicit_and_never_nan_or_implied_zero():
    c = confusion([True, True, False, False], [True, False, True, False])
    assert all(c[k] == 1 for k in ("accept_accurate", "accept_inaccurate", "reject_accurate", "reject_inaccurate"))
    assert c["inaccurate_among_accepted"] == c["accurate_among_rejected"] == .5
    assert confusion([False], [True])["inaccurate_among_accepted"] is None
    with pytest.raises(ValueError):
        confusion([], [])


def test_repeatable_but_wrong_normalizer_is_identified_without_relaxing_gate():
    truth = TARGETS[0].truth()
    moments = {k: truth[k] for k in ("mean", "second_moment", "right_mass")}
    logz = truth["log_normalizer"]+math.log(.5)
    rows = [{"log_normalizer": logz, "errors": truth_errors(moments, logz, truth)} for _ in range(32)]
    summary = method_summary(rows, truth)
    assert summary["rse"] == 0 and summary["known_relative_Z_error"] == pytest.approx(.5)
    assert all(g["qualified"] and g["repeatability_qualified_but_Z_error_over_20pct"] for g in summary["groups_of_four"])
    with pytest.raises(ValueError):
        method_summary(rows[:4], truth)


def test_spearman_is_descriptive_and_constant_case_is_null():
    assert rank_correlation([1., 2., 3.], [3., 2., 1.]) == pytest.approx(-1.)
    assert rank_correlation([1., 1.], [2., 3.]) is None
    with pytest.raises(ValueError):
        rank_correlation([1., math.nan], [2., 3.])


@pytest.mark.parametrize("sd,weights", [(0., (.5, .5)), (1., (.5, .5)), (.5, (.6, .5)), (.5, (-.2, 1.2))])
def test_invalid_known_target_is_not_silently_renormalized(sd, weights):
    with pytest.raises(ValueError):
        KnownMixtureTarget("bad", (-1., 1.), weights, sd)


def test_invalid_weighted_moment_bank_is_rejected():
    with pytest.raises(ValueError):
        weighted_moments(torch.zeros((4, 1), dtype=torch.float64), torch.ones(4, dtype=torch.float64))
