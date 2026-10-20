import math
from dataclasses import replace

import pytest
import torch

from src.path_integral.gaussian_span_marginalization import sample_gaussian_mixture
from src.path_integral.matched_window_diagnostics import (
    FEATURES,
    contribution_diagnostics,
    merge_contribution_diagnostics,
    paired_contributions_with_features,
    price_only_window_guide,
)
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.volterra_excursion_guide import volterra_monitoring_operator
from tests.test_p1_recovery_p2_parallel import problem
from tests.test_physical_window_guide import KWARGS


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("rho", [-.7, .7, 0.])
def test_matched_guide_only_changes_price_and_has_independent_dense_density_oracle(rho):
    p = replace(problem(32), rho=rho)
    a = PhysicalWindowGuide.create(p, kind="point", **KWARGS)
    b = price_only_window_guide(p, **KWARGS)
    assert b.centres == a.centres and b.template_kinds == a.template_kinds
    assert torch.equal(b.vol_directions, a.vol_directions) and torch.equal(b.feature_weights, a.feature_weights)
    assert torch.equal(b.spec.weights, a.spec.weights)
    dense = b.feature_weights[:, :-1] @ volterra_monitoring_operator(p)
    torch.testing.assert_close(b.vol_directions*b.feature_variances.sqrt()[:, None], dense, rtol=2e-12, atol=2e-12)
    assert b.price_window_steps == tuple(min(4, 32-c) for c in a.centres)
    torch.testing.assert_close(b.spec.means.square().sum(1), a.spec.means.square().sum(1))
    z = torch.randn((20, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(25001))
    full = torch.distributions.MultivariateNormal(torch.zeros(64, dtype=torch.float64), torch.eye(64, dtype=torch.float64))
    logs = torch.stack([full.log_prob(z-m) for m in b.spec.means], 1)
    torch.testing.assert_close(b.log_q_over_p(z), torch.logsumexp(logs+b.spec.weights.log(), 1)-full.log_prob(z), rtol=1e-12, atol=1e-12)


def test_label_sampler_features_and_paired_payoffs_do_not_change_estimator():
    p = problem(32)
    b = price_only_window_guide(p, **KWARGS)
    sample = sample_gaussian_mixture(b.spec, 256, gaussian_generator=torch.Generator().manual_seed(25002), label_generator=torch.Generator().manual_seed(25003))
    torch.testing.assert_close(sample.target_coordinates, b.sample(256, path_seed=25002, label_seed=25003), rtol=0, atol=0)
    eps = torch.randn((256, 16), dtype=torch.float64, generator=torch.Generator().manual_seed(25004))
    pair = PairedVolterraProbability.create(p)
    values, features = paired_contributions_with_features(pair, b, sample.target_coordinates, eps, sample.labels)
    torch.testing.assert_close(values, pair.log_components(torch.cat((sample.target_coordinates, eps), 1))-b.log_q_over_p(sample.target_coordinates)[:, None], rtol=1e-13, atol=1e-13)
    assert features.shape == (256, len(FEATURES))
    terminal, integrated, _ = pair.fine.quantities(sample.target_coordinates)
    torch.testing.assert_close(features[:, 1], (terminal-math.log(p.spot)+.5*integrated)/p.rho, rtol=1e-12, atol=1e-12)


def test_ledger_merges_without_using_component_density_or_modifying_raw_mean():
    labels = torch.tensor([0, 0, 1, 2, 1, 3])
    logs = torch.tensor([-1000., -999., -980., -995., -981., -999.], dtype=torch.float64)
    features = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    full = contribution_diagnostics(logs, labels, features, components=5)
    merged = merge_contribution_diagnostics([contribution_diagnostics(logs[s], labels[s], features[s], components=5) for s in (slice(0, 3), slice(3, 6))])
    assert merged["count"] == full["count"] and merged["label_counts"] == full["label_counts"]
    assert merged["label_log_sums"] == full["label_log_sums"]
    assert merged["top"] == full["top"]
    torch.testing.assert_close(torch.tensor(merged["observed_weighted_feature_means"]), torch.tensor(full["observed_weighted_feature_means"]))
    present = torch.tensor([x for x in merged["label_log_sums"] if x is not None], dtype=torch.float64)
    assert float(torch.logsumexp(present, 0)) == pytest.approx(float(torch.logsumexp(logs, 0)))
    assert merged["label_log_sums"][-1] is None
    with pytest.raises(ValueError):
        contribution_diagnostics(logs, labels+5, features, components=5)


def test_matched_price_windows_converge_in_brownian_embedding():
    for n in (32, 64):
        a = price_only_window_guide(problem(n), **KWARGS)
        b = price_only_window_guide(problem(2*n), **KWARGS)
        embed = a.price_directions[:, ::2].repeat_interleave(2, 1)/math.sqrt(2)
        torch.testing.assert_close(embed, b.price_directions[:, ::2], rtol=0, atol=2e-12)
