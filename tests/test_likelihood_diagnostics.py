from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.likelihood import likelihood_diagnostics


def test_constant_likelihood_is_perfectly_normalized() -> None:
    diagnostics = likelihood_diagnostics(np.zeros(100), np.ones(100))
    assert diagnostics.mean_likelihood == pytest.approx(1.0)
    assert diagnostics.normalization_z_score == pytest.approx(0.0)
    assert diagnostics.likelihood_ess == pytest.approx(100.0)
    assert diagnostics.contribution_ess == pytest.approx(100.0)
    assert diagnostics.top_one_percent_weight_share == pytest.approx(0.01)


def test_log_domain_diagnostics_do_not_underflow_relative_quantities() -> None:
    logs = np.array([-1000.0, -1001.0, -1002.0, -1003.0])
    diagnostics = likelihood_diagnostics(logs)
    assert diagnostics.mean_likelihood == 0.0
    assert np.isfinite(diagnostics.log_mean_likelihood)
    assert 1.0 < diagnostics.likelihood_ess < 4.0
    assert 0.0 < diagnostics.max_normalized_weight < 1.0


def test_contribution_diagnostics_detect_concentration() -> None:
    log_weights = np.zeros(1000)
    payoff = np.zeros(1000)
    payoff[0] = 100.0
    payoff[1:10] = 1.0
    diagnostics = likelihood_diagnostics(log_weights, payoff)
    assert diagnostics.likelihood_ess_fraction == pytest.approx(1.0)
    assert diagnostics.contribution_ess_fraction is not None
    assert diagnostics.contribution_ess_fraction < 0.01
    assert diagnostics.max_contribution_share is not None
    assert diagnostics.max_contribution_share > 0.9


def test_invalid_payoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        likelihood_diagnostics(np.zeros(10), np.full(10, -1.0))
