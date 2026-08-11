from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_zero_variance_target_identity_on_discrete_probability_space() -> None:
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    q = torch.tensor([0.25, 0.15, 0.35, 0.25], dtype=torch.float64)
    g = torch.tensor([0.01, 0.2, 0.6, 0.95], dtype=torch.float64)
    probability = torch.dot(p, g)
    likelihood = p / q
    contribution = g * likelihood
    relative_variance = (
        torch.dot(q, contribution.square()) - probability.square()
    ) / probability.square()
    zero_variance_target = p * g / probability
    chi_square = torch.sum((zero_variance_target - q).square() / q)
    assert abs(float(relative_variance - chi_square)) < 2e-15


def test_defensive_second_moment_bound_on_discrete_probability_space() -> None:
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    residual = torch.tensor([0.55, 0.05, 0.1, 0.3], dtype=torch.float64)
    payoff = torch.tensor([0.0, 0.2, 0.6, 1.0], dtype=torch.float64)
    delta = 0.17
    q = delta * p + (1.0 - delta) * residual
    likelihood = p / q
    second_moment = torch.dot(q, (payoff * likelihood).square())
    probability = torch.dot(p, payoff)
    assert float(torch.max(likelihood)) <= 1.0 / delta + 1e-14
    assert float(second_moment) <= float(probability / delta) + 1e-14


def test_conditionalization_is_a_variance_reduction_in_toy_model() -> None:
    # W has two states under Q.  Conditional Bernoulli randomness represents the
    # analytically integrated independent price driver.
    q = torch.tensor([0.35, 0.65], dtype=torch.float64)
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    event_probability_given_w = torch.tensor([0.05, 0.8], dtype=torch.float64)
    likelihood = p / q
    conditional = event_probability_given_w * likelihood
    conditional_second_moment = torch.dot(q, conditional.square())
    hard_second_moment = torch.dot(
        q,
        event_probability_given_w * likelihood.square(),
    )
    assert conditional_second_moment <= hard_second_moment


def test_theorem_ledger_prevents_open_asymptotic_claims() -> None:
    ledger = yaml.safe_load(
        (ROOT / "configs/g11_v15/theorem_ledger_v1.yaml").read_text(encoding="utf-8")
    )
    allowed = set(ledger["allowed_statuses"])
    statuses = {key: item["status"] for key, item in ledger["theorems"].items()}
    assert set(statuses.values()) <= allowed
    assert all(statuses[f"T15-{index}"] == "proved" for index in range(1, 5))
    assert statuses["T15-5"] == "open"
    assert all(statuses[f"T16-{index}"] == "proved" for index in range(1, 5))
    assert ledger["gates"]["G5"]["pass"] is False
    assert ledger["gates"]["G5_fixed_grid"]["pass"] is True
    assert ledger["gates"]["G5_continuous_probability"]["pass"] is True
    assert "asymptotically_optimal" in ledger["claim_locks"]["prohibited"]
