from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_claim_contract_keeps_unproved_claims_locked() -> None:
    values = yaml.safe_load((ROOT / "configs/g11_v15/claim_contract_v1.yaml").read_text())
    assert values["primary_asymptotic_regime"] == "small_noise"
    assert values["small_time_claim_authorized"] is False
    assert values["continuous_exact_simulation_claim_authorized"] is False
    assert values["external_novelty_review"]["submission_lock"] is True
    assert values["gates"]["qualification"] == "locked"
    assert values["nonnegotiable"]["final_estimator"] == "ordinary_importance_sampling"


def test_contracted_independent_price_control_is_the_minimum_norm_solution() -> None:
    weights = torch.tensor([0.4, 0.7, 1.1, 0.9], dtype=torch.float64)
    rho = -0.65
    correlated_return = -0.15
    threshold = -0.8
    gap = correlated_return - threshold
    scale = (1.0 - rho**2) ** 0.5
    optimum = -(gap / (scale * torch.dot(weights, weights))) * weights
    achieved = correlated_return + scale * torch.dot(weights, optimum)
    contracted_cost = gap**2 / (2.0 * (1.0 - rho**2) * torch.dot(weights, weights))
    assert abs(float(achieved) - threshold) < 1e-14
    assert abs(0.5 * float(torch.dot(optimum, optimum)) - float(contracted_cost)) < 1e-14
    # Every feasible perturbation orthogonal to the volatility direction costs more.
    orthogonal = torch.tensor([0.7, -0.4, 0.0, 0.0], dtype=torch.float64)
    orthogonal -= torch.dot(orthogonal, weights) / torch.dot(weights, weights) * weights
    competitor = optimum + orthogonal
    assert abs(float(torch.dot(weights, competitor) - torch.dot(weights, optimum))) < 1e-14
    assert torch.dot(competitor, competitor) > torch.dot(optimum, optimum)
