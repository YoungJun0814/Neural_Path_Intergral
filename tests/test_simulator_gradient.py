"""Gradient flow tests — simulate_controlled must be differentiable through
the control so that NeuralImportanceSampler.train_step actually updates."""
from __future__ import annotations

import math

import torch

from src.losses.distribution_match import mmd_loss, moment_match_loss, standardized_moments
from src.neural_engine import NeuralImportanceSampler, NeuralSDESimulator
from src.physics_engine import MarketSimulator
from src.utils import set_seed


def test_is_loss_has_gradient_through_market_simulator():
    set_seed(0)
    sim = MarketSimulator(mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, device="cpu")
    sampler = NeuralImportanceSampler(sim, hidden_dim=16, n_layers=2, u_bound=1.0)

    def payoff_fn(S_T):
        return torch.clamp(S_T - 100.0, min=0.0)

    params_before = [p.detach().clone() for p in sampler.parameters()]
    optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-3)

    info = sampler.train_step(
        S0=100.0, T=0.1, dt=1 / 50.0, num_paths=200, optimizer=optimizer,
        payoff_fn=payoff_fn, v0=0.04,
    )
    assert math.isfinite(info["loss"])
    assert math.isfinite(info["mean_estimate"])
    assert info["ess"] > 0
    # At least one parameter should have changed
    diffs = [float((p_new.detach() - p_old).abs().sum()) for p_new, p_old in zip(sampler.parameters(), params_before)]
    assert max(diffs) > 0, "no parameter moved — gradient did not flow"


def test_is_loss_has_gradient_through_neural_sde():
    set_seed(1)
    nsde = NeuralSDESimulator(hidden_dim=16, n_layers=2, device="cpu")
    sampler = NeuralImportanceSampler(nsde, hidden_dim=16, n_layers=2, u_bound=1.0)

    def payoff(S_T):
        return torch.clamp(S_T - 100.0, min=0.0)

    optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-3)
    info = sampler.train_step(
        S0=100.0, T=0.1, dt=1 / 50.0, num_paths=200, optimizer=optimizer,
        payoff_fn=payoff, v0=0.04,
    )
    assert math.isfinite(info["loss"])


def test_moment_match_loss_differentiable():
    set_seed(2)
    x = torch.randn(200, requires_grad=True)
    y = torch.randn(200)
    loss = moment_match_loss(x, y)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mmd_loss_differentiable():
    set_seed(3)
    x = torch.randn(100, requires_grad=True)
    y = torch.randn(100) + 0.1
    loss = mmd_loss(x, y)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(loss) > 0  # distributions are different


def test_standardized_moments_shape():
    x = torch.randn(1000)
    m, s, sk, k = standardized_moments(x)
    assert m.dim() == 0 and s.dim() == 0 and sk.dim() == 0 and k.dim() == 0
    assert abs(float(m)) < 0.1
    assert abs(float(s) - 1.0) < 0.1
