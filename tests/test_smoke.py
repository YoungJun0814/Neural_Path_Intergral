"""Smoke tests: imports resolve and simulators run end-to-end."""
from __future__ import annotations

import torch

from src.ai_calibrator import NeuralCalibrator
from src.ml_models import BlackScholesModel
from src.neural_engine import DiffNet, DriftNet, NeuralSDESimulator, VolNet
from src.physics_engine import FractionalBrownianMotion, MarketSimulator, RBergomiSimulator
from src.utils import git_hash, pick_device, set_seed


def test_imports_resolve():
    """All public symbols we rely on downstream must import cleanly."""
    assert DriftNet is not None
    assert DiffNet is not None
    assert VolNet is not None
    assert NeuralSDESimulator is not None
    assert MarketSimulator is not None
    assert RBergomiSimulator is not None
    assert FractionalBrownianMotion is not None
    assert NeuralCalibrator is not None
    assert BlackScholesModel is not None


def test_pick_device_returns_torch_device():
    dev = pick_device(None)
    assert isinstance(dev, torch.device)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.rand(4)
    set_seed(123)
    b = torch.rand(4)
    assert torch.allclose(a, b)


def test_git_hash_returns_string():
    h = git_hash()
    assert isinstance(h, str) and len(h) > 0


def test_market_simulator_heston_shapes():
    """Heston simulation should return (num_paths, n_steps+1)-shaped tensors."""
    set_seed(0)
    device = "cpu"
    sim = MarketSimulator(mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, device=device)
    S, v = sim.simulate(S0=100.0, v0=0.04, T=0.1, dt=0.01, num_paths=64)
    assert S.shape == (64, 11)
    assert v.shape == (64, 11)
    assert torch.isfinite(S).all()
    assert torch.isfinite(v).all()
    assert (v >= 0).all()
    assert (S > 0).all()


def test_black_scholes_closed_form_sanity():
    """BS ATM call price with σ=0.2, T=1, r=0.05 should be around 10.45."""
    bs = BlackScholesModel(sigma=0.2)
    price = bs.price(S0=100.0, K=100.0, T=1.0, r=0.05)
    assert 9.5 < price < 11.5, f"BS call price unexpectedly {price}"
