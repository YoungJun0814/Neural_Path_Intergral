"""Smoke test for the kurtosis penalty path in ``NeuralSDESimulator.train_step``."""
from __future__ import annotations

import torch

from src.neural_engine import NeuralSDESimulator
from src.utils import set_seed


def test_kurtosis_penalty_runs():
    """The training step should execute, produce a finite loss, and update weights."""
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simulator = NeuralSDESimulator(hidden_dim=16, n_layers=2, device=device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in simulator.parameters()]

    loss = simulator.train_step(
        target_prices=[8.0, 5.0, 2.5],
        strikes=[95.0, 100.0, 105.0],
        T=0.25,
        S0=100.0,
        r=0.05,
        optimizer=optimizer,
        target_kurtosis=6.0,
        kurtosis_weight=0.1,
    )

    assert isinstance(loss, float)
    assert loss == loss  # finite (not NaN)
    assert loss < 1e12  # not overflowing

    # At least one parameter should have changed
    changed = any((b - a).abs().sum().item() > 0 for a, b in zip(before, simulator.parameters()))
    assert changed, "train_step did not update parameters"
