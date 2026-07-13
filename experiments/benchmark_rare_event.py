"""Rare-event pricing benchmark — DriftNet vs. several IS baselines.

Compares, for a deep-OTM put option under Heston:

1. Standard Monte Carlo (baseline)
2. Exponential tilting / Esscher drift (classical IS)
3. NeuralImportanceSampler (research method)

The benchmark outputs the work-normalized VRF and ESS from
``src.evaluation.backtest.efficiency_metrics``.

This file is a smoke benchmark and does NOT establish training convergence.
NIS is given a short budget so the benchmark runs end-to-end in minutes.
Production-quality comparisons must include a validated CEM baseline, measured
wall-clock costs, independent seeds, and more training epochs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.backtest import efficiency_metrics
from src.neural_engine import NeuralImportanceSampler
from src.physics_engine import MarketSimulator
from src.utils import set_seed


def _mc_estimate(
    sim: MarketSimulator, S0: float, v0: float, T: float, dt: float, N: int, K: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S, _ = sim.simulate(S0=S0, v0=v0, T=T, dt=dt, num_paths=N)
    payoff = torch.clamp(K - S[:, -1], min=0.0).numpy()
    weights = np.ones(N)
    return payoff, weights, payoff  # estimates = payoff·w = payoff


def _is_constant_drift(sim, S0, v0, T, dt, N, K, u_value):
    def ctrl(t, S, v, A):
        return torch.full_like(S, float(u_value))

    S, _v, log_w, _, _ = sim.simulate_controlled(
        S0=S0,
        v0=v0,
        T=T,
        dt=dt,
        num_paths=N,
        control_fn=ctrl,
        apply_v_drift_correction=True,
    )
    payoff = torch.clamp(K - S[:, -1], min=0.0)
    weight = torch.exp(log_w)
    est = payoff * weight
    return payoff.numpy(), weight.numpy(), est.numpy()


def _is_neural(sim, S0, v0, T, dt, N, K, n_train_steps: int = 50):
    sampler = NeuralImportanceSampler(sim, hidden_dim=32, n_layers=3, u_bound=1.0)
    optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-3)

    def payoff_fn(S_T):
        return torch.clamp(K - S_T, min=0.0)

    train_start = time.perf_counter()
    for _step in range(n_train_steps):
        sampler.train_step(
            S0=S0,
            T=T,
            dt=dt,
            num_paths=min(2000, N),
            optimizer=optimizer,
            payoff_fn=payoff_fn,
            v0=v0,
        )
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    with torch.no_grad():
        ctrl = sampler.get_control_fn()
        S, _v, log_w, _, _ = sim.simulate_controlled(
            S0=S0,
            v0=v0,
            T=T,
            dt=dt,
            num_paths=N,
            control_fn=ctrl,
            apply_v_drift_correction=True,
        )
    payoff = torch.clamp(K - S[:, -1], min=0.0)
    weight = torch.exp(log_w)
    est = payoff * weight
    eval_seconds = time.perf_counter() - eval_start
    return payoff.numpy(), weight.numpy(), est.numpy(), train_seconds, eval_seconds


def run_benchmark(N: int = 10_000, S0: float = 100.0, K: float = 85.0, T: float = 0.25) -> dict:
    sim = MarketSimulator(mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, device="cpu")
    dt = 1 / 252.0
    v0 = 0.04

    set_seed(0)
    start = time.perf_counter()
    pay_mc, w_mc, est_mc = _mc_estimate(sim, S0, v0, T, dt, N, K)
    elapsed_mc = time.perf_counter() - start
    set_seed(1)
    start = time.perf_counter()
    pay_esscher, w_esscher, est_esscher = _is_constant_drift(sim, S0, v0, T, dt, N, K, u_value=-0.4)
    elapsed_esscher = time.perf_counter() - start
    set_seed(2)
    pay_nis, w_nis, est_nis, train_nis, eval_nis = _is_neural(
        sim, S0, v0, T, dt, N, K, n_train_steps=30
    )

    result = {"N": N, "S0": S0, "K": K, "T": T}
    for name, est, w, elapsed in [
        ("monte_carlo", est_mc, w_mc, elapsed_mc),
        ("is_esscher", est_esscher, w_esscher, elapsed_esscher),
        ("is_neural", est_nis, w_nis, eval_nis),
    ]:
        mean = float(est.mean())
        se = float(est.std(ddof=1) / math.sqrt(N))
        ess = float(w.sum() ** 2 / max((w**2).sum(), 1e-12))
        result[name] = {
            "mean": mean,
            "se": se,
            "ess": ess,
            "var": float(est.var(ddof=1)),
            "eval_seconds": elapsed,
            "eval_seconds_per_path": elapsed / N,
        }
    result["is_neural"]["train_seconds"] = train_nis

    # Online efficiency treats the trained proposal as reusable. End-to-end
    # efficiency charges this evaluation batch for the full training cost.
    eff_online = efficiency_metrics(
        estimates_mc=est_mc,
        estimates_is=est_nis,
        weights_is=w_nis,
        cost_mc=elapsed_mc / N,
        cost_is=eval_nis / N,
    )
    eff_end_to_end = efficiency_metrics(
        estimates_mc=est_mc,
        estimates_is=est_nis,
        weights_is=w_nis,
        cost_mc=elapsed_mc / N,
        cost_is=(train_nis + eval_nis) / N,
    )
    result["efficiency_mc_vs_nis"] = {
        "online_vrf": eff_online.vrf,
        "single_batch_end_to_end_vrf": eff_end_to_end.vrf,
        "ess_is": eff_online.ess_is,
        "var_mc": eff_online.var_mc,
        "var_is": eff_online.var_is,
    }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=85.0)
    p.add_argument("--T", type=float, default=0.25)
    args = p.parse_args()
    print(json.dumps(run_benchmark(N=args.N, S0=args.S0, K=args.K, T=args.T), indent=2))
