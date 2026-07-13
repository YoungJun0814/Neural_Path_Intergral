"""Phase 2.2 unbiasedness verification protocol.

For a fixed control u (constant or linear-in-S), simulate the option price
(call payoff) under both the natural P-measure and the controlled Q-measure
and check that the IS estimator falls within the 99.7% (3σ) CLT band of the
MC estimator.

Run:
    python experiments/unbiasedness_check.py --N 50000 --u 0.5

Outputs a JSON report to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics_engine import MarketSimulator
from src.utils import set_seed


def constant_control(u_value: float):
    def fn(t, S, v, A):
        return torch.full_like(S, float(u_value))

    return fn


def linear_control(slope: float, S0: float):
    def fn(t, S, v, A):
        # u = -slope * (S/S0 - 1) — push back toward S0 (mean-reversion control)
        return torch.tanh(-slope * (S / S0 - 1.0))

    return fn


def main(
    N: int = 50_000,
    u: float = 0.5,
    T: float = 0.5,
    K: float = 100.0,
    linear_slope: float | None = None,
) -> dict:
    sim = MarketSimulator(mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, device="cpu")

    set_seed(0)
    S_mc, _ = sim.simulate(S0=100.0, v0=0.04, T=T, dt=1 / 252.0, num_paths=N)
    pay_mc = torch.clamp(S_mc[:, -1] - K, min=0.0)
    mean_mc = float(pay_mc.mean())
    std_mc = float(pay_mc.std())
    se_mc = std_mc / math.sqrt(N)

    if linear_slope is not None:
        ctrl = linear_control(linear_slope, S0=100.0)
        ctrl_label = f"linear(slope={linear_slope})"
    else:
        ctrl = constant_control(u)
        ctrl_label = f"constant({u})"

    set_seed(1)
    S_is, _v, log_w, _bh, _ = sim.simulate_controlled(
        S0=100.0,
        v0=0.04,
        T=T,
        dt=1 / 252.0,
        num_paths=N,
        control_fn=ctrl,
        apply_v_drift_correction=True,
    )
    weighted = torch.clamp(S_is[:, -1] - K, min=0.0) * torch.exp(log_w)
    mean_is = float(weighted.mean())
    std_is = float(weighted.std())
    se_is = std_is / math.sqrt(N)

    diff = mean_is - mean_mc
    band = 3.0 * math.sqrt(se_mc**2 + se_is**2)
    inside = abs(diff) < band

    weights_np = torch.exp(log_w).numpy()
    ess = float((weights_np.sum() ** 2) / max((weights_np**2).sum(), 1e-12))

    # Repeat with v-drift correction OFF for comparison
    set_seed(1)
    S_bad, _, log_w_bad, _, _ = sim.simulate_controlled(
        S0=100.0,
        v0=0.04,
        T=T,
        dt=1 / 252.0,
        num_paths=N,
        control_fn=ctrl,
        apply_v_drift_correction=False,
    )
    weighted_bad = torch.clamp(S_bad[:, -1] - K, min=0.0) * torch.exp(log_w_bad)
    mean_bad = float(weighted_bad.mean())

    report = {
        "N_mc": N,
        "N_is": N,
        "T": T,
        "K": K,
        "control": ctrl_label,
        "mean_mc": mean_mc,
        "se_mc": se_mc,
        "mean_is_corrected": mean_is,
        "se_is": se_is,
        "mean_is_uncorrected": mean_bad,
        "diff_corrected": diff,
        "3sigma_band": band,
        "passes_3sigma": bool(inside),
        "ess": ess,
        "vrf_naive": (std_mc**2) / max(std_is**2, 1e-12),
    }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=20_000)
    p.add_argument("--u", type=float, default=0.5)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--linear-slope", type=float, default=None)
    args = p.parse_args()
    out = main(N=args.N, u=args.u, T=args.T, K=args.K, linear_slope=args.linear_slope)
    print(json.dumps(out, indent=2))
