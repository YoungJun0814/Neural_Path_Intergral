"""Training objectives for NPI importance samplers.

Notation follows ``docs/formulation.md``:

* ``g(S_T)``      — target functional (payoff / indicator / …).
* ``E_T``         — Radon–Nikodym density  dP/dQ  accumulated along a path.
* ``u``           — control; we simulate under Q and compute log E_T on the fly.

Two publication-path objectives are exposed:

``variance_minimization_objective``
    ``L_VM(u) = E^Q[(g·E_T)^2]``  (Asmussen & Glynn 2007).

``kl_regularized_objective``
    Entropy-regularized stress generation with a smooth terminal reward. This
    is not a variance-minimization objective and must be evaluated separately
    with the hard event and likelihood weights.

The previous helper named ``cem_step`` paired elite labels from one pilot
batch with independent states from a second batch. That invalid helper remains
removed; the validated trajectory-likelihood implementation lives in
``src.training.cem``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn.functional as F


class ControlledSimulator(Protocol):
    """Structural type shared by analytic and neural controlled simulators."""

    def simulate_controlled(self, **kwargs: Any) -> tuple[Any, ...]: ...


# -----------------------------------------------------------------------------
# 1.  Variance-minimization objective
# -----------------------------------------------------------------------------


def variance_minimization_objective(
    simulator: ControlledSimulator,
    control_fn: Callable,
    *,
    S0: float,
    T: float,
    dt: float,
    num_paths: int,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    v0: float | None = None,
    discount: float = 0.0,
) -> dict:
    """Compute L_VM and diagnostics for the current control.

    Returns a dict with keys ``loss``, ``mean_estimate``, ``ess``, ``log_w_mean``.
    The caller is responsible for calling .backward() / .step().
    """
    kwargs = {"S0": S0, "T": T, "dt": dt, "num_paths": num_paths, "control_fn": control_fn}
    if v0 is not None:
        # MarketSimulator requires v0; NeuralSDESimulator accepts it optionally
        kwargs["v0"] = v0
    out = simulator.simulate_controlled(**kwargs)
    S = out[0]
    log_w = out[2]

    S_T = S[:, -1]
    payoff = payoff_fn(S_T)
    weight = torch.exp(log_w)
    reweighted = payoff * weight * math.exp(discount)

    loss = (reweighted**2).mean()

    with torch.no_grad():
        mean_est = reweighted.mean()
        ess = (weight.sum() ** 2) / (weight**2).sum().clamp_min(1e-12)

    return {
        "loss": loss,
        "mean_estimate": mean_est,
        "ess": ess,
        "log_w_mean": log_w.mean().detach(),
    }


# -----------------------------------------------------------------------------
# 2.  KL-regularized crash generation
# -----------------------------------------------------------------------------


def kl_regularized_objective(
    simulator: ControlledSimulator,
    control_fn: Callable,
    *,
    S0: float,
    T: float,
    dt: float,
    num_paths: int,
    barrier_K: float,
    v0: float | None = None,
    kl_weight: float = 1e-2,
    hinge_scale: float = 25.0,
) -> dict:
    """Smooth stress-generation objective with a KL(Q||P) regularizer.

    Surrogate for 1{S_T < K}:

        σ_soft(S_T) = softplus( scale · (K − S_T) / K )

    which is differentiable and continues to reward paths below the barrier.
    It is a proposal-shaping reward, not a smooth log-indicator or an estimator
    of crash probability. ``kl_weight`` balances aggressiveness against
    deviation from the base measure.

    ``KL(Q||P)`` is computed as ``½ E^Q[∫ u² dt]`` (docs/formulation.md §3.2);
    the log_w returned by ``simulate_controlled`` is ``−∫u dW^Q − ½∫u² dt``
    so ``−log_w`` has the correct expectation in the limit.
    """
    kwargs = {"S0": S0, "T": T, "dt": dt, "num_paths": num_paths, "control_fn": control_fn}
    if v0 is not None:
        kwargs["v0"] = v0
    out = simulator.simulate_controlled(**kwargs)
    S = out[0]
    log_w = out[2]
    S_T = S[:, -1]

    hinge = F.softplus(hinge_scale * (barrier_K - S_T) / barrier_K)
    crash_term = -hinge.mean()  # we *maximize* hinge, so minimize −hinge

    kl_est = -log_w.mean()  # equal to ½ E^Q[∫u² dt] in expectation

    loss = crash_term + kl_weight * kl_est

    with torch.no_grad():
        frac_below = (S_T < barrier_K).float().mean()
        ess = (torch.exp(log_w).sum() ** 2) / (torch.exp(2 * log_w)).sum().clamp_min(1e-12)

    return {
        "loss": loss,
        "crash_surrogate": -crash_term.detach(),
        "frac_below": frac_below,
        "kl": kl_est.detach(),
        "ess": ess,
    }
