"""Training objectives for NPI importance samplers.

Notation follows ``docs/formulation.md``:

* ``g(S_T)``      — target functional (payoff / indicator / …).
* ``E_T``         — Radon–Nikodym density  dP/dQ  accumulated along a path.
* ``u``           — control; we simulate under Q and compute log E_T on the fly.

Three objectives are exposed:

``variance_minimization_objective``
    ``L_VM(u) = E^Q[(g·E_T)^2]``  (Asmussen & Glynn 2007).

``kl_regularized_objective``
    ``L_KL(u) = −E^Q[log σ(g − τ)] + λ·KL(Q||P)``  with a softplus-hinge
    surrogate for crash generation (``g = 1{S_T<K}``).

``cem_step``
    One step of the cross-entropy method: simulate a pilot batch under
    current Q, reweight, fit control to weighted samples.  Included as a
    baseline (Rubinstein & Kroese 2004).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F


Simulator = object  # anything with .simulate_controlled(...) -> (S, v, log_w, …)


# -----------------------------------------------------------------------------
# 1.  Variance-minimization objective
# -----------------------------------------------------------------------------

def variance_minimization_objective(
    simulator: Simulator,
    control_fn: Callable,
    *,
    S0: float,
    T: float,
    dt: float,
    num_paths: int,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    v0: Optional[float] = None,
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

    loss = (reweighted ** 2).mean()

    with torch.no_grad():
        mean_est = reweighted.mean()
        ess = (weight.sum() ** 2) / (weight ** 2).sum().clamp_min(1e-12)

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
    simulator: Simulator,
    control_fn: Callable,
    *,
    S0: float,
    T: float,
    dt: float,
    num_paths: int,
    barrier_K: float,
    v0: Optional[float] = None,
    kl_weight: float = 1e-2,
    hinge_scale: float = 25.0,
) -> dict:
    """Soft-hinge crash-probability objective with KL(Q||P) regularizer.

    Surrogate for 1{S_T < K}:

        σ_soft(S_T) = softplus( scale · (K − S_T) / K )

    which is differentiable and pushes paths toward the barrier without
    committing to a hard indicator.  ``kl_weight`` balances aggressiveness
    against deviation from the base measure.

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


# -----------------------------------------------------------------------------
# 3.  Cross-entropy method (CEM) baseline
# -----------------------------------------------------------------------------

def cem_step(
    simulator: Simulator,
    control_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    S0: float,
    T: float,
    dt: float,
    num_paths: int,
    target_event: Callable[[torch.Tensor], torch.Tensor],
    v0: Optional[float] = None,
    quantile: float = 0.9,
    pilot_passes: int = 1,
) -> dict:
    """One CEM iteration.

    * Pilot: simulate ``num_paths`` under current control; identify the
      upper-``quantile`` of ``target_event(S_T)`` values.
    * Update: regress control toward the elite paths' drift using MSE on
      the path-level ``u_t`` predicted values vs. a data-driven target.

    This is a minimal CEM implementation and is provided purely as a
    benchmark; production IS uses ``variance_minimization_objective``.
    """
    control_fn_builder = (lambda net: (lambda t, S, v, A: net(S, v, t, A)))
    control_fn = control_fn_builder(control_net)

    elite_losses = []
    for _ in range(pilot_passes):
        with torch.no_grad():
            kwargs = {"S0": S0, "T": T, "dt": dt, "num_paths": num_paths, "control_fn": control_fn}
            if v0 is not None:
                kwargs["v0"] = v0
            out = simulator.simulate_controlled(**kwargs)
            S = out[0]
            S_T = S[:, -1]
            score = target_event(S_T)
            threshold = torch.quantile(score, quantile)
            elite_mask = score >= threshold

        # Regress control toward an "elite" drift signal:
        # u_t target ≡ +1 for elite paths, 0 otherwise (simple classifier surrogate)
        optimizer.zero_grad()
        kwargs2 = dict(kwargs)
        out2 = simulator.simulate_controlled(**kwargs2)
        S2 = out2[0]
        # Use mid-trajectory state as features
        mid_idx = S2.shape[1] // 2
        S_mid = S2[:, mid_idx]
        v_dummy = torch.full_like(S_mid, float(v0) if v0 is not None else 0.04)
        u_pred = control_net(S_mid, v_dummy, T * 0.5)
        target_u = elite_mask.float()
        loss = F.mse_loss(u_pred, target_u - 0.5)  # center around 0
        loss.backward()
        optimizer.step()
        elite_losses.append(float(loss.item()))

    return {"loss": sum(elite_losses) / max(len(elite_losses), 1), "elite_frac": float((1.0 - quantile))}
