"""Finite known-answer law-of-total-variance oracle for auxiliary risk sampling."""

from __future__ import annotations

from typing import Any

import torch


def finite_risk_decomposition(px: torch.Tensor, pu: torch.Tensor, g: torch.Tensor,
                              dq: torch.Tensor, dgx: torch.Tensor, dh: torch.Tensor) -> dict[str, Any]:
    tensors = (px, pu, g, dq, dgx, dh)
    if (any(t.dtype != torch.float64 or t.device.type != "cpu" or t.requires_grad
            or not torch.isfinite(t).all() for t in tensors)
            or px.ndim != 1 or pu.ndim != 1 or len(px) < 1 or len(pu) < 1
            or g.shape != (len(px), len(pu)) or dh.shape != g.shape
            or dq.shape != px.shape or dgx.shape != px.shape
            or any(not bool((t > 0).all()) for t in (px, pu, dq, dgx, dh))
            or (g < 0).any() or (g > 1).any()
            or abs(float(px.sum())-1) > 1e-12 or abs(float(pu.sum())-1) > 1e-12
            or abs(float(px @ dq)-1) > 1e-12 or abs(float(px @ dgx)-1) > 1e-12
            or not torch.allclose(dh @ pu, torch.ones_like(px), rtol=0, atol=1e-12)):
        raise ValueError("finite normalized positive densities and bounded finite-grid payoff required")
    m2 = g.square() @ pu
    target = float(px @ (m2/dq))
    gx, h = px*dgx, pu[None, :]*dh
    contribution = g.square()/(dq[:, None]*dgx[:, None]*dh)
    if not torch.isfinite(contribution).all():
        raise FloatingPointError("finite risk contribution range unresolved")
    conditional = (h*contribution).sum(1)
    analytic_conditional = m2/(dq*dgx)
    mean = float(gx @ conditional)
    between = float(gx @ (analytic_conditional-target).square())
    within = float(gx @ (h*(contribution-analytic_conditional[:, None]).square()).sum(1))
    total = float((gx[:, None]*h*(contribution-target).square()).sum())
    if not torch.isfinite(torch.tensor([target, mean, between, within, total], dtype=torch.float64)).all():
        raise FloatingPointError("finite risk second-moment range unresolved")
    if abs(mean-target) > 1e-12 or not torch.allclose(conditional, analytic_conditional, rtol=1e-12, atol=1e-13):
        raise FloatingPointError("conditional risk expectation identity failed")
    if abs(total-between-within) > 1e-12*max(1., total):
        raise FloatingPointError("total risk variance decomposition failed")
    return {"M2_of_fixed_event_q": target, "risk_estimator_mean": mean,
        "total_risk_estimator_variance": total, "marginal_variance": between,
        "conditional_variance": within, "conditional_m2": m2.tolist(),
        "not_population_volterra_result": True}


def marginal_risk_known_answer() -> dict[str, Any]:
    px = torch.tensor([.8, .2], dtype=torch.float64)
    pu = torch.tensor([.5, .5], dtype=torch.float64)
    g = torch.tensor([[.01, .02], [.5, .9]], dtype=torch.float64)
    dq = torch.ones(2, dtype=torch.float64)
    dh = torch.ones((2, 2), dtype=torch.float64)
    m2 = g.square() @ pu
    ideal_h = g.square()/m2[:, None]
    ideal_gx = m2/(px @ m2)
    plain = finite_risk_decomposition(px, pu, g, dq, dq, dh)
    conditional = finite_risk_decomposition(px, pu, g, dq, dq, ideal_h)
    joint = finite_risk_decomposition(px, pu, g, dq, ideal_gx, ideal_h)
    return {"plain": plain, "ideal_conditional_only": conditional, "ideal_joint": joint,
        "conditional_only_leaves_positive_marginal_variance": conditional["marginal_variance"] > 0,
        "ideal_conditional_variance_zero": conditional["conditional_variance"] < 1e-28,
        "ideal_joint_variance_zero": joint["total_risk_estimator_variance"] < 1e-28,
        "not_new_theorem_or_implemented_volterra_G": True}
