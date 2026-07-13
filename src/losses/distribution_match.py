"""Distribution-matching losses for base NeuralSDE training.

Reference: ``docs/formulation.md §3.4``.

We expose three losses, from cheapest/least-informative to most-informative:

1. ``moment_match_loss`` — weighted squared error between standardized moments
   (mean, std, skew, kurtosis).  Fast but biased; useful as a warm-up.
2. ``sliced_wasserstein_distance`` — 1D Wasserstein-2 along random directions,
   O(n log n) per projection.  For univariate returns this reduces to the
   sorted-vector distance.
3. ``mmd_loss`` — empirical squared MMD with Gaussian / multi-scale kernel.
   Proper divergence on a universal RKHS (Gretton et al. 2012); O(n²).

All functions take ``torch.Tensor`` inputs and are fully differentiable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

# -----------------------------------------------------------------------------
# 1.  Moment matching
# -----------------------------------------------------------------------------


def standardized_moments(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mean, std, skew, kurt) where kurt is raw (Normal = 3)."""
    mu = x.mean()
    std = x.std(unbiased=False) + 1e-8
    z = (x - mu) / std
    skew = (z**3).mean()
    kurt = (z**4).mean()
    return mu, std, skew, kurt


def moment_match_loss(
    x_model: torch.Tensor,
    x_data: torch.Tensor,
    weights: Sequence[float] = (50.0, 100.0, 0.5, 0.01),
    targets: tuple[float | None, float | None, float | None, float | None] = (
        None,
        None,
        None,
        None,
    ),
) -> torch.Tensor:
    """Weighted squared error on (mean, std, skew, kurt).

    If a ``targets[k]`` is None, use the empirical value from ``x_data``.
    """
    m_m, m_s, m_sk, m_kt = standardized_moments(x_model)
    d_m, d_s, d_sk, d_kt = standardized_moments(x_data)
    tgt = [
        d_m if targets[0] is None else torch.tensor(float(targets[0]), device=x_model.device),
        d_s if targets[1] is None else torch.tensor(float(targets[1]), device=x_model.device),
        d_sk if targets[2] is None else torch.tensor(float(targets[2]), device=x_model.device),
        d_kt if targets[3] is None else torch.tensor(float(targets[3]), device=x_model.device),
    ]
    w = weights
    return (
        w[0] * (m_m - tgt[0]) ** 2
        + w[1] * (m_s - tgt[1]) ** 2
        + w[2] * (m_sk - tgt[2]) ** 2
        + w[3] * (m_kt - tgt[3]) ** 2
    )


# -----------------------------------------------------------------------------
# 2.  Sliced / 1-D Wasserstein
# -----------------------------------------------------------------------------


def _sorted_wasserstein_1d(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
    """W_p between 1-D empirical distributions (equal sample sizes OK; else
    we interpolate via quantile matching on a common grid).
    """
    x_s, _ = torch.sort(x)
    y_s, _ = torch.sort(y)
    n = min(x_s.numel(), y_s.numel())
    if x_s.numel() != n:
        q = torch.linspace(0.0, 1.0, steps=n, device=x.device)
        x_s = torch.quantile(x, q)
    if y_s.numel() != n:
        q = torch.linspace(0.0, 1.0, steps=n, device=y.device)
        y_s = torch.quantile(y, q)
    if p == 1:
        return (x_s - y_s).abs().mean()
    return ((x_s - y_s) ** 2).mean()


def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 64,
    p: int = 2,
) -> torch.Tensor:
    """Sliced Wasserstein-p distance.  For 1-D inputs reduces to the sorted
    distance (``num_projections`` is ignored).
    """
    if x.dim() == 1 and y.dim() == 1:
        return _sorted_wasserstein_1d(x, y, p=p)
    d = x.shape[-1]
    projs = torch.randn(num_projections, d, device=x.device)
    projs = projs / (projs.norm(dim=1, keepdim=True) + 1e-12)
    x_p = x @ projs.T  # (N, K)
    y_p = y @ projs.T
    losses = [_sorted_wasserstein_1d(x_p[:, k], y_p[:, k], p=p) for k in range(num_projections)]
    return torch.stack(losses).mean()


# -----------------------------------------------------------------------------
# 3.  Maximum Mean Discrepancy
# -----------------------------------------------------------------------------


def _gaussian_kernel_matrix(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    if a.dim() == 1:
        a = a.unsqueeze(-1)
    if b.dim() == 1:
        b = b.unsqueeze(-1)
    d2 = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
    return torch.exp(-d2 / (2.0 * sigma**2))


def mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Sequence[float] | None = None,
    unbiased: bool = True,
) -> torch.Tensor:
    """Empirical squared MMD with multi-scale Gaussian kernel.

    ``sigmas`` defaults to a log-spaced ladder based on the median heuristic
    computed on the concatenation of x and y.
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if y.dim() == 1:
        y = y.unsqueeze(-1)

    if sigmas is None:
        with torch.no_grad():
            combined = torch.cat([x, y], dim=0)
            # Median of pairwise distances (subsample for speed)
            n = min(combined.size(0), 500)
            idx = torch.randperm(combined.size(0))[:n]
            sub = combined[idx]
            d = torch.cdist(sub, sub).flatten()
            d = d[d > 0]
            med = d.median().item() if d.numel() > 0 else 1.0
        sigmas = [med * s for s in (0.25, 0.5, 1.0, 2.0, 4.0)]

    n, m = x.size(0), y.size(0)
    loss = x.new_zeros(())
    for sigma in sigmas:
        Kxx = _gaussian_kernel_matrix(x, x, sigma)
        Kyy = _gaussian_kernel_matrix(y, y, sigma)
        Kxy = _gaussian_kernel_matrix(x, y, sigma)
        if unbiased:
            Kxx = Kxx - torch.diag(torch.diagonal(Kxx))
            Kyy = Kyy - torch.diag(torch.diagonal(Kyy))
            loss = loss + Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
        else:
            loss = loss + Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return loss / len(sigmas)
