"""
Utility helpers: reproducibility, device selection, git hashing, and visualization.
"""

from __future__ import annotations

import os
import random
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# 1. Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    # Some legacy experiments still call the module-level NumPy RNG. Keep it
    # synchronized until those call sites are migrated to explicit Generators.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------------------------------------------------------
# 2. Device & Environment
# -----------------------------------------------------------------------------


def pick_device(override: str | None = None) -> torch.device:
    """Return an appropriate torch.device, honoring an optional override."""
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_hash(short: bool = True) -> str:
    """Return the current git commit hash (or 'unknown' if unavailable)."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            cmd = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# -----------------------------------------------------------------------------
# 3. Tensor utilities
# -----------------------------------------------------------------------------


def to_numpy(tensor):
    """Safely convert a PyTorch Tensor to a NumPy array (CPU copy)."""
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor


# -----------------------------------------------------------------------------
# 4. Dataset
# -----------------------------------------------------------------------------


class HestonDataset(Dataset):
    """Random Heston parameter combinations for synthetic training data."""

    def __init__(
        self, num_samples: int = 1000, T: float = 1.0, dt: float = 1 / 252, device: str = "cuda"
    ):
        self.num_samples = num_samples
        self.T = T
        self.dt = dt
        self.device = device
        self.kappas = torch.rand(num_samples, device=device) * 4.0 + 0.1
        self.thetas = torch.rand(num_samples, device=device) * 0.5 + 0.01
        self.xis = torch.rand(num_samples, device=device) * 0.9 + 0.1
        self.rhos = torch.rand(num_samples, device=device) * 1.8 - 0.9

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "kappa": self.kappas[idx],
            "theta": self.thetas[idx],
            "xi": self.xis[idx],
            "rho": self.rhos[idx],
        }


# -----------------------------------------------------------------------------
# 5. Visualization
# -----------------------------------------------------------------------------


def plot_weighted_paths(S_paths, weights, num_to_plot: int = 30, title: str = "Weighted Paths"):
    """Visualize top 'classical paths' with highest importance weights."""
    sorted_indices = torch.argsort(weights, descending=True)
    top_indices = sorted_indices[:num_to_plot]
    S_cpu = to_numpy(S_paths)
    weights_cpu = to_numpy(weights)

    plt.figure(figsize=(10, 6))
    max_weight = weights_cpu[top_indices[0]]
    for idx in top_indices:
        alpha = float(weights_cpu[idx] / max_weight)
        plt.plot(S_cpu[idx], color="blue", alpha=max(0.1, min(1.0, alpha)), linewidth=1)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.grid(True, alpha=0.3)
    plt.show()
