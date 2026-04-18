"""Loss functions for distribution matching and objective learning."""

from .distribution_match import (
    mmd_loss,
    moment_match_loss,
    sliced_wasserstein_distance,
    standardized_moments,
)

__all__ = [
    "mmd_loss",
    "moment_match_loss",
    "sliced_wasserstein_distance",
    "standardized_moments",
]
