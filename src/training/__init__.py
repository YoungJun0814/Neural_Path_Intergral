"""Training-objective implementations for variance-minimization IS, KL-regularized
crash generation, and the cross-entropy method (CEM) baseline.

See ``docs/formulation.md §3``.
"""

from .objectives import (
    cem_step,
    kl_regularized_objective,
    variance_minimization_objective,
)

__all__ = [
    "variance_minimization_objective",
    "kl_regularized_objective",
    "cem_step",
]
