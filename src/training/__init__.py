"""Training objectives and validated importance-sampling baselines.

See ``docs/formulation.md §3``.
"""

from .cem import (
    CEMBatch,
    CEMIteration,
    CEMResult,
    HestonTerminalLossSampler,
    fit_constant_control_cem,
)
from .markov_control import (
    MarkovianHestonControl,
    MarkovObjectiveDiagnostics,
    MarkovTrainingEpoch,
    MarkovTrainingResult,
    load_markovian_control_checkpoint,
    markov_control_objective,
    markov_control_state_sha256,
    save_markovian_control_checkpoint,
    train_markovian_control,
)
from .objectives import (
    kl_regularized_objective,
    variance_minimization_objective,
)

__all__ = [
    "CEMBatch",
    "CEMIteration",
    "CEMResult",
    "HestonTerminalLossSampler",
    "fit_constant_control_cem",
    "MarkovianHestonControl",
    "MarkovObjectiveDiagnostics",
    "MarkovTrainingEpoch",
    "MarkovTrainingResult",
    "markov_control_state_sha256",
    "save_markovian_control_checkpoint",
    "load_markovian_control_checkpoint",
    "markov_control_objective",
    "train_markovian_control",
    "variance_minimization_objective",
    "kl_regularized_objective",
]
