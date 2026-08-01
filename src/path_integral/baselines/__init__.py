"""Faithful rBergomi baseline implementations for the G11 V8 program."""

from .cem import CEMTrainingConfig, train_cem_proposal
from .conditional_rbergomi import (
    evaluate_conditional_terminal_units,
    freeze_conditional_rbergomi_proposal,
)
from .coupling_flow_is import FlowTrainingConfig, train_coupling_flow_proposal
from .crude_antithetic import (
    evaluate_latent_is_units,
    freeze_crude_or_antithetic_proposal,
)
from .large_deviation_is import (
    LargeDeviationTrainingConfig,
    train_large_deviation_proposal,
)
from .rbergomi_common import BaselineUnitBatch, RBergomiBaselineProblem
from .smoothing_rqmc import (
    evaluate_smoothing_rqmc_units,
    freeze_smoothing_rqmc_proposal,
)

__all__ = [
    "BaselineUnitBatch",
    "CEMTrainingConfig",
    "FlowTrainingConfig",
    "LargeDeviationTrainingConfig",
    "RBergomiBaselineProblem",
    "evaluate_conditional_terminal_units",
    "evaluate_latent_is_units",
    "evaluate_smoothing_rqmc_units",
    "freeze_conditional_rbergomi_proposal",
    "freeze_crude_or_antithetic_proposal",
    "freeze_smoothing_rqmc_proposal",
    "train_cem_proposal",
    "train_coupling_flow_proposal",
    "train_large_deviation_proposal",
]
