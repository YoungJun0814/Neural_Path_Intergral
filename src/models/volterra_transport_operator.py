"""Structure-preserving amortized predictor for V15 transport modes.

The network is only a proposal initializer.  Exact Gaussian likelihoods and a
positive natural component preserve estimator validity independently of prediction
quality; deterministic correction or natural fallback controls proposal quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_modes import (
    ModeSearchConfig,
    RBergomiModeSearchResult,
    find_rbergomi_conditional_modes,
)
from src.path_integral.finite_rank_gaussian_transport import (
    CurvatureTransportConfig,
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
    build_curvature_transport,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.volterra_action import RBergomiConditionalAction


@dataclass(frozen=True)
class VolterraTransportOperatorConfig:
    feature_dimension: int
    modes: int
    rank: int
    hidden_features: int = 64
    maximum_coefficient_norm: float = 30.0
    minimum_precision_eigenvalue: float = 0.05
    maximum_precision_eigenvalue: float = 20.0

    def __post_init__(self) -> None:
        counts = (self.feature_dimension, self.modes, self.rank, self.hidden_features)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts):
            raise ValueError("operator dimensions must be positive integers")
        if not math.isfinite(self.maximum_coefficient_norm) or (
            self.maximum_coefficient_norm <= 0.0
        ):
            raise ValueError("maximum coefficient norm must be finite and positive")
        if not 0.0 < self.minimum_precision_eigenvalue <= self.maximum_precision_eigenvalue:
            raise ValueError("precision spectrum bounds are invalid")
        if not math.isfinite(self.maximum_precision_eigenvalue):
            raise ValueError("maximum precision eigenvalue must be finite")


@dataclass(frozen=True)
class VolterraTransportPrediction:
    coefficients: torch.Tensor
    mode_weights: torch.Tensor
    precision_matrices: torch.Tensor
    precision_eigenvalues: torch.Tensor


class VolterraTransportOperator(torch.nn.Module):
    """Fixed-slot set decoder with bounded coefficients and SPD precisions."""

    def __init__(self, config: VolterraTransportOperatorConfig) -> None:
        super().__init__()
        self.config = config
        triangular = config.rank * (config.rank + 1) // 2
        output_features = config.modes * (config.rank + triangular + 1)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(config.feature_dimension, config.hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_features, config.hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_features, output_features),
        )
        self.double()

    def forward(self, features: torch.Tensor) -> VolterraTransportPrediction:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dimension:
            raise ValueError("operator features have the wrong shape")
        if features.device.type != "cpu" or features.dtype != torch.float64:
            raise ValueError("operator features must be CPU float64")
        if not torch.isfinite(features).all():
            raise ValueError("operator features must be finite")
        raw = self.encoder(features)
        batch = features.shape[0]
        modes = self.config.modes
        rank = self.config.rank
        triangular = rank * (rank + 1) // 2
        cursor = 0
        coefficients = raw[:, cursor : cursor + modes * rank].reshape(batch, modes, rank)
        cursor += modes * rank
        norms = torch.linalg.vector_norm(coefficients, dim=2, keepdim=True)
        coefficient_factor = torch.clamp(
            self.config.maximum_coefficient_norm / torch.clamp(norms, min=1e-300),
            max=1.0,
        )
        coefficients = coefficients * coefficient_factor
        cholesky_raw = raw[:, cursor : cursor + modes * triangular].reshape(
            batch,
            modes,
            triangular,
        )
        cursor += modes * triangular
        logits = raw[:, cursor : cursor + modes]
        mode_weights = torch.softmax(logits, dim=1)
        lower = torch.zeros((batch, modes, rank, rank), dtype=torch.float64)
        row, column = torch.tril_indices(rank, rank)
        lower[:, :, row, column] = cholesky_raw
        diagonal = torch.arange(rank)
        lower[:, :, diagonal, diagonal] = torch.nn.functional.softplus(
            lower[:, :, diagonal, diagonal]
        ) + math.sqrt(self.config.minimum_precision_eigenvalue)
        raw_precision = lower @ lower.transpose(-1, -2)
        eigenvalues, eigenvectors = torch.linalg.eigh(raw_precision)
        clipped = torch.clamp(
            eigenvalues,
            min=self.config.minimum_precision_eigenvalue,
            max=self.config.maximum_precision_eigenvalue,
        )
        precision = eigenvectors @ torch.diag_embed(clipped) @ eigenvectors.transpose(-1, -2)
        return VolterraTransportPrediction(
            coefficients=coefficients,
            mode_weights=mode_weights,
            precision_matrices=precision,
            precision_eigenvalues=clipped,
        )


def encode_rbergomi_transport_task(
    problem: RBergomiBaselineProblem,
    *,
    epsilon: float,
) -> torch.Tensor:
    if not isinstance(problem.task, TerminalThresholdTask):
        raise TypeError("V15 operator currently supports terminal threshold tasks")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must lie in (0, 1]")
    return torch.tensor(
        [
            problem.hurst,
            math.log(problem.eta),
            math.atanh(problem.rho),
            math.log(problem.xi),
            math.log(problem.maturity),
            math.log(problem.task.level / problem.spot),
            math.log(epsilon),
        ],
        dtype=torch.float64,
    )


@dataclass(frozen=True)
class OperatorTransportCertificate:
    proposal: DefensiveFiniteRankGaussianMixture
    corrected_modes: RBergomiModeSearchResult | None
    used_natural_fallback: bool
    maximum_gradient_norm: float | None
    exact_likelihood: bool
    estimator_unbiased_when_frozen: bool


def correct_and_build_operator_transport(
    action: RBergomiConditionalAction,
    prediction: VolterraTransportPrediction,
    *,
    mode_search: ModeSearchConfig,
    transport_config: CurvatureTransportConfig | None = None,
) -> OperatorTransportCertificate:
    """Correct one prediction; fall back to the reference law on any failed gate."""

    if prediction.coefficients.shape[0] != 1:
        raise ValueError("operator correction accepts one task at a time")
    if prediction.coefficients.shape[2] != action.basis.rank:
        raise ValueError("operator prediction and action ranks differ")
    supplied = tuple(row.detach().clone() for row in prediction.coefficients[0])
    try:
        corrected = find_rbergomi_conditional_modes(
            action,
            config=mode_search,
            supplied_starts=supplied,
        )
        tolerance = 10.0 * mode_search.solver.gradient_tolerance
        valid = tuple(
            mode
            for mode in corrected.modes
            if mode.gradient_norm <= tolerance
            and mode.hessian_eigenvalues is not None
            and float(torch.min(mode.hessian_eigenvalues))
            > (transport_config or CurvatureTransportConfig()).positive_curvature_tolerance
        )
        if not valid:
            raise ValueError("corrector did not certify a positive-curvature stationary mode")
        filtered = RBergomiModeSearchResult(
            modes=valid,
            raw=corrected.raw,
        )
        proposal = build_curvature_transport(
            action,
            filtered,
            config=transport_config,
        )
        return OperatorTransportCertificate(
            proposal=proposal,
            corrected_modes=filtered,
            used_natural_fallback=False,
            maximum_gradient_norm=max(mode.gradient_norm for mode in valid),
            exact_likelihood=True,
            estimator_unbiased_when_frozen=True,
        )
    except (FloatingPointError, RuntimeError, ValueError):
        natural = FiniteRankGaussianComponent.natural(action.basis.dimension)
        proposal = DefensiveFiniteRankGaussianMixture(
            components=(natural,),
            weights=torch.ones(1, dtype=torch.float64),
        )
        return OperatorTransportCertificate(
            proposal=proposal,
            corrected_modes=None,
            used_natural_fallback=True,
            maximum_gradient_norm=None,
            exact_likelihood=True,
            estimator_unbiased_when_frozen=True,
        )
