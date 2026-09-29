import torch

from src.models.volterra_transport_operator import (
    VolterraTransportOperator,
    VolterraTransportOperatorConfig,
    correct_and_build_operator_transport,
    encode_rbergomi_transport_task,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis
from src.path_integral.cameron_martin_modes import ActionSolverConfig, ModeSearchConfig
from src.path_integral.finite_rank_gaussian_transport import CurvatureTransportConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.volterra_action import RBergomiConditionalAction
from src.training.volterra_transport_operator import (
    VolterraOperatorTeachers,
    VolterraOperatorTrainingConfig,
    train_volterra_transport_operator,
)


def test_operator_outputs_obey_structure_contract() -> None:
    config = VolterraTransportOperatorConfig(
        feature_dimension=7,
        modes=3,
        rank=4,
        hidden_features=12,
        maximum_coefficient_norm=2.0,
        minimum_precision_eigenvalue=0.1,
        maximum_precision_eigenvalue=3.0,
    )
    model = VolterraTransportOperator(config)
    prediction = model(torch.randn((5, 7), dtype=torch.float64))
    assert prediction.coefficients.shape == (5, 3, 4)
    assert torch.max(torch.linalg.vector_norm(prediction.coefficients, dim=2)) <= 2.0 + 1e-12
    assert torch.max(torch.abs(torch.sum(prediction.mode_weights, dim=1) - 1.0)) < 2e-15
    assert torch.min(prediction.precision_eigenvalues) >= 0.1 - 1e-12
    assert torch.max(prediction.precision_eigenvalues) <= 3.0 + 1e-12
    symmetry_error = prediction.precision_matrices - prediction.precision_matrices.transpose(-1, -2)
    assert torch.max(torch.abs(symmetry_error)) < 2e-14


def test_teacher_training_reduces_structure_aware_loss() -> None:
    torch.manual_seed(5)
    config = VolterraTransportOperatorConfig(
        feature_dimension=3,
        modes=2,
        rank=2,
        hidden_features=24,
    )
    model = VolterraTransportOperator(config)
    features = torch.randn((12, 3), dtype=torch.float64)
    coefficients = torch.stack(
        (
            features[:, :2],
            -0.5 * features[:, :2],
        ),
        dim=1,
    )
    weights = torch.full((12, 2), 0.5, dtype=torch.float64)
    precision = torch.eye(2, dtype=torch.float64).expand(12, 2, 2, 2).clone()
    teachers = VolterraOperatorTeachers(
        features=features,
        coefficients=coefficients,
        mode_weights=weights,
        precision_matrices=precision,
        mode_mask=torch.ones((12, 2), dtype=torch.bool),
    )
    result = train_volterra_transport_operator(
        model,
        teachers,
        seed=19,
        config=VolterraOperatorTrainingConfig(epochs=250, learning_rate=0.01),
    )
    assert result.final_loss < 0.05 * result.initial_loss


def test_operator_correction_emits_exact_defensive_transport() -> None:
    problem = RBergomiBaselineProblem(
        task_id="v15-operator",
        task=TerminalThresholdTask(level=60.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )
    basis = build_blp_cameron_martin_basis(steps=problem.steps, modes_per_driver=2)
    action = RBergomiConditionalAction(problem=problem, basis=basis)
    model = VolterraTransportOperator(
        VolterraTransportOperatorConfig(
            feature_dimension=7,
            modes=2,
            rank=basis.rank,
            hidden_features=12,
        )
    )
    prediction = model(encode_rbergomi_transport_task(problem, epsilon=1.0).unsqueeze(0))
    certificate = correct_and_build_operator_transport(
        action,
        prediction,
        mode_search=ModeSearchConfig(
            methods=("lbfgs", "trust-ncg"),
            random_starts=1,
            random_seed=221,
            start_scale=1.0,
            solver=ActionSolverConfig(maximum_iterations=100, gradient_tolerance=1e-6),
        ),
        transport_config=CurvatureTransportConfig(defensive_mass=0.2),
    )
    assert certificate.exact_likelihood
    assert certificate.estimator_unbiased_when_frozen
    assert not certificate.used_natural_fallback
    sample = certificate.proposal.sample(2_000, path_seed=445, label_seed=446)
    assert float(torch.max(torch.exp(sample.log_p_over_q))) <= 5.0 + 2e-12
