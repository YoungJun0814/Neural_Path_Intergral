import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.cameron_martin_modes import ActionSolverConfig, ModeSearchConfig
from src.path_integral.conditional_transport_adaptation import (
    ConditionalTransportAdaptationConfig,
)
from src.path_integral.finite_rank_gaussian_transport import CurvatureTransportConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import (
    evaluate_rbergomi_cm_transport,
    train_rbergomi_cm_transport,
)


def test_conditional_adaptation_preserves_exact_mixture_contract() -> None:
    problem = RBergomiBaselineProblem(
        task_id="adaptation-test",
        task=TerminalThresholdTask(level=60.0),
        spot=100.0,
        maturity=1.0,
        steps=4,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )
    basis = build_mesh_compatible_blp_hybrid_basis(
        steps=4,
        maturity=1.0,
        hurst=0.12,
        drift_modes=3,
        bridge_modes=1,
    )
    trained = train_rbergomi_cm_transport(
        problem,
        basis=basis,
        mode_search=ModeSearchConfig(
            methods=("lbfgs",),
            random_starts=1,
            random_seed=900,
            start_scale=1.0,
            solver=ActionSolverConfig(maximum_iterations=80, gradient_tolerance=1e-6),
        ),
        transport_config=CurvatureTransportConfig(
            defensive_mass=0.15,
            asymptotic_safety_mass=0.02,
            safety_spectrum_decay=2.0,
            safety_spectrum_scale=4.0,
        ),
        adaptation_config=ConditionalTransportAdaptationConfig(
            iterations=2,
            samples_per_iteration=512,
            minimum_ess_fraction=0.1,
            final_components=3,
        ),
        adaptation_seed=901,
    )
    assert len(trained.proposal.components) == 5
    assert trained.adaptation_effective_sample_sizes
    assert len(trained.adaptation_effective_sample_sizes) == 3
    assert all(value >= 51.2 - 1e-8 for value in trained.adaptation_effective_sample_sizes)
    evaluated = evaluate_rbergomi_cm_transport(
        problem,
        trained.proposal,
        sample_count=20_000,
        path_seed=902,
        label_seed=903,
    )
    assert evaluated.maximum_likelihood_bound_violation == 0.0
    assert abs(float(torch.mean(evaluated.likelihood)) - 1.0) < 0.06
