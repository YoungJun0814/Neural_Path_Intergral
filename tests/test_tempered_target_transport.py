import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import evaluate_rbergomi_cm_transport
from src.path_integral.tempered_conditional_smc import TemperedSMCConfig
from src.path_integral.tempered_target_transport import (
    TemperedTargetTransportConfig,
    fit_tempered_target_transport,
)


def test_tempered_target_fit_preserves_exact_defensive_mixture() -> None:
    problem = RBergomiBaselineProblem(
        task_id="tempered-fit-test",
        task=TerminalThresholdTask(level=50.0),
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
    fitted = fit_tempered_target_transport(
        problem,
        basis,
        config=TemperedTargetTransportConfig(
            smc=TemperedSMCConfig(
                particles=256,
                temperatures=tuple((index / 8) ** 2 for index in range(9)),
                mutation_steps=2,
                pcn_scale=0.35,
                replicates=2,
                seed=445,
                retain_final_particles=True,
            ),
            components=2,
        ),
    )
    assert len(fitted.proposal.components) == 4
    assert fitted.proposal.defensive_mass == 0.15
    assert fitted.fitted_particle_count == 512
    assert fitted.training_cost.algorithmic_work_units > 0.0
    evaluated = evaluate_rbergomi_cm_transport(
        problem,
        fitted.proposal,
        sample_count=10_000,
        path_seed=446,
        label_seed=447,
    )
    assert evaluated.maximum_likelihood_bound_violation == 0.0
    assert torch.isfinite(evaluated.contribution).all()
