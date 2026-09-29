import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import evaluate_rbergomi_cm_transport
from src.path_integral.rbergomi_local_volterra_transport import (
    LocalVolterraTransportTrainingConfig,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig
from src.path_integral.tempered_conditional_smc import TemperedSMCConfig
from src.path_integral.tempered_target_transport import TemperedTargetTransportConfig
from src.path_integral.v14_tempered_hybrid_transport import (
    V14TemperedHybridConfig,
    fit_v14_tempered_hybrid_transport,
)


def test_v14_tempered_hybrid_is_an_exact_frozen_balance_mixture() -> None:
    problem = RBergomiBaselineProblem(
        task_id="hybrid-test",
        task=TerminalThresholdTask(level=50.0),
        spot=100.0,
        maturity=1.0,
        steps=3,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )
    basis = build_mesh_compatible_blp_hybrid_basis(
        steps=3,
        maturity=1.0,
        hurst=0.12,
        drift_modes=2,
        bridge_modes=1,
    )
    trained = fit_v14_tempered_hybrid_transport(
        problem,
        basis,
        target_seed=6001,
        local_seed=6002,
        config=V14TemperedHybridConfig(
            target=TemperedTargetTransportConfig(
                smc=TemperedSMCConfig(
                    particles=64,
                    temperatures=(0.0, 0.25, 1.0),
                    mutation_steps=1,
                    pcn_scale=0.3,
                    replicates=2,
                    seed=6001,
                    retain_final_particles=True,
                ),
                components=2,
                clustering="replicate_kmeans",
            ),
            local=LocalVolterraTransportTrainingConfig(
                target_powers=(0.1,),
                shifted_weights=(1.0,),
                defensive_weight=0.2,
                smc=AdaptiveResidualSMCConfig(
                    particles=64,
                    target_ess_fraction=0.7,
                    pcn_scale=0.25,
                    pcn_sweeps_per_stage=1,
                    maximum_stages=8,
                ),
            ),
            target_mass=0.4,
        ),
    )
    assert trained.training_cost.algorithmic_work_units > 0.0
    assert trained.local_component_count == 2
    evaluated = evaluate_rbergomi_cm_transport(
        problem,
        trained.proposal,
        sample_count=20_000,
        path_seed=6003,
        label_seed=6004,
    )
    assert evaluated.maximum_likelihood_bound_violation == 0.0
    assert torch.isfinite(evaluated.contribution).all()
