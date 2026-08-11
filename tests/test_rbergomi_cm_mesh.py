import math

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_mesh import (
    evaluate_adjacent_conditional_mesh_pair,
    evaluate_dct_cameron_martin_drift,
    run_conditional_mesh_study,
)
from src.path_integral.rbergomi_cm_mlmc import (
    MLMCLevelPilot,
    allocate_pilot_frozen_mlmc,
)
from src.path_integral.rbergomi_residual_mlmc import RBergomiAdjacentResidualProblem


def _template(steps: int = 8) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="v15-mesh",
        task=TerminalThresholdTask(level=75.0),
        spot=100.0,
        maturity=0.5,
        steps=steps,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )


def test_adjacent_full_conditionalization_matches_raw_price_driver_mean() -> None:
    template = _template()
    problem = RBergomiAdjacentResidualProblem(
        task_id=template.task_id,
        task=template.task,
        spot=template.spot,
        maturity=template.maturity,
        fine_steps=8,
        hurst=template.hurst,
        eta=template.eta,
        xi=template.xi,
        rho=template.rho,
    )
    generator = torch.Generator().manual_seed(1102)
    local = torch.randn((20_000, problem.local_dimension), dtype=torch.float64, generator=generator)
    conditional = evaluate_adjacent_conditional_mesh_pair(problem, local)
    price = torch.randn(
        (20_000, problem.fine_steps),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(1103),
    )
    paths = problem.simulate(torch.cat((local, price), dim=1))
    hard_fine = template.task.hard_event_from_log_spot(
        paths.fine.log_spot,
        paths.fine.step_dt,
    ).to(torch.float64)
    difference = float(torch.mean(hard_fine - conditional.fine_probability))
    standard_error = math.sqrt(float(torch.var(hard_fine - conditional.fine_probability)) / 20_000)
    assert abs(difference) < 4.0 * standard_error + 5e-4
    assert torch.var(conditional.fine_probability) <= torch.var(hard_fine) + 1e-12


def test_mesh_study_keeps_bias_and_sampling_error_separate() -> None:
    study = run_conditional_mesh_study(
        _template(),
        steps=(4, 8, 16),
        sample_count=2_000,
        seed=991,
    )
    assert study.steps == (4, 8, 16)
    assert len(study.levels) == 3
    assert len(study.adjacent) == 2
    assert all(item.correction.standard_error > 0.0 for item in study.adjacent)
    assert all(-1.0 <= item.correlation <= 1.0 for item in study.adjacent)
    assert math.isfinite(study.final_correction_signal_to_noise)


def test_dct_coefficients_represent_mesh_consistent_constant_drift() -> None:
    coefficients = torch.tensor([[0.7], [-0.4]], dtype=torch.float64)
    coarse = evaluate_dct_cameron_martin_drift(coefficients, steps=16, maturity=0.5)
    fine = evaluate_dct_cameron_martin_drift(coefficients, steps=64, maturity=0.5)
    assert torch.max(torch.abs(coarse - coarse[0])) < 2e-15
    assert torch.max(torch.abs(fine - fine[0])) < 2e-15
    assert torch.max(torch.abs(coarse[0] - fine[0])) < 2e-15


def test_pilot_frozen_allocation_meets_sampling_budget() -> None:
    pilots = (
        MLMCLevelPilot(level=0, mean=0.03, variance=0.2, work_per_sample=1.0),
        MLMCLevelPilot(level=1, mean=0.005, variance=0.04, work_per_sample=2.0),
        MLMCLevelPilot(level=2, mean=0.001, variance=0.008, work_per_sample=4.0),
    )
    allocation = allocate_pilot_frozen_mlmc(pilots, target_rmse=0.01)
    assert allocation.predicted_sampling_variance <= allocation.sampling_variance_budget
    assert allocation.bias_gate_pass
    assert all(count >= 2 for count in allocation.sample_counts)
