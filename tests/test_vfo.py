"""Causality and branch-contract tests for the VFO architecture."""

from __future__ import annotations

import torch

from src.path_integral.controllers import VolterraFollmerOperator
from src.path_integral.memory import SOEKernelBank, fit_positive_soe_kernel
from src.physics_engine import RBergomiSimulator


def _vfo() -> VolterraFollmerOperator:
    torch.manual_seed(3101)
    return VolterraFollmerOperator(
        H=0.1,
        rho=-0.7,
        eta=1.2,
        xi=0.04,
        maturity=0.25,
        barrier=80.0,
        minimum_dt=1.0 / 64.0,
        soe_terms=6,
        hidden_dim=16,
        residual_dim=8,
        control_bound=(6.0, 6.0),
    ).double()


def test_positive_soe_fit_and_update_contract() -> None:
    fit = fit_positive_soe_kernel(
        H=0.1,
        minimum_lag=1.0 / 128.0,
        maximum_lag=1.0,
        terms=10,
    )
    assert torch.all(fit.rates > 0.0)
    assert torch.all(fit.weights >= 0.0)
    assert fit.relative_l2_error < 0.20
    assert fit.maximum_relative_error < 0.50

    bank = SOEKernelBank(
        H=0.1, minimum_lag=1.0 / 64.0, maximum_lag=0.25, terms=6
    ).double()
    state = bank.initial_state(3, device=torch.device("cpu"), dtype=torch.float64)
    increment = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    updated = bank.update(state, increment, 1.0 / 64.0)
    rates = bank.rates
    expected_gain = -torch.expm1(-rates / 64.0) / (rates / 64.0)
    assert torch.allclose(updated, increment[:, None] * expected_gain, atol=1e-14)


def test_vfo_starts_at_null_control_with_closed_memory_gates() -> None:
    control = _vfo()
    control.reset_for_simulation(
        batch_size=4, device=torch.device("cpu"), dtype=torch.float64
    )
    output = control(
        0.0,
        torch.full((4,), 100.0, dtype=torch.float64),
        torch.full((4,), 0.04, dtype=torch.float64),
        torch.zeros(4, dtype=torch.float64),
    )
    diagnostics = control.branch_diagnostics()
    assert torch.equal(output, torch.zeros_like(output))
    assert diagnostics.structural_gate == 0.0
    assert diagnostics.residual_gate == 0.0
    assert diagnostics.residual_energy_fraction == 0.0


def test_vfo_stage_freeze_contract() -> None:
    control = _vfo()
    control.set_stage("structural")
    assert all(parameter.requires_grad for parameter in control.instantaneous.parameters())
    assert all(parameter.requires_grad for parameter in control.structural.parameters())
    assert not any(parameter.requires_grad for parameter in control.residual_cell.parameters())
    assert control.structural_gate_parameter.requires_grad
    assert not control.residual_gate_parameter.requires_grad

    control.set_stage("residual")
    assert not any(parameter.requires_grad for parameter in control.instantaneous.parameters())
    assert not any(parameter.requires_grad for parameter in control.structural.parameters())
    assert all(parameter.requires_grad for parameter in control.residual_cell.parameters())
    assert all(parameter.requires_grad for parameter in control.residual_head.parameters())
    assert control.residual_gate_parameter.requires_grad

    control.set_stage("joint")
    assert all(parameter.requires_grad for parameter in control.parameters())


def _replay_controls(
    control: VolterraFollmerOperator,
    target_driver_one: torch.Tensor,
) -> torch.Tensor:
    batch, steps = target_driver_one.shape
    control.reset_for_simulation(
        batch_size=batch, device=target_driver_one.device, dtype=target_driver_one.dtype
    )
    controls = []
    for step in range(steps):
        controls.append(
            control(
                step / 64.0,
                torch.full((batch,), 100.0, dtype=target_driver_one.dtype),
                torch.full((batch,), 0.04, dtype=target_driver_one.dtype),
                torch.zeros(batch, dtype=target_driver_one.dtype),
            )
        )
        control.observe_target_increment(target_driver_one[:, step], 1.0 / 64.0)
    return torch.stack(controls, dim=1)


def test_suffix_perturbation_cannot_change_earlier_controls() -> None:
    control = _vfo()
    control.set_stage("structural")
    with torch.no_grad():
        control.structural_gate_parameter.fill_(0.8)
    base = torch.zeros(2, 8, dtype=torch.float64)
    changed = base.clone()
    changed[:, 4:] = 0.5
    base_controls = _replay_controls(control, base)
    changed_controls = _replay_controls(control, changed)
    # Increment 4 is observed after control 4; the first possible change is 5.
    assert torch.equal(base_controls[:, :5], changed_controls[:, :5])
    assert not torch.equal(base_controls[:, 5:], changed_controls[:, 5:])


def test_rbergomi_simulator_resets_vfo_memory_between_batches() -> None:
    simulator = RBergomiSimulator(
        H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu"
    )
    control = _vfo()
    control.set_stage("structural")
    with torch.no_grad():
        control.structural_gate_parameter.fill_(0.5)
    torch.manual_seed(3102)
    first = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 32.0,
        num_paths=32,
        control_fn=control,
        record_augmented=True,
    )
    torch.manual_seed(3102)
    second = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 32.0,
        num_paths=32,
        control_fn=control,
        record_augmented=True,
    )
    assert torch.equal(first.spot, second.spot)
    assert torch.equal(first.variance, second.variance)
    assert torch.equal(first.controls, second.controls)


def test_vfo_memory_update_uses_target_not_proposal_increment() -> None:
    simulator = RBergomiSimulator(
        H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu"
    )
    control = _vfo()
    control.set_stage("instant")
    # Give B0 a known nonzero constant through the final bias.
    with torch.no_grad():
        control.instantaneous[-1].bias.copy_(
            torch.tensor([-0.1, 0.05], dtype=torch.float64)
        )
    torch.manual_seed(3103)
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.05,
        dt=0.025,
        num_paths=5,
        control_fn=control,
        record_augmented=True,
    )
    assert result.target_brownian_increments is not None
    expected = control.soe_bank.initial_state(
        5, device=torch.device("cpu"), dtype=torch.float64
    )
    for step in range(result.target_brownian_increments.shape[1]):
        expected = control.soe_bank.update(
            expected, result.target_brownian_increments[:, step, 0], result.step_dt
        )
    assert control._soe_state is not None
    assert torch.allclose(control._soe_state, expected, atol=1e-14)
