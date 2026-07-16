"""Finite-grid convention tests for the downside-excursion task."""

from __future__ import annotations

import torch

from src.path_integral import DownsideExcursionTask, TimePiecewiseTwoDriverControl


def _task() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=80.0,
        stress_level=90.0,
        minimum_occupation=0.50,
        hit_scale=2.0,
        occupation_scale=0.10,
    )


def test_right_endpoint_occupation_and_hard_event_are_exact() -> None:
    spot = torch.tensor(
        [
            [100.0, 89.0, 79.0, 95.0, 88.0],
            [100.0, 85.0, 82.0, 95.0, 96.0],
            [100.0, 79.0, 95.0, 96.0, 97.0],
        ],
        dtype=torch.float64,
    )
    running_minimum, occupation, hit = _task().prefix_state(spot, step_dt=0.25)
    assert torch.equal(
        occupation[:, -1], torch.tensor([0.75, 0.50, 0.25], dtype=torch.float64)
    )
    assert torch.equal(running_minimum[:, -1], torch.tensor([79.0, 82.0, 79.0]))
    assert torch.equal(hit[:, -1], torch.tensor([True, False, True]))
    assert torch.equal(_task().hard_event(spot, 0.25), torch.tensor([True, False, False]))


def test_cem_score_nonnegative_set_equals_the_hard_event() -> None:
    spot = torch.tensor(
        [
            [100.0, 90.0, 80.0, 89.0, 95.0],
            [100.0, 90.0, 81.0, 89.0, 79.0],
            [100.0, 90.0, 79.0, 95.0, 96.0],
        ],
        dtype=torch.float64,
    )
    task = _task()
    assert torch.equal(task.score(spot, 0.25) >= 0.0, task.hard_event(spot, 0.25))
    soft = task.soft_payoff(spot, 0.25)
    assert torch.all((soft > 0.0) & (soft < 1.0))


def test_prefix_state_is_causal_under_suffix_perturbations() -> None:
    original = torch.tensor([[100.0, 95.0, 89.0, 79.0, 85.0]], dtype=torch.float64)
    changed = original.clone()
    changed[:, 3:] = torch.tensor([[120.0, 130.0]], dtype=torch.float64)
    original_state = _task().prefix_state(original, 0.25)
    changed_state = _task().prefix_state(changed, 0.25)
    for before, after in zip(original_state, changed_state, strict=True):
        assert torch.equal(before[:, :3], after[:, :3])


def test_piecewise_control_uses_left_endpoint_time_segments() -> None:
    control = TimePiecewiseTwoDriverControl(
        ((1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)),
        maturity=1.0,
    )
    spot = torch.ones(5, dtype=torch.float64)
    times = torch.tensor([0.0, 0.249999, 0.25, 0.75, 1.0], dtype=torch.float64)
    result = control(times, spot, spot, spot)
    expected = torch.tensor(
        [[1.0, -1.0], [1.0, -1.0], [2.0, -2.0], [4.0, -4.0], [4.0, -4.0]],
        dtype=torch.float64,
    )
    assert torch.equal(result, expected)
