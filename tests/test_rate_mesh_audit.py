import torch

from src.path_integral.rate_mesh_audit import (
    omitted_hybrid_drift_gradient_norm,
    omitted_mode_gradient_norm,
    omitted_tail_gradient_norm,
    pad_channel_coefficients,
    pad_hybrid_coefficients,
    pad_nested_coefficients,
)


def test_channel_coefficient_padding_preserves_each_channel_block() -> None:
    values = torch.tensor([1.0, 2.0, -3.0, -4.0], dtype=torch.float64)
    padded = pad_channel_coefficients(values, old_modes=2, new_modes=4)
    torch.testing.assert_close(
        padded,
        torch.tensor([1.0, 2.0, 0.0, 0.0, -3.0, -4.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_omitted_gradient_norm_excludes_retained_channel_modes() -> None:
    gradient = torch.tensor([9.0, 8.0, 3.0, 4.0, -7.0, -6.0, 0.0, 12.0], dtype=torch.float64)
    value = omitted_mode_gradient_norm(
        gradient,
        retained_modes=2,
        expanded_modes=4,
    )
    assert abs(value - 13.0) < 1e-14


def test_nested_padding_and_tail_gradient() -> None:
    coefficients = torch.tensor([1.0, -2.0], dtype=torch.float64)
    torch.testing.assert_close(
        pad_nested_coefficients(coefficients, new_modes=4),
        torch.tensor([1.0, -2.0, 0.0, 0.0], dtype=torch.float64),
    )
    gradient = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert abs(omitted_tail_gradient_norm(gradient, retained_modes=2) - 5.0) < 1e-14


def test_hybrid_padding_preserves_bridge_block_and_omitted_drift_norm() -> None:
    coefficients = torch.tensor([1.0, 2.0, -3.0, -4.0], dtype=torch.float64)
    padded = pad_hybrid_coefficients(
        coefficients,
        old_drift_modes=2,
        new_drift_modes=4,
        bridge_modes=2,
    )
    torch.testing.assert_close(
        padded,
        torch.tensor([1.0, 2.0, 0.0, 0.0, -3.0, -4.0], dtype=torch.float64),
    )
    gradient = torch.tensor([9.0, 8.0, 3.0, 4.0, -7.0, -6.0], dtype=torch.float64)
    value = omitted_hybrid_drift_gradient_norm(
        gradient,
        retained_drift_modes=2,
        expanded_drift_modes=4,
        bridge_modes=2,
    )
    assert abs(value - 5.0) < 1e-14
