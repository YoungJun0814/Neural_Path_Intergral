import pytest
import torch

from experiments.post_audit_residual_oracle_numerical_review import density_invariance_review
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec


def fixture():
    means = torch.tensor([[0., 0., 0.], [2., 0., -1.]], dtype=torch.float64)
    spec = GaussianMixtureShiftSpec(means, torch.tensor([.1, .9], dtype=torch.float64))
    points = torch.tensor([[.3, u, -.7] for u in (-4., -2., 0., 2., 4.)], dtype=torch.float64)
    return spec, points, [[.1, .9], [.5, .5]]


def test_exact_algebraic_identity_and_dense_roundoff_are_separate():
    spec, points, weights = fixture()
    result = density_invariance_review(spec, points, weights, 1)
    assert result["exact_zero_means_and_fixed_other_coordinates"]
    assert result["maximum_dense_log_ratio_difference"] <= result["float64_diagnostic_tolerance"]
    assert result["number_of_weight_vectors"] == 2
    assert result["not_rigorous_interval_bound"]


def test_nonzero_mean_coordinate_cannot_pass_with_small_numerical_change():
    spec, points, weights = fixture()
    spec.means[1, 1] = 1e-100
    with pytest.raises(ValueError, match="algebraic"):
        density_invariance_review(spec, points, weights, 1)


def test_other_coordinate_changes_are_not_called_blind_coordinate_probes():
    spec, points, weights = fixture()
    points[0, 0] += .01
    with pytest.raises(ValueError, match="algebraic"):
        density_invariance_review(spec, points, weights, 1)


def test_unnormalized_or_incomplete_density_family_rejected():
    spec, points, _ = fixture()
    for weights in ([[.1, .1]], [[1.]], [[0., 1.]]):
        with pytest.raises(ValueError, match="weights"):
            density_invariance_review(spec, points, weights, 1)
