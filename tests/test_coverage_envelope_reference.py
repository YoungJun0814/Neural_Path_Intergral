import math

import pytest
import torch

from src.path_integral.collective_variance_reference import CollectiveVarianceReference
from src.path_integral.conditional_excursion_reference import ConditionalExcursionReference
from src.path_integral.coverage_envelope_reference import CoverageEnvelopeReference
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def reference():
    units, indices = torch.tensor([[1., 0.]], dtype=torch.float64), torch.tensor([1])
    return CoverageEnvelopeReference(ConditionalExcursionReference(units, indices, thresholds=(1., 2.)),
                                     CollectiveVarianceReference(units, indices))


def test_envelope_density_equals_independent_closed_formula():
    r = reference()
    z = torch.tensor([[0., 0.], [1.5, 0.], [2.5, 2.], [-3., 4.]], dtype=torch.float64)
    y, price = z[:, 0], z[:, 1]
    excursion = torch.full((len(z),), .1, dtype=torch.float64)
    for t in (1., 2.):
        tail = .5*math.erfc(t/math.sqrt(2))
        tail_price = .5*math.erfc(1/math.sqrt(2))
        excursion += .9/4*(y >= t).to(torch.float64)/tail
        excursion += .9/4*((y >= t) & (price >= 1)).to(torch.float64)/(tail*tail_price)
    collective = .1+.9/25*torch.exp(.48*z.square().sum(1))
    radial = .1+.9/2*torch.exp(.25*z.square().sum(1))
    torch.testing.assert_close(r.log_q_over_p(z), (.4*excursion+.4*collective+.2*radial).log())


def test_envelope_corrected_mass_and_bounded_oracle():
    r = reference()
    z = r.sample(131072, path_seed=6271, label_seed=6272)
    d = r.log_q_over_p(z)
    assert bool((d >= math.log(.1)-1e-12).all())
    weights = torch.exp(-d)
    assert abs(float(weights.mean())-1) < .025
    oracle = GaussianBumpOracle((1.,), ((1., 2.),), (1.,))
    contribution = torch.exp(oracle.log_value(z)-d)
    mean, se = float(contribution.mean()), float(contribution.std(unbiased=True)/math.sqrt(len(z)))
    assert abs(mean-math.exp(oracle.log_endpoint(1))) < max(.005, 4*se)
    torch.testing.assert_close(z, r.sample(131072, path_seed=6271, label_seed=6272), rtol=0, atol=0)


def test_moderate_excursion_is_not_only_defensive_density():
    units, indices = torch.tensor([[1., 0.]], dtype=torch.float64), torch.tensor([1])
    r = CoverageEnvelopeReference(ConditionalExcursionReference(units, indices, thresholds=(2., 3., 4., 5., 6.)),
                                  CollectiveVarianceReference(units, indices))
    x = torch.tensor([[3.71754, 1.069897]], dtype=torch.float64)
    old = ConditionalExcursionReference(units, indices)
    assert float(old.log_q_over_p(x)) == pytest.approx(math.log(.1))
    assert float(r.log_q_over_p(x)) > float(old.log_q_over_p(x))+4


def test_radial_sampling_scale_is_float64_exact_replay():
    r, count, path_seed, label_seed = reference(), 8192, 6431, 6432
    generator = torch.Generator().manual_seed(label_seed)
    labels = torch.multinomial(torch.tensor([.4, .4, .2], dtype=torch.float64), count,
                               replacement=True, generator=generator)
    child_labels = torch.randint(1, 2**62, (3,), generator=generator).tolist()
    child_paths = torch.randint(1, 2**62, (3,), generator=torch.Generator().manual_seed(path_seed)).tolist()
    rows = torch.arange(count)[labels == 2]
    z = torch.randn((len(rows), r.dimension), dtype=torch.float64,
                    generator=torch.Generator().manual_seed(child_paths[2]))
    inflated = torch.rand(len(rows), dtype=torch.float64,
                          generator=torch.Generator().manual_seed(child_labels[2])) >= .1
    expected = z.clone()
    expected[inflated] *= math.sqrt(2.)
    torch.testing.assert_close(r.sample(count, path_seed=path_seed, label_seed=label_seed)[rows], expected,
                               rtol=0, atol=0)
