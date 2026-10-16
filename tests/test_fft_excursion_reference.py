from dataclasses import replace

import pytest
import torch

from src.path_integral.adjacent_latent_adapter import AdjacentLatentAdapter
from src.path_integral.conditional_excursion_reference import ConditionalExcursionReference
from src.path_integral.coverage_envelope_reference import CoverageEnvelopeReference
from src.path_integral.fft_excursion_reference import (
    AdjacentBlockLatentAdapter,
    FFTCoverageEnvelopeReference,
    FFTExcursionReference,
)
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("steps", [16, 32, 64])
def test_fft_excursion_matches_dense_geometry_and_density(steps):
    p = replace(problem(), steps=steps)
    fast, dense = FFTExcursionReference.create(p), ConditionalExcursionReference.create(p)
    torch.testing.assert_close(fast.reference.units, dense.units, rtol=1e-12, atol=1e-12)
    z = dense.sample(4096, path_seed=3101, label_seed=3102)
    torch.testing.assert_close(fast.log_q_over_p(z), dense.log_q_over_p(z), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(fast.sample(512, path_seed=3103, label_seed=3104),
                               dense.sample(512, path_seed=3103, label_seed=3104), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("steps", [16, 32, 64])
def test_block_adjacent_adapter_matches_dense(steps):
    p = replace(problem(), steps=steps)
    fast, dense = AdjacentBlockLatentAdapter.create(p), AdjacentLatentAdapter.create(p)
    generator = torch.Generator().manual_seed(3105)
    z = torch.randn((512, 2*steps), dtype=torch.float64, generator=generator)
    eps = torch.randn((512, steps//2), dtype=torch.float64, generator=generator)
    torch.testing.assert_close(fast.coarse(z, eps), dense.coarse(z, eps), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("steps", [16, 32, 64])
def test_fft_envelope_matches_dense_geometry_density_and_sampler(steps):
    p = replace(problem(), steps=steps)
    fast, dense = FFTCoverageEnvelopeReference.create(p), CoverageEnvelopeReference.create(p)
    torch.testing.assert_close(fast.envelope.units, dense.units, rtol=1e-12, atol=1e-12)
    z = dense.sample(4096, path_seed=3111, label_seed=3112)
    torch.testing.assert_close(fast.log_q_over_p(z), dense.log_q_over_p(z), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(fast.sample(512, path_seed=3113, label_seed=3114),
                               dense.sample(512, path_seed=3113, label_seed=3114), rtol=1e-12, atol=1e-12)
