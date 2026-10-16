"""Original finite-grid mu and raw M2: exact last-pair cache and unbiased nesting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture
from src.path_integral.gaussian_mixture_marginal import require_shift_mixture
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal


@dataclass(frozen=True)
class LastPairCache:
    a: torch.Tensor
    b: torch.Tensor
    c_squared: torch.Tensor
    log_q_prefix_components: torch.Tensor
    q_pair_means: torch.Tensor

    def log_mu_mean(self) -> torch.Tensor:
        return torch.special.log_ndtr(self.a / torch.sqrt(self.c_squared + self.b.square()))

    def log_inner_risk(self, pair: torch.Tensor) -> torch.Tensor:
        if (pair.ndim != 3 or pair.shape[0] != len(self.a) or pair.shape[2] != 2
                or pair.shape[1] < 1 or pair.dtype != torch.float64 or pair.device.type != "cpu"
                or not torch.isfinite(pair).all()):
            raise ValueError("inner coordinates must have shape (outer,L,2), finite CPU float64")
        log_g = torch.special.log_ndtr((self.a[:, None] - self.b[:, None] * pair[:, :, 0])
                                      / torch.sqrt(self.c_squared[:, None]))
        log_q = torch.logsumexp(self.log_q_prefix_components[:, None, :]
                               + pair @ self.q_pair_means.T, 2)
        values = 2 * log_g - log_q
        if not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite cached original-M2 contribution")
        return values


def last_pair_cache(problem: RBergomiBaselineProblem, prefix: torch.Tensor,
                    q: DefensiveFiniteRankGaussianMixture) -> LastPairCache:
    if not isinstance(problem.task, TerminalThresholdTask):
        raise ValueError("last-pair reference supports terminal downside only, epsilon=1")
    require_shift_mixture(q)
    if q.dimension != problem.local_dimension:
        raise ValueError("q dimension mismatch")
    if (prefix.ndim != 2 or prefix.shape[1] != problem.local_dimension - 2
            or len(prefix) < 1 or prefix.dtype != torch.float64 or prefix.device.type != "cpu"
            or not torch.isfinite(prefix).all()):
        raise ValueError("invalid last-pair prefix")
    full = torch.cat((prefix, torch.zeros((len(prefix), 2), dtype=torch.float64)), 1)
    paths = problem.simulate_local(full)
    integrated = problem.step_dt * paths.variance[:, :-1].sum(1)
    a = math.log(problem.task.level) - paths.log_spot[:, -1]
    b = problem.rho * torch.sqrt(paths.variance[:, -2] * problem.step_dt)
    c_squared = (1 - problem.rho**2) * integrated
    means = torch.stack([component.mean for component in q.components])
    cached_density = prefix @ means[:, :-2].T - .5 * means.square().sum(1) + q.weights.log()
    if not torch.isfinite(a).all() or not torch.isfinite(c_squared).all() or bool((c_squared <= 0).any()):
        raise FloatingPointError("nonfinite last-pair coefficients")
    return LastPairCache(a, b, c_squared, cached_density, means[:, -2:])


def nested_log_means(log_inner: torch.Tensor, log_outer_ratio: torch.Tensor) -> torch.Tensor:
    if (log_inner.ndim != 2 or log_inner.shape[1] < 1 or log_outer_ratio.shape != (len(log_inner),)
            or any(v.dtype != torch.float64 or v.device.type != "cpu" or not torch.isfinite(v).all()
                   for v in (log_inner, log_outer_ratio))):
        raise ValueError("invalid nested log contributions")
    return torch.logsumexp(log_inner, 1) - math.log(log_inner.shape[1]) - log_outer_ratio


def block_log_inner_risk(problem: RBergomiBaselineProblem, outer: torch.Tensor,
                         pair: torch.Tensor, q: DefensiveFiniteRankGaussianMixture,
                         block: int) -> torch.Tensor:
    """Recompute the complete finite-grid future when a nonterminal pair changes."""
    require_shift_mixture(q)
    if (not isinstance(problem.task, TerminalThresholdTask) or q.dimension != problem.local_dimension
            or isinstance(block, bool) or not isinstance(block, int) or not 0 <= block < problem.steps
            or outer.ndim != 2 or outer.shape[1] != problem.local_dimension - 2
            or pair.ndim != 3 or pair.shape[0] != len(outer) or pair.shape[2] != 2
            or pair.shape[1] < 1 or any(v.dtype != torch.float64 or v.device.type != "cpu"
                                       or not torch.isfinite(v).all() for v in (outer, pair))):
        raise ValueError("invalid fixed-block nested reference")
    indices = [i for i in range(problem.local_dimension) if i not in (2 * block, 2 * block + 1)]
    full = torch.empty((len(outer), pair.shape[1], problem.local_dimension), dtype=torch.float64)
    full[:, :, indices] = outer[:, None, :]
    full[:, :, 2 * block:2 * block + 2] = pair
    flat = full.reshape(-1, problem.local_dimension)
    logg = evaluate_rbergomi_conditional_terminal(problem, flat).payoffs.log_left_probability
    return (2 * logg - q.log_q_over_p(flat)).reshape(len(outer), pair.shape[1])
