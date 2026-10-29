"""FR2 bank/family contrasts using existing exact shift-mixture primitives."""

from __future__ import annotations

import math
from typing import Any

import torch
from scipy.stats import norm

from .cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from .finite_grid_geometry_protocol import EvaluationBudget, GeometryProtocol, GeometryTask, Role
from .finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from .physical_window_guide import PhysicalWindowGuide
from .proposal_family_diagnostics import refit_identity_mixture
from .research_result_contract import Moments, canonical_digest
from .volterra_conditional_payoffs import evaluate_lognormal_terminal_payoffs
from .weighted_bank_mixture import fit_weighted_bank_mixture

ARMS = ("parent", "static", "ce-preserve", "ce-compress", "static-preserve", "static-compress")
CONTRASTS = (("ce-preserve", "ce-compress"), ("static-preserve", "static-compress"),
             ("ce-preserve", "static-preserve"), ("ce-compress", "static-compress"))
PER_WHOLE = 3*2048+2*4096+6*32768+2*65536


def identity_mixture(means: torch.Tensor, weights: torch.Tensor) -> DefensiveFiniteRankGaussianMixture:
    d = means.shape[1]
    return DefensiveFiniteRankGaussianMixture(tuple(
        FiniteRankGaussianComponent(m.clone(), torch.empty(d, 0, dtype=torch.float64),
                                    torch.empty(0, dtype=torch.float64)) for m in means), weights.clone())


def log_payoff(task: GeometryTask, cache: CachedVolterraTerminal, z: torch.Tensor) -> torch.Tensor:
    if cache.problem != task.problem():
        raise ValueError("mechanism payoff binding changed")
    if task.payoff == "digital":
        return cache.log_probability(z)
    mean, integrated, _ = cache.quantities(z)
    strike = task.spot*task.strike_ratio
    return evaluate_lognormal_terminal_payoffs(
        mean, (1-task.rho**2)*integrated, strike=strike).log_put_value-math.log(strike)


def stratified_moments(strata: list[torch.Tensor]) -> dict[str, Any]:
    """Equal known masses; rows are IID within, not across, sampling strata."""
    if (not strata or any(x.ndim != 2 or len(x) < 2 or x.dtype != torch.float64
                          or not torch.isfinite(x).all() for x in strata)
            or len({x.shape[1] for x in strata}) != 1):
        raise ValueError("finite same-column independent strata required")
    weight = 1/len(strata)
    mean = sum((weight*x.mean(0) for x in strata), torch.zeros(strata[0].shape[1], dtype=torch.float64))
    covariance = sum((weight**2*torch.cov(x.T)/len(x) for x in strata),
                     torch.zeros((len(mean), len(mean)), dtype=torch.float64))
    se = covariance.diag().clamp_min(0).sqrt()
    return {"mean": mean.tolist(), "covariance_of_mean": covariance.tolist(), "se": se.tolist(),
            "rse": (se/mean).tolist(), "counts": [len(x) for x in strata],
            "known_stratum_masses": [weight]*len(strata)}


def diagnostic_contrasts(h: dict[str, Any], p: dict[str, Any]) -> list[dict[str, Any]]:
    """Exploratory family-adjusted normal diagnostics, not tail-coverage proofs."""
    critical = float(norm.ppf(1-.05/(2*len(CONTRASTS)*6*2)))
    output = []
    for candidate, baseline in CONTRASTS:
        a, b = ARMS.index(candidate), ARMS.index(baseline)
        checks = []
        for ref in (h, p):
            mean = ref["mean"]
            cov = ref["covariance_of_mean"]
            delta = .9*mean[b]-mean[a]
            se = math.sqrt(max(0., .81*cov[b][b]+cov[a][a]-1.8*cov[a][b]))
            checks.append({"ten_percent_reduction_lower": delta-critical*se,
                           "point_reduction": 1-mean[a]/mean[b],
                           "precision": max(ref["rse"][a], ref["rse"][b]) <= .10})
        agreement = True
        for i in (a, b):
            radius = critical*math.hypot(h["se"][i], p["se"][i])
            agreement &= abs(h["mean"][i]-p["mean"][i])+radius <= .2*min(h["mean"][i], p["mean"][i])
        output.append({"candidate": candidate, "baseline": baseline, "references": checks,
                       "cross_reference_agreement": agreement,
                       "signal": agreement and all(v["precision"] and v["ten_percent_reduction_lower"] > 0 for v in checks),
                       "not_population_dominance_proof": True})
    return output


def whole(protocol: GeometryProtocol, task_id: str, rep: int,
          budget: EvaluationBudget) -> dict[str, Any]:
    task = protocol.authorize(task_id, "FR2", "training")
    cache = CachedVolterraTerminal.create(task.problem())
    d = 2*task.steps
    seeds: dict[str, int] = {}

    def seed(role: Role, stream: str) -> int:
        value = protocol.seed(task_id, role, rep, "bank-family-"+stream)
        if value in seeds.values():
            raise ValueError("mechanism stream reuse")
        seeds[role+":"+stream] = value
        return value

    def draw(q: DefensiveFiniteRankGaussianMixture, n: int, role: Role, stream: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        budget.reserve(n)
        sample = q.sample(n, path_seed=seed(role, stream+"-path"), label_seed=seed(role, stream+"-label"))
        g = log_payoff(task, cache, sample.samples)
        budget.check_time()
        return sample.samples, g, sample.log_q_over_p

    q0 = identity_mixture(torch.zeros(1, d, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    parent_ess = []
    for iteration in range(3):
        z, g, density = draw(q0, 2048, "training", f"parent-{iteration}")
        weights = torch.softmax(g-density, 0)
        parent_ess.append(float(1/weights.square().sum()))
        q0 = fit_weighted_bank_mixture(z, weights, clusters=2, covariance_rank=0).proposal
    if len(q0.components) != 3:
        raise ValueError("K2 parent was not retained; family contrast would be confounded")
    guide = PhysicalWindowGuide.create(task.problem(), kind="mixed-window",
                                      amplitudes=[2., 4., 6.], price_amplitudes=[0., 1., 2., 3.])
    static = identity_mixture(guide.spec.means, guide.spec.weights)
    learned_mean = sum((q0.weights[i]*c.mean for i, c in enumerate(q0.components) if i > 0),
                       torch.zeros(d, dtype=torch.float64))/.9
    compressed = identity_mixture(torch.stack((torch.zeros(d, dtype=torch.float64), learned_mean)),
                                  torch.tensor([.1, .9], dtype=torch.float64))
    proposals = {"parent": q0, "static": static}
    training = []
    for bank_name, bank in (("ce", q0), ("static", static)):
        z, g, density = draw(bank, 4096, "training", bank_name+"-bank")
        weights = torch.softmax(g-density, 0)
        row: dict[str, Any] = {"bank": bank_name, "event_weight_ess": float(1/weights.square().sum()), "fits": {}}
        for family, initial in (("preserve", q0), ("compress", compressed)):
            fitted, diagnostics = refit_identity_mixture(initial, z, g, -density, steps=40)
            proposals[f"{bank_name}-{family}"] = fitted
            row["fits"][family] = diagnostics
        training.append(row)
    densities = {name: CachedShiftDensity.create(q) for name, q in proposals.items()}
    finals = []
    for name in ARMS:
        _, g, density = draw(proposals[name], 32768, "final", name)
        values = (g-density).exp()
        moments = Moments.from_values(values)
        finals.append({"arm": name, "mean": moments.mean, "se": moments.standard_error,
                       "rse": moments.standard_error/moments.mean,
                       "precision_5percent": moments.standard_error <= .05*moments.mean,
                       "components": len(proposals[name].components),
                       "digest": canonical_digest({"means": [c.mean.tolist() for c in proposals[name].components],
                                                    "weights": proposals[name].weights.tolist()})})

    def bank_scores(bank: DefensiveFiniteRankGaussianMixture, count: int, design: str,
                    stream: str) -> torch.Tensor:
        z, g, _ = draw(bank, count, "heldout", stream)
        rows = []
        for begin in range(0, count, 4096):
            chunk, log_g = z[begin:begin+4096], g[begin:begin+4096]
            log_q = torch.stack([densities[a].log_q_over_p(chunk) for a in ARMS], 1)
            log_gbank = (torch.logaddexp(log_q[:, 0], log_q[:, 1])-math.log(2)
                         if design == "H" else torch.zeros(len(chunk), dtype=torch.float64))
            scores = (2*log_g[:, None]-log_gbank[:, None]-log_q).exp()
            rows.append(torch.cat((scores, (log_g-log_gbank).exp()[:, None]), 1))
        return torch.cat(rows)

    h = stratified_moments([bank_scores(q0, 32768, "H", "H-parent"),
                            bank_scores(static, 32768, "H", "H-static")])
    natural = identity_mixture(torch.zeros(1, d, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    p = stratified_moments([bank_scores(natural, 65536, "p", "P")])
    return {"task_id": task_id, "whole": rep, "estimand_digest": task.estimand_digest,
            "seeds": seeds, "parent_ess": parent_ess, "training": training, "finals": finals,
            "risk_H": h, "risk_P": p, "contrasts": diagnostic_contrasts(h, p),
            "event_reference_column": len(ARMS), "not_confirmation": True}
