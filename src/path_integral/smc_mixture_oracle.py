"""Independent one-dimensional references for the conditional-risk SMC bridge."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.integrate import quad

from .conditional_risk_target_smc import RiskEvaluation
from .gaussian_span_marginalization import GaussianMixtureShiftSpec


def mixture_spec() -> GaussianMixtureShiftSpec:
    return GaussianMixtureShiftSpec(torch.tensor([[0.], [2.]], dtype=torch.float64),
        torch.tensor([.2, .8], dtype=torch.float64))


def mixture_evaluation(x: torch.Tensor) -> RiskEvaluation:
    if (x.ndim != 2 or x.shape[1] != 1 or x.dtype != torch.float64
            or x.device.type != "cpu" or x.requires_grad or not torch.isfinite(x).all()):
        raise ValueError("detached finite CPU float64 X1 required")
    dq = torch.logaddexp(torch.full_like(x[:, 0], math.log(.2)), math.log(.8)+2*x[:, 0]-2)
    return RiskEvaluation(-.4*(x[:, 0]-1).square()-.3, dq, x.clone(), 0.)


def scalar_log_target(x: float, beta: float, *, wrong_single_Dq: bool = False) -> float:
    if not math.isfinite(x) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("finite scalar and beta in [0,1] required")
    log_p = -.5*x*x-.5*math.log(2*math.pi)
    terms = (math.log(.2)-.5*x*x, math.log(.8)-.5*(x-2)**2)
    maximum = max(terms)
    log_q = maximum+math.log(math.fsum(math.exp(t-maximum) for t in terms))-.5*math.log(2*math.pi)
    log_m2 = -.4*(x-1)**2-.3
    power = 1 if wrong_single_Dq else 2
    return (1-power*beta)*log_q+power*beta*log_p+beta*log_m2


def reference_moments(beta: float, *, wrong_single_Dq: bool = False) -> dict[str, Any]:
    scalar_log_target(0., beta, wrong_single_Dq=wrong_single_Dq)
    evaluations = 0

    def integrand(x: float, power: int) -> float:
        nonlocal evaluations
        evaluations += 1
        return math.exp(scalar_log_target(x, beta, wrong_single_Dq=wrong_single_Dq))*x**power

    adaptive, errors = [], []
    for power in range(3):
        halves = [quad(integrand, a, b, args=(power,), epsabs=1e-12, epsrel=1e-12, limit=200)
                  for a, b in ((-math.inf, 0.), (0., math.inf))]
        adaptive.append(math.fsum(v[0] for v in halves))
        errors.append(math.fsum(v[1] for v in halves))
        if power == 0:
            right_raw = halves[1][0]
    finite = []
    for order in (256, 512):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        raw = [0., 0., 0.]
        right = 0.
        for left, right_edge in ((-12., 0.), (0., 12.)):
            points = (left+right_edge)/2+(right_edge-left)/2*nodes
            values = np.array([math.exp(scalar_log_target(float(x), beta, wrong_single_Dq=wrong_single_Dq)) for x in points])
            evaluations += order
            masses = weights*values*(right_edge-left)/2
            raw = [v+float(np.dot(masses, points**power)) for power, v in enumerate(raw)]
            if left == 0:
                right = float(masses.sum())
        finite.append(raw+[right])
    exact_raw = adaptive+[right_raw]
    disagreement = max(abs(a-b) for row in finite for a, b in zip(row, exact_raw, strict=True))
    z, first, second = adaptive
    if z <= 0 or not all(math.isfinite(x) for x in exact_raw):
        raise FloatingPointError("finite positive oracle normalizer required")
    return {"beta": beta, "wrong_single_Dq": wrong_single_Dq, "normalizer": z,
        "mean": first/z, "second_moment": second/z, "right_mass": right_raw/z,
        "raw_moments_and_right_mass": exact_raw, "adaptive_error_proxies": errors,
        "maximum_independent_raw_disagreement": disagreement,
        "reference_gate": disagreement <= 1e-10 and max(errors) <= 1e-10,
        "scalar_reference_evaluations": evaluations,
        "not_outward_rounded_or_arbitrary_dimensional_certificate": True}


def wrong_endpoint_closed_form() -> dict[str, float]:
    variance, mean = 1/1.8, .8/1.8
    return {"normalizer": math.exp(-.3-.4/1.8)/math.sqrt(1.8), "mean": mean,
        "second_moment": variance+mean**2,
        "right_mass": .5*math.erfc(-mean/math.sqrt(2*variance))}


def reference_suite() -> dict[str, Any]:
    rows = [reference_moments(beta) for beta in (0., .25, .5, .75, 1.)]
    wrong = reference_moments(1., wrong_single_Dq=True)
    closed = wrong_endpoint_closed_form()
    wrong_error = max(abs(wrong[k]-v) for k, v in closed.items())
    initial_error = max(abs(rows[0][k]-v) for k, v in {"normalizer": 1., "mean": 1.6, "second_moment": 4.2}.items())
    separation = abs(closed["normalizer"]/rows[-1]["normalizer"]-1)
    t = 12.
    phi, tail = math.exp(-t*t/2)/math.sqrt(2*math.pi), .5*math.erfc(t/math.sqrt(2))
    return {"rows": rows, "wrong_endpoint": wrong, "wrong_closed_form": closed,
        "wrong_closed_form_error": wrong_error, "initial_analytic_error": initial_error,
        "wrong_endpoint_relative_normalizer_separation": separation,
        "endpoint_raw_absolute_tail_bounds_real_arithmetic": [10*tail, 10*phi, 10*(t*phi+tail)],
        "pass": all(r["reference_gate"] for r in rows+[wrong]) and wrong_error <= 1e-10
            and initial_error <= 1e-10 and separation >= .1,
        "scalar_reference_evaluations": sum(r["scalar_reference_evaluations"] for r in rows+[wrong])}


def final_moments(result: dict[str, Any]) -> dict[str, float]:
    x = torch.tensor(result["final_x"], dtype=torch.float64)[:, 0]
    weights = torch.tensor(result["final_weights"], dtype=torch.float64)
    return {"mean": float(weights @ x), "second_moment": float(weights @ x.square()),
        "right_mass": float(weights @ (x > 0).to(torch.float64))}


def oracle_summary(records: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    if len(records) != 16 or {r["replicate"] for r in records} != set(range(16)):
        raise ValueError("sixteen complete independently seeded oracle islands required")
    if (not math.isfinite(reference["normalizer"]) or reference["normalizer"] <= 0
            or any(not math.isfinite(r["log_normalizer"]) or any(not math.isfinite(v)
                for v in r["moments"].values()) for r in records)):
        raise ValueError("finite oracle statistics and positive reference required")
    logs = torch.tensor([r["log_normalizer"] for r in records], dtype=torch.float64)
    values = logs.exp()
    estimate, se = float(values.mean()), float(values.std(unbiased=True)/4)
    truth = reference["normalizer"]
    rse = se/estimate
    moments = {k: math.fsum(r["moments"][k] for r in records)/16 for k in ("mean", "second_moment", "right_mass")}
    errors = {k: abs(v-reference[k]) for k, v in moments.items()}
    checks = {"normalizer_rse": rse <= .025, "normalizer_relative_error": abs(estimate/truth-1) <= .04,
        "normalizer_four_SE": abs(estimate-truth) <= 4*se+1e-4*truth,
        "target_mean": errors["mean"] <= .06, "target_second": errors["second_moment"] <= .10,
        "target_right_mass": errors["right_mass"] <= .04}
    return {"normalizer": estimate, "normalizer_se": se, "rse": rse,
        "relative_normalizer_error": abs(estimate/truth-1), "moments": moments,
        "absolute_moment_errors": errors, "checks": checks, "pass": all(checks.values()),
        "finite_sample_diagnostic_not_unbiased_normalized_particle_mean_or_confirmation": True}
