# V14 Exact Tempered Residual Flow Mixture Plan

Date: 2026-08-11

## Falsified V13 mechanism

V13's beta-one flow reaches high-`g` residuals but assigns them likelihood ratios as
small as `1e-9`--`1e-8`. The natural defensive component does not reach the
transition shell often enough at `1e-5`. A single flow therefore leaves a density
gap between natural and over-concentrated target regions.

## V14 proposal

For fixed powers `0 < beta_1 < ... < beta_K <= 1`, independently train exact flows
against densities proportional to `g(r)^beta_k p_R(r)`. Each frozen component is a
defensive exact flow `q_k`. The final proposal is

`q_mix(r) = sum_k omega_k q_k(r)`.

Its exact ordinary likelihood is

`p_R(r)/q_mix(r) = 1 / sum_k omega_k q_k(r)/p_R(r)`.

If component `k` has defensive mass `delta_k`, then

`p_R/q_mix <= 1 / sum_k omega_k delta_k`.

No SMC weight or final estimate is self-normalized. All SMC populations and mixture
weight screening remain training-only.

## Gates

1. exact component and balance-mixture likelihood;
2. likelihood normalization and analytic defensive bound;
3. held-out training coverage across contribution quantiles and component strata;
4. candidate reference accuracy;
5. empirical-Bernstein tail forecast without resource censoring;
6. training-inclusive work against conditional rBergomi, RQMC, and V10R1;
7. qualification only after a frozen development pass.
