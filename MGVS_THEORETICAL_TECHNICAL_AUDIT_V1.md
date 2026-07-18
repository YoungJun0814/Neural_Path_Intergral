# MGVS 이론·기술 감사 보고서 V1

작성일: 2026-07-18
대상: Monotone Gaussian Volterra Smoothing(MGVS), kappa=1 BLP rBergomi 유한격자 추정기

## 1. 감사 결론

현재 구현은 **선언된 유한격자 문제와 결정론적 시간 제어 범위 안에서 이론적으로
정합적**이다. 사건 임계값, Girsanov 우도 분해, 한 차원 Gaussian 적분, fine/coarse
telescoping, Rao--Blackwell 분산 비증가, FFT BLP convolution의 여섯 핵심 명제를 코드와
독립 수치 oracle로 확인했다.

다만 다음 범위를 넘겨 주장하면 오류다.

1. 연속시간 barrier가 아니라 선언된 discrete monitoring grid의 exact estimator다.
2. analytic smoothing은 deterministic time-only control에 한정된다.
3. 분산 비증가는 보장되지만 wall-clock work 또는 MLMC rate 개선은 자동 결과가 아니다.
4. 개발 regime 한 곳의 성공은 frozen multi-regime 논문 증거가 아니다.
5. one-dimensional smoothing 자체는 선행연구가 있어 단독 novelty가 아니다.

현 상태는 **논문 핵심 알고리즘과 초기 falsification gate를 통과한 research prototype**이며
top-journal 제출본은 아니다.

## 2. 정확한 확률모형과 적용 범위

Fine grid `t_i=ih`, `i=0,...,N`에서 simulator의 target BLP law는

\[
X_{i+1}=X_i+(\mu-\tfrac12V_i)h
+\sqrt{V_i}(\rho\Delta W^{1,P}_i+\rho_\perp\Delta W^{2,P}_i),
\qquad \rho_\perp=\sqrt{1-\rho^2}.
\]

`W1`과 recent-cell 보조 적분이 rough Volterra variance를 만들고 `W2`는 가격에만
들어간다. Proposal 아래에서

\[
\Delta W_i^{j,P}=\Delta W_i^{j,Q}+u_{i,j}h
\]

이며 `u_i`는 deterministic schedule이다. 구현의 target-over-proposal likelihood는

\[
L=\frac{dP}{dQ}
=\exp\left[-\sum_i u_i^\top\Delta W_i^Q
-\frac12h\sum_i\lVert u_i\rVert^2\right].
\]

Self-normalization은 사용하지 않는다. 필수 가정은 `0<H<1/2`, `xi>0`, `eta>=0`,
`-1<rho<1`, kappa=1 BLP finite grid, deterministic time-only two-driver control,
`||q||_2=1`과 모든 `q_i>0`, right-endpoint occupation, initial point를 포함하는 hit이다.

## 3. T9-1: affine monotone representation

Proposal price normal을

\[
\xi_i=\Delta W_i^{2,Q}/\sqrt h,\qquad
Z=q^\top\xi,\qquad R=(I-qq^\top)\xi
\]

로 분해한다. `Z~N(0,1)`, `R`과 `Z`는 독립이고 `q^T R=0`이다. Volterra variance는
augmented `W1` randomness와 deterministic `u1`에만 의존하므로 `(W1, BLP local, R)`을
고정하면 `V_i`는 `Z`와 독립이다. 따라서

\[
X_k(Z)=A_k+B_kZ,\qquad
B_k=\rho_\perp\sqrt h\sum_{i<k}\sqrt{V_i}q_i>0\quad(k\ge1).
\]

Adjacent coarse grid에서도

\[
B_j^c=\rho_\perp\sqrt h
\sum_{m<j}\sqrt{V_m^c}(q_{2m}+q_{2m+1})>0.
\]

Fine과 coarse가 같은 fine-coordinate `Z`에서 동시에 단조 증가한다. 비균일 positive
direction을 포함한 pathwise replay test가 통과했다.

## 4. T9-2: hit-plus-occupation의 scalar threshold

`X_k(Z)=A_k+B_kZ`, `B_k>0`이면 `S_k<=K`는

\[
Z\le c_k(K):=(\log K-A_k)/B_k
\]

와 동치다.

- 초기 spot이 hit barrier 이하이면 `a_hit=+infinity`;
- 그 외에는 `a_hit=max_k c_k(K_hit)`;
- 필요한 occupation endpoint 수는
  `r=ceil((tau-1e-15)/h)`;
- `1<=r<=N`이면 `a_occ`는 `c_k(K_stress)`의 r번째 큰 값;
- `r<=0`이면 `a_occ=+infinity`, `r>N`이면 `a_occ=-infinity`.

두 조건은 AND이므로

\[
H=1\{Z\le a\},\qquad a=\min(a_{hit},a_{occ}).
\]

5,000 synthetic affine paths와 실제 rBergomi fine/coarse paths에서 기존
`DownsideExcursionTask.hard_event`와 bitwise 일치했다.

## 5. T9-3: controlled Gaussian integration과 부호

`w_i=sqrt(h)u_{i,2}`를 `w=bq+w_perp`, `b=q^T w`로 분해하면

\[
L=L_\perp\exp(-bZ-\tfrac12b^2),
\]

\[
\log L_\perp
=-\sum_i u_{i,1}\Delta W_i^{1,Q}
-\frac h2\sum_i u_{i,1}^2
-w_\perp^TR-\frac12\lVert w_\perp\rVert^2.
\]

따라서

\[
E_Q[1\{Z\le a\}L\mid W^1,\text{BLP local},R]
=L_\perp\int_{-\infty}^{a}e^{-bz-b^2/2}\phi(z)dz
=L_\perp\Phi(a+b).
\]

부호는 `a-b`가 아니라 `a+b`다. Fine-minus-coarse correction은

\[
L_\perp[\Phi(a_f+b)-\Phi(a_c+b)].
\]

동일 tail CDF의 cancellation을 막기 위해 signed log-CDF/log-survival 구현을 사용한다.
`NaN`, `+/-infinity`, 동일 threshold, swapped arguments, `11 sigma` tail을 테스트했다.

## 6. T9-4: finite-grid telescoping

각 smoothed value는 원래 importance contribution의 조건부 기댓값이므로

\[
E_Q\widetilde H_0=E_PH_0,\qquad
E_Q\widetilde\Delta_l=E_P(H_l-H_{l-1}).
\]

Adjacent coupling이 두 BLP marginal을 정확히 가지므로

\[
E_Q[\widetilde H_0+\sum_{l=1}^L\widetilde\Delta_l]=E_PH_L.
\]

여기서 exact는 finest BLP finite-grid target에 대한 것이다. 연속 rough Bergomi
functional의 discretization bias가 사라진다는 뜻이 아니다.

## 7. T9-5: Rao--Blackwell inequality

원래 contribution을 `Y`, conditioning sigma-field를
`G=sigma(W1, BLP local, R)`라 하면 MGVS는 `Y_tilde=E[Y|G]`다. 따라서

\[
Var(Y)=Var(E[Y|G])+E[Var(Y|G)]\ge Var(Y_tilde).
\]

Signed fine-minus-coarse correction에도 동일하게 적용된다. 이는 population theorem이며
finite sample variance가 매 seed에서 반드시 작다는 명제는 아니다. 개발 실험은 level별
pooled paired bootstrap 95% interval을 사용했다.

## 8. T9-6: FFT BLP marginal preservation

BLP historical weight를 `g_0=0`, `g_r=b_{r+1}`로 쓰면

\[
y_n=\sum_{j=0}^{n}x_jg_{n-j}
\]

인 linear convolution이다. 길이 `2N-1` 이상으로 zero-pad한 FFT는 circular aliasing
없이 같은 합을 계산한다. Recent-cell singular Gaussian integral은 FFT에 넣지 않고 기존
exact BLP local covariance를 유지한다.

Frozen CPU/float64 audit 결과:

- same-innovation maximum path error: `1.1369e-13`;
- fitted log-cost slope: `0.656`;
- one-sided 95% slope upper: `0.772 < 1.35`;
- 1,024 steps reference-loop speedup: `108.4x`;
- batch/chunk replay 통과.

이는 유한한 측정 window의 구현 결과이며 wall-clock exponent theorem이 아니다.

## 9. 구현 후 기술 감사

다음 방어 조건을 코드가 강제한다.

- FFT engine은 `deterministic_schedule(times)` 없는 controller를 실행 전에 거부;
- reference engine도 recorded control이 path별로 다르면 거부;
- `|rho|=1`, nonpositive/non-unit `q`, grid mismatch 거부;
- fine/coarse에 동일 fine `q`와 하나의 exact likelihood 사용;
- raw event와 threshold event가 하나라도 다르면 중단;
- likelihood, affine path, residual projection reconstruction error 기록;
- self-normalized estimator 미제공.

10,000 paths x 5 seeds, levels `16--512`, frozen G0 CEM control의 개발 결과는 다음과 같다.

- geometric raw/smoothed correction work ratio: `1.876`;
- 개선 seed: `5/5`;
- mean smoothed correction variance slope: `0.203`;
- seed-slope one-sided 95% lower: `0.021`;
- 모든 pooled variance-ratio 95% upper: `<0.55`;
- deepest pooled variance ratio: 약 `0.191`;
- deepest raw/smoothed excess kurtosis: 약 `19,163 / 4,055`;
- maximum exactness diagnostic: 약 `1.1e-14`;
- 모든 mean-consistency와 likelihood-normalization gate 통과.

이는 development evidence이며 untouched multi-regime confirmatory result가 아니다.

## 10. 선행연구와 novelty 경계

BLP discretization 기반은 Bennedsen--Lunde--Pakkanen의
[Hybrid scheme](https://arxiv.org/abs/1507.03004)이다. One-dimensional numerical
smoothing으로 discontinuous MLMC payoff의 rate와 kurtosis를 개선하는 발상은
[Bayer--Ben Hammouda--Tempone](https://arxiv.org/abs/2003.05708)에 존재한다.
Conditional-expectation smoothing도 고전적인 variance-reduction 원리다.

따라서 허용 가능한 novelty 후보는 다음의 결합과 exact specialization이다.

1. augmented BLP rough-Volterra law에서 volatility-independent price coordinate 분리;
2. deterministic two-driver importance likelihood를 포함한 analytic `Phi(a+b)` 공식;
3. discrete barrier와 occupation의 공동 사건을 exact random threshold로 환원;
4. 동일 coordinate에서 exact adjacent fine/coarse correction smoothing;
5. stable tail arithmetic, replay gates, FFT BLP engine을 포함한 end-to-end rare-event MLMC.

이 조합의 최초성은 최종 systematic review와 전문가 검토 전에는 확정할 수 없다.
“Gaussian smoothing 자체의 발명” 또는 “새 물리 법칙”으로 주장해서는 안 된다.

## 11. 남은 이론 위험

1. T9-7: barrier random threshold의 `L^p` grid convergence 증명;
2. T9-8: correction variance-rate theorem과 가정;
3. occupation order statistic의 anti-concentration/near-tie 문제;
4. CEM likelihood의 큰 kurtosis와 finite-sample CI 불안정성;
5. independent price Brownian 및 monotone downside event에 대한 의존;
6. state-dependent feedback control에서는 closed form이 깨지는 한계.

## 12. 저널 수준 판정

- 재현 가능한 numerical-method paper의 핵심 prototype: **충족**;
- SIFIN/Quantitative Finance 제출 준비: **아직 미충족**;
- top mathematical-finance journal 준비: **미충족**.

SIFIN급에는 최소 T9-7 또는 barrier-only variance theorem, T9-8/9 complexity 결과,
12 core + 6 stress frozen regimes, 10-seed end-to-end baseline 비교, probability
`1e-4--1e-6` coverage, independent reproduction이 필요하다. 더 높은 수학 저널에는
rough-Volterra class의 일반 정리와 occupation extension까지 요구될 가능성이 높다.

## 13. 감사 추적 파일

- `src/path_integral/gaussian_smoothing.py`
- `src/path_integral/rbergomi_smoothing.py`
- `src/path_integral/rbergomi_fft.py`
- `src/evaluation/smoothed_multilevel.py`
- `tests/test_gaussian_smoothing.py`
- `tests/test_rbergomi_smoothing.py`
- `tests/test_rbergomi_fft.py`
- `tests/test_smoothed_multilevel.py`
- `results/g9_fft_scaling_2026-07-18.json`
- `results/g9_mgvs_development_v3_2026-07-18.json`
