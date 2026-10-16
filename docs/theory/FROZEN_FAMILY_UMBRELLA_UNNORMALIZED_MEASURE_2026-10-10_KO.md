# Frozen-family umbrella SMC: 수학적 계약과 구현 검토

2026-10-10. 실제 rare-event pilot 이전 검토. 새 일반 SMC 정리의 발견 주장 아님.
기존 unnormalized Feynman–Kac particle measure를 우리 여러 frozen estimand에
적용하는 구현·비용 공유 후보이다. 성능과 저널 novelty는 별도로 입증해야 한다.

## 1. 서로 다른 q의 위험값을 하나로 바꾸지 않고 계산 공유

같은 p=N(0,I2N), conditional payoff0<g<=1에서 q1...qK는 각각 정확한 defensive floor
δ_j>=.1을 가진 별도 frozen proposal이다. D_j=q_j/p.

- H0=g, Z0=μ.
- Hj=δ_j*g²/D_j (j=1...K), Zj=δ_j*M2(q_j).

각0<Hj<=1. 새로운 frozen positive weights w0...wK, sum w=1에 대해
H=Σw_jHj,0<H<=1. 이는 같은 사건의 참값이나 q_j를 바꾸지 않고 reference target만 바꾼다.
H/p같은 density 오기를 피한다. reference target은 pH, endpoint normalizer Z=E_pH=Σw_jZj다.

균등 w도 가능하지만 Zj가 서로 매우 다르면 큰 normalizer가 mixture를 지배한다.
개발 pilot의 positive fixed estimates c_j로 w_j∝1/c_j를 정할 수 있다.
c_j가 부정확해도 expectation identity는 유지되며 효율은 악화될 수 있다.
현재 whole SMC의 데이터로 c_j나 weights를 재조정하지 않는다.

## 2. 올바른 endpoint estimator

πβ∝pH^β의 fixed-schedule SMC를 실행한다. product normalizer를 Zhat,
terminal normalized particle weights를 Wi라 하면 각 component의 estimator는

Zhat_j = Zhat * Σ_i Wi Hj(Xi)/H(Xi).

원래 관심값은 μhat=Zhat_0, M2hat(q_j)=Zhat_j/δ_j다.
**ΣWi Hj/H만 쓰면 정규화된 π_H의 expectation이지 원래 Zj가 아니다.**
Zhat를 반드시 유지한다. 일반 final self-normalized IS ratio의 불편성을 주장하는 것이 아니다.
SMC 내부 normalized weights와 product normalizer를 곱한 unnormalized empirical measure다.

단일 fixed whole-run의 여러 output은 서로 상관되어 있다. particle를 IID 표본으로 세지 않는다.
같은 output의 SE는 독립 whole-run의 Zhat_j를 평범하게 평균하여 계산한다.
component와 서로 다른 q의 값을 평균내어 한 risk로 대체하지 않는다.

## 3. 왜 기대값이 맞는가: finite-particle induction

γβ(f)=∫p(x)H(x)^β f(x)dx를 두고 γhatβ(f)=Zhatβ ΣWi f(Xi).
initial Gaussian empirical measure는 E γhat0(f)=γ0(f).

1. G=H^(βnext-βold)로 weighting하면 normalization factors가 상쇄되어
   γhatnext(f)=γhatold(G f)가 된다.
2. 고정 resampling 시 conditional expected offspring empirical measure는
   이전 normalized weighted empirical measure다. product normalizer를 곱하면
   unnormalized measure의 conditional expectation을 유지한다.
3. 다음 β의 invariant MH kernel K를 particle마다 적용하면 expectation은
   γhatold(G Kf)이다. induction으로 γold(GKf)=γnext(Kf)=γnext(f).
   skipped-resampling 단계에도 weight를 유지하므로 같은 선형 관계가 성립한다.
4. final에서 f=Hj/H를 선택하면 γ1(f)=∫pHj=Zj.

따라서 고정 schedules, unbiased resampling, invariant kernels, integrability 아래
finite-particle unnormalized-measure identity를 얻는다. start/transition/target를
현재 final 결과에 맞춰 선택하거나 failed whole을 삭제하면 이 실험 계약 밖이다.
일반적 배경은 [Del Moral–Doucet–Jasra SMC samplers](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).
위 짧은 induction은 현재 구현의 weighting→resampling→mutation 순서에 대한 검토다.

## 4. Ratio bound와 구현 invariant

H>=w_jHj이므로 0<Hj/H<=1/w_j. 따라서 endpoint reweighting의 integrability는
frozen positive w_j로 확보된다. 또 samplewise Σw_j Zhat_j=Zhat가 성립한다.
작은 w_j의1/w_j는 클 수 있으므로 이 bound만으로 relative precision을 보장하지 않는다.
Zhat 자체의 변동·mixing·rare mode coverage도 여전히 필요하다.

코드는 logsumexp로 H와 endpoint vector를 계산한다. 모든 component logH와
particle logweight가 finite CPUfloat64이고 정상화 조건에 맞는지 검사한다.
각 mutation/resampling에 component logpotentials도 함께 갱신한다.
마지막 component evaluation을 다시 계산하지 않도록 최신 값을 유지한다.
따라서 하나의 g 평가로 K+1개 target를 만들고 계산량은 q density K개를 포함한다.
potential-count가 K배 줄어도 실제 wall-time이 K배 줄었다고 주장하지 않는다.
mixture weights는 함수 진입 시 clone해 freezing한다.

Gaussian prior block/full pCN의 MH ratio는 βΔlogH다. guide, bank, independence
proposal을 쓰지 않는 arm과 guided arm을 구분한다. original g/qs를 변경하지 않는다.
future learned q를 자동 적용하지 않는다.

guided arm은 normalized r 시작의 Fβ=r^(1-β)(pH)^β를 사용한다.
increment=Δβ(logH-logDr), prior-reversible mutation MH=βΔlogH+(1-β)ΔlogDr,
r-independence MH=βΔ(logH-logDr). final F1=pH이므로 같은 endpoint component
formula가 성립한다. induction의 초기 measure만 r, G만(H/Dr)^Δβ로 바뀐다.
각 β-invariant kernel과 unbiased resampling 조건은 동일하다.
proposal bridge 한 component와 scalar core의 numerical identity도 검사한다.

## 5. 검증과 잔여 위험

- terminal weighted-sum identity 및 ratio bound.
- constant component들의 exact normalizers (Zhat factor 누락 방지).
- 서로 다른 Gaussian-bump component들의 알려진 analytic endpoints.
- 한 component일 때 scalar guide-free SMC와 동일한 결과/calls.
- 실제 family의 frozen anchors/q/source와 independent whole-run replay는 별도 수행.

실제 family가 모두 같은 rare mode를 놓치면 낮은 sample RSE가 다시 실패할 수 있다.
umbrella가 reference의 새로운 독립 메커니즘이라고 자동 주장하지 않는다.
기존 static IID / guided SMC / guide-independent SMC corroboration은 유지한다.
P1/P2/large-training pass는 이 코드의 구현·oracle 통과만으로 바뀌지 않는다.
