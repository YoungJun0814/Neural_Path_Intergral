# 다음 판단 실행계획 — 해석적 분할·배치 적분·공동 residual·marginal risk

2026-10-21 America/New_York. 이전 §8의 네 판단을 별도 bounded 개발 단계로 수행한다.
기존 source/계획/결과, 특히 Q1 `NO_ADOPTION_PRECISION_UNRESOLVED`는 변경하지 않는다.
commit/push·새 operator·대규모 학습·추가 무작위 표본은 이번 범위가 아니다.

## 1. RQ2: 계산 구조 개선

기존 single-residual payoff는 t=sqrt(Vlast)=exp((logV0+beta*u)/2)에서

```
h(t)=(a+b*t²+c*t)/sqrt(k*(d+e*t²)),  g(u)=Phi(h(t)),
a=logK-A0, b=dt/2, c=-rho*w, d=I0>0, e=dt>0, k=1-rho²>0.
```

dh/dt의 부호는 `P(t)=b*e*t³+(2*b*d-e*a)*t+c*d`의 부호와 같다.
P'는 t>0에서 최대 하나의 turning point만 가지므로 양의 root는 최대2개다.
단순한 uniform grid만 신뢰하지 않고 t-domain의 monotone bracket마다 root를 찾는다.
beta=0은 상수 경우로 처리한다. beta<0은 이번 구현 범위에서 거부한다.
near-multiple root/overflow/부호 불확실은 unresolved로 남긴다.

- 고정 [-16,16], 폭1 기본 panel에 모든 검출 critical u를 추가한다.
- 16/32/64-order composite Gauss–Legendre를 profiles×panels×nodes로 벡터화한다.
- 두 모멘트는 같은 g nodes를 공유한다. 결과는 log-space에서 집계한다.
- 고정 numerical proxy는 두 연속 order 차이 + 상대1e-12 floor + Gaussian tail이다.
  모든 profile에서 상대1e-6을 요구하며 absolute-only 통과는 없다.
- 각 monotone panel endpoint min/max g와 Gaussian panel mass로 moment envelope를 계산한다.
  정확한 극값·실수 연산에서는 전역적 적분 부등식이다. 구현의 root/CDF/roundoff는
  interval arithmetic가 아니므로 반환값을 floating-point certified interval로 부르지 않는다.
- near-multiple root 또는 envelope 밖 추정치이면 수치 gate는 실패한다.
- 꼬리 upper는각모멘트2Phi(-16). 이 bound와 내부 numerical proxy를 구분한다.

고정 validation set은 Q0의6개 profile과 Q1의256개 profile **전부**다. 새대표표본이라고 부르지 않는다.
parent SHA:
Q0 `53c604c5faafa573ae380605342a130f68d54dcde6cefb30393a380bc482bb6b`,
Q1 `8645b470b638ff9860ea831915c1bf10e2111d66ca1780f31640c2c6c008ade1`.
원래 두 모멘트와 상대1e-6 이내 일치, 모든 context gate통과를 요구한다.

Known-answer: beta=0의 상수CDF 확률 두 profile. 반례: 두 turning point/near-double root/
tail-dominated/잘못된 parameter/overflow를 tests에서 고정한다.

## 2. 계산 비용 대조

같은262개 profile의 old adaptive 함수와 new batch 함수를 대조한다.
양셀zero두profile로양함수각각1회 warmup 후, **3회 AB/BA/AB** 순서로 반복한다.
모든 반복의 numerical consistency를 확인한다. median wall ratio≥2이고 모든수치gate통과면
`ADOPT_BATCH_DIAGNOSTIC_ONLY`다. 그렇지않으면 구현채택보류다.
이것은 scalar kernel benchmark일 뿐 event model의 fixed-precision speedup이 아니다.
기존 Q1 timing을 새 timing으로 덮어쓰지 않는다. 벤치마크 중 full pytest를 병행하지 않는다.

RQ2 production scalar cap12,000,000 / wall600초, full physics0.
audit는 배치 적분 전체262개재생 + 두zero의oldadaptive재생, scalar cap3,000,000 / wall120초.
모든 padding/endpoint/warmup node를 평가 수에 포함한다.
새 디렉터리 one-shot, 실행 전 source/plan/theory/parent를 동결한다.
실패·부분완료·소비예산을 보존하고 관측 후 order/tolerance 변경 또는 추가 반복하지 않는다.

## 3. RQ3: 57/61 공동 residual 법칙

현재 finite-grid hybrid Volterra kernel에서 odd coordinate2j+1은 독립 local integral residual이며,
그 coordinate는 V_(j+1)에만 들어간다. 역사 convolution은 dW채널만 사용한다.
따라서57은V29,61은V31,63은미사용V32에만영향을준다.
이 법칙을 연속 Volterra process의 독립성이라고 일반화하지 않는다.

다른좌표를고정하면 V29(U)=v29*exp(beta*U), V31(W)=v31*exp(beta*W)이고,
두분산을제외한A_base,I_base에대해

```
g(U,W)=Phi((logK-A_base+.5*dt*(V29+V31)
           -rho*(sqrt(V29)*dW29+sqrt(V31)*dW31))
          /sqrt((1-rho²)*(I_base+dt*(V29+V31)))).
```

두좌표모두모든mean에서정확0인지확인하여q/G의conditionalbaseGaussianfactorization을검사한다.
payoff는일반적으로두좌표에분리가능하지않다. 단순두1Dmoment곱으로근사하지않는다.

고정context: 양셀각 zero, Q1 block0 row0, Q1 block7 row15의3개.
각각한fullprofile+[-2,0,2]²9개probe=60physics.
audit양셀zero의동일10개씩20physics. 합80/cap100, wall각120/60초.
오차는logg1e-8이하를요구한다. 기존실험경로는통계적confirmation으로세지않는다.
이번에2D대규모적분·학습·채택은하지않는다.

## 4. RQ4: 남는 marginal risk를 분리

q=q_X*p_U, G=G_X*H(U|X), Dq=q_X/p_X, DGX=G_X/p_X, DH=H/p_U이면

```
Hrisk=g²/(Dq*DGX*DH),
E[Hrisk|X]=m2(X)/(Dq*DGX),
Var_G(Hrisk)=Var_GX(m2/(Dq*DGX))+E_GX Var_H(Hrisk|X).
```

이 law of total variance 때문에 U만 완벽하게 처리해도 X방향의 기여 집중은 남는다.
ideal conditional H*=g²*p_U/m2는두번째항만제거한다. m2>0/support가정과정규화상수문제를명시한다.
최적marginalrisk분포는G_X* ∝ p_X*m2/Dq이며 event-optimalq와다르다.
일반원리를새정리/실제표본생성가능성이라고주장하지않는다.

Q1의기존q conditionalM2 RSE약66%는이잔여문제의개발용관측이고population하한이아니다.
계산법개선으로samplinggate가통과한것처럼취급하지않는다.
다음G학습을하려면별도training/selection/final/독립reference seed와전체density,
비용·rare-mode·정밀도gate를동결해야한다. 이번에는그계약을작성하고새G를자동fit하지않는다.

## 5. 완료조건

이론식/finite-grid의존관계/known-answer·반례/성능대조·대표재생을검증한다.
full regression과lint/types,원본hash를확인하고결과/제한/다음채택판단을기록한다.
P2,D3,대규모학습승인은항상false. 적분코어의개발채택과모델성능채택은분리한다.
