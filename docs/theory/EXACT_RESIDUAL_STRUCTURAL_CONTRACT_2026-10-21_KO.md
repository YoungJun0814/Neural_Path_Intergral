# Residual 구조 진단의 수학적 계약과 주장 경계

2026-10-21. 일반 IS 항등식의 적용·검증 문서이며 새 정리의 독창성 주장이 아니다.

## 1. 사건 proposal과 위험 평가 target은 다르다

기준밀도p, 비음수g, μ=∫gp>0, 사건 proposal q에 대해

```
M₂(q)=∫g²p²/q,  r_q=g²p²/(qM₂(q)).
```

G에서 H_q=g²p²/(qG)를 평균내면 M₂(q)를 추정한다. 독립 표본수n에서
그 추정치의 상대분산은 `(∫r_q²/G−1)/n`이다. 좋은 사건q와 좋은 위험평가G는 같은 최적화
문제가 아니다. G=r_q는 이론적 zero-variance 평가분포이나 일반적으로 미지의 정규화 상수와
표본 생성 문제가 있어 production에서 바로 사용할 수 있다는 뜻이 아니다.

## 2. 변경하지 않는 residual이 있는 proposal family

Z=(X,Y), p=p_Xp_Y라 하고 q=q_Xp_Y로 제한한다. 모든 mixture 평균이 한 선형공간S에 있고
각 성분 공분산이I이면, 직교 Gaussian 좌표X∈S,Y∈S⊥에서 이 factorization이 성립한다.
모든 평균에서 좌표j가 정확0인 경우는 그 좌표만 분리한 특수 경우다.

A(x)=E_pY[g(x,Y)²], a(x)=q_X(x)/p_X(x)라 두면

```
M₂(q)=E_pX[A(X)/a(X)],  E_pX[a(X)]=1.
(E_pX[sqrt(A(X))])² <= E_pX[A(X)/a(X)] * E_pX[a(X)] = M₂(q).
```

두 번째 줄은 Cauchy–Schwarz다. relaxed unrestricted q_X에서 equality는
`a=√A/E√A`일 때 성립한다. 실제 defensive mixture family가 이 밀도를 포함하지 않을 수
있으므로 equality/attainability를 실제모델에 주장하지 않는다. support와μ>0 가정을 유지한다.
g가 bounded이면A와E√A는유한하다.

`m(x)=E[g|X=x]`라 하면 `√A(x)≥m(x)`이므로 relaxed floor도μ² 이상이다.
조건부분산이p_X-positive 집합에서양수이면그부등식은엄격하다. 다만 엄격한양수gap과
실제로큰상대gap은다르며, 현재Volterra의양적하한은추정하거나입증하지않았다.

분리가능한g(x,y)=g_X(x)g_Y(y)에서는 relaxed floor/μ²=`E[g_Y²]/E[g_Y]²`다.
이 예에서는 X 쪽 가중치를 아무리 잘 조정해도 Y 쪽 오차를 제거하지 못함을 정확식으로 보일 수 있다.
실제Volterra payoff를분리가능하다고가정하지않는다.

## 3. Gaussian bump의 완전한 대조식

p=N(0,I_d), g=exp(−||z−c||²/(2s²)), q_m=N(m,I_d)에 대해 completing-square로

```
μ = (s²/(1+s²))^(d/2) exp(−||c||²/(2(1+s²))),
m* = c/(1+s²),
log(M₂(q_m)/μ²)
 = d log((s²+1)/sqrt(s²(s²+2))) + (s²+1)/(s²+2)||m−m*||².
```

첫항은 이 단일 identity-covariance Gaussian family의 covariance 제한이고 두번째는평균오차다.
mixture 전체의하한으로첫항을일반화하면안된다. 여러평균을섞으면표현가능한공분산/형태가달라진다.

고정q_m의r_q는 `N((2c−s²m)/(s²+2), s²/(s²+2)I)`다.
r=N(a,vI),G=N(b,wI)의 `∫r²/G`는2w>v일때

```
(w/sqrt(v(2w−v)))^d exp(||a−b||²/(2w−v)).
```

2w≤v이면 이 적분은 발산한다. 이는 해당 Gaussian toy의 조건이며,
현재production의방어적bounded integrand에서무한분산을발견했다는뜻이아니다.

## 4. 현재 N32에서 마지막 사용 분산의 residual

local 좌표는 `(z_0,z_1),...,(z_62,z_63)`의32쌍이다. lower triangular local Cholesky로
첫채널은dW, 두번째는local Volterra integral에기여한다. 이들은독립적인두물리Brownian이아니다.
V_i는구간i의시작분산이다. terminal law는V_0,...,V_31을사용하고V_32는사용하지않는다.

현재guide mean의61번성분이전부0이면 q/G의density ratio는U=z61과무관하다.
U는local integral index30에들어가 V31을변화시키고, 그이후V32는미사용이다.
다른좌표를고정하면V31(U)=V31(0)exp(βU), `β=η√(2H)L11`다.

이전31개구간의terminal/log값A0,integrated varianceI0, 마지막dW를w라하면

```
g(U)=Φ((logK−A0+.5dt V31(U)−ρ√V31(U)w)
       /sqrt((1−ρ²)(I0+dt V31(U)))).
```

63번은local integral index31에들어가V32만변화시키므로g와무관해야한다.
이를수식과full evaluator의probe로구분한다. 단순히mean이0인좌표모두를유용한새방향으로보지않는다.

## 5. 조건부 적분을 다음 후보로 검토할 때의 정확한 목표

U를q/G 모두p_U=N(0,1)로남길때다른좌표X에대해 `m1(X)=E_U[g]`, `m2(X)=E_U[g²]`다.

- event estimator를조건부평균으로바꾸면 `m1/D_q`이며목표μ는유지된다.
- **기존q의M₂**를평가하려면 `m2/(D_GD_q)`를사용해야한다.
- `m1²/(D_GD_q)`는Rao–Blackwellized event estimator의M₂다. 기존q의M₂와같지않다.
- 둘의차이는 `∫p_X Var_U(g|X)/D_q ≥0`다. 이는조건부기대값의일반적분산감소다.
- 수치quadrature는정확한조건부기대값이아니다. g의대수식이정확한것과적분이정확한것을구분해야한다.
  production에서는quadrature bias/오차전파/계산비용을다시검증해야하며이번진단은그완료증거가아니다.

이분리는새G를적합하기전에검토할구체적수학경로를제공한다. 그러나6개조건부context의
효과크기를global기댓값·speedup으로확장할수없다.

## 6. 선행연구와 novelty 경계

Gaussian proposal의low-dimensional projection과covariance 방향선택은이미연구된주제다.
[El Masri–Morio–Simatos, Optimal projection for parametric importance sampling in high dimensions](https://computo-journal.org/published-202402-elmasri-optimal/)는 Gaussian family의KL기준방향선택을다룬다.
이번진단은그논문의새정리를재발견했다고주장하지않으며,KL최적화를M₂최적으로치환하지않는다.

[Beh–Shadmi–Simatos, Insight from the Kullback–Leibler divergence into adaptive importance sampling schemes](https://arxiv.org/abs/2309.16828)의표본복잡도결과에는확률이0에서떨어져있다는가정등이있다.
그결과를현재극희귀Volterra셀의자동정밀도보장으로적용하지않는다.

현단계기여는기존항등식과정확toy를사용해 **구체적구현의표현불변좌표와payoff의존을분리한검증**이다.
최상위저널용새기여의완성은아니며,Volterra특화양적분석과독립실험·비용검증이추가로필요하다.
