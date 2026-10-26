# Marginal G 최적화 식의 적용 범위 — 최종 검토 보완

2026-10-21 America/New_York. 해석적 residual 실행 후의 이론 검토 부록이다.
[실행 전 동결 계약](ANALYTIC_RESIDUAL_AND_MARGINAL_RISK_CONTRACT_2026-10-21_KO.md)을
소급 수정하지 않는다. 그 계약 §3의 `G_X* ∝ p_X m2/Dq`에는 아래의 적용 조건이 필요하다.
수치 실험의 값·threshold·채택 판정은 변경하지 않으며, 이번에 G를 학습하지 않았다.

## 1. 기준과 가정

기준 확률밀도는 `p(x,u)=p_X(x)p_U(u)`, 사건 제안은 `q=q_X p_U`,
평가 제안은 `G=G_X H(u|x)`로 둔다. 아래의 비율은 모두 **정규화된 실제 밀도**의 비율이다.

```
Dq(x)  = q_X(x) / p_X(x)
DGX(x) = G_X(x) / p_X(x)
DH(u|x)= H(u|x) / p_U(u)
m1(x)  = E_pU[g(x,U)]
m2(x)  = E_pU[g(x,U)^2]
M2(q)  = E_pX[m2(X)/Dq(X)]
```

`0 <= g <= 1`, 필요한 support/절대연속성, `Dq,DGX,DH>0`인 기여 영역,
정규화 및 해당 분산의 유한성을 가정한다. `M2(q)>0`인 비자명한 경우를 다룬다.
영기여 집합에서는 별도 정의가 필요하고, ideal 식이 positive defensive mass를 자동 보존하지는 않는다.

## 2. 항상 성립하는 분해

직접 risk 평가량을 `Y=g²/(Dq DGX DH)`라 하면

```
E_H[Y | X=x] = m2(x)/(Dq(x) DGX(x)),
E_G[Y]       = M2(q),
Var_G(Y)     = Var_GX(m2/(Dq DGX)) + E_GX[Var_H(Y|X)].
```

따라서 U 방향만 정밀하게 처리하는 것은 두 번째 항을 줄이는 작업이다.
첫 번째 항이 실제로 얼마나 남는지는 별도 population 실험 없이는 알 수 없다.
Q1의 큰 관측 RSE는 유한한 개발 자료이지 첫 번째 항의 엄밀한 하한이 아니다.

## 3. `G_X ∝ p_X m2/Dq`가 맞는 경우

첫째, U를 먼저 적분한 risk 평가량 `Y_RB=m2/(Dq DGX)`를 직접 쓰는 경우다.
이 경우 자유로운 정규화 밀도 전체에서

```
G_X,RB*(x) = p_X(x) m2(x)/(Dq(x) M2(q))
```

가 zero-variance 분포다. 단, m2의 계산과 정규화 상수 및 표본 생성이 알려져 있다는
이상적인 수학 문제의 해이지, 구현 가능한 sampler를 얻었다는 결론은 아니다.

둘째, 모든 관련 X에서 ideal conditional 분포

```
H*(u|x) = g(x,u)^2 p_U(u) / m2(x)
```

를 정확하게 쓸 수 있는 경우다. 이때 `Y=Y_RB`가 H*-거의 확실히 성립하여 같은 식이 적용된다.
`m2(x)=0`이면 이 식을 그대로 나눠 쓸 수 없다.

## 4. 임의의 고정 H에 대한 일반식

H가 ideal이 아니고 U도 실제로 표본 추출한다면

```
B_H(x) = E_pU[g(x,U)^4 / DH(U|x)],
E_G[Y^2] = E_pX[B_H(X)/(Dq(X)^2 DGX(X))].
```

정규화 조건 `E_pX[DGX]=1` 아래 Cauchy–Schwarz를 적용하면,
관련 적분이 유한하고 양수인 자유 밀도 문제에서

```
G_X,H*(x) ∝ p_X(x) sqrt(B_H(x))/Dq(x)
```

다. **일반적인 고정 H에서는 m2가 아니라 sqrt(B_H)가 목표다.**
`H=H*`일 때만 `B_H=m2²`가 되어 앞 절의 식으로 환원된다.
`G_X ∝ p_X m2/Dq`를 쓰면 어떤 H에서도 between-X 분산항은 제거할 수 있지만,
임의의 H에서 total variance까지 최소화한다고 주장할 수는 없다.

제약된 Gaussian mixture/defensive family에서는 위 자유 밀도 최적형이 그대로 feasible하지 않을 수 있다.
KL fitting과 직접 second-moment fitting도 서로 다른 projection 문제다.
그 차이를 숨기고 하나의 학습 목표라고 부르지 않는다.

## 5. 혼동하면 안 되는 세 가지 목표

| 최적화하려는 대상 | 제약/평가량 | 자유로운 X 밀도의 이상형 |
|---|---|---|
| 원래 사건 estimator의 second moment | `q=q_X p_U`, estimator `g/Dq` | `q_X ∝ p_X sqrt(m2)` |
| U를 적분한 사건 estimator | estimator `m1/Dq` | `q_X ∝ p_X m1` |
| 고정 q의 조건부 risk 평가 | estimator `m2/(Dq DGX)` | `G_X ∝ p_X m2/Dq` |

세 행은 다른 estimator/목적함수다. 특히 `m1²`로 기존 q의 `m2` 목표를 대체하지 않는다.
여기서 q를 바꾸면 평가해야 할 M2(q) 자체가 달라지고, G만 바꾸면 평가 효율만 바뀐다.

## 6. 다음 구현에서 강제할 조건

1. 먼저 estimator를 고정한다. numerical m2를 쓰는 Rao–Blackwell risk와
   finite-sample H를 쓰는 direct risk를 같은 학습 목표로 혼합하지 않는다.
2. m2가 numerical이면 적분 오차·tail truncation과 sampling 오차를 분리한다.
   작은 order 차이나 numerical envelope만으로 exact unbiasedness를 주장하지 않는다.
3. actual sampler와 normalized density를 함께 구현하고 모든 defensive/mixture mass를 포함한다.
4. 조건부 근사 H가 들어가면 위 B_H 식과 실제 density로 목표를 다시 유도한다.
5. training/selection/final/reference를 분리하고, final 자료로 구조·허용오차·예산을 바꾸지 않는다.
6. 이것은 기존 조건부기대값·total variance·Cauchy–Schwarz의 적용이다.
   새 원리의 발견, 새 G의 구현, 또는 Volterra 성능 입증으로 해석하지 않는다.

여기서 다루는 static Gaussian path proposal는 전체 유한차원 밀도를 직접 평가하는 문제다.
이를 stochastic drift의 Girsanov likelihood로 구현한 것으로 혼동하지 않는다.
후에 시간별 drift control을 도입한다면 adaptedness와 해당 change-of-measure의 조건을
별도로 증명해야 하며, 이번 static density identity가 그 증명을 대신하지 않는다.

이번 finite oracle는 정확한 H*를 쓰는 toy이므로 그 결과와 모순되지 않는다.
부록은 기존 계약의 해석 범위를 좁히는 것이며, 실패한 Q1 sampling gate를 통과시키는 변경이 아니다.
