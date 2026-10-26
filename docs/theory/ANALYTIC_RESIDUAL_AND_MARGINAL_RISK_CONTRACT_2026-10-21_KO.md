# 해석적 residual 분할과 marginal risk 계약

2026-10-21. 기존조건부확률·totalvariance의적용이며새독창성정리주장이아니다.

## 1. Critical point와 envelope

g(u)=Phi(h(t(u))), t(u)=exp((logv+beta*u)/2)>0,
h(t)=(a+b*t²+c*t)/sqrt(k*(d+e*t²))라한다.
b=dt/2>0,e=dt>0,d=Ibase>0,k=1-rho²>0이다.

직접미분하면

```
dh/dt=[b*e*t³+(2*b*d-e*a)*t+c*d]
       /[sqrt(k)*(d+e*t²)^(3/2)].
```

beta>0이면dt/du>0이고Phi'>0이므로g의stationarypoint는P(t)의양의root와같다.
P'(t)=3*b*e*t²+(2*b*d-e*a)는양의t에서단조증가한다.
따라서P는양의축에서한번까지감소후증가하거나계속증가하며양의root는최대2개다.
구간끝과유효한P'root를bracket경계로두면각구간에는root가최대하나다.
beta=0이면g는상수이고beta<0은부호관계는달라져도extrema위치는변환가능하나이번API범위밖이다.

전체criticalpoint로분할한panel[a_i,b_i]에서는g가monotone이다.
l_i=min(g(a_i),g(b_i)),u_i=max(g(a_i),g(b_i)),w_i=Phi(b_i)-Phi(a_i)라하면p=1,2에대해

```
Σ w_i*l_i^p <= E[g(U)^p; |U|<=L] <= Σ w_i*u_i^p,
E[g(U)^p] <= Σ w_i*u_i^p + 2*Phi(-L).
```

이는정확실수연산·완전한criticalpoint집합을가정한수학적부등식이다.
부동소수점rootfinder/CDFendpoint는outward-roundedinterval계산이아니므로
구현의근사envelope를rigorouslycertifiedinterval이라고표현하지않는다.
near-multipleroot는실수root의수/위치가수치적으로불안정하므로unresolved로보존한다.
작은quadrature-order차이만으로미검출peak가없음을증명하지않는다.

## 2. 실제finite-grid residual 의존성

현재BLP hybrid law는local=(dW,localIntegral), driver_i=√(2H)*(history(dW)_i+localIntegral_i),
V_(i+1)=xi*exp(eta*driver_i-.5*eta²*VarDriver_i)로구성한다.
lowertriangularCholesky의oddindependentcoordinate2i+1은dW를바꾸지않는다.
따라서그coordinate는localIntegral_i와V_(i+1)만바꾼다.
terminalestimator는V0..V_(N−1)만쓰므로2N−1은payoff에들어가지않는다.
이finite-grid모형의특성을정확continuousVolterra law의독립성으로옮기면안된다.

N32에서57,61을독립표준정규U,W로분리한다. 모든mean의두성분이0이고성분공분산이I이면
q(X,U,W)=q_X(X)phi(U)phi(W), 평가G도같이분리된다.
V29와V31을제외한terminal log·적분분산을A,I로저장하면실행계획의jointCDF식이바로따른다.
분산의합과sqrt합을공유하므로g(U,W)는일반적으로곱으로분리되지않는다.
수치rank49를exactbasis선택의증명으로사용하지않는다.

## 3. Conditional risk만으로 해결할 수 없는 부분

기준p=p_Xp_U, 사건q=q_Xp_U, 평가G=G_XH라한다.
필요한절대연속성·정규화·finite second moment를가정한다.
M2(q)=E_G[g²/(Dq*DGX*DH)]이다.
U를먼저적분하면conditionalmean=m2(X)/(Dq*DGX)이며,
totalvariance는실행계획식과같다. 따라서conditionalrisk평가만개선하면남는marginalvariance항이있다.

m2>0인X에서H*=g²p_U/m2가conditionalvariance를0으로하지만,
이를계산·정규화·표본생성하는문제가해결됐다는뜻은아니다.
0payoff영역의support문제와defensivemass를제거해도된다고추론하지않는다.

G_X를자유롭게고를수있는relaxedproblem에서positiveintegrand의zero-variance분포는

```
G_X*(x)=p_X(x)*m2(x)/(Dq(x)*M2(q)).
```

이는event-optimalq*=g*p/mu와다른target이다.
G의learning용가중치목표/실제밀도/실제sampling분포를혼동하지않는다.
conditioning뒤m1²를m2대신쓰면기존q의M2가아닌새conditionalevent의M2를학습하게된다.

## 4. 다음 marginal-G 연구의 최소 계약

- 이번자료의재사용은개발진단에한정;publicationconfirmation이나새independenttest로쓰지않는다.
- geometry/feature/mixture구조선택은training/selection자료에서만하고최종seed를봉인한다.
- U,W를분리한새G라도fullnormalizeddensity와defensivemass를모든weight에반영한다.
- m2를numerical하게학습하면오차proxy·truncationtail·samplingSE를별도전파한다.
- event probability, 기존q M2, learnedG성능을각각독립검증하고frozenall-q범위를명시한다.
- 희귀모드누락,최대기여,block/LOO,trainingESS,독립mechanismreference를함께검사한다.
- totalwall에는offline/fit/selection/inference/I/O/실패실험을포함하고,
  동일accuracy자격전에는fixedprecision효율을선언하지않는다.

이번에는위계약을고정할뿐새G를학습하거나실제실용개선을확정하지않는다.
