# 방어적 fixed-dictionary mixture 재배분: 이론·수치 검토

2026-10-20. 기존 원리의 적용 검토이며 새로운 정리라는 주장이 아니다.
실험 조건은 사전 계획, 결과는 별도 결과 보고서를 따른다.

## 정의

p=N(0,I_2N), 0≤g≤1은 원래 finite-grid terminal CDF다.
component r_j=N(m_j,I), d_j=r_j/p=exp(m_j·z−||m_j||²/2), m_0=0.
shift 개수 K=372, 원래 G1은 w^u=(.1,.9/K,...,.9/K)다.

학습 가능한 영역은 v≥0, Σv=1,
w_0=.1, w_j=.1/K+.8v_j (j≥1)이다. Dw=Σw_j d_j.
원래 균등 G1은 v_j=1/K일 때 재현되며 Dw≥.1이다.
component means, 물리 법칙, 원래 학습 모델 q는 바꾸지 않는다.

## 정확성

고정 w에 대해 r_w=p Dw에서 X_w=g/Dw를 평가하면
E_r_w X_w=∫p g=μ다. denominator는 component 단독이 아니라 전체 mixture다.
0≤X_w≤10이므로 finite variance는 존재하지만 목표 상대 정밀도는 보장하지 않는다.

학습 bank B로 정한 w(B)와 final 표본을 분리하면
E[μhat | B]=μ이고, 따라서 E μhat=μ다.
동일 B에서 새 IID final n개를 사용하면
Var(μhat | B)=[F(w(B))−μ²]/n,
F(w)=E_p[g²/Dw]다. 독립 final 없이 training bank로 최종 평균/SE를 계산하면
이 계약을 그대로 적용할 수 없다. bank를 보고 count를 임의 연장하는 것도 별도 문제다.

## Convex 학습 목적과 그 한계

원래 균등 r_u=p Du에서 학습 표본을 만들고
Fhat_B(w)=n_B^(-1) Σ g(z_i)²/[Du(z_i) Dw(z_i)]를 최소화한다.
고정 w에 대해 E_B Fhat_B(w)=F(w)다. 그러나 최소화 후 w(B)를 대입한 training
목적은 모집단 F(w(B))의 unbiased 추정량이라고 할 수 없다. 별도 validation과 final이 필요하다.

각 표본의 목적은 양수 선형식의 역수이므로 convex다. 모집단도 convex이고
∂F/∂w_j=−E_p[g² d_j/Dw²],
∂²F/∂w_j∂w_k=2E_p[g² d_j d_k/Dw³]이다.
고정 유한 Gaussian shift와 양의 defensive bound 아래 관련 Gaussian 지수 모멘트가
유한하므로 이 영역에서 미분과 적분의 교환을 정당화할 수 있다. Hessian은 양의 준정부호다.
이는 convex mixture IS의 기존 원리이며 [He & Owen](https://arxiv.org/abs/1411.3954)을 참고한다.

shift floor 때문에 Du≤9Dw다. 따라서 F(w)≤9F(w^u)지만, 이는 원래 균등보다
반드시 좋다는 뜻이 아니다. 마찬가지로 convexity에 의해 두 학습 weight의 평균은
F((w1+w2)/2)≤[F(w1)+F(w2)]/2를 만족하나 최고 단일 fit보다 좋다는 보장은 없다.

Dw≥.1에서 F(w)≤10μ이므로 relative variance의 일반 상한은 [10/μ−1]/n이다.
μ가 극히 작으면 매우 느슨하다. 방어적 floor를 tail coverage/rare-event efficiency
증명으로 바꿔 표현해서는 안 된다.

## 구현 검토

Frank–Wolfe는 w가 아니라 simplex v에서 실행한다. ∇_v F=.8∇_w_shift F다.
가장 작은 gradient의 vertex를 선택하며 경험적 gap은 ∇F(v)·(v−s)≥0이다.
코드는 이 값을 F로 나눈 relative gap을 기록한다. 이는 경험적 목적에 관한
float64 진단이며 모집단 optimum이나 directed-rounding certificate가 아니다.

row별 log-component 최대값을 빼서 density를 계산하고, objective contribution의
softmax로 gradient의 공통 양의 배율만 제거한다. 목적값 자체는 logsumexp−log n의
ordinary mean으로 계산한다. 이 softmax를 최종 probability의 self-normalization으로
사용하지 않는다. 선분 위 목적의 미분은 단조이므로 고정 bisection line search를 쓴다.

모든 bank row를 유지한다. 비유한 density, 상승한 목적, 직접 log-domain 목적과의
불일치에서 실패시킨다. 최종 값은 원래 full density로 계산하고 clipping/표본 제거는 없다.
학습 최대 반복 256회와 허용 gap은 실험 전에 고정했다. 이번 네 fit은 반복 상한에
도달했으므로 수렴 기준을 충족한 정확한 최적해라고 부르지 않는다.

## 독립 검사와 남은 범위

작은 문제의 ordinary-mean 목적, convexity, 유한차분 gradient, floor, 목적 비증가,
공통 log-payoff rescaling 불변성, 별도 Gauss–Hermite full-mixture probability oracle,
잘못된 입력/미분 입력 거부를 검사했다. seed 역할 및 전체 unit 수 계약도 검사한다.

같은 dictionary에서 weight만 바꾸므로 새로운 물리적 경로 모드의 존재/부재는 알 수 없다.
두 fit의 weight 차이는 실제 mixture 분포 거리나 target 편향과 같지 않다.
보존된 top-16의 진단은 관측된 큰 기여의 설명이며 전체 clustering/coverage를 대체하지 않는다.
q의 M2, guide-free 교차검증, mesh/continuum 편향, 저널 독창성은 여전히 별도다.
