# 경로 dictionary 보존형 모델: 목적함수와 추정 계약

2026-10-20. 이번 실험은 기존 finite-grid conditional g와 Gaussian p를 유지한다.
N32에서 q의 mixture weights만 학습하며 fine-grid/continuum gate를 바꾸지 않는다.

## 1. 정확한 밀도와 가중치 domain

고정 shifts m0=0,m1,…,m372와 identity covariance Gaussian을 사용한다.
D_j(z)=exp(m_jᵀz−||m_j||²/2), D_w=Σw_jD_j=q_w/p다.
w0=.1, wj=.1/372+.8vj, v≥0,Σvj=1로 둔다. 이는 정규화된 full mixture이며
각 성분은 제거되지 않는다. uniform G의 shift weight .9/372도 domain 안이다.

새 Z_i~q_w에서 X_i=g(Z_i)/D_w(Z_i)의 ordinary mean은 frozen training/selection에
조건부로 μ=E_p[g]를 추정한다. 정확한 likelihood와 독립 final이 필요하다.
component label의 density 하나만 쓰거나 최종 contribution을 clipping하면 안 된다.

q_w≥.1p, G≥.1p로 0≤X≤10, H_w=g²/(D_GD_w)≤100이다. 이 절대 상한은 작은
μ 또는 M2에 대한 충분한 상대 정밀도를 뜻하지 않는다.

## 2. 서로 다른 convex 목적

training Z_i~G에 대해 event weight t_i=g_i/D_Gi를 사용한다.
event-KL와 같은 minimizer를 갖는 empirical objective는
L(v)=−Σ(t_i/Σt)log D_w(Z_i)다. normalized training weights는 optimization scale
선택일 뿐 final estimator의 self-normalization이 아니다. −log의 convexity와 D_w의
affine성 때문에 convex다. unknown μ를 최종 likelihood에 넣지 않는다.

risk-M2는 Jhat(v)=n⁻¹Σg_i²/[D_Gi D_wi]이며 ordinary IS bank mean이다.
1/x의 convexity 때문에 convex다. log Jhat도 같은 minimizer를 갖지만 일반적인
“log가 convexity를 보존한다”는 주장을 하지 않는다. optimizer는 Jhat의 상대
gradient를 써서 같은 방향과 line-search root를 계산한다.

row scaling a_ij=exp(logD_j−rowmax_i), d_i=Σw_ja_ij라 하자.
event gradient는 −.8Σnormalized(t_i)a_ij/d_i다.
risk gradient를 Jhat로 나눈 값은 −.8Σsoftmax(2logg−logD_G−rowmax−logd)_i a_ij/d_i다.
두 softmax 모두 gradient/fit scale 용도이며 추정 target을 바꾸지 않는다.

Frank–Wolfe 방향은 simplex vertex minus current v다. event line derivative는
−Σnormalized(t_i)delta_i/(1+s delta_i), risk의 상대 line derivative는
−Σnormalized(riskterm_i)delta_i/(1+s delta_i)²다.
delta=(d_vertex−d_current)/d_current이고 s∈[0,1]이다. 양의 floor 때문에 denominator가
양수여야 한다. nonfinite나 유효 범위 실패는 row를 버리거나 clipping하지 않고 거부한다.

event gap은 normalized-target objective의 directional FW gap이고, risk gap은 Jhat로
나눈 relative FW gap이다. 둘은 이름이 비슷해도 같은 loss 단위가 아니다.
iteration cap 도달은 수렴 성공과 구분하며 population optimum이라고 쓰지 않는다.
이 fixed-mixture convexity와 optimizer는 기존 원리이며 신규성은 주장하지 않는다.

## 3. 선택과 baseline 보존

train 후 q_fit을 고정하고 별도 selection IID G에서 α∈{0,.25,.5}의
qα=(1−α)G+αq_fit를 비교한다. 두 종류 모두 selection 목적은 empirical M2다.
따라서 결과는 “각 training objective+같은 risk selection policy”의 비교다.
selection minimum은 독립 성능 평가가 아니며 새 heldout/final이 필요하다.

α0은 정확한 baseline이다. 모든 실제 qα≥(1−α)G≥.5G이므로
M2(qα)≤M2(G)/(1−α)≤2M2(G)다. 이 bound는 우월성을 보장하지 않는다.
각 q의 natural mass .1과 normalized full density를 유지한다.

## 4. 위험 비교와 비용

선택 이후 별도 IID G에서 H_q=g²/(D_GD_q)를 세 q에 대해 같은 경로에서 계산한다.
E_G[H_q]=M2(q). paired difference는 같은 경로의 covariance를 반영해야 한다.
raw IID path와 equal-size block mean을 사용한 normal/delta 진단은 개발용이며
finite-sample tail certificate가 아니다. 큰 bank도 미관측 contribution을 배제하지 못한다.

같은 g,p에서 Var_q(X)=M2(q)−μ²이므로 M2 순서는 μ 참값 없이 비교할 수 있다.
하지만 비용효율은 단순 M2만으로 정해지지 않는다. per-draw cost와 setup/train/fit/
selection 비용, 목표 정확도를 함께 고려해야 한다. 이번에는 fixed-count workflow
시간을 기록하며 동일 accuracy 자격 없는 방법의 fixed-precision speedup은 주장하지 않는다.

같은 geometry에서 five-fit 반복이 보인 신호와 신규 task/fine grid의 성능은 다르다.
기존 failed gate,새 개발 gate,publication confirmation을 끝까지 구분한다.
