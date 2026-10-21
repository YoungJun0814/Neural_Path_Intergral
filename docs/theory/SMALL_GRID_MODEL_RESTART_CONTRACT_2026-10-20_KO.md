# 작은 격자 모델 재시작: 독립 개발과 위험 비교의 수학적 계약

2026-10-20. 원래 N32 사건 proposal을 개선하는 제한된 개발 실험의 계약이다.
N4096 reference 인증을 기다리는 연구 순서와 이 작은 격자의 개발 순서를 분리한다.
이는 기존 실패 기록이나 P1/P2, fine-grid 정확도, mesh/continuum 관문을 통과시킨다는
뜻이 아니다. 이 문서는 실행 결과가 아니며, 표본 수와 판정 규칙은 실행 전 동결한다.

## 1. 대상과 두 경로

- 개발 경로: 기존 두 셀의 원래 N32, full conditional payoff g, latent dimension 64를
  유지하고 새 전체 부모 학습을 셀마다 5회 수행한다. 부모 q0는 새 SMC bank에서
  natural 질량 .1 + identity-covariance Gaussian shift 최대 2개로 적합한다.
- 인증 경로: 기존 fine N4096 reference, 독립 평가, mesh/continuum 및 P1/P2 요구를
  별도로 유지한다. 개발 결과가 좋더라도 이 경로의 false/unresolved 상태는 바뀌지 않는다.
- static G는 N32에서 다시 구성한 고정 mixed-window PhysicalWindowGuide이다.
  기존 물리적 dictionary의 373개 성분과 natural 질량 .1을 유지하며, 실용적인 사건
  proposal baseline과 독립 IID bank sampler의 두 역할을 구분해 기록한다.
- 기존 N32의 mu 자료는 사건·격자·모델 parameter binding이 일치할 때 보조 비교에
  사용할 수 있다. N4096 mu나 과거에 동결된 다른 q의 M2를 새 q0의 참값으로 쓰지 않는다.

## 2. 원래 위험과 두 학습 bank 가중치

p는 원래 Gaussian density, 0 <= g <= 1, f=gp, mu=integral f라 둔다. 정규화된
proposal q에 대해

\[
J(q)=M_2(q)=\int \frac{f(z)^2}{q(z)}\,dz,
\qquad \operatorname{Var}_q(gp/q)=J(q)-\mu^2.
\]

q0를 먼저 동결한 뒤, q0의 SMC bank와 독립인 하나의 IID training bank Z_i~G를 쓴다.
이 같은 bank에서 event와 risk 교정의 가중치를 각각

\[
w_i^{\rm event}=\frac{g(Z_i)p(Z_i)}{G(Z_i)},\qquad
w_i^{\rm risk}=\frac{g(Z_i)^2p(Z_i)^2}{q_0(Z_i)G(Z_i)}
\]

로 계산한다. D_q=q/p, D_G=G/p로 두면 log-space에서는 log g-log D_G 및
2 log g-log D_q0-log D_G이다.
bank 가중치의 정규화는 적합에만 허용한다. 최종 mu, M2 추정에 empirical
self-normalization이나 clipping을 쓰지 않는다. 여기서 M2는 이미 조건부 적분된
사건 estimator g의 위험이므로 integrand는 g²다. 원래 indicator의 조건부 제곱
모멘트 E[1_event²|local]=g로 바꾸면 다른 위험을 평가하게 된다.

두 교정 r_event, r_risk는 같은 최대 K=2 shift, identity covariance, natural 질량 .1,
같은 fitting 규칙을 사용한다. 이는 제한된 가중 적합이지 exact KL projection이나
population M2 최소화의 보장이 아니다. 작은 cluster가 제거되면 실제 성분 수를 기록한다.
SMC particle ESS, IID target-weight ESS 및 독립 전체 학습 반복 수는 서로 다른 진단이다.

## 3. exact risk target의 의미와 보장되는 범위

J0=J(q0)>0가 유한하고 q0가 f의 support에서 양수일 때, 정규화된 exact risk density를

\[
r_0(z)=\frac{f(z)^2}{q_0(z)J_0}
\]

로 정의한다. mu>0일 때 unrestricted 최적 사건 proposal은 q*=f/mu이며,
r0는 일반적으로 q*가 아니다. r0 자체가 natural 질량 .1을 가진다고 가정하지 않는다.

일반적인 정규화된 교정 r와 q_alpha=(1-alpha)q0+alpha r에 대해
integral f^2 |r-q0|/q0^2 < infinity이면, 우미분의 적분 교환이 가능하며

\[
\left.\frac{dJ(q_\alpha)}{d\alpha}\right|_{0+}
=\int\frac{f^2(q_0-r)}{q_0^2}
=J_0\left[1-E_{r_0}\!\left(\frac{r}{q_0}\right)\right].
\]

특히 r=r0이고 chi-square divergence가 유한하면

\[
J'(0+)=-J_0\chi^2(r_0\Vert q_0)\le0.
\]

현재 q0>=.1p 및 g<=1 계약에서는 r0/q0<=100/J0이므로 이 적분가능성은 충족된다.
더 강하게, g=0인 영역에서 0/0 기여를 0으로 정의하면

\[
J(r_0)=J_0\int_{\{f>0\}}q_0\le J_0.
\]

따라서 1/q의 convexity로 모든 alpha in [0,1]에 대해

\[
J((1-\alpha)q_0+\alpha r_0)
\le(1-\alpha)J_0+\alpha J(r_0)\le J_0.
\]

g>0 q0-a.e.이면 J(r0)=J0이다. 그러나 finite IID bank를 K2 Gaussian mixture로
압축한 r_risk에는 이 exact-target 보장이 이전되지 않는다. 그 적합은 국소 미분의
부호도 보장하지 않으며, alpha=.25 또는 .5의 개선은 독립 평가가 결정한다.
이 항등식·convexity 관측의 독창성은 주장하지 않는다. 선행문헌 검토 전에는 새로운
정리나 새로운 방법이라는 표현을 사용하지 않는다.

## 4. 사전 고정 혼합과 정확한 밀도

각 교정 branch에서 alpha in {0,.25,.5}만 후보로 두고, q_alpha의 full normalized
mixture density를 사용한다. q0와 실제 교정 r가 각각 .1p 이상이므로 q_alpha>=.1p이다.
또한 alpha<1이면 J(q_alpha)<=J(q0)/(1-alpha)이며, 현재 후보 alpha<=.5에서는

\[
J(q_\alpha)\le J(q_0)/(1-\alpha)\le2J(q_0)
\]

이지만, 이것은 악화 상한이지 개선 보장이 아니다. alpha=0은 q0 그 자체다.

q0와 r가 각각 natural 1개 + shift 2개이면 비자명한 blend의 표현은 성분 6개이며,
공통 natural 성분을 합치면 최대 5개이다. 따라서 교정의 fitting family가 K2라는
말과 최종 proposal이 K2라는 말을 혼동하지 않는다. alpha=0, 제거된 cluster 및
동일 mean의 병합은 더 적은 성분을 줄 수 있다. 동일한 밀도는 같은 proposal로 표시하고,
같은 밀도에서 발생한 독립 표본 평균/RSE 차이를 모델 개선으로 세지 않는다.

## 5. 서로 독립인 selection, final 및 paired M2 진단

부모 적합 -> 독립 training G bank -> 교정 동결 -> 별도 IID selection G bank ->
alpha 선택 및 모든 proposal 동결 -> fresh final의 순서를 지킨다. event/risk는
같은 selection bank를 공유할 수 있지만 train/final과 섞지 않으며, 결과를 본 후
alpha grid나 fit 설정을 늘리지 않는다. q0 tie를 우선하는 사전 고정 tie 규칙을 둔다.

selection에서는 ordinary raw 위험 기여

\[
H_q(Z)=\frac{g(Z)^2p(Z)^2}{q(Z)G(Z)},\quad Z\sim G,
\qquad E_G H_q=J(q)
\]

의 평균으로 각 branch의 alpha를 정한다. selection 최솟값은 선택된 위험의 불편
추정값이 아니므로 최종 개선 수치로 보고하지 않는다.

선택에 사용하지 않은 새 held-out IID G bank에서 q0, static G, 선택된 event/risk
proposal의 H_q를 같은 경로에서 모두 계산한다. 각 비교의 기본 통계는
D_i=H_a(Z_i)-H_b(Z_i), mean(D), sample_sd(D)/sqrt(n)이다.
M2 비율은 보조 통계이며, 그 SE에는 같은 경로의 covariance를 포함한다. 각 M2의
SE를 독립인 것처럼 더하거나 비율의 분모를 참값으로 취급하지 않는다. leave-one-block,
최대 기여 share와 block 민감도는 미관측 tail의 부재를 증명하지 않는 진단이다.

q>=.1p, G>=.1p 및 g<=1이면 0<=H_q<=100이다. 이 절대 유계성은 작은 M2의
상대 정밀도나 작은 bank의 tail coverage를 보장하지 않는다. 낮은 직접 사건 RSE만으로
paired 위험 진단의 불안정성을 덮지 않는다. 신호가 불명확하면 unresolved로 남긴다.

mu는 모든 proposal에 공통이므로 M2의 순서/차이를 비교하는 데 mu 참값은 필요 없다.
그러나 accuracy가 자격을 얻지 못한 결과로 fixed-precision 총 wall-time 우위를
인증할 수는 없다. cost-weighted variance는 c_q[J(q)-mu^2]이며 c_q J(q)와 다르다.
기존의 accuracy 인증 관문은 이 개발 실험의 시작 조건이 아니고, 성능 인증에는 유지된다.

## 6. 고정 예산의 개발 pilot과 판정 범위

실행 전 제안하는 bounded CPU pilot은 다음과 같다.

- 기존 2셀 x 새 전체 부모 학습 5회. 잘 된 parent만 고르지 않는다.
- 부모당 training IID G 8,192개, 별도 selection IID G 16,384개.
- 최종 4 arm은 q0 / static G / 선택된 event correction / 선택된 risk correction이다.
  각 arm에서 새 IID 8 block x 32,768개 = 262,144개를 사용한다.
- 모든 proposal 동결 후 별도 paired held-out G 131,072개를 쓴다. 이 bank는
  training, selection 및 각 arm의 direct final과 독립이며 후속 재선택에 쓰지 않는다.
- 기존 부모 SMC가 768 x [1+47 x 2] = 72,960 payoff 평가를 쓰는 경우, 위 네 arm을
  모두 실행하는 상한은 10 x [72,960+8,192+16,384+4 x 262,144+131,072]
  = 12,771,840 payoff 평가이다. 이는 density/fit 비용이나 실패 후 소비를 제외한 수치다.

이 수는 원하는 상대 정밀도의 보증이 아니라 고정된 반증 기회다. 모든 계획된 반복,
실패, 불명확한 비교를 보존하고, parent 단위의 paired 결과와 전체 분포를 보고한다.
8 IID block을 8개 새 parent 학습으로 세지 않는다. 5회 학습의 불확실성은 큰 sample
count의 조건부 IID SE와 별개다. 여러 셀/비교의 단일 CI들을 전역 확인적 유의성으로
해석하지 않으며, 선택 후 개선 claim에는 사전 고정한 비교와 다중성 처리가 필요하다.

위험 교정이 event 교정이나 q0를 이겼다는 개발 신호와, 강한 static G보다 실용적으로
좋다는 신호를 분리한다. 어느 것도 자동 certification은 아니다. 비용 기록은 부모 학습,
G geometry setup, bank 생성/payoff/full density, event/risk fit, selection의 모든 후보
density, 최종 sample/payoff/full density를 분리한다. 공유 계산의 실제 소비와 각 방법을
단독 배치할 때 필요한 비용을 함께 기록하고, static setup을 0으로 숨기거나 static에
불필요한 부모 SMC 비용을 부과하지 않는다. 같은 static final을 반복 간 재사용한다면
그 의존성과 공유 비용을 명시하며 독립 parent 비교로 중복 집계하지 않는다.

결론은 N32 모델 개발의 제한된 채택/기각/불명확 판정이다. original-q 개선, 보조 M2
평가 효율, fine-reference 정확도 및 continuum/P1/P2는 끝까지 다른 주장으로 남긴다.
