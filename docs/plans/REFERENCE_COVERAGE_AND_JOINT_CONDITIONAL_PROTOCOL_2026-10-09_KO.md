# 독립 coverage 대조 및 exact joint conditional 구현 계획

2026-10-09. 전날 실패 결과를 변경하지 않는 후속 개발 프로토콜이다. 현재 q·finite grid·payoff·학습 bank는 동결한다. 본 문서는 아래 새 실험을 실행하기 전에 작성한다.

## 1. 우선순위와 범위

1. 기존 guide와 별개인 IID proposal 하나로 rare contribution coverage를 점검한다.
2. cached evaluator와 full-r static baseline을 유지한다.
3. 고정 orthogonal 좌표 분할에서 shift-mixture의 exact marginal·conditional law를 구현·검증한다.
4. 결과에 따라 다음 방향을 보고한다. P1 미통과 상태의 P2, 새 학습, 봉인 confirmation, 차원 증가, 자동 연장, seed 교체는 금지한다.

이번 후보는 **defensive local variance-inflation mixture**다. d=2N whitened Gaussian을 시간별 서로 겹치지 않는 2좌표 block으로 나눈다. r=δp+(1−δ)/N Σ_t r_t, δ=0.1, r_t의 해당 block 분산만 λ=25이며 나머지는 1이다. toy는 각 1좌표를 block으로 한다. 평균·mode·q·과거 fitted guide를 읽어 방향을 정하지 않는다. 양·음 극단을 대칭적으로 탐색한다. 이는 기존 Gaussian mixture IS의 진단용 적용이지 새 원리라는 주장이 아니다.

## 2. 수학적 오류 방지

block k의 크기를 b라 하면 log(r_k/p)=−b log(λ)/2+(1−1/λ)||z_k||²/2다. 전체 밀도는 logsumexp로 합산한다. 샘플링 label과 Gaussian 잡음은 별도 deterministic seed를 사용한다. 선택된 label의 밀도만 쓰지 않고 전체 혼합 밀도를 사용한다.

μ contribution은 g/(r/p), original M₂ contribution은 g²/[(q/p)(r/p)]다. q는 그대로다. self-normalization, clipping, pilot 선택, component label을 event mode로 오해하는 처리는 없다. r≥0.1p, q≥δq p이므로 유한분산은 확보하지만 tiny mean의 작은 relative error는 보장되지 않는다. 단일 block 확대는 여러 시간대의 동시 극단을 잘 탐색하지 못할 수 있다. 이것을 의도적으로 반증 가능한 한계로 남긴다.

## 3. joint conditional 기반

고정 orthogonal U에서 x=zU, x=(A,B)다. identity-covariance shift-mixture의 means는 mU로 변환한다. r_A/p_A=Σw_j exp(A·m_Aj−||m_Aj||²/2), conditional component 확률 α_j(A)는 이 항들을 정규화한 값이다. r(B|A)/p_B=Σα_j(A) exp(B·m_Bj−||m_Bj||²/2).

log(r/p)=log(r_A/p_A)+log(r(B|A)/p_B)를 검사한다. 전체 모형 mean norm을 marginal에 잘못 남기거나 conditional weight를 원래 w로 고정하지 않는다. conditional sampled H_q p_B/r(B|A)의 평균은 E_pB H_q다. L=1은 full-r IS의 분해이며 exact 평균만 RB다. nested finite L의 성능은 자동 보장하지 않는다. 변환은 고정이며 표본별 spike 선택은 허용하지 않는다. nonzero covariance-rank는 이번 구현에서 명시적으로 거부한다.

## 4. 고정 작은 실험

- toy 3종(centered/shifted/bimodal), μ와 q=p M₂. 자연 IID와 variance mixture를 비교한다.
- 실제 canonical·높은 η, parent training rep 0의 q만 사용한다. 기존 full-r excursion guide와 새 proposal의 독립 IID 표본을 비교한다.
- 모든 방법마다 8개 독립 반복 × 32,768 draw, batch 4,096. μ/M₂는 같은 draw를 재사용하는 **paired targets**이며 서로 독립이라는 주장을 하지 않는다. 서로 다른 방법·반복·batch의 label/path stream은 분리한다.
- 실험 phase(준비·I/O·summary 포함, 사전 입력 읽기·source archive는 별도 보고) wall cap 900초, 실제 Volterra FFT+CDF 상한 1,050,000회(샘플 1,048,576회와 별도 준비·oracle 포함). toy 호출은 별도로 계수한다. CPU float64, thread 1. 실패 시 결과 보존 후 차단한다. 자동 연장 없음.
- source/config/input q binding·seed·raw log contribution shard와 계산 비용을 저장한다. reference 단위는 IID draw, 반복 민감도는 whole independent batch/replicate로 구분한다.
- toy oracle 상대 오차, RSE, 최대 기여 및 고정 영역 기여를 기록한다. 실제는 시간대 최대 variance 위치 기반 partition의 weighted μ/M₂ 기여를 기록한다. proposal label과 실제 geometry는 별개다.
- 실제 두 방법이 모두 RSE≤2%, 최대 single contribution≤1%, 독립 반복 mean 민감도≤10%, family-adjusted mean 동등성을 충족하는지 개발 gate로 확인한다. 이것은 unseen-tail 인증이나 production qualification이 아니다. 비용 비교도 accuracy 자격 미충족 셀에서는 승자 선정에 사용하지 않는다.
- 결과와 상관없이 P2=false, production reference=false. 새로운 방법의 우수성이 확인되더라도 더 넓은 fixed-q와 독립 메커니즘 확인을 다시 사전 고정해야 한다.

## 5. 검증과 전달

SciPy Gaussian logpdf와 density 비교, quadrature normalization, defensive bound, sampling covariance, conditional posterior weight·rotation·전체 density factorization·독립 inner 평균을 검사한다. 원 simulator 표본 경로를 재검산한다. 저장된 모든 log contribution의 summary/seed/source hash를 재감사한다. 관련 테스트 후 전체 pytest·Ruff·CI mypy를 실행한다.

현재 요청은 우선순위 구현이며 새 commit/push는 명시 요청 전 하지 않는다. 원본 2026-10-08 자료와 실패 상태는 보존한다.

참고: [Owen–Zhou, Safe and Effective Importance Sampling](https://artowen.su.domains/reports/seis.pdf)은 방어적 혼합분포의 기존 연구다. 이번 구현은 해당 논문의 control-variate 알고리즘을 구현했다는 뜻이 아니다. [Owen, variance reduction](https://artowen.su.domains/mc/Ch-var-basic.pdf)의 conditional Monte Carlo 원리와 별개로 outer coverage·계산 비용은 실험 대상이다.
