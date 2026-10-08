# S2 bank 안정성 대조·독립 위험 평가 실행 보고서

작성일: 2026-10-07

실행 범위: [기존 개선계획](MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md)의 S2 및 [S0/S1 결과](STRUCTURAL_IMPROVEMENT_S0_S1_EXECUTION_2026-10-07_KO.md)의 다음 관문 1–2.

## 1. 결론과 현재 결정

**SMC island 배분만으로 문제가 해결된다는 가설은 이번 개발 실험에서 지지되지 않았다.** 정확도 자격을 얻은 후보는 없고, canonical의 독립 위험 평가도 충분히 정밀하지 않았다. 따라서 새 구조·차원·operator·대규모 confirmation을 추가하지 않았으며, 어떤 island 설정도 최종 우승 모델로 선택하지 않는다.

중요한 새 관측은 다음과 같다.

1. 가중 SMC bank의 particle ESS가 약 600/768로 높아 보여도, 고유 초기 조상은 약 45–82개 수준이다. 이를 600개 독립 학습 표본으로 해석하면 안 된다.
2. 동일 예산의 1/3/6-island 배분은 ancestry를 크게 바꾸지 못했다. 새 IID bank의 target ESS는 설정에 따라 달랐지만, 최종 정확도·독립 위험까지 개선됐다고 입증하지 못했다.
3. 여러 canonical proposal에서 risk-SMC M₂가 직접 IS의 M₂보다 수백–천 배 크게 추정됐다. **위험한 희귀 기여를 작은 직접 평가가 놓칠 수 있다는 신호**다. risk-SMC 자체 RSE도 높으므로, 이 배수를 참 분산이나 엄밀한 lower bound로 사용하지 않는다.
4. 기존 부모와 mixture 보존 재적합의 고정 proposal 20개를 새 난수로 평가해도 이러한 불일치가 관찰됐다. 단일 shift 압축만이 유일한 병목은 아니다.

계획에 있는 risk-SMC 불안정 시 `unresolved` 규칙을 적용했다. bank ancestry만을 기준으로 구조 변경을 채택하거나, 낮은 직접 RSE만으로 논문용 성능 데이터를 만들지 않았다.

## 2. 구현과 봉인된 설계

실행기: [post_audit_r2_bank_risk_diagnostics.py](../../experiments/post_audit_r2_bank_risk_diagnostics.py)

### A. 새 부모 전체 학습과 island 배분 대조

- 셀: canonical, 높은 η의 두 기존 development 셀.
- 부모 전체 학습: 셀별 5회 × 3개 설정 = **30개 새 부모**.
- bank 배분: **1×768 / 3×256 / 6×128**.
- 동일 N=32, conditional dimension=64, bridge 48단계, power=4, mutation 2회, pCN scale=0.35, 4단계마다 stratified resampling.
- 마지막 β=1 단계는 기존 구현과 같이 mutation하지 않는다. terminal rejuvenation은 이번 대조에 추가하지 않았다.
- 각 부모의 potential 평가 수는 **768×[1+47×2]=72,960회**로 동일하다.
- fitting family는 natural 질량 0.1 + 학습 Gaussian 성분 2개, identity covariance를 유지했다. 30개 실제 fit 모두 학습 성분 2개를 유지했다.
- fitting에 쓰지 않는 새 IID bank 진단 4,096개, 새 직접 IS 평가 131,072개.
- 부모마다 별도 risk-SMC: 128 particles × 독립 whole replicate 8개, 위와 같은 bridge/mutation 설정.
- 총 상한: potential 평가 1,200만 회, wall 600초. 실패나 budget 초과를 삭제하지 않는다.

같은 potential work가 같은 wall-time을 뜻하지는 않는다. 큰 batch와 여러 작은 island의 Python/밀도 연산 overhead는 다르며 실제 시간은 별도로 기록한다.

### B. 기존 고정 proposal의 독립 위험 대조

- 출처: S0/S1의 5회 부모 학습 artifact.
- 각 셀·부모 반복의 `parent-as-is`와 `same-family-refit` 모두 사용했다. 유리한 반복만 선택하지 않았다.
- **20개 고정 proposal**에 새 직접 IS 65,536개와 별도 risk-SMC 8회/128 particles를 적용했다.
- 부모를 새로 학습하지 않았으므로 이 실험을 20회 전체 학습 안정성 검사로 세지 않는다.
- 새 결과는 원본 artifact SHA256 및 개별 proposal digest와 연결한다.
- 총 상한: potential 평가 400만 회, wall 300초.

A와 B는 직접 표본 수가 다르다. 두 실험 간 RSE 숫자를 같은 계산량 성능 순위로 비교하지 않는다. 각 실험 안의 방법 간 대조만 matched-count 개발 진단이다.

## 3. 위험 평가의 수학적 계약

고정 q에서 X=g p/q라 두면,

\[
M_2(q)=E_q[X^2]=\int g^2p^2/q.
\]

q가 natural 성분 δp를 포함하고 0<g≤1이면,

\[
h=\delta g^2p/q,\qquad 0<h\le1,\qquad E_p[h]=\delta M_2(q).
\]

따라서 SMC는 `p h^β`를 target으로 삼고, **whole-run normalizer Ẑ/δ**로 고정 q의 M₂를 평가한다. q는 평가 중 학습하거나 갱신하지 않는다. terminal 입자의 normalized weight나 ESS를 normalizer estimator로 대체하지 않는다.

- 직접 M₂는 새 IID q 표본의 **mean(X²)**이며 ordinary IS이다.
- 직접 M₂의 SE와 사건확률 mean(X)의 SE는 서로 다른 통계다. 실행기와 감사기는 두 batch moment를 별도로 저장·재계산한다.
- risk-SMC의 SE 단위는 **독립 전체 실행 8개**다. 각 실행의 128개 terminal particle을 독립 반복으로 세지 않는다.
- 고정 schedule, unbiased stratified resampling, Gaussian reference 보존 pCN 및 `β[log h(candidate)−log h(current)]` MH를 사용한다. 기존 weighted SMC의 weight 운반 로직은 변경하지 않았다.
- `log h=log δ+2log g−log(q/p)`를 full-mixture density로 계산한다. clipping/self-normalization을 최종 추정에 적용하지 않는다.
- 상수로 potential을 축소하면 normalizer는 축소되지만 normalized target·SMC geometry가 같음을 시험했다. 작은 Z만으로 문제가 어렵다고 결론 내리지 않는다.
- log 값은 유효하지만 ordinary float 평균으로 변환할 때 underflow/overflow가 발생하면 `unresolved_risk_numerical_range`를 기록한다. 0 또는 Inf를 정상 위험값으로 쓰지 않는다.

`risk M₂ / reference_mean²`와 두 M₂의 SE-scaled 차이는 **탐색적 지표**다. noisy historical reference의 비율은 unbiased 상대 second moment가 아니며, whole replicate 8개로 CLT/신뢰구간의 정확성을 인증하지 않는다.

## 4. A: island 대조 결과

- 30개 모두 실행 완료, potential 평가 **9,162,240회**.
- 물리적 실행시간 **226.86초**.
- 정확도 qualification **0/30**. risk-SMC 개발용 RSE≤20%는 **3/30**. 이는 oracle 자격이 아니다.

아래는 각 셀·설정의 부모 학습 5회 **중앙값**이다. 각 열의 중앙값이 같은 반복에서 나온다는 뜻은 아니다.

| 셀 | island | 고유 초기 조상 합 | IID bank target ESS /4096 | 직접 IS RSE | risk-SMC RSE | risk/direct M₂ 비 |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 1 | 45 | 24.72 | 8.19% | 56.99% | 467.92 |
| canonical | 3 | 47 | 7.15 | 6.72% | 51.75% | 1080.44 |
| canonical | 6 | 48 | 23.56 | 5.58% | 53.61% | 1093.59 |
| 높은 η | 1 | 79 | 22.26 | 13.52% | 21.37% | 2.90 |
| 높은 η | 3 | 82 | 12.14 | 7.05% | 27.04% | 49.43 |
| 높은 η | 6 | 82 | 17.92 | 14.26% | 22.84% | 6.85 |

조상 합은 island별 label에 island identity를 붙인 개수다. 섬끼리 label이 우연히 같은 숫자여도 같은 조상으로 합치지 않는다. 조상 합이나 가중 particle ESS 자체는 최종 effective iid sample size가 아니다.

가중 bank particle ESS 중앙값은 canonical에서 약 611–618/768, 높은 η에서 671–675/768이다. 반면 새 IID bank의 target ESS는 낮다. 기존 bank의 좋은 weight ESS만으로 proposal이 희귀 기여 영역 전체를 잘 표현한다고 판단할 수 없다.

부모 학습·fit의 offline 시간 중앙값:

| 셀 | 1 island | 3 islands | 6 islands |
|---|---:|---:|---:|
| canonical | 1.22초 | 1.69초 | 2.54초 |
| 높은 η | 0.94초 | 1.36초 | 1.99초 |

이 표는 overhead 진단이지 fixed-precision 총비용 우위가 아니다. 어느 설정도 정확도 자격을 통과하지 못했다. 같은 potential work를 wall-time 개선이라고 발표하지 않는다.

## 5. B: 고정 부모·재적합 위험 대조 결과

- 20개 모두 실행 완료, potential 평가 **3,256,320회**.
- 물리적 실행시간 **84.85초**.
- 정확도 qualification **0/20**. risk-SMC 개발용 RSE≤20%는 **5/20**.

| 셀 | 고정 proposal | 직접 IS RSE 중앙값 | risk-SMC RSE 중앙값 | risk/direct M₂ 비 중앙값 | risk 개발 정밀도 통과 |
|---|---|---:|---:|---:|---:|
| canonical | 부모 그대로 | 5.98% | 34.42% | 1159.49 | 0/5 |
| canonical | 같은-family 재적합 | 17.67% | 54.71% | 401.27 | 0/5 |
| 높은 η | 부모 그대로 | 9.70% | 25.17% | 32.14 | 2/5 |
| 높은 η | 같은-family 재적합 | 16.61% | 18.33% | 121.73 | 3/5 |

불리한 관측도 남긴다. canonical 부모 반복 1의 같은-family에서는 risk/direct M₂ 비가 **0.56**, risk RSE가 약 **59.6%**였다. 모든 반복에서 risk-SMC가 더 큰 값을 주거나 정확하다고 가정할 수 없다. 이 사례도 제외하지 않았다.

이 진단으로 부모 q의 표현 한계와 fitting bank 문제를 압축 손실과 분리할 필요가 커졌다. 그러나 아직 true M₂를 충분한 정밀도로 확보하지 못했으므로, 어느 fitting이 실제로 더 좋은지 최종 순위를 매기지 않는다.

## 6. 구현 재검토에서 수정한 부분

1. **실패 계산량 기록:** 성공한 SMC 결과가 반환된 뒤에만 평가 수를 더하면, 내부 수치 오류로 중단된 계산이 빠질 수 있었다. payoff 호출 입구에서 실제 호출 표본 수를 기록하도록 수정했다. SMC 내부에서도 wall 상한을 확인한다. 강제 오류 검사에서 48/16/8개 첫 호출이 각 실패 job에 보존됨을 확인했다. 실제 A 실험은 모두 성공 반환했으므로 이 수정으로 과거 수치가 달라지지 않는다.
2. **실패 stage 시간:** `finally`로 실행 중 stage의 시간까지 저장한다. 실패를 0시간 성공처럼 집계하지 않는다.
3. **risk 통계 단위:** source/proposal binding에 더해 whole-normalizer 통계를 독립 재계산한다. particle SE로 잘못 바꾸는 경로가 없다.
4. **정확도·누락 감사:** 실제 봉인된 job 집합과 artifact record를 비교한다. 삭제된 실패 job, 중복 job, 바뀐 고정 proposal/reference, batch moment 불일치, stale qualification, 예산 초과를 거부한다.
5. **수치 범위:** log-space 값과 float mean의 표현 가능 범위를 구분했다. 강제 underflow/overflow 검사에서는 위험값을 조용히 0/Inf로 변환하지 않는다.

각 결과는 실행 당시 source ZIP에 묶여 있다. 이후 감사·범위 처리 보완으로 현재 source digest가 달라도 과거 artifact를 다시 쓰지 않는다. 감사기는 그 ZIP과 당시 설정을 검증한다.

## 7. 다음 행동 — 기존 계획 안의 중단·전환

### 지금 하지 않는 것

- 더 많은 island/차원/rank/operator를 임의로 탐색하지 않는다.
- 직접 RSE가 가장 낮은 seed나 설정을 성공 모델로 채택하지 않는다.
- 불안정한 risk-SMC를 새로운 oracle/reference로 등록하지 않는다.
- risk 보정 mixture의 성능이나 novelty를 주장하지 않는다.
- 20회 확인 학습·24개 미사용 task 확인 실험·저널용 fixed-precision 비용 표로 넘어가지 않는다.

### 다음 제한된 개발 묶음

1. **risk 평가의 자체 안정성:** 동일 고정 q에서 기존 계획의 bridge/resampling·mutation 대조를 소수 설정으로 봉인하고, whole replicate를 늘릴 예산을 throughput으로 먼저 계산한다. 조건부 payoff의 사건확률 reference와 risk normalizer를 구분한다. 현재 M₂ 배수는 그 예산의 정확한 forecast가 아니다.
2. **geometry 밖 기여 진단:** 학습 geometry와 독립 calibration threshold를 고정한 뒤, 학습 PCA complement·최근접 중심 거리로 “학습 geometry 밖”을 표시한다. mathematical support나 모든 mode 발견을 인증하지 않는다. risk 탐색과 직접 IS의 기여 영역을 비교한다.
3. **보정 후보를 실행할 조건:** 위험 bank로 r을 적합하는 단계는 독립 selection/final과 동등 비용 event-bank 대조가 준비된 경우에만 진행한다. qα=(1−α)q0+αr의 α=0 baseline을 보존하고, 제한된 사전 고정 α 외에 결과를 보고 무한 튜닝하지 않는다.
4. **reference/allocation:** 최소 두 estimator의 정확도와 reference SE 비율을 확보한 뒤에만 fixed-precision 실제 총비용을 비교한다. 필요 reference 예산이 현재 노트북 상한을 넘으면 unresolved 및 예상 비용을 보고한다.

이번 결과는 risk 중심 진단을 우선해야 한다는 탐색적 근거다. 새로운 수학적 기여나 논문 게재 가능성의 입증은 아니다. 위 제한된 대조에서도 신호가 없으면 계획의 scope/연구 질문 재검토 규칙을 따른다.

## 8. 파일과 검증 상태

- [bank/island 설정](../../configs/post_audit/r2_bank_islands_risk_v1.yaml), [결과](../../results/post_audit/r2_bank_islands_risk_v1.json)
- [고정 proposal 설정](../../configs/post_audit/r2_frozen_proposal_risk_v1.yaml), [결과](../../results/post_audit/r2_frozen_proposal_risk_v1.json)
- [위험 potential·whole replicate 통계](../../src/path_integral/conditional_second_moment.py)
- [oracle·프로토콜·강제 실패 검사](../../tests/test_r2_bank_risk_diagnostics.py)

두 artifact의 source snapshot·proposal·seed·산술 감사는 통과했다(A: 30 jobs/1360 seed streams, B: 20 jobs/480 seed streams). 전체 Ruff와 mypy(168개 소스 파일)는 통과했다.

- 전체 pytest: **1,099개 통과**, 167.58초. 새 위험 통계·matched-work·oracle·강제 실패·수치 범위 검사를 포함한다.
- 커밋·푸시: 하지 않음.
- 검토한 계약 내에서 오류를 탐지·수정했으며, 모든 이론적·기술적 오류의 부재를 보장하지 않는다.
