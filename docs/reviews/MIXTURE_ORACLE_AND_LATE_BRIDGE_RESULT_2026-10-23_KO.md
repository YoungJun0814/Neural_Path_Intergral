# 혼합분포 정답 검증과 late-bridge 한정 대조 — 결과·재검토

2026-10-23 America/New_York.
[사전 계획](../plans/MIXTURE_ORACLE_AND_LATE_BRIDGE_2026-10-23_KO.md)에 따른 별도 개발 실험이다.
이전 SMC STOP을 소급 통과시키거나 원본을 변경하지 않는다. commit/push는 하지 않는다.

**결론:** 혼합분포 oracle(A)는4조건/64반복과 전체 재생 감사를 통과했다.
실제 late-bridge 대조(B)는16개를 완료했지만 `STOP_LATE_BRIDGE_BANK_UNQUALIFIED`다.
개발용 normalizer 반복 기준은 통과했고 bank 계보 기준은 남았으며, G/P2 채택은 없다.

## 1. 이번 변경의 범위

새 모델/차원/operator를 추가하지 않았다.
먼저 nontrivial mixture에서 기존 SMC의 target/normalizer 연결을 독립 정답과 확인한다(A).
A와 그 전체 재생 감사가 통과할 때만 실제 Volterra에서 **temperature 배치 하나**를 대조한다(B).
사건 proposal q, scalar 적분, kernel, particle 수, stage 수와 resampling index는 고정한다.

## 2. A: nontrivial mixture 정답 검증

`p=N(0,1)`, `q=0.2N(0,1)+0.8N(2,1)`,
`m2(x)=exp[-0.4(x-1)^2-0.3]`로 q≠p인 상황을 만들었다.
SMC와 별도의 scalar 코드로 `q^(1-2beta) p^(2beta) m2^beta`를 적분했다.
무한구간 adaptive quadrature와 유한구간 Gauss–Legendre256/512를 대조했다.
beta0의 analytic mixture moment, beta1의 Gaussian tail bound와 잘못된 endpoint의 closed form도 확인했다.
이는 이1D 문제의 강한 구현 진단이며 다차원/연속시간/모든 potential 정확도의 증명이 아니다.

정답 기준값은 다음과 같다.

| beta1 endpoint | 값 |
|---|---:|
| Z1 | 0.9978230506953805 |
| E[X] | 0.01159450548606075 |
| E[X²] | 0.3963254726548829 |
| P(X>0) | 0.5121596687459116 |

beta1에서 독립 raw 적분의 최대 차이는3.02e-14였다.
잘못된 single-Dq endpoint `p*m2`의 normalizer는 올바른 값과 **55.69%** 달랐다.
따라서 q=p인 trivial 검사만으로 density 보정 오류가 가려지는 상황을 보완했다.
비외향 반올림 적분 error proxy와 실수 산술의 tail bound를 혼동하지 않는다.

512particle,32stage,조건당16개 독립 island의 사전4조건은 모두 통과했다.

| Schedule / kernel arm | Z 상대오차 | 독립 island RSE | 평균 오차 | 2차 moment 오차 | 오른쪽 질량 오차 |
|---|---:|---:|---:|---:|---:|
| Quartic / prior | 1.174% | 0.521% | 0.00648 | 0.00491 | 0.00370 |
| Quartic / component | 0.283% | 0.674% | 0.00211 | 0.00406 | 0.00557 |
| Linear / prior | 0.420% | 0.741% | 0.01419 | 0.00602 | 0.00679 |
| Linear / component | 0.872% | 0.650% | 0.00234 | 0.00260 | 0.00689 |

RSE≤2.5%, 상대 Z오차≤4%,4-SE 차이 검사 및 세 target moment의 절대오차 gate를 모두 만족했다.
normalized particle 평균의 유한-N bias가0이라고 주장하지 않는다.
이 비교는 단순 oracle이므로 어느 kernel/schedule이 실제 Volterra에서 더 좋다는 근거가 아니다.

생산 Gaussian potential contexts3,178,496, scalar reference evaluations17,136,
Volterra physics0. source freeze4.369초, reference1.661초,
island compute+I/O21.710초, end-to-end28.067초였다.
host 독점/정밀도 일치 speedup benchmark는 아니다.

64개 oracle 전부의 동일-seed 재생 감사도 통과했다.
감사 Gaussian contexts3,178,496, wall21.413초였고, 생산+감사 Gaussian contexts는6,356,992였다.
reference suite도 감사에서 다시 계산했다(각17,136 scalar evaluations).
Volterra physical은0이며 단위 테스트 계산은 별도다.
[A 생산](../../results/post_audit/mixture_smc_oracle_v1/session_20261023/result.json),
[독립 기준값](../../results/post_audit/mixture_smc_oracle_v1/session_20261023/reference.json),
[A 감사](../../results/post_audit/mixture_smc_oracle_v1/session_20261023/audit.json).

## 3. B: 고정 work의 실제 Volterra 대조

기존 quartic `(j/32)^4`와 선형 `j/32`를 비교한다.
beta>0.5인 stage 수는6개에서16개로 바뀐다. 나머지 설정은 동일하다.
기존 표본을 control로 재사용하지 않고, 양 셀/두 schedule/4개 독립 반복의 새16개 island를 사용한다.
선형 schedule이 최적이라는 가정이나 acceptance를 높이면 해결된다는 가정은 하지 않는다.

전체16개 island가 numerical/resource/technical 실패 없이 완료됐다.
198,656개 proposed context의 개별 수치 gate가 통과했고 최대 상대 numerical proxy는1.787e-12였다.
이 proxy는 엄밀한 bias certificate가 아니다.

**사전 최종 판정: `STOP_LATE_BRIDGE_BANK_UNQUALIFIED`.**
실제 G fitting0, event q 변경false, P2/D3/large training false다.

| 셀 / schedule | 개별 bank 통과 /4 | 독립 normalizer RSE | 최대 island 질량 | logZ spread | normalizer 자격 |
|---|---:|---:|---:|---:|---|
| Canonical / quartic | 0 | 17.39% | 32.96% | 0.669 | 통과 |
| Canonical / linear | 3 | 16.13% | 36.55% | 0.654 | 통과 |
| High eta / quartic | 1 | 16.39% | 35.11% | 0.832 | 통과 |
| High eta / linear | 1 | 17.02% | 36.34% | 0.715 | 통과 |

이번에는 양 셀의 두 schedule이 모두 **개발용 normalizer 반복 기준**을 통과해,
사전 combined-SE 차이 검사를 수행했다. 양 셀 모두 `no_detected_contradiction=true`였다.
이는 equivalence/정답 인증/다른 메커니즘의 reference와 교차검증 성공이 아니다.
개별 bank gate는 독립 조건이므로 normalizer RSE 통과로 대체하지 않았다.
bank 자격이 미달해 실제 G fitting의 fixed-precision total-wall 성능 비교는 없다.

### 주후보 linear의 실패 항목

| 셀 / 반복 | 초기 조상 수 | 조상 ESS | 최대 조상 질량 | nominal ESS | 판정/실패 항목 |
|---|---:|---:|---:|---:|---|
| Canonical /0 | 25 | 13.47 | 11.94% | 114.19 | 통과 |
| Canonical /1 | 26 | 10.14 | 20.89% | 121.41 | 통과 |
| Canonical /2 | 23 | 7.09 | 28.92% | 122.23 | 조상 ESS/질량 |
| Canonical /3 | 27 | 13.84 | 16.71% | 120.54 | 통과 |
| High eta /0 | 27 | 14.92 | 13.97% | 116.85 | 통과 |
| High eta /1 | 34 | 10.93 | 25.06% | 120.50 | 조상 질량 |
| High eta /2 | 29 | 6.59 | 25.74% | 98.53 | 조상 ESS/질량 |
| High eta /3 | 32 | 9.64 | 26.99% | 97.08 | 조상 질량 |

primary8개 모두 distinct128, max particle 질량≤2.23%, movement gate 통과였다.
late acceptance35.04–39.57%, median cumulative squared jump128.12–159.47였다.
quartic보다 acceptance/Euclidean jump가 작아도 최종 nominal ESS는 높았다.
따라서 acceptance나 nominal ESS 하나로 실제 bank 신뢰성을 판단할 수 없다.
high-eta 반복1은25% 기준을0.058 percentage point 넘은 borderline 실패지만,
관측 후 그 기준을 완화하거나 반복1만 살리지 않았다.
다른 실패의 조상 ESS6.59/7.09도 무시하지 않았다.

K3 최소 존재 gate는 두 셀 모두 통과했다.
그러나 high-eta 세 번째 cluster 질량은 반복별9.39%,7.20%,62.40%,35.37%로 달랐다.
clustering은 동일 bank의 사후적 기술 진단이며 quantitative mode coverage/held-out 검증이 아니다.

### 새 표본 없는 trace 분석: 집중이 사라졌나, 위치가 바뀌었나?

저장된 stage만 기술적으로 분석했다. 새 random draw0, gate/판정 변경0이다.
quartic의 stage4 직전 nominal ESS는두 셀 모두거의128이었고,
마지막에는canonical44.59–66.61/high-eta58.12–74.78로 떨어졌다.
linear는 stage4(beta0.125)에서canonical61.69–77.74/high-eta71.70–82.10으로 먼저 감소했고,
마지막에는canonical114.19–122.23/high-eta97.08–120.50였다.

즉, 후반 weight 집중은 완화되는 양상이지만 초기 선택 부담이 생겼다.
남은 초기 조상 수도 linear23–34, quartic31–42로 달랐다.
resampling과 mutation으로 현재 particle 좌표/weight가 퍼져도 초기 lineage는 다시 늘어나지 않는다.
이것은 **집중의 시간적 위치가 바뀌는 양상**이지 충분한 mixing/인과적 기전의 증명이 아니다.

두 schedule은 독립 seed이며 조건당4개뿐이다. linear bank 통과4/8 대 quartic1/8은
관측된 개발 count이지 통계적으로 확정된 우위나 confirmation 성공이 아니다.
새 quartic도 normalizer 기준을 통과했으므로, 이전 실패 run 대비 RSE가 작아진 것을
linear 변경의 효과로 귀속할 수 없다.
이전 high-eta component run은 한 island가71.57%를 차지했다.
이번에 그런 극단값을 덜 봤다고 해서 population tail을 충분히 봤다는 결론도 낼 수 없다.

### B 생산 비용

physical198,656, scalar749,226,240으로 사전 cap 안이었다.
동일 potential context 수이지만 scalar node 수는 profile별 panel/padding 때문에 조금 다르며,
quartic374,912,256/linear374,313,984였다.
setup4.092초, source freeze4.001초, protocol I/O0.00786초,
compute+island I/O572.764초(그중 island I/O0.419초), 집계0.0516초,
result I/O0.0142초, end-to-end581.626초였다.
회귀검증과 일부 병렬 실행했으므로 독점 host 또는 wall-speedup 결과가 아니다.

재생 감사도 **통과**했다.16개 final bank 전부를 다시 평가하고,
양 셀의 linear 반복0 두 개를 처음부터 끝까지 재생했다.
최종 logm2 최대 차이는1.776e-15였고, 지정된 full replay의 stage/trace/판정은 동일했다.
나머지14개 island는 final state 재평가와 hash/trace/집계 확인 범위이며,
rejected proposal 전부를 재생했다고 하지 않는다.

감사 physical26,880/scalar101,210,112/wall91.900초.
B 생산+감사 physical **225,536**, scalar **850,436,352**였다.
A의 Gaussian contexts와 B의 Volterra physical을 동일한 work 단위처럼 합산하지 않는다.
감사는 별도 cap30,000/120,000,000/600초 안에서 수행했다.
[B 생산](../../results/post_audit/late_bridge_schedule_v1/session_20261023/result.json),
[B end-to-end 비용](../../results/post_audit/late_bridge_schedule_v1/session_20261023/end_to_end.json),
[B 재생 감사](../../results/post_audit/late_bridge_schedule_v1/session_20261023/audit.json).

## 3.1. 채택/중단/다음 연구 판단

- **채택:** nontrivial-mixture 독립 정답 검증 및 그 전체 재생 절차.
  linear는 후반 weight 진단을 위한 비교 설정으로만 보존한다.
- **미채택:** 현재linear bank를 production G 학습 bank로 사용, 좋은4개만 골라 fitting,
  사후 ancestry threshold 변경, P2/D3/대규모 학습, 논문 성능 우위 주장.
- **중단:** 이번 “temperature 재배치 하나로 해결” 가설의 추가 tuning.
  같은 결과를 통과시키려고 schedule/seed/particle budget을 계속 바꾸지 않는다.

현재 근거는 **기본 target 보정의 명백한 오류 가능성은 줄였지만,
신뢰할 수 있는 rare-contribution bank가 확보되지는 않았다**는 것이다.
이는 SMC나 연구 전체가 불가능하다는 증명은 아니며, top-journal 준비 완료도 아니다.

다음 우선순위는 또 다른 operator를 붙이는 일이 아니라 다음 둘을 먼저 분리하는 것이다.

1. **검증 기준의 연구:** initial ancestry는 보수적인 개발 기준이지 이론적 필요조건은 아니다.
   genealogy gate가 실제 target-moment/normalizer 오차를 얼마나 예측하는지
   rare multimodal known-answer와 독립 반복에서 검증하는 별도 한정 설계가 필요하다.
   이 결과를 이용해 현재 실패를 소급 통과시키지는 않는다.
2. **실제 Volterra 기준의 연구:** 현재 두 schedule은 같은 SMC 계열이다.
   다른 메커니즘의 accuracy-qualified 기준 estimator와의 교차검증, 더 많은 독립 반복과
   새 confirmation 없이는 population M2/누락 모드의 해결을 주장할 수 없다.

이 검증 투자가 타당한지 먼저 판단하고, 연구를
`새 proposal 모델의 성능 우위`로 계속 밀지, `정확한 조건부 분해+오류 진단/한계 분석`으로
범위를 축소할지 재설정해야 한다. 후자는 가능한 논문 기여 방향일 뿐 채택/게재 보장은 아니다.
현재 구현만으로 모델의 박사급/최상위 저널 수준을 입증했다고 말할 수 없다.

## 4. 코드·이론 검토와 잔여 한계

새 core는 [smc_mixture_oracle.py](../../src/path_integral/smc_mixture_oracle.py),
실행기는 [post_audit_mixture_oracle_late_bridge.py](../../experiments/post_audit_mixture_oracle_late_bridge.py)다.
사전36개 테스트를 통과했다. 기존 frozen numeric source는 수정하지 않았다.
이번에는 K3 표준화 floor를 **variance1e-8, 즉 std1e-4**로 명시했다.

1D oracle 통과는 구현 전체의 무오류/Volterra 모드 완전성/SMC 정밀도의 증명이 아니다.
4개 독립 Volterra 반복의 RSE가 작아도 공통 누락 모드 가능성은 남는다.
두 schedule의 자격/비반증 검사도 **같은 SMC 계열 내부 대조**이며,
다른 메커니즘의 독립 기준 estimator와 교차검증을 완료했다는 뜻이 아니다.
initial ancestry gate는 보수적인 개발 기준이지 necessary mixing theorem이 아니다.
bank gate와 실제 fresh G fit/selection/final/reference의 검증도 별개다.

## 5. 재현성·전체 회귀검증

전체 회귀검증 **1,828개 통과**(476.86초), source statement coverage85%
(22,860 statements/3,440 missed), 새 oracle core98%(89/2)다.
Ruff CI 범위, source219개 Mypy와 새 실행기1개 Mypy도 통과했다.
이는 local Windows/Python3.10 검증이며 새 remote Linux/Python3.11 CI 결과가 아니다.
알려진 Requests dependency warning은 로그에 남겼고 dependency는 변경하지 않았다.
실행 전 import 정렬 및 Python typing 오류를 수정하고, 수정 후 테스트/정적 검사를 다시 통과한 뒤 봉인했다.
봉인 후 numeric source/plan을 변경하지 않았다. 테스트 통과가 무오류/이론적 타당성의 완전한 증명은 아니다.
[검증 receipt](../../results/post_audit/late_bridge_schedule_v1/session_20261023/validation/regression_review.json),
[전체 테스트 로그](../../results/post_audit/late_bridge_schedule_v1/session_20261023/validation/pytest-output.txt),
[정적 검사 로그](../../results/post_audit/late_bridge_schedule_v1/session_20261023/validation/static_checks-output.txt).
전체 회귀검증은 B 생산과 일부 병렬 실행했다. 실제 wall은 기록하되 독점 host benchmark나
schedule 간 matched-wall 효율로 해석하지 않는다. 단위 테스트의 물리 계산은 연구 표본 예산과 별도다.

8개 원본 anchor/source archive, archive member9,029개, current frozen file1,159개의
해시 검토를 통과했다. 공개파일270개/strict JSON213개/문서 로컬링크208개,
Git filter 이후 producer byte 보존221개와 validation log hash8개를 확인했다.
credential pattern/위험 archive 경로/100MiB 이상 파일/trailing whitespace 검출은0이었다.
이는 exhaustive secret 부재 보장은 아니다. 새80개 island artifact와 양 감사/result binding도 일치했다.
[무결성 receipt](../../results/post_audit/late_bridge_schedule_v1/session_20261023/validation/integrity_review.json)에
정확한 범위와 해시를 남겼다. `git diff --check` 통과, staged0,
HEAD `f556424360b8307284b23a6cfc0efcdee7380060` 유지, 새 commit/push0이다.

A result SHA256: `71772960f208158868a18656c999d7ad0626ed02560569aacbd56486fc39e5d2`.
A source ZIP SHA256: `12e2e80a67b635ffda3db0e38f84097b7bdfa7cd3bacd178eff86ceb6978b704` (1,136files).
A audit SHA256: `4a5fb38a5a29c53f4f96c386d74d9bf29e819cbbdb72af55f5e5bfde89046f19`.
B result SHA256: `24b2b05b03e1e706f30c81d5f51c469f818930ab7ff835cc067a0e9719aaa721`.
B source ZIP SHA256: `616b7b3191cc011803efda232934743cb6f17b0d42b4cb272a9fd51406cb56de` (1,138files).
B audit SHA256: `574f9cfaff2524201b0eb3cc18e9ece23153ccbb1ba78f3a0a616d38e9d26026`.
