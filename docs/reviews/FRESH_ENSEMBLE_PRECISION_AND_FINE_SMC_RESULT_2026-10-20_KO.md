# 새 앙상블 정밀도 및 fine-only guide-free SMC 결과

2026-10-20. main 9cafe59 이후 후속 개발. 이번 단계는 커밋·푸시하지 않는다.
main 9cafe59의 GitHub CI는 Python 3.10/3.11 모두 성공했다.

## 1. 목표와 검증 범위

이전 ensemble pilot을 새 표본으로 검증하고, 같은 fine N4096 확률을 계산하는
guide-free SMC와 비교한다. 원래 q, q의 M2, mesh/continuum, 새 bank 재학습 및
저널용 confirmation은 이번 단계가 대신하지 않는다. 전체 P1/P2/대규모 학습 자격은 false다.

I 단계는 이전 pilot의 raw/독립 block RSE 중 큰 값과 safety3로 사전에 표본 수를
고정했다. canonical 59×8,192=483,328개, high-eta 43×8,192=352,256개다.
과거 표본과 합치지 않고, 결과가 좋을 때 조기 종료하거나 실패했을 때 count를 늘리지 않는다.

G 단계는 원래 fine CDF에 대한 gradient pCN-L SMC다. coarse/augmentation을 제거하고
fine μ만 직접 target으로 삼는다. 두 설정은 particles128, β64/mutation2/resample4 또는
β128/mutation1/resample8이다. cell/arm별 독립 whole8개, 총32개를 수행한다.
G1 means/weights는 이 커널에 입력하지 않는다. forward/reverse Gaussian MH 비율,
SMC normalizer product, initial ancestry 기록을 유지한다.

계획: [사전 고정 실험](../plans/FRESH_ENSEMBLE_PRECISION_AND_FINE_ONLY_SMC_2026-10-20_KO.md).
검토: [수학 계약](../theory/FINE_ONLY_REFERENCE_AND_PRECISION_CONTRACT_2026-10-20_KO.md).

## 2. 실행 상태

초기 테스트 6개와 새 runner 타입/lint 검사를 통과한 뒤 고정 실험을 완료했다.
I 전체102개 block과 G 전체32개 whole을 마쳤고 수치 replay도 통과했다.
각 단계 완료와 전체 통계 자격 통과는 구분한다.

모든 preflight 포함 physical forward 1,353,152회, backward 517,056회로 사전 고정했다.
추가 replay는 I4개 block 및 G4개 whole, physical97,280회/backward64,512회다.
실제 physical forward는 1,353,152회, backward는 517,056회로 계획과 같다.
총139개 journal unit을 완료했다. 추가 replay는 최종 성능 표본과 합치지 않는다.

## 3. I 단계: 새 fixed-count 앙상블 결과

| 셀 | 새 final 표본 | 평균 | raw RSE | 독립 block RSE | 최대 단일 sample 기여 |
|---|---:|---:|---:|---:|---:|
| canonical | 483,328 | 1.37653e-8 | 0.902% | 0.766% | 0.191% |
| high-eta | 352,256 | 9.82276e-7 | 0.907% | 1.058% | 0.283% |

두 셀 모두 사전에 정한 raw RSE≤1.5%, 최대 block 비중≤15%, leave-one-block-out
이동≤5%, block RSE≤2×raw RSE를 통과했다. 최대 block 비중은 1.910% / 2.767%,
LOO 이동은 0.322% / 0.452%다. block 수는 각각 59 / 43이다.

score wall 합은 약 276.99 / 189.17초다. 준비·source 동결·감사·이전 bank 학습을
모두 포함한 total-work speedup 측정이 아니다. `iid-stage.json`은 완료한 I 자료만
보존하고 독립 corroboration 또는 P2 통과를 선언하지 않는다.

정밀도·민감도 기준을 충족한 것은 진전이지만, 관측 RSE가 미관측 tail이나 bias를
모두 포함한 상대오차 보증은 아니다. 같은 frozen weights의 새 final이지 새로운
학습 반복/confirmation이 아니다. G의 정밀도·일치 결과와 replay는 별도다.

## 4. G 단계: fine-only 독립 SMC는 실패

각 표 행은 독립 whole8개이며 particle 표준오차가 아니다.

| 셀 | 설정 | 평균 | whole RSE | 최종 initial ancestry 범위 | mutation acceptance 범위 |
|---|---|---:|---:|---:|---:|
| canonical | 64points/2moves | 1.36960e-8 | 16.31% | 6–14 / 128 | 83.5–84.0% |
| canonical | 128points/1move | 8.62264e-9 | 24.20% | 8–14 / 128 | 82.7–83.9% |
| high-eta | 64points/2moves | 8.40085e-7 | 13.38% | 11–14 / 128 | 82.1–82.9% |
| high-eta | 128points/1move | 8.41818e-7 | 11.58% | 9–15 / 128 | 82.1–83.4% |

네 행 모두 RSE뿐 아니라 최대 whole 비중≤15%와 LOO 이동≤5%에도 미달했다.
최대 whole 비중은 약18.67–29.40%, LOO 이동은 약7.06–19.31%였다.
높은 acceptance를 mixing, rare-mode 탐색 또는 정상적인 계보의 증거로 볼 수 없다.
fine-only로 차원과 nuisance/coarse 비용을 줄여도 신뢰성 문제가 해결되지는 않았다.
과거 paired umbrella와는 target bridge/입자 수가 달라 직접적인 성능 승패를 주장하지 않는다.

### 앙상블과의 비교

4개 family 조정 10% 일치 불확실성 상한은 다음과 같다.

| 셀/SMC 설정 | 조정 차이 상한 | 판정 |
|---|---:|---|
| canonical 64/2 | 41.30% | 동등성 불확실 |
| canonical 128/1 | 103.24% | 동등성 불확실 |
| high-eta 64/2 | 48.89% | 동등성 불확실 |
| high-eta 128/1 | 44.24% | 동등성 불확실 |

네 쌍 모두 양쪽 accuracy 자격이 충족되지 않았다. 모두 넓은 구간 때문에 unresolved이며,
조정된 구간이 허용 범위 밖에 완전히 놓인 detected disagreement로 분류된 것은 없다.
일부 point estimate가 크게 다르더라도 어느 방법이 진실인지 또는 영구적 bias가 있는지
확정하지 않는다. ensemble도 미관측 tail에 대한 확정 보증을 얻은 것은 아니다.

이전 ensemble pilot과 새 IID의 개발용 일치 상한은 canonical7.98%, high-eta7.04%로
10% 기준을 통과했다. 그러나 동일 frozen weights이고 이전 pilot으로 count를 정한
비교이므로 독립 탐색 corroboration이나 저널 confirmation으로 세지 않는다.

## 5. 실패 위치를 분리한 추가 진단

별도 source-bound 분석은 완료한 모든 SMC whole의 단계 기록을 사용한다. 새 physical은 0회다.
초기 조상이 64개 이하로 줄어든 첫 β 범위는 약.00397–.01612였고,
32개 이하의 첫 β는 대체로 .016–.036 부근이었다. β=1에 도달하기 훨씬 전의 퇴화다.
최소 stage ESS는 whole에 따라 약34.3–55.7 / 128이었다.

두 설정은 resampling calendar의 실제 β 위치가 비슷하다. quadratic schedule에서
첫 resampling β는 (4/63)²≈.00403 또는 (8/127)²≈.00397다.
따라서 β points를 두 배로 하고 mutation을 절반으로 바꾸는 것만으로는
초기 선택 압력이나 누적 rejuvenation 비용이 근본적으로 달라지지 않는다.
이 관측이 초기 퇴화의 유일한 원인이나 rare-event mean 편향을 증명하지는 않는다.

작은 pilot의 관측 RSE를 safety3로 외삽하면, 새 정밀도 whole 수는 설정별 약
1,431–6,249개, physical 약2,326만–1억158만 회다. 단순 score 시간 예측은
약4.93–20.56시간이며 준비·검증·모드 누락 위험까지 해결하지 않는다.
이 예측으로 대량 실행을 승인하지 않았다.

추가 분석: `results/post_audit/fine_smc_genealogy_analysis_v1/session_20261020/result.json`.
원래 whole의 hash와 source archive에 묶었다. stage 위치 진단은 post-hoc 설명이며
새 사전 등록 통계 가설의 검정 결과가 아니다.

## 6. 재현·비용 및 주장 범위

I 각 cell의 첫/마지막 block4개와 G 각 cell/arm의 rep0 whole4개를 재현했다.
ordinary moments, 전체 physical bins, seed와 SMC normalizer/stage 진단을 대조했다.

- 감사 상태: `fresh_ensemble_fine_only_smc_source_seed_feature_stage_replay_pass`.
- result SHA256: `c2d8311f99deefaef7f880ffa4dc4417307aa2922e1c242af6fdddfbc8acd126`.
- 원천/결과/감사: `results/post_audit/fresh_ensemble_fine_smc_v1/session_20261020/`.

runner phase I는 약476.64초, G는 약388.64초다. 준비·단계 I/O를 포함하되
CLI import/source 동결과 추가 audit는 제외한다. accuracy가 다른 방법의 실행 기록이지
fixed-precision total-wall speedup 비교가 아니다. 과거 bank 학습 비용을 이번
score에 포함했다고 주장하지 않는다.

두 메커니즘은 탐색·추정 방식이 다르지만 원래 finite-grid physics/CDF 구현을 공유한다.
따라서 이 비교만으로 공유된 simulator 오류까지 독립 검증한 것은 아니다.
작은 grid의 별도 oracle 검사는 유지하며 continuum 주장은 계속 제한한다.

## 7. 이제 실제로 남은 우선순위

1. **해결된 부분:** 현재 frozen-bank ensemble 두 셀의 새 표본 정밀도·민감도 조건.
   이를 원래 q의 성능 향상, unseen-tail 보증, all-q 기준 또는 P2 완료로 확대하지 않는다.
2. **가장 중요한 남은 부분:** guide-free independent reference의 early-stage 퇴화.
   전체 whole을 수천 개 추가하는 방식은 우선하지 않는다.
3. 다음 개발 실험은 동일 입자·forward/backward 예산·resampling 규칙 아래
   초기 β≤.04 부근에 계산을 더 배분하는 fixed schedule 대조를 우선한다.
   quadratic 대 더 강한 초기 집중 schedule을 비교한다면 실제 resampling β와
   mutation 배분도 공개한다. β points 수만 늘리는 대조와 구분한다.
4. schedule을 데이터로 보정한다면 별도 calibration pilot으로 정한 뒤 primary
   run 전에 동결한다. 같은 particle로 temperature를 적응 선택한 뒤 고정 schedule의
   normalizer 성질을 검토 없이 주장하지 않는다. 원래 MH 보정/target/weight는 유지한다.
5. 독립 기준의 precision·일치 자격 뒤에 all-q M2, 새 bank ablation, mesh/continuum을
   별도 진행한다. 대규모 신경망은 아직 우선순위가 아니다.

이번 요청의 fixed-count IID·독립 SMC 대조·수치 감사·실패 분석은 완료했다.
초기 schedule 재설계의 새 primary 실험, 독립 reference 자격 및 P2는 미완료다.

## 8. 최종 검증 기록

전체 pytest: **1,491 passed, 275.72초**. 새 gate/target/seed/replay/분석 검사 9개를
포함한다. 테스트 통과는 모든 이론적 주장이나 미관측 tail의 무오류 증명이 아니다.
이번 단계는 커밋·푸시하지 않았다.

- CI 범위 Ruff: 통과.
- CI 범위 Mypy: 204 source files 통과.
- 새 producer/analysis 직접 타입 검사: 2 source files 통과.
- 원래 result/audit SHA, source에 동결한 numerical runtime hash와 추가 분석의
  모든 whole 결과 anchor hash를 재확인했다.
- source-bound genealogy 분석을 원래 journal에서 zero-physics로 다시 계산해 일치시켰다.
- 모든 P1/P2/large-training flag는 false로 확인했다. `git diff --check` 통과.
- HEAD는 9cafe59 유지, staging은 비어 있다. 새 코드·문서·결과는 로컬 변경이다.
- 기존 범위 밖 보조 파일의 lint/type 문제와 requests 의존성 경고가 사라졌다고
  주장하지 않는다. 수치 실험 중 의존성 버전이나 OS 정책을 바꾸지 않았다.
