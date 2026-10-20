# README/main 반영 및 bank ensemble·전체 경로 진단 결과

2026-10-20. 이전 실패 결과와 자격 조건을 유지한 후속 개발 실험.

## 1. GitHub 반영

README에 최신 RSE 비교표, reference 개선과 원래 q의 구분, 미해결 P2 조건을 추가했다.
이전까지의 구현·감사·실패 기록을 main 커밋
`a63825fc2542bf753ca43c04ccb969553a61f81f`로 푸시했다.
원격 refs/heads/main이 같은 SHA임을 확인했고, 해당 커밋의 GitHub CI는
Python 3.10/3.11 모두 성공했다.

CI: https://github.com/YoungJun0814/Neural_Path_Intergral/actions/runs/39495536819 .
이 CI 결과는 위 커밋에 대한 것이며 이후 새 구현의 원격 CI 결과와 혼동하지 않는다.

## 2. 다음 우선순위의 구현

이전 두 independent bank가 학습한 weight의 단순 평균을 만들었다.
균등 G1, fit0, fit1, ensemble 네 방법을 같은 373개 means와 원래 fine N4096 법칙에서
비교한다. 새 bank fitting이나 앙상블 비율 selection은 없으며, 기존 bank와 모든
과거 final에서 분리된 새 namespace의 표본을 생성한다.

모든 최종 경로에 대해 적분 변동성, 점별 집중도, T/32 창의 최대 에너지 비중,
peak 시점을 계산하고 결과 확인 전에 고정한 96개 joint bins에 기록한다.
상위 16개만 보는 것과 달리 전체 표본의 구간별 기여를 복원할 수 있다.
이는 관측된 finite-grid 분할이지 모집단 mode clustering/coverage 보장은 아니다.

각 bin의 평균과 SE는 bin 밖의 0도 포함한 전체 표본 수로 계산한다. 빈 bin은
확률 0/분산 0의 근거로 사용하지 않는다. 전체 bins가 원래 ordinary mean과 moments를
복원하는지 검사한다. 원래 q의 M2, guide-free 기준, mesh/continuum 조건은 바꾸지 않는다.

[사전 고정 계획](../plans/BANK_ENSEMBLE_FULL_PATH_COVERAGE_2026-10-20_KO.md)과
[수학·기술 검토](../theory/BANK_ENSEMBLE_FULL_PATH_COVERAGE_REVIEW_2026-10-20_KO.md)를 따른다.

## 3. 고정 실험 완료 결과

두 cell × 네 방법 × 8 독립 blocks × 8,192 draws를 모두 완료했다.
최종 물리 평가는 524,288회, preflight 포함 524,800회다. 각 표 행은 65,536개
새 final draw를 사용한다. 서로 다른 행의 stream은 별도다. 과거 final과 합치지 않았다.
전체 runner phase는 약 382.88초다. CLI import/source 동결 등 phase 시작 전 시간과
별도 audit는 포함하지 않으며, 동일 정확도의 total-wall benchmark도 아니다.

| 셀 | 방법 | 평균 | raw RSE | block RSE | 최대 단일 기여 비중 |
|---|---|---:|---:|---:|---:|
| canonical | uniform | 1.33823e-8 | 5.88% | 5.42% | 2.88% |
| canonical | fit0 | 1.35251e-8 | 2.63% | 2.93% | 0.52% |
| canonical | fit1 | 1.37704e-8 | 2.67% | 1.29% | 0.98% |
| canonical | ensemble | 1.34362e-8 | **2.31%** | 2.34% | 0.65% |
| high-eta | uniform | 9.30328e-7 | 3.87% | 4.77% | 0.95% |
| high-eta | fit0 | 9.81492e-7 | 2.02% | 2.22% | 0.68% |
| high-eta | fit1 | 9.43673e-7 | 2.12% | 1.81% | 0.57% |
| high-eta | ensemble | 9.60293e-7 | **1.93%** | 2.01% | 0.57% |

8개 행 모두 목표 raw RSE 1.5%에 미달했다. block RSE는 8개의 독립 block 평균으로
계산한 민감도 진단이며 raw IID RSE를 대체하는 작은 값만 골라 자격을 부여하지 않는다.

ensemble은 두 cell에서 관측 RSE가 가장 낮았다. uniform 대비 관측 RSE² 비는
6.48 / 4.02였다. 이는 정확한 모집단 분산비나 속도 향상의 증명이 아니다.
ensemble의 실제 score 시간은 셀별 약 49.19 / 47.46초다. 전체 방법의 시간 범위는
41.22–49.19초이나 fixed-order 실행 및 accuracy 미달 상태이므로 속도 우위를 주장하지 않는다.

### 방법 간 일치 검사

ensemble 대 uniform/fit0/fit1을 두 cell에서 비교하는 총 6개 family에 조정했다.
canonical fit0와 high-eta fit0/fit1의 3개 비교가 개발용 10% 일치 기준을 통과했다.
canonical uniform/fit1, high-eta uniform은 미달했다. 특히 uniform 비교 상한은
17.08% / 14.58%였다. 구간이 넓어 동등성을 입증하지 못한 것이며 확정 편향을
검출했다는 뜻은 아니다. 같은 geometry의 비교이므로 guide-independent corroboration은 아니다.

## 4. 모든 경로에서 관측한 잔여 기여 구조

ensemble의 전체 96개 bins를 합산한 관측 기여 비중이다. 각 표는 별개의 주변 집계로
합산되며, 표 안의 다른 행들을 독립 사건으로 더하는 것은 아니다.

| 관측 구간 | canonical | high-eta |
|---|---:|---:|
| 적분 변동성 I가 1–2 | 68.61% | 55.27% |
| 적분 변동성 I가 2–4 | 2.81% | 39.98% |
| peak가 마지막 1/4 시간에 발생 | 62.30% | 57.12% |
| 어느 T/32 창도 전체 에너지의 25% 미만 포함 | 75.95% | 44.58% |
| 최대 T/32 창 에너지 비중이 25–50% | 23.65% | 47.50% |
| 최대 T/32 창 에너지 비중이 50% 이상 | 0.40% | 7.92% |

관측된 제곱 기여 합에서도 C32<.5 구간이 약 99.80% / 89.13%를 차지했다.
따라서 **아주 좁은 spike만을 추가하면 문제가 풀린다는 가설은 현재 자료의 지지를
충분히 받지 못한다.** 넓고 지속되는 변동성 구간과 후반부 peak도 중요하다.
이것은 관측 구간 진단이며 “실제 확률의 정확한 비율”, 인과적 기전 증명 또는
미관측 spike의 부재를 뜻하지 않는다. 원래 price-direction 탐색 누락도 별도일 수 있다.

ensemble의 최대 단일 표본은 관측 제곱 기여 합의 약 7.75% / 8.24%였다.
이번 fresh 표본에서 과거 high-eta 1의 52.65% 집중은 반복되지 않았지만, 과거 표본을
이상치로 삭제하거나 문제가 영구히 사라졌다고 하지 않는다.
top-16 기여 비중은 약 5.67% / 4.28%다. 이는 관측 tail sensitivity의 개선 신호다.

canonical ensemble에는 7개, high-eta ensemble에는 3개 joint bin의 표본이 없었다.
빈 bin은 unobserved로 유지했다. 단순 관측 bin 수는 coverage의 우열 척도가 아니다.
전체 경로를 binning해도 bin 내부의 희귀 구조를 놓칠 수 있으므로 population coverage
증명이나 학습된 physical mode clustering으로 표현하지 않는다.

## 5. 재현 감사 및 별도 집계

각 cell/arm block 0, 총 8개 block을 다시 계산했다. 추가 물리 평가 65,536회로
seed, ordinary moments, 모든 bin count/first/second moments, feature 가중 요약과
top-path 진단을 재현했다. 원천 snapshot·설정·전체 family membership·비용도 검사했다.

- 감사 상태: `bank_ensemble_all_path_source_seed_feature_moment_replay_pass`.
- 결과 SHA256: `7b2ef92daa4c00a0ff8b52c239e00d3c044b92f0a7201cffaf43575977f18cf7`.
- 결과/감사: `results/post_audit/bank_ensemble_full_path_v1/session_20261020/`.

그다음 source-bound zero-physics 분석으로 전체 bins의 주변 집계와 다음 정밀도 예산을
별도로 기록했다. 새 물리 계산은 0회다.
`results/post_audit/bank_ensemble_full_path_analysis_v1/session_20261020/result.json`.
집계는 원래 joint moments를 재구성하며, projection은 향후 성공/효율 보장이 아니다.

## 6. 다음 판단

**ensemble을 다음 precision 개발의 고정 후보로 유지한다.** 단, 이번에는 두 개의
과거 bank에서 만든 frozen weights만 평가했으며 여러 bank 쌍을 새로 학습한 반복은 아니다.
새 학습 반복에 대한 일반화까지 입증했다고 하지 않는다.

우선순위는 다음과 같이 갱신한다.

1. ensemble의 새 fixed-count precision 검증과, 서로 다른 탐색 메커니즘의 기준과
   교차검증할 예산을 사전 고정한다. 이번 결과에 draw를 선택적으로 덧붙이지 않는다.
   raw RSE 기반 safety 3 예측은 canonical 475,136개, high-eta 327,680개의 새 draw다.
   해당 score 시간의 단순 예측은 약 357 / 237초이나 누락된 tail·block 변동·독립
   기준 비용·준비 비용·확인 반복을 보장하지 않으며 이 보고서로 자동 실행 승인하지 않는다.
2. geometry를 바꾼다면 좁은 spike만 늘리지 않고 넓은 에너지 구간, 후반부 peak,
   price-direction을 대조하는 최소 후보를 새 bank/selection/final로 검증한다.
3. guide-free SMC의 독립 whole-run 불일치와 계보는 여전히 별도 해결 대상이다.
   ensemble 성공 신호로 과거 SMC 경고를 무효화하지 않는다.
4. all-q M2, 새로운 training-bank ablation 및 mesh/continuum 조건을 얻기 전에는
   원래 q의 성능 개선이나 P2/대규모 학습을 승인하지 않는다.

이번 단계의 구현·고정 실험·수치 감사·전체 경로 진단을 완료했지만, 남은 모든
reference 및 학습 문제를 해결했다고 선언하지 않는다. 전역 자격은 모두 false다.

## 7. 검증 기록

전체 pytest: **1,482 passed, 254.72초**. 초기 8개 oracle/실행 계약 검사와
source-bound 분석 검사 2개를 포함한다. 테스트 통과는 모든 입력에 대한 무오류 증명이나
통계적 qualification의 증명이 아니다.

- CI 범위 Ruff 통과.
- CI 범위 Mypy: 204 source files 통과.
- 새 core/producer/analysis 직접 타입 검사: 3 source files 통과.
- 부모 result/audit의 SHA, analysis anchor SHA 및 현재 numerical runtime hash를 재확인했다.
- 저장된 분석을 원래 parent에서 zero-physics로 재계산해 일치시켰다.
- 모든 P1/P2/large-training flag가 false임을 재확인했다.
- `git diff --check` 통과. 기존 범위 밖 보조 파일의 lint/type 문제와 requests
  의존성 경고가 사라졌다고 주장하지 않는다.
