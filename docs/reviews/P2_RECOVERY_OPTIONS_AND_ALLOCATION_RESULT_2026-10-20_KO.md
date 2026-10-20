# 현재 병목의 해결책 우선순위와 방어적 재배분 실행 결과

2026-10-20. 커밋·푸시 없음. 결과에 따른 판단과 미해결 사항을 구분한다.

## 1. 결론과 우선순위

현재 중요한 문제는 모델의 크기가 아니라 정확한 기준 확률, 경로 탐색의 신뢰성,
bank 일반화 및 격자 오차다. 최신 조건부 주변화는 수치 검증을 통과했으나 통계적
정밀도와 총비용 우위를 입증하지 못했다. 그러므로 해당 계산을 무작정 가속하기보다
기존 경로 후보를 얼마나 효율적으로 뽑는지 먼저 검증한다.

다음 순위는 성공 확률의 수치 추정이 아니다. 기존 관측, 비용, 변경 위험, falsifiability를
종합한 실행 우선순위다. 1순위의 성공도 다른 미해결 자격을 자동으로 해결하지 않는다.

| 순위 | 해결책 | 기대 효과 | 주요 실패 가능성 |
|---|---|---|---|
| 1 | 고정 G1 후보의 방어적 second-moment mixture 재배분 | 불필요한 후보에 쓰는 표본 감소; 물리 법칙/전체 밀도 유지 | 작은 bank가 희귀 기여를 놓치면 학습 비율이 과적합 |
| 2 | 물리적 경로 특성/기여 집중 분석과 guide-free 독립 기준 | 서로 다른 탐색의 blind spot 점검 | 높은 비용; 작은 RSE도 일치하지 않을 수 있음 |
| 3 | 새 bank로 window·scale geometry 재설계 | 기존 dictionary 밖의 중요한 경로 탐색 | 후보 증가만으로는 누락 문제 해결 안 됨; 독립 확인 필요 |
| 4 | 조건부 FFT solver의 검증된 preconditioner | 표본당 계산비 감소 | 경로 탐색이나 reference 불일치를 해결하지 못함 |
| 5 | 차원/신경망 확대와 대규모 학습 | 검증이 끝난 뒤 표현력 개선 가능 | 현재는 잘못된 reference를 더 잘 학습할 위험 |

## 2. 1순위의 구현과 이론 검토

기존 G1의 373개 component means를 그대로 두고, natural mass .1과 모든 shift에
나누는 floor mass .1을 남긴 채 .8만 학습한다. 따라서 모든 후보와 원래 p의 support를
보존하고 full-mixture 확률 기여는 10 이하로 제한된다. tail coverage가 입증됐다는 뜻은 아니다.

학습 목적은 원래 균등 G1 bank에서의 ordinary mean g²/(D0 Dw)다. w에 대해 convex이며,
독립 bank에서 고정 w를 평가하면 추정량의 second moment와 대응한다. 학습 후 같은
bank의 최솟값은 낙관적일 수 있으므로 최종 성능 추정에 쓰지 않는다.

Frank–Wolfe의 log-domain gradient scaling은 최적화 계산만 위한 것이다.
확률/M2를 empirical self-normalization한 것이 아니다. 최종 평가에서는 새 r_w 표본에서
g/Dw의 ordinary mean을 사용한다. component 단독 denominator나 clipped likelihood는 없다.

이 원리는 기존 mixture IS 최적화에 기반하며 독창적 정리라고 주장하지 않는다.
[He & Owen의 원 논문](https://arxiv.org/abs/1411.3954)을 참고했다. 우리의 새로운
학술 기여는 별도로 선행연구 대비와 일반화 이론/실험으로 입증해야 한다.

### 통계 계약

학습 bank B와 최종 표본을 분리하면 E[μhat | B]=μ다. 따라서 bank에 적응한 proposal를
사용해도 조건부로 ordinary IS는 unbiased다. 그러나 B가 누락한 모드가 있으면 큰
분산이 남을 수 있고 작은 최종 표본은 그 분산을 낮게 관측할 수 있다. 방어적 floor와
unbiasedness는 목표 상대 정밀도나 실용적인 비용을 보장하지 않는다.

## 3. 고정 실행

[사전 실행계획](../plans/DEFENSIVE_MIXTURE_ALLOCATION_RECOVERY_2026-10-20_KO.md)에
따라 canonical/high-eta × 독립 학습 반복 2개를 모두 수행한다.
각 반복에서 train 4,096, validation 2,048개를 분리한다. validation은 선택에 사용하지 않는다.
uniform/learned 각각 8 독립 block × 4,096개의 새 final 표본을 사용한다.
두 preflight를 포함한 총 physical 평가는 287,232회다.

모든 결과가 모이기 전 prefix를 성공으로 판정하지 않는다. 모든 qualification flag는
false다. 같은 geometry이므로 독립 기준 탐색 메커니즘 2개로 세지 않는다.

### 완료 결과

고정 71개 unit, 총 physical 287,232회를 완료했다. 그중 final은 262,144회다.
runner phase 시간은 약 161.73초다. import/source 동결 등 phase 시작 전 시간과
별도 replay 비용은 포함하지 않으며, 동일 accuracy의 total-wall benchmark도 아니다.
각 표 행의 final 표본은 32,768개로 동일하다.

| 셀/독립 학습 반복 | uniform RSE | learned RSE | uniform 최대 단일 기여 | learned 최대 단일 기여 |
|---|---:|---:|---:|---:|
| canonical 0 | 10.09% | 4.22% | 7.91% | 1.49% |
| canonical 1 | 8.55% | 4.43% | 5.90% | 1.96% |
| high-eta 0 | 5.66% | 2.62% | 2.24% | 0.54% |
| high-eta 1 | 5.34% | 5.11% | 1.85% | 3.73% |

4개 경우 모두 관측 RSE는 낮아졌다. 그러나 high-eta 1은 변화가 작고 집중도가
악화했다. 1.5% 정밀도는 **8개 행 모두 실패**했다. family 4개 조정 방법 간 일치
상한도 20.06–38.65%로 모두 10% 자격을 얻지 못했다. 구간이 넓어 동등성을
입증하지 못한 것이며, 차이가 확정 편향으로 검출됐다는 뜻은 아니다.

관측 RSE² 비는 5.72 / 3.73 / 4.66 / 1.09다. 이를 보장된 분산 감소율이나
속도 향상으로 해석하지 않는다. 각 방법의 실제 score 시간은 약 16.6–17.9초로
비슷했지만, 모든 accuracy가 미달이므로 fixed-precision 속도 우위는 주장하지 않는다.

학습 fit은 반복당 약 1.01–1.05초였다. train+fit+validation은 4.06–4.32초였다.
검증 bank의 learned/uniform second moment 비는 0.182 / 0.159 / 0.562 / 0.364였다.
4개 fit 모두 256회 상한에 도달했고, 상대 경험적 FW gap은 약 3.44–6.69%였다.
따라서 1e-6 수렴 기준을 달성한 최적해라고 표현하지 않는다. 이는 해당 bank의
경험적 최적화 진단이며 모집단 최적성이나 bank 일반화 보장이 아니다.

### 재현 감사

독립 학습 bank 4개와 validation 4개를 모두 재생성하고, 학습 weights/목적/gap을
재계산했다. learned 대표 final block 4개도 replay해 moments, tail path 진단 및
seed digest를 일치시켰다. 감사 추가 physical은 40,960회다.

- 감사: `defensive_allocation_source_bank_fit_final_replay_pass`.
- 결과 SHA256: `881949bb35d4b03c264b280de4933d49386a5a5f627398ce4bc3501e95a8d5a0`.
- 결과/감사: `results/post_audit/defensive_allocation_v1/session_20261020/`.
- 원래 q, 그 M2, 기존 reference 결과 및 고정 실패 기록은 변경하지 않았다.

## 4. 2순위 중 추가 물리 계산 없이 완료한 집중도 진단

감사된 최종 요약을 원천으로 새 post-hoc 분석을 수행했다. 보존한 최대 기여 16개와
전체 raw moments만 사용했으며 새 physical 평가는 0회다.

high-eta 1 learned의 **한 표본이 관측 제곱 기여 합의 52.65%**를 차지했다.
상위 16개가 81.71%를 차지한다. 비교해 high-eta 0 learned는 각각 4.03% / 28.68%다.
이는 표본의 분산 추정이 tail 한두 개에 민감할 수 있다는 관측 경고다. 제곱 기여
비중을 모집단 분산의 확정 비율이나 실제 희귀 모드의 확률이라고 부르지 않는다.

문제의 최대 기여 경로는 적분 변동성 약 1.988, 최대 왼쪽 분산 약 2,149,
발생 시점 약 0.515T였다. 다른 큰 기여 경로와 다른 spike 구조를 탐색할 이유는
있지만, 이것만으로 원인·새 물리 모드·고정 window 폭의 결함을 증명하지는 못한다.
clipping으로 해당 경로를 제거하지 않는다.

학습 weight 벡터 간 total variation은 canonical 약 0.648, high-eta 약 0.569였다.
component가 겹치므로 이는 실제 경로 분포 간 거리와 같지 않다. 은닉 모드 누락이나
target 편향의 증명도 아니다. 단지 독립 bank에서 후보별 비율이 많이 바뀌었다는 진단이다.

상위 16개는 큰 기여로 선택된 표본이다. 전체 경로 clustering, calibrated bin,
관측되지 않은 tail의 부재, population coverage를 대신하지 않는다. 후속 geometry
선택에 활용하면 이 final 자료는 개발 자료가 되며 새로운 confirmation을 요구한다.

분석 결과: `results/post_audit/defensive_allocation_tail_diagnosis_v1/session_20261020/result.json`.

## 5. 결과에 따른 다음 판단

**재배분은 유망한 reference 개발 후보로 보존하되 단독 해결책으로 채택하지 않는다.**
다음에는 더 큰 신경망보다 bank 일반화와 잔여 tail coverage를 우선한다.

1. 독립 bank 여러 개를 사용한 공통/ensemble weights와 기존 두 fit을 새 표본에서
   비교한다. 단순 평균 weights에는 second moment의 convexity에 따른 평균 대비
   상한이 있으나, 최고 단일 fit 대비 개선이나 unseen mode coverage는 보장하지 않는다.
   같은 final을 다시 사용해 ensemble 성공을 주장하지 않는다.
2. 전체 경로 특성의 개발용 replay/새 bank로 spike duration·발생 시점별 기여를
   분석한다. 현재 top-only 진단으로 새 geometry를 확정하지 않는다.
3. 그 결과로 최소한의 multiscale 후보를 고정하고, train/selection/final을 다시
   분리해 평가한다. 누락 문제를 component 수 증가로 자동 해결했다고 보지 않는다.
4. 그 뒤 guide-free reference와 동일 target의 새 독립 교차검증을 수행한다.
   충분한 precision 없이 큰 SMC 일괄 증설이나 P2/대규모 학습으로 건너뛰지 않는다.

이번에 실제 완료한 범위는 1순위 구현·고정 실험·replay와 2순위의 관측 집중도 분석이다.
새 ensemble/geometry 실험, guide-free 새 confirmation, all-q/mesh/P2 자격은 미완료다.

## 6. 검증 기록

전체 pytest: **1,472 passed, 280.29초**. 기존 1,460개에 새 검사 12개를 추가했다.
실험 producer를 동결한 뒤 추가한 유한차분/rescaling 검사도 이 전체 실행에 포함됐다.
이는 모든 입력·모든 주장의 무오류 증명이나 통계적 자격 증명이 아니다.

수식, 조건부 unbiasedness, convexity, defensive bound 및 수치 계산의 한계는
[별도 이론 검토](../theory/DEFENSIVE_MIXTURE_ALLOCATION_REVIEW_2026-10-20_KO.md)에 정리했다.

- CI 범위 Ruff 검사 통과.
- CI 범위 Mypy: 203 source files 통과.
- 새 core/producer/postprocessor의 직접 타입 검사: 3 source files 통과.
- 원래 결과와 audit의 SHA, snapshot에 포함된 numerical runtime 203개 파일의 hash,
  false인 P1/P2/large-training flag를 별도로 재확인했다.
- `git diff --check` 통과. HEAD와 staging 유지. 커밋·푸시하지 않았다.
- 이전 보고서에 기록한 루트 보조 스크립트 lint/옛 experiment 타입 오류 및 기존
  requests 의존성 경고가 없어졌다고 주장하지 않는다. 이번 범위 밖 파일은 수정하지 않았다.
