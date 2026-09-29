# R1.5/R2 선행조건 실행·검토 보고서

작성일: 2026-09-28~29

범위: `N=32`, 동일 조건부 `2N=64` Gaussian Volterra 경로, canonical `H=.05, η=1.5, K=1` 및 높은 η `H=.05, η=2, K=1`.
상태: **개발 실험. 새 operator/rank 확장 또는 저널용 confirmation이 아니다.**

## 결론

1. **기준 추정의 정밀도는 개선했다.** 사전 고정한 SMC bridge·resampling·mutation 대조에서 원래 조상 중앙값은 256명 중 canonical 4→16, 높은 η 5→27명으로 개선됐다. 새 독립 SMC 기준 RSE는 각각 2.09%, 1.81%다. SMC로 학습한 proposal을 사용하되 최종 추정 메커니즘은 다른 exact-density defensive IS를 새 524만 개 최종 표본으로 실행해 각각 3.37%, 2.89% RSE와 사전 설정한 25% 동등성 구간 안의 일치를 얻었다. **SMC 학습에도 의존하지 않는 CE-only IS는 같은 표본 수에서 정밀도 gate에 실패**했으므로 이 교차검증을 완전히 독립된 세 번째 메커니즘의 확인으로 확대하지 않는다.
2. **학습 bank 신뢰성은 아직 해결되지 않았다.** 동일한 512개 IID bank 20회 시험에서 가장 좋은 median target ESS는 canonical 18.2, 높은 η 10.3이다. 하위 10% 부근은 여전히 1~4명에 불과하다. 새 bank·새 최종 표본의 rank/basis/KL–M₂ ablation도 3개 학습 반복 × 두 bank 원천 × 두 셀에서 자격 통과 방법이 없다. SMC 입자의 final-weight ESS와 원래 조상 수는 IID target-ESS와 다르다.
3. **현재 방법의 고정 정밀도 총시간 우위는 없다.** 새로 학습한 proposal에서 canonical IS RSE 8.07%로 자격 미달이므로 시간 비율은 보류했다. 높은 η는 두 방법이 통과했으나 SMC 82.7초, IS 86.9초로 IS가 약 5% 느렸다. 이는 노트북의 한 번의 개발 실행이며 속도 우위나 열위의 통계적 결론이 아니다.
4. **R2 복잡도 확대와 R3 저널용 confirmation은 보류한다.** 교차검증 성공은 특정 frozen proposal을 충분히 오래 평가했을 때의 정확도 증거이지, 새 학습 반복이 안정적이라는 증거가 아니다.

## 1. 추정 대상과 비협상 조건

유한격자 `N=32`에서 독립 가격 Brownian driver를 조건부 적분한 `g_N(z)∈[0,1]`를 대상으로 `μ_N=E_{p_N}[g_N(Z)]`를 추정했다. 모든 SMC·CE·혼합 IS가 **같은 64차원 조건부 target**을 사용한다. SMC의 고정 온도 경로는 `p_N(z)g_N(z)^β`이고 pCN 돌연변이는 현재 `β` target에 불변인 Metropolis kernel이다. resampling을 건너뛴 단계에서는 정규화 가중치를 계속 운반한다. 고정 일정 resampling의 조건부 기대 offspring 수는 부모 weight에 비례하며, 정상적인 potential·불변 kernel 조건하에서 normalizer 추정이 불편이다. 실제 구현의 정확성은 analytic toy와 Monte Carlo 테스트로 별도 확인했다.

IS는 학습 후 proposal `q`를 동결하고 독립 최종 경로에서 `g_N p_N/q`의 **ordinary mean**을 사용한다. 전체 Gaussian mixture의 정확한 `q`를 분모에 쓰고, 자연 Gaussian 성분 질량 `δ=.1`을 남긴다. 이로써 유한격자에서 support와 `p_N/q≤10`은 확보되지만 상대 효율은 보장되지 않는다. 학습 bank의 normalized weight는 fit에만 사용했고 최종 self-normalized IS, clipping, 유리한 표본 선별은 하지 않았다.

## 2. SMC 설계와 독립 기준

개발 설정을 먼저 고정했다: 256입자, 48단계, bridge power 2/4, 매 단계 또는 4단계마다 resampling, multinomial/stratified, pCN mutation 1/2회, scale .55/.35. 각 후보·셀에서 독립 전체 SMC 실행 24회로 diversity/비용-분산 gate를 적용했다. 선택된 `stronger-mutation`은 power4, stratified 4단계 간격, pCN 2회, scale .35다. 디자인 pilot과 기준 실행은 다른 seed를 사용한다.

| 셀 | 원래 일정 조상 중앙값 | 선택 일정 조상 중앙값 | 새 SMC 반복 수 | 새 SMC μ | 새 SMC RSE | 독립 IS μ | 독립 IS RSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 4 | 16 | 512 | 4.89424e-8 | 2.09% | 4.68307e-8 | 3.37% |
| 높은 η | 5 | 27 | 256 | 4.77434e-6 | 1.81% | 4.65122e-6 | 2.89% |

첫 SMC 기준은 canonical RSE 5.1%로 사전 5% 기준을 근소하게 넘었다. 그 결과를 새 기준과 합치지 않고 별도 fixed-count 재실행으로 정밀도를 높였다. 두 SMC 실행 간 표준화 차이는 canonical 1.11, 높은 η 0.35다. 이전 CE 기반 IS는 canonical 7.0%, 높은 η 17.3% RSE로 교차검증에 실패했다. 그 뒤 학습한 SMC-bank 혼합 IS도 65.5만 개 표본에서 11.3%/11.9% RSE로 실패했다. 이 실패 결과를 보존한 뒤 **고정한 proposal과 640×8192개의 새 최종 IID 경로**로 독립 IS를 재실행했다. IS와 SMC 평균의 상대 차이는 약 -4.3%/-2.6%이고, 95% 차이 상한은 사전 25% equivalence margin의 약 47%/37%다.

정밀한 평균 일치에도 독립 IS의 상위 1% 경로가 총 추정량의 약 67%/77%를 담당한다. 높은 η의 `E_q[p/q]` 정상화 점검은 관측값 약 1.00298, 표본 SE 기준 약 2.5σ 편차다. 단일 진단의 결정적 밀도 오류 증거는 아니지만, 원인 미확인 QC 주의사항으로 유지한다. 이 결과는 알려지지 않은 다른 rare mode의 부재를 증명하지 않는다.

SMC 학습 bank에 의존하지 않는 이전 **CE-only portfolio**도 고정하고, 기존 실패 표본과 합치지 않은 새 640×8192개 IID 표본으로 추가 교차검증했다. canonical 평균 4.96231e-8·RSE 5.07%, 높은 η 평균 4.74675e-6·RSE 12.81%다. 평균은 두 SMC 기준에 각각 약 +1.4%/-0.6%로 가깝지만 **둘 다 사전 RSE≤5% gate를 통과하지 못했다**. 특히 높은 η에서는 상위 1% 표본이 총 기여의 93.6%, 최대 한 경로가 11.4%를 담당한다. 이 계열의 낮아 보이는 분산과 mode 누락 위험을 배제하지 못하며, 평균의 우연한 일치를 독립 confirmation으로 승격하지 않는다.

## 3. 학습 bank와 rare-mode 분석

선택된 SMC의 기준 반복 512/256회에서 원래 조상 중앙값은 15/27명이다. 혼합 proposal을 위한 새 3개 SMC bank의 원래 조상은 canonical 11·9·11명, 높은 η 22·28·29명이다. `pooled_weighted_bank_ess≈620/674`는 각 SMC 실행의 **최종 입자 가중치 ESS를 이어 붙인 값**이다. resampling 이후 상관된 입자를 IID로 간주한 target ESS가 아니다.

SMC bank에서 K=1/2/4와 identity/low-rank covariance 혼합을 개발 pilot로 대조했다. 두 셀의 pilot 제약을 통과한 공통 후보는 K=2 identity였고, 새 SMC 학습 bank와 새 IID 최종 표본에서 다시 평가했다. 각 final path를 SMC 학습 bank에서만 정의한 4개 고정 weighted-PCA/k-means cluster에 배정했다. 총 기여 비율이 canonical 13%/26%/37%/23%, 높은 η 17%/43%/18%/23%라는 사실은 네 발견된 cluster가 추정에 모두 관여함을 보여줄 뿐, 미발견 mode가 없음을 인증하지 않는다.

기존 CE portfolio, 선택된 SMC 혼합, CE/SMC 가중치 75/25·50/50·25/75를 **동일한 512개 IID bank를 20회씩** 새로 뽑아 비교했다.

| 셀 | CE median ESS | SMC median ESS | 가장 높은 hybrid median ESS | 최고 후보의 하위 10% ESS |
|---|---:|---:|---:|---:|
| canonical | 9.58 | 13.56 | 18.24 (50/50) | 1.99 |
| 높은 η | 4.47 | 10.26 | 9.22 (50/50) | 1.54 |

사전 규칙은 두 셀에서 각각 가장 좋은 순수 원천보다 median ESS가 1.5배 이상이고 하위 10% ESS가 2 이상일 때만 hybrid 채택이었다. **통과한 조합은 없다.** 높은 η의 CE-only bank 20회 중 4회는 네 고정 cluster 중 하나의 target 기여가 1% 미만이었다. SMC-only에서는 이 단순 지표의 누락 사례가 없었지만, 이는 cluster가 포착하지 못한 모드를 배제하지 않는다.

## 4. 새 bank·새 최종 표본 ablation

기존 결과의 bank를 재사용하지 않았다. CE portfolio와 SMC 학습 혼합 `q` 각각에서 셀당 3개의 독립 4096경로 IID bank를 생성하고, 각 bank에 DCT/target-PCA × rank16/32 × KL/M₂의 여덟 fit을 수행했다. 각 fit은 별도의 8192경로 최종 평가를 받았다. KL과 M₂는 같은 defensive identity-covariance mean-shift family·같은 bank·같은 rank를 사용한다. M₂ 목적은 알려진 bank density `G` 아래 `E_G[g²p²/(Gq_θ)]`의 경험 근사이며, fitted bank의 경험 loss를 최종 성능으로 읽지 않았다.

4096 bank의 target ESS 평균은 canonical CE 42.8, SMC 35.8; 높은 η CE 10.0, SMC 25.8이다. 특히 높은 η SMC 원천의 개별 ESS는 2.5~61.2로 크게 흔들렸다. 셀별·fit별 accuracy gate를 통과한 결과는 **0개**다. 일부 KL/M₂ 또는 rank가 단일 평가에서 낮은 RSE처럼 보여도, 동일한 μ를 정확하게 추정했는지 먼저 확인하지 않으면 잘못 낮아 보이는 분산이다. 따라서 basis/rank/M₂ 우월성을 채택하지 않았다.

## 5. accuracy 자격이 있을 때만 총 wall-time

별도 frozen-workflow 실행에서 SMC는 canonical 256, 높은 η 128개의 독립 전체 SMC 반복을 사용했다. IS는 매 셀에서 새 SMC 학습 bank 3개 → K=2 identity 혼합 fit → 640×8192개의 독립 최종 경로를 사용했다. SMC의 비교 기준은 앞 단계의 독립 IS이고, IS의 비교 기준은 앞 단계의 독립 SMC다. 양쪽 RSE≤5%, 기준 RSE≤5%, 95% 차이 상한≤25% 기준, 그리고 두 새 방법 간 동등성이 모두 통과해야 시간 비율을 보고한다. 학습 bank 생성은 offline, 혼합 적합은 fit, 최종 draw·payoff·density는 inference에 포함한다. 설정은 실행 전 동결해 runtime selection=0으로 기록했다. 이전 개발 검색·실패 재시도 비용은 별도이며 이번 per-task frozen-workflow 시간에 몰래 0으로 합산한 것은 아니다.

| 셀 | SMC 새 RSE/시간 | IS 새 RSE | IS offline / fit / inference / 합계 | 공동 자격 | 허용되는 결론 |
|---|---:|---:|---:|---|---|
| canonical | 3.09% / 118.0초 | 8.07% | 1.38 / 0.01 / 75.08 / 76.47초 | 실패 | 시간비 발표 금지 |
| 높은 η | 2.55% / 82.74초 | 3.60% | 1.79 / 0.02 / 85.08 / 86.89초 | 통과 | 단일 실행에서 IS 약 5% 느림 |

이 시간은 Windows 노트북 CPU, PyTorch float64, 단일 thread에서 순차 실행해 얻었다. 전력·열·백그라운드 부하 반복 통제가 없고, 학습 반복의 분포를 추정할 만큼 충분한 독립 총시간 실행도 없다. **저널용 속도 주장이나 최적 알고리즘 선정 근거로 사용하지 않는다.**

## 6. 이론·기술 검토와 보류 판정

- weighted SMC는 고정 bridge와 고정 resampling calendar, unbiased stratified offspring 수, 현재 tempered target 불변 pCN kernel, 건너뛴 단계의 누적 particle weight를 사용한다. 모든 SMC particle을 독립 표본으로 SE 계산하지 않고 **독립 전체 SMC 실행**을 단위로 사용했다.
- SMC 은행의 원래 조상 수 증가는 계보 다양성 개선이지 posterior mode 완전성 증명이 아니다. weighted cluster 분석은 training-only geometry로 held-out path를 분류했지만 cluster 경계의 질과 미관측 모드는 여전히 한계다.
- 모든 최종 IS 표본은 training/selection과 별도 seed 역할을 갖고, 저장된 전체 Gaussian mixture density를 사용한다. 정상화·analytic Gaussian oracle·defensive support·동일 조건부 target 검사를 통과했다.
- 경험 KL/M₂ fit은 bank ESS가 매우 낮아 과적합 가능성이 크다. 새로운 수학적 정리나 canonical 전체 parameter 영역의 성능 우월성을 도출하지 않았다.
- hash-bound 입력 연결, seed ledger, SMC 반복 mean/SE/RSE, IS cluster mean/RSE, ablation RSE, 자격 gate, total-time stage 합계를 독립 감사기로 재계산했다. 실행 중 소스 추가로 각 v1 결과의 전체 source-tree digest는 최종 tree와 다르다. 감사기가 이를 mismatch로 명시한다. 같은 코드·설정·seed로 **9개 산출물의 timing을 제외한 전체 수치를 재실행해 일치**했다. 감사기의 재실행은 timing-derived ratio를 수치 동일성 검사에서 제외하되, 그 ratio와 각 stage 합계를 별도 산술 검사한다. 이는 특정 개발 환경의 재현성이지 독립 외부 검증이나 통계적 주장의 진실성 보장이 아니다.

## 7. 다음 gate

**R2 대형 모델로 진입하지 않는다.** 다음 최소 작업은 (a) 높은 η의 CE bank target ESS 하위 분위와 SMC bank 간 proposal 품질 변동을 함께 줄이는 작은 학습기 변경, (b) 기존 4-cluster 밖 희귀 기여 발견을 평가하는 독립 mode 진단, (c) 동일 512-bank 반복에서 개선된 ESS뿐 아니라 **새 fit 5회 이상**의 독립 final μ/RSE 및 전체 비용 개선을 동시에 확인하는 것이다. 한 번의 우연한 frozen proposal로 IS를 길게 돌린 결과를 새 학습 알고리즘의 안정성으로 바꾸지 않는다. 이 gate가 통과한 뒤에야 방법·hyperparameter·실패 처리와 미사용 셀을 봉인하고, 더 많은 독립 학습 반복의 R3 confirmation을 생산한다.

## 검증 기록과 파일

- [SMC 설계](../../results/post_audit/r15_reference_design_v1.json), [첫 교차검증의 실패](../../results/post_audit/r15_reference_crosscheck_v1.json), [새 독립 SMC 기준](../../results/post_audit/r15_reference_refinement_v1.json)
- [혼합 proposal 개발](../../results/post_audit/r15_mixture_design_v1.json), [새 bank의 독립 혼합 IS](../../results/post_audit/r15_mixture_independent_v1.json), [고정 proposal 대규모 IS](../../results/post_audit/r15_is_precision_v1.json), [CE-only 추가 교차검증 실패](../../results/post_audit/r15_ce_precision_v1.json)
- [새 bank ablation](../../results/post_audit/r15_fresh_ablation_v1.json), [hybrid bank 실패](../../results/post_audit/r15_hybrid_bank_pilot_v1.json), [총시간 실행](../../results/post_audit/r15_fixed_precision_total_work_v1.json)
- 재현/검증: `python -m experiments.post_audit_r15_audit --replay` (기존 결과를 덮어쓰지 않는 순수 재실행), `pytest -q`, `ruff check src tests experiments main.py train_driftnet.py`, `git diff --check`.

**최종 검증 상태:** 저장된 10개 artifact의 입력 hash 연결·총 9,050개 seed ledger·산술 gate 감사 통과. 수치 replay는 감사기의 timing-field 분류 수정 전 7개, 수정 후 2개, CE-only 추가 1개를 각각 수행하여 **10/10 일치**했다. `ruff check src tests experiments main.py train_driftnet.py` 통과. CE-only 추가 후 `pytest -q` 전체 **1,079개 통과**. 최종 tree에 대한 외부 독립 실행과 R3 confirmation은 미실행이다. 실행 중 새로운 source/config 파일이 추가됐으므로 v1 artifact의 저장된 전체 source-tree digest와 최종 tree는 10/10 불일치하며, 수치 replay와 이 provenance 한계를 함께 보고해야 한다.
