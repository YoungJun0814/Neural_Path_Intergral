# Conditional marginal-G 개발 결과 — 학습 bank 신뢰성 STOP

2026-10-21 America/New_York.
[사전 실행계획](../plans/MARGINAL_G_CONDITIONAL_RISK_DEVELOPMENT_2026-10-21_KO.md)을
동결한 뒤 새 자료로 실행하고 별도 재생 감사를 완료했다.
**결론: `STOP_MARGINAL_TRAINING_BANK_UNRELIABLE`.**
G fitting/selection/final은 실행하지 않았고 q/G 모델 채택, P2/D3, 대규모 학습 승인은 없다.
커밋·푸시는 하지 않았다.

## 1. 쉽게 설명하면

지금 모델은 두 역할로 나뉜다.

- **q:** 드문 사건 경로를 뽑아 사건확률을 계산하는 기존 제안분포.
- **G:** 그 q의 위험도, 즉 M2(q)를 검사할 경로를 뽑는 별도 평가분포.

이번에는 q를 바꾸지 않고 G를 학습할 준비를 했다. residual61을 먼저 적분해,
일부 좌표의 운에 덜 좌우되는 학습 점수를 얻었다. 새 G는 나머지62차원에만 작용하고
61/미사용63은 표준정규로 남기도록 구현했다.

그런데 **새 2,048개 학습 경로 중 극소수에 학습 점수가 몰렸다.**
이 자료로 평균·공분산·cluster를 학습하면 그 몇 개 경로의 우연을 모델 구조로 굳힐 위험이 있다.
그래서 사전에 정한 gate가 실제 fitting을 막았다.
“더 큰 신경망이 필요하다”는 결과가 아니라, **무엇으로 학습시켜야 할지부터 해결해야 한다**는 결과다.

## 2. 실제 수행 범위와 미실행 범위

완료:

1. 같은 q의 M2 평가라는 estimand를 고정하고 수학식을 점검했다.
2. m2 기반 target, f² 기반 selection objective, exact normalized sampler/density를 구현했다.
3. 학습용 Gaussian의 reduced62→full64 embedding과 residual factorization을 검증했다.
4. 26개 known-answer/반례/gate/밀도/seed/예산/직렬화 사전 테스트를 통과했다.
5. 두 셀×2rep×2,048개의 **새 training contexts**, 총8,192개를 생성했다.
6. 두 셀×두 raw 평가분포×4,096개, 총16,384개의 독립 대조 경로를 생성했다.
7. 전체 저장 자료의 seed/density/통계/판정 재계산과 사전 고정 full-law 재생을 통과했다.

미실행:

- covariance-rank4/two-cluster 실제 G fitting: **0회**.
- selection/final: **0개**.
- 새 사건q 또는 실용 평가G 채택: **없음**.
- 새로운 대규모 학습, P2/D3, fixed-precision benchmark: **없음**.

G fitting·selection·final의 구현 경로는 준비하고 일부 unit/known-answer 검사를 했지만,
**그 성공 branch 전체의 production end-to-end 검증을 완료한 것은 아니다.**
다음 실제 사용 전에는 그 경로를 별도 봉인된 작은 end-to-end 검사로 다시 검증해야 한다.

## 3. 수학적 대상 재확인

현재 q는 N32,373성분, natural mass0.1과 shifted mass0.9/372의 고정 혼합분포다.
U=61, unused63을 제외한 X는62차원이며57은 X에 포함한다.

```
Dq=q_X/p_X, Ds=S_X/p_X, DG=G_X/p_X,
m2(X)=E[g(X,U)^2 | X], f(X)=m2(X)/Dq(X),
M2(q)=E_pX[f(X)].
```

training S=q의 fitting log mass는 `logm2-logDq-logDs=logm2-2logDq`다.
최종평가 예정량은 ordinary mean `m2/(Dq DG)`다.
normalized softmax는 empirical fitting의 가중치/ESS에만 사용하고 최종 M2를 self-normalize하지 않는다.

selection 예정 objective는 `E_S[m2²/(Dq² Ds DG_alpha)]`다.
m1²로 원래q의 M2를 대체하거나 f와 f²를 바꾸지 않는다.
자유 marginal optimum `G_X ∝ p_X m2/Dq`의 범위는
[이론 부록](../theory/ANALYTIC_RESIDUAL_MARGINAL_G_SCOPE_ADDENDUM_2026-10-21_KO.md)에 따른다.
이번에는 임의의 H를 sampling하는 문제와 혼동하지 않는다.

새 core는 [`conditional_marginal_g.py`](../../src/path_integral/conditional_marginal_g.py),
runner는 [`post_audit_conditional_marginal_g.py`](../../experiments/post_audit_conditional_marginal_g.py)다.
기존 source/result/threshold를 소급 수정하지 않았다.

## 4. 새 training bank의 결과

각 bank는8개 독립block×256개였다. 기준은 ESS≥32, max-path≤10%, max-block≤50%,
구조적 numerical pass와 aggregate error proxy≤1e-6였다.

| 셀 / rep | ESS / 2,048 | 최대 경로 질량 | 최대 block 질량 | 판정 |
|---|---:|---:|---:|---|
| Canonical / 0 | 2.363 | 62.12% | 62.47% | 실패 |
| Canonical / 1 | 4.261 | 43.58% | 64.35% | 실패 |
| High eta / 0 | 8.571 | 30.96% | 45.56% | 실패 |
| High eta / 1 | 14.558 | 15.81% | 29.33% | 실패 |

**네 bank 모두 ESS와 max-path 기준에 실패**했고, canonical 둘은 max-block 기준에도 실패했다.
좋은 bank만 골라 학습하지 않았고 미실행 예산을 추가 n이나 다른 target에 재배정하지 않았다.

수치 측면에서는8,192개 profile의 개별 상대 gate가 모두 통과했다.
near-multiple/envelope/Jensen 등의 구조적 검사는 모두 통과했고,
가중 aggregate numerical error proxy는 약 **1.0002e-12–1.0019e-12**였다.
따라서 이번 STOP은 “적분 결과가 유실됐다/underflow를0으로 셌다”는 종류가 아니다.
다만 numerical proxy는 엄밀한 quadrature error bound가 아니다.

natural 성분에서 생성된 경로는 각 bank의189–230개였다.
그 draw들이 차지한 관측 risk mass는 약2.12%,10.89%,3.59%,3.09%였다.
component label은 서로 겹치는 mixture 성분의 생성 라벨이지 고유한 물리적 rare mode가 아니다.
이 수치로 발견되지 않은 모드가 없다고 주장하지 않는다.

네 bank의 ordinary M2 점추정은 각각
`1.498e-11, 5.865e-12, 1.232e-8, 6.880e-9`였다.
모두 집중 gate에 실패했으므로 population M2의 정답/수렴값으로 사용하지 않는다.
가중 ESS는 확률 자체의 신뢰구간이나 population effective sample size의 인증이 아니다.

## 5. 적분을 사용하지 않은 raw 대조 결과

두 방법은 같은 q의 M2를 다른 평가분포에서 추정한다.

- raw G=q: `g²/Dq²`.
- raw G=p: `g²/Dq`.

각각8block×512개로, training과 seed가 다르다.
서로 다른 proposal와 수치오류 경로의 대조이지 독립적인 정답 인증은 아니다.

| 셀 / raw 평가분포 | Raw RSE | Block RSE | 최대 한 경로 기여 | Accuracy 자격 |
|---|---:|---:|---:|---|
| Canonical / q | 53.85% | 52.23% | 51.18% | 실패 |
| Canonical / p | 100.00% | 100.00% | 100.00% | 실패 |
| High eta / q | 98.05% | 97.93% | 98.04% | 실패 |
| High eta / p | 98.50% | 98.28% | 98.49% | 실패 |

LOO와 최대 기여 기준도 통과하지 못했다.
자연분포 reference의 관측 평균은 극히 작았고, q reference와의 점추정 비율은
canonical 약 **1.92e24**, high eta 약 **2.95e7**이었다.
이를 상대적인 q/G 성능 개선율로 해석하면 안 된다.
**같은 기대값의 추정치가 현재 예산에서 극도로 불안정한 것**이다.

### `mean_consistency: true`를 왜 일치 성공으로 읽으면 안 되는가

원본의 이 flag는 고정된 “차이/combined estimated SE≤3.291” 검사다.
매우 큰 SE에서는 수십 자릿수 차이도 이 검사로 반증하지 못할 수 있다.
이번에는 두 셀 모두 이 flag가true였지만, precision gate가 별도로false라서
`qualified`는 양 셀 모두false였다.

따라서 정확한 해석은 **“저정밀도 점추정을 이 검사가 반증하지 못함”**이지
“독립 estimator들이 일치했다”가 아니다. 다음 사용에서는 이 라벨을
`no_detected_contradiction`처럼 명확히 구분하고, accuracy gate 이전의 agreement 성공 주장을 금지해야 한다.
원본 flag/threshold는 바꾸지 않았다.

## 6. 이번 결과로 말할 수 있는 것과 없는 것

말할 수 있는 것:

- 새 bank에서도 단일 residual 적분만으로 학습 질량 집중이 해결되지는 않았다.
- 현재 q로 IID importance bank를 만들어 G를 학습하는 경로는 이 고정 예산/규칙에서 실패했다.
- 단순한 covariance/cluster/operator 확대 전에 bank 생성 방식을 바꿔야 한다는 근거가 강해졌다.
- 원래 적분의 커널 속도 개선을 이용해 더 넓은 새 진단을 수행할 수 있었다.

말할 수 없는 것:

- 모든 conditional integration, 모든 marginal G, 또는 연구 주제 자체가 불가능하다.
- q가 population에서 나쁘다는 정량 결론이나 q/G가 CEM을 이긴다는 결론.
- 두 개 rep로 안정적인 learning reproducibility를 확보했다.
- 불확실한 natural reference가 실제 M2보다 낮으므로 biased한 estimator라는 증명.
- 현재 numerical m2가 exact finite-grid integration이거나 연속 Volterra law까지 exact하다는 주장.

현재 진전은 새로운 우승 모델이 아니라 **실패 메커니즘을 더 분명하게 분리한 것**이다.
계속 같은 q-IID bank의 표본 수/모델 크기만 늘리는 것은 이번 결과가 지지하는 우선순위가 아니다.

## 7. 재생 감사와 비용

보존된64개 artifact의 hash/완전성, 모든 경로의 seed/실제 sampling/density/라벨/score,
training 및 reference 통계·gate·최종 STOP 판정을 다시 계산했다.
full 재생은 다음과 같다.

- 각 training bank block0의 처음16개, 총64개 full profile.
- 그 block의 처음128개 저장 profile를 batch 재적분해 padding까지 맞춘 결과 비교.
- 각 셀/raw방법 block0의512개, 총2,048개 raw payoff.

전체 24,576개 payoff를 다시 실행한 것은 아니다.
실행되지 않은 fitting/selection/final branch의 production 재생도 아니다.

| 작업 | Physical evaluations | Scalar nodes |
|---|---:|---:|
| 새 training | 8,192 | 30,891,264 |
| 새 raw reference | 16,384 | 0 |
| 감사 | 2,112 | 1,926,144 |
| **주기 합계** | **26,688** | **32,817,408** |

scalar는 g/적분 node 수이며 rootfinder/density/FLOP 전체가 아니다.
그 연산은 wall에 포함됐다. unit/full regression 비용은 연구실험 예산과 별도다.
profile는 각context한번full, rawreference도각draw한번full 평가했다.

source freeze4.239초, setup3.721초, compute+artifact I/O30.981초,
그중 artifact I/O5.075초, 최종 result I/O0.0119초였다.
end-to-end receipt 직전 wall은 **39.267초**였다. 감사 wall은14.571초였다.
이는 full accuracy를 맞춘 효율 대조가 아니라 이번 실패 실험의 비용 기록이다.
정적 검사 프로세스와 시간대가 겹쳤으므로 독점 host benchmark라고도 부르지 않는다.

## 8. 결과를 반영한 다음 우선순위

1. **동일 q-IID bank 기반 fitting은 보류한다.** 이번 실패 bank로 G를 억지로 fit하거나
   n/ESS 기준을 사후 변경하지 않는다. rank/cluster/차원을 늘리는 것도 먼저 하지 않는다.
2. **조건부 risk target을 직접 탐색하는 bank 생성법을 별도 설계한다.**
   `r_X ∝ p_X m2/Dq`를 목표로 한 tempered/SMC 또는 동등한 target-aware 방식이 다음 후보다.
   현재 batch 코어를 이용할 수 있지만, 성능을 선취할 수는 없다.
3. **SMC의 uniform resampled weight를 높은 training ESS로 오인하지 않는다.**
   independent islands, 고유 ancestor/lineage 집중, mutation 이동, rare-mode 반복,
   실제 fitted density의 fresh final 성능을 모두 별도 gate로 둬야 한다.
4. **이론 identity와 known-answer transition부터 검증한다.**
   예를 들어 q에서 시작하는 bridge의 endpoint를 `q_X(m2/Dq²)^beta`로 두면 beta1에서
   정확히 `p_X m2/Dq`가 된다. 이를 `q_X(m2/Dq)^beta`로 쓰면 다른 target이므로 잘못이다.
   실제 구현에서는 numerical m2 오차, reference/q의 normalizer, mutation detailed balance를 다시 유도해야 한다.
5. **새 bank가 독립적 평가 gate를 얻기 전에는 대규모 G 학습/P2로 가지 않는다.**
   작은 SMC 통과를 저널용 성능 확인으로 바꾸지 말고 별도 training/selection/final/reference를 봉인한다.

이 후보도 실패할 수 있다. 다음 한정된 target-aware bank 실험마저 genealogy/정밀도에 실패하면,
또 다른 operator를 추가하기 전에 주장 축소·타깃 변경·negative-result 논문화 또는 중단을 다시 판단해야 한다.
그 후속 실험은 **이번에 실행하지 않았다**.

## 9. 재현 anchor와 검증 상태

- Result SHA256: `ecd0a93de8dc90a24a3a3e91972c125178986c573c15a8f1ff6e1c33fcf4afbd`
- Source ZIP SHA256: `0d63aa038fdb77814000f558cec563a5ed9da65c077954ff8fc522db59a668bd`
- Audit SHA256: `2b7a6281dd914843ea4bd5c3f8bbf575906c9ac3b4bc03100dc207a706460e9b`
- Snapshot의 source/config/test/선택된 문서 및 anchor:1,130개 파일.
- 64개 bank artifact 총45,984,924bytes. result는 파일목록/hash를 참조하고 경로자료를 중복 삽입하지 않았다.

[원본 result](../../results/post_audit/conditional_marginal_g_v1/session_20261021/result.json),
[audit](../../results/post_audit/conditional_marginal_g_v1/session_20261021/audit.json),
[end-to-end receipt](../../results/post_audit/conditional_marginal_g_v1/session_20261021/end_to_end.json).

26개 사전 테스트와 전체 **1,750개 회귀 테스트 /428.51초 /exit0**가 통과했다.
전체 source statement coverage는 **85%**(22,615 statements /3,431 missed),
새 marginal-G 코어는 **94%**(113 statements /7 missed)다.
CI범위 Ruff, Mypy217 source files와 새 runner 별도 검사도 통과했다.
이는 statement 검증 범위이며 branch 전수 증명이나 오류 부재/실험 성공의 보증이 아니다.
기존 RequestsDependencyWarning은 남아 있고 의존성을 변경하지 않았다.
이번 미커밋 연구 변경의 원격 Linux/Python3.11 CI는 아직 실행·확인하지 않았다.

### 기술 재검토의 남은 제한

- 앞서 명시한 G fitting 성공 branch는 아직 production end-to-end 미검증이다.
- result/artifact 저장은 numerical try/except 안에서 보호하지만,
  source freeze/setup/초기 protocol 저장 단계의 실패까지 동일 failure receipt로 감싸는 것은 아직 아니다.
  이번 실행에서는 해당 단계도 성공했다. 다음 실행기에는 이 시작 단계 I/O failure-injection 검사도 필요하다.
- numerical envelope와 aggregate error는 rigorous outward-rounded bounds가 아니다.
- `mean_consistency`의 큰-SE 비반증을 일치 성공으로 표시하면 안 된다.

연구 계산 후 `.gitattributes`에 validation 출력의 exact-byte 보존 규칙을 추가했다.
검증 receipt의 SHA가 향후 Git LF 변환으로 달라지는 것을 막는 조치이며,
동결된 numeric source/result/archive를 바꾸지 않았다.
기존과 새 stdout 파일에서 filtered Git blob이 raw-byte blob과 같음을 확인했다.

최종 receipt 추가 전135개 파일,5개 archive의5,619 members를 검사했다.
알려진 credential 패턴·비정상 archive 경로·100MiB 이상 파일·trailing whitespace는 발견하지 못했다.
당시 strict JSON99개/local Markdown link176개가 통과했고,
다섯 source ZIP/anchor 및1,142개 고유 동결 파일의 현재 hash가 일치했다.
이는 모든 비밀정보 부재의 보증이나 repository 전체의 보안 감사가 아니다.

[무결성 receipt](../../results/post_audit/conditional_marginal_g_v1/session_20261021/validation/integrity_review.json),
[회귀/정적 검사 receipt](../../results/post_audit/conditional_marginal_g_v1/session_20261021/validation/regression_review.json),
[pytest 출력](../../results/post_audit/conditional_marginal_g_v1/session_20261021/validation/pytest-output.txt),
[Ruff/Mypy 출력](../../results/post_audit/conditional_marginal_g_v1/session_20261021/validation/static_checks-output.txt)을 보존했다.

최종 README/두 receipt/바이트 보존 규칙을 반영한138개 파일에서 strict JSON101개,
local Markdown link180개, 다섯 anchor/archive와1,142개 동결 파일의 hash를 다시 확인했다.
현재 runtime tree도 새 실행의 동결 digest와 일치했고, 로그 hash/Git raw-byte 보존 및
`git diff --check`가 통과했다. staged 변경은 없고 HEAD는 `f556424` 그대로다.
커밋·푸시는 하지 않았다.
