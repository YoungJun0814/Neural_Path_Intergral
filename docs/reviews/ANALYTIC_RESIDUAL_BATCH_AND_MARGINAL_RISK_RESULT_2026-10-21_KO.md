# 해석적 residual·배치 적분·joint 법칙·marginal risk 실행 결과

2026-10-21 America/New_York. 이전 Q0/Q1 결과 §8의 다음 판단을 수행한 보고서다.
결론은 **계산 도구의 제한적 채택, 새 모델·P2·대규모 학습의 채택 보류 유지**다.
원본 source/계획/결과를 소급 수정하지 않았고 이번 연구 변경은 커밋/푸시하지 않았다.

## 1. 이번에 무엇이 달라졌는가

| 질문 | 실행 결과 | 허용하는 결론 |
|---|---|---|
| 경로별 adaptive 적분 비용을 줄일 수 있는가? | 같은 262개 profile에서 batch/adaptive 수치 일치, 커널 wall 중앙값 비율 73.27 | `ADOPT_BATCH_DIAGNOSTIC_ONLY` |
| residual 57/61을 동시에 분리할 수 있는가? | 현재 N32 hybrid 법칙의 식 유도, 양 셀 zero context의 full-law/replay 일치 | 제한된 joint scalar 도구와 finite-grid 의존관계 확인 |
| U만 처리하면 risk 집중이 없어지는가? | finite known-answer에서 conditional 항만 제거되고 marginal 항이 남음 | 다음 연구는 X 방향 평가분포도 다뤄야 함 |
| 사건 q 또는 실제 G가 개선됐는가? | 새 q/G fitting·새 무작위 표본·population 대조 없음 | 개선됐다고 주장하지 않음 |
| P2/D3/대규모 학습으로 넘어가는가? | 이전 Q1 정밀도 실패 그대로 | **아니오** |

쉽게 말하면, **확률을 검사하는 계산기는 빨라졌지만 희귀 기여 경로를 충분히 확보하는 문제는 아직 풀리지 않았다.**
73배는 모델 전체가 아니라 같은 적분 문제를 푸는 두 커널의 계측 비율이다.

## 2. 계획과 절차 변경의 기록

실행 전 고정한 문서는
[원래 실행계획](../plans/ANALYTIC_RESIDUAL_BATCH_AND_MARGINAL_RISK_2026-10-21_KO.md)과
[수학 계약](../theory/ANALYTIC_RESIDUAL_AND_MARGINAL_RISK_CONTRACT_2026-10-21_KO.md)이다.
후속 이론 검토의 적용 범위 보완은
[marginal-G 부록](../theory/ANALYTIC_RESIDUAL_MARGINAL_G_SCOPE_ADDENDUM_2026-10-21_KO.md)에 따로 남겼다.

### 원래 실행의 저장 오류

첫 실행은 kernel/joint 계산 뒤 `result.json` 저장에서
`TypeError: Object of type bool_ is not JSON serializable`로 실패했다.
critical-point flag에 NumPy bool이 포함됐고, 원래 writer 호출이 numerical try/except 밖에 있었다.
따라서 자동 failure 파일도 생기지 않았다. 이 누락을 숨기거나 성공 결과처럼 다루지 않았다.

- 원래 protocol/source ZIP을 보존했다.
- 별도 `serialization_failure.json`에 오류와 복구 가능한 범위를 기록했다.
- 첫 실행의 개별 수치·timing·정확한 kernel 평가 횟수는 저장되지 않아 **복구하지 못했다**.
- joint 60 physical/54 scalar는 최종 writer까지 도달한 control flow에서 **재구성한 비용**이다.
  여섯 context의 성공 자료가 남았다는 의미가 아니다.

[복구 부록](../plans/ANALYTIC_RESIDUAL_SERIALIZATION_RECOVERY_2026-10-21_KO.md)을 고정한 뒤
새 wrapper/source/directory에서 NumPy scalar를 Python scalar로 바꾸어 다시 실행했다.
수치식·order·cutoff·tolerance·AB/BA/AB 순서는 바꾸지 않았다.
전체 payload 직렬화, 수치값 보존, NaN 거부 테스트도 추가했다.

복구는 **사후 절차 변경**이며 scalar 예산이 늘었다.
첫 timing 중 좋은 결과를 골랐다는 뜻은 아니지만, 첫 실행과 독립된 사전 확정 confirmation이라고도 부를 수 없다.
physical 총 cap100을 지키기 위해 joint full 검증은 원래 여섯 context에서 **두 zero context로 축소**했다.

## 3. 해석적 분할과 배치 적분

### 왜 단순히 적분 node만 줄이지 않았는가

단일 residual `U~N(0,1)`의 조건부 사건확률은

```
t = exp((logv + beta*u)/2) > 0,
h(t) = (a + b*t² + c*t)/sqrt(k*(d + e*t²)),
g(u) = Phi(h(t)),
P(t) = b*e*t³ + (2*b*d-e*a)*t + c*d.
```

여기서 `dh/dt`의 부호는 P의 부호다. `b,e,d,k>0`이며
`P'(t)=3*b*e*t²+(2*b*d-e*a)`는 양의 t에서 단조증가한다.
P는 양의 축에서 turning point가 최대 하나, 양의 root가 최대 두 개다.
계획의 P' turning-point 표현은 **P'의 양의 zero/P의 turning point**라는 의미다.

이를 이용해 critical u에서 panel을 분할하고 같은 g node로 m1/m2를 함께 계산했다.
고정 `[-16,16]`, unit panel, 16/32/64 Gauss–Legendre rule, log-space 집계를 사용했다.
두 order 차이와 상대 1e-12 floor, Gaussian tail bound를 numerical proxy에 포함했다.
near-double root나 overflow 등 미해결 경우는 성공으로 넘기지 않는다.

새 구현:
[`analytic_residual_batch.py`](../../src/path_integral/analytic_residual_batch.py).
이는 production event estimator를 교체한 것이 아니라 별도 진단 코어다.

### 262개 고정 profile 결과

Q0의 여섯 개와 Q1의 256개 **전부**를 재사용했다. 새로운 대표성 표본이 아니다.

| 항목 | 값 |
|---|---:|
| 상대 numerical gate 통과 | 262 / 262 |
| frozen adaptive reference와 최대 상대 차이 | 7.106e-15 |
| numerical proxy+tail 최대 상대값 | 1.015e-12 |
| 알려진 상수-CDF 두 경우의 최대 log error | 3.802e-16 |
| 검출 critical point 수 0개 / 1개 | 135 / 127 |
| near-multiple case | 관측 profile에는 0개; 별도 실패 반례 테스트 |

두 critical point를 가진 별도 고정 test도 기존 adaptive 결과와 일치했다.
tail이 작은 moment를 지배하는 경우에는 absolute-only 방식으로 통과하지 않는 것을 확인했다.

### Envelope의 중요한 한계

monotone panel의 endpoint g min/max와 Gaussian panel mass로 상·하 envelope를 계산했다.
정확한 extrema·실수 연산에서는 적분 부등식이지만 현재 root/CDF는 outward-rounded 연산이 아니다.
따라서 **rigorous certified interval이 아니다**.

모든 reference는 이 numerical envelope 안에 있었지만 envelope 자체가 매우 좁지는 않았다.
`log(upper/lower)`의 중앙값/최댓값은 m1에서 **0.115 / 9.715**, m2에서 **0.366 / 10.597**이었다.
이를 1e-6 상대 오차의 엄밀한 인증이라고 쓰면 잘못이다.
작은 order 차이, 기존 adaptive와의 일치, known-answer는 서로 보완하는 개발 검증이며
모든 미래 profile의 오차 보증이나 수치적 불편성의 증명은 아니다.
정확한 finite-grid density/조건부 identity, numerical quadrature, 연속시간 이산화 오차는
서로 다른 층위다. 이 결과의 구조적 exactness가 뒤의 두 오차까지 0이라는 뜻은 아니다.

## 4. 계산 비용 대조

같은 262개 profile에 양 함수 warmup 후 고정 순서로 세 번 측정했다.
benchmark 동안 전체 pytest를 함께 실행하지 않았다.

| 반복 | 순서 | Adaptive wall (s) | Batch wall (s) |
|---|---|---:|---:|
| 1 | A→B | 38.0574 | 0.588851 |
| 2 | B→A | 50.4127 | 0.632733 |
| 3 | A→B | 43.1477 | 0.518397 |

`median(A)/median(B)=73.2744`. 세 반복 모두 numerical gate를 통과했다.
각 반복의 scalar node 수는 A **1,456,994**, B **985,644**였다.
warmup/padding/endpoint를 포함한 전체 recovery kernel work는 **8,344,518**이다.
kernel stage wall은 **134.626초**다.

비율은 scalar node 수 감소뿐 아니라 Python callback 제거와 배열 연산 효과를 포함한다.
예산 계측 callback의 overhead도 포함한다. 완전히 독점된 host의 benchmark가 아니고,
source archive·setup·최종 I/O·전체 테스트 비용까지 포함한 end-to-end 대조도 아니다.
세 개 timing은 충분한 독립 performance confirmation이 아니다.

**기존 Q1의 253배/955배 비용 proxy를 73으로 나누어 새 모델 성능을 추정하지 않는다.**
새 profile 추출과 실제 estimator의 총비용, sampling precision이 따로 검증돼야 한다.
이번에는 matched-wall/fixed-precision 모델 대조를 하지 않았다.

## 5. 57/61 공동 residual과 63 제외

현재 N32 hybrid kernel의 독립 local Gaussian 좌표에서 odd `2j+1`은
해당 local integral과 `V_(j+1)`만 바꾼다. 역사 convolution은 dW 채널을 쓴다.

| 좌표 — zero-based | 바꾸는 분산 | 사건 payoff에 사용되는가? |
|---|---|---|
| 57 | V29 | 예 |
| 61 | V31 | 예 |
| 63 | V32 | 아니오: 현재 terminal estimator는 V0..V31 사용 |

모든 성분 mean의 57/61이 정확히 0이고 성분 covariance가 I인 현재 dictionary에서는
어떤 mixture weight에서도 `q=q_X phi(U)phi(W)`로 분리된다. G도 같은 조건에서만 마찬가지다.
이 identity는 수치 rank49로부터 추측한 것이 아니다.
이는 **현재 이산 hybrid 법칙**이며 연속 Volterra 과정의 독립성으로 일반화하지 않는다.

V29/V31을 제외한 A/I를 저장하여 두 분산만 갱신하는 joint CDF식을 구현했다.
payoff의 분모와 terminal 항을 공유하므로 `g(U,W)`를 두 1D 함수의 곱으로 놓지 않았다.

복구에서는 양 셀 zero context 각각 3×3 probe를 full law와 대조했다.
최대 log-payoff 차이는 canonical **6.822e-13**, high eta **9.095e-13**이었다.
감사에서 같은 두 context를 full replay했다.
별도 unit test는 nonzero 고정 context, unused63 불변성, odd 좌표별 단일 분산 변화를 다뤘다.
unit test를 원래 여섯 research context의 대체 성공 자료로 세지는 않는다.

이번에 joint 2D 적분기·학습기·production sampler·성능 우위는 구현/입증하지 않았다.

## 6. 남아 있는 marginal risk와 목표 구분

`X=나머지 경로`, `U=분리한 residual`이라 두면

```
Var_G(g²/(Dq DGX DH))
  = Var_GX(m2/(Dq DGX)) + E_GX Var_H(g²/(Dq DGX DH)|X).
```

이 분해가 핵심이다. U를 완벽하게 처리해도 **X에서 중요한 경로를 드물게 뽑는 문제**는 남을 수 있다.
유한 두-X/두-U known-answer를 구현하여 이 항들을 직접 전수 합산했다.

| Toy 평가분포 | Total variance | Marginal 항 | Conditional 항 |
|---|---:|---:|---:|
| 기존 p | 0.060581628 | 0.044901610 | 0.015680018 |
| ideal conditional H*만 적용 | 0.044901610 | 0.044901610 | 0 |
| ideal joint G | 약 4.57e-35 | roundoff 수준 | roundoff 수준 |

세 경우의 같은 고정 q target은 `M2=0.1062`다.
이것은 **toy의 exact finite sum**이지 Volterra population variance 측정이나 새 G 모델의 성능이 아니다.
새 코어는 [`marginal_risk_contract.py`](../../src/path_integral/marginal_risk_contract.py)다.

### 최종 이론 검토에서 보완한 조건

동결 계약의 `G_X* ∝ p_X m2/Dq`는 U를 적분한 risk estimator,
또는 정확한 ideal conditional H*를 사용할 때의 식으로 해석해야 한다.
일반적인 고정 H에서는

```
B_H(X)=E_pU[g^4/DH],
G_X,H* ∝ p_X sqrt(B_H)/Dq.
```

따라서 임의의 H에 m2 목표를 그대로 붙여 total variance 최적이라고 부르면 틀린다.
이 적용 조건을 [별도 이론 부록](../theory/ANALYTIC_RESIDUAL_MARGINAL_G_SCOPE_ADDENDUM_2026-10-21_KO.md)에
명시했다. 이번 oracle의 H는 ideal이므로 저장된 결과는 이 보완과 일치한다.
기존 수치식/관측 판정은 바꾸지 않았다.

사건 q 자체의 자유 marginal optimum `p_X sqrt(m2)`,
조건부 사건 estimator의 optimum `p_X m1`,
고정 q의 조건부 risk G optimum `p_X m2/Dq`도 서로 다르다.
이들은 기존 Cauchy–Schwarz/zero-variance 원리의 적용이지 이번 연구의 독창성 정리가 아니다.

## 7. 기술 검토와 재생 범위

완료한 수치 audit는 다음과 같다.

1. 262개 profile의 id/digest와 두 parent result anchor 결합 검사.
2. 전체 262개 batch 결과를 그대로 재계산하여 저장 결과와 일치 확인.
3. 양 셀 zero의 기존 adaptive 적분 독립 재계산.
4. warmup·padding·전체 scalar count·AB/BA/AB 순서·양수 유한 wall·채택 decision 재계산.
5. 양 셀 zero joint full replay와 유한 marginal-risk oracle 재계산.
6. protocol/source archive/current frozen file hash 검사.

audit는 3회 wall 값을 독립 host에서 재측정한 것이 아니며,
원래 저장 실패한 실행의 수치 결과를 복원한 것도 아니다.
새로운 near-double/2-root/constant/tail/budget/finite-density/overflow/JSON roundtrip 테스트를 추가했다.
전체 회귀검증과 최종 무결성 receipt는 아래 최종 검증 절에 기록한다.

## 8. 실패·복구를 포함한 연구 비용

| 단계 | Physical evaluations | Scalar evaluations |
|---|---:|---:|
| 저장 실패한 첫 kernel | 0 | 정확 횟수 유실: 상한 예약 12,000,000 |
| 저장 실패한 첫 joint | 60 — control-flow 재구성 | 54 — 재구성 |
| 복구 kernel | 0 | 8,344,518 |
| 복구 joint | 20 | 18 |
| kernel audit | 0 | 1,016,602 |
| joint audit | 20 | 18 |

physical 총계 **100** 중 60은 재구성,40은 보존된 실행/audit 기록이다.
첫 kernel 외 scalar 합계는 **9,361,210**이고, 첫 kernel의 전체 예약을 더한 보수적 상한은
**21,361,210**이다. 이를 실제 사용량의 정확한 총계라고 부르지 않는다.

원래 scalar 전체 예산 안에서 끝났다고 주장하지 않는다. 복구 부록의 예산 확대가 있었다.
unit/full regression 내부의 물리 계산은 연구 실험 예산과 별도다.
여기 scalar 단위는 g/적분 node 평가 수다. rootfinder의 다항식 연산이나 density 연산을
같은 단위로 환산한 총 FLOP 수가 아니며, 해당 실행 overhead는 계측 wall에 포함된다.
새 random draw, 새 training bank, 새 G fitting은 **0**이다.
앞선 Q0/Q1의 576 physical/1,713,716 scalar는 별도 주기 비용이며 덮어쓰지 않았다.

## 9. 다음 우선순위 — 무엇을 채택하고 무엇을 멈추는가

**채택:** 해석적 critical-point 분할, batch 진단 적분, 상대 numerical gate,
exact finite-grid residual 의존관계, marginal/conditional risk 분해와 estimator별 target 구분.

**채택하지 않음:** 새 q/G 모델, joint2D production, 73배 모델 speedup 주장,
기존 실패 bank를 새 confirmation으로 취급하는 것, P2/D3/대규모 학습.

다음 단계는 다음 순서로 한정하는 것이 타당하다.

1. **Marginal-G 개발 문제를 먼저 정확히 고정한다.** 원래 q는 동결하고,
   conditional m2를 쓰는 평가 estimator인지 실제 H sampling인지 먼저 선택한다.
   목표는 q 모델의 성능 변경이 아니라 고정 q의 M2를 신뢰할 수 있게 평가하는 것이다.
2. **작은 범위에서 numerator/density/numerical error의 알려진 답을 먼저 검증한다.**
   actual normalized density와 sampler가 같은 G인지, natural defensive mass가 포함됐는지,
   support와 conditional factorization이 유지되는지 확인한다. 적분 실패는 누락하지 않는다.
3. **새 bounded 계획을 동결한 뒤에만 training/selection/final/reference를 새로 만든다.**
   현재 262 profile는 설계·진단 자료로만 쓴다. 기존 관측을 보고 sample 수를 자동 증액하지 않는다.
   raw/block RSE, 최대 기여, LOO, rare-mode, training ESS, 독립 mechanism reference를 함께 고정한다.
4. **적어도 두 방법이 같은 accuracy 자격을 얻은 경우에만 total-wall을 비교한다.**
   offline/fit/selection/inference/I/O/실패·복구 비용을 포함한다. 미자격 점추정으로 우승자를 고르지 않는다.
5. **57/61 joint 확장은 차순위 가설이다.** 현재 full 연구 검증은 두 zero context뿐이고
   X-risk가 해결되지 않았으므로, 무조건적인 2D 적분/차원 확대를 먼저 수행할 근거는 없다.

이 방향도 성공을 보장하지 않는다. 새로운 marginal G가 신뢰 가능한 risk 평가를 못 만들면
같은 target의 더 큰 학습을 반복하는 대신 제한된 negative result/방법론 논문 경로까지 재평가해야 한다.
현재는 최상위 저널 수준의 model novelty·population efficiency·실용성 입증이 완료된 상태가 아니다.

## 10. 재현 anchor

- Q0 result: `53c604c5faafa573ae380605342a130f68d54dcde6cefb30393a380bc482bb6b`
- Q1 result: `8645b470b638ff9860ea831915c1bf10e2111d66ca1780f31640c2c6c008ade1`
- 복구 result: `a0a6a3b39c475fcf5fa92a2a6a652fdf21c3da1a311c56dbda2b0888f2896f54`
- 복구 source ZIP: `157501d98380093bf6a048a93ce8da492dfd28741ccf95aea7a6712114166564`
- 복구 audit: `544f600916288b3df0c808254ace3cb795bdbab5b770059ddcfba7e6121b6ef5`

원래 실패 디렉터리:
[`session_20261021`](../../results/post_audit/analytic_residual_batch_v1/session_20261021/serialization_failure.json).
복구 디렉터리:
[`session_20261021_serialization_recovery`](../../results/post_audit/analytic_residual_batch_v1/session_20261021_serialization_recovery/result.json).
실행 runner와 복구 runner를 둘 다 보존하며, 실패한 original runner를 성공한 production entry point처럼 쓰지 않는다.

## 11. 최종 검증

전체 회귀검증은 **1,724개 통과 / 406.74초 / exit0**다. 이번 추가 테스트는22개다.
전체 source coverage는 **85%**(22,502 statements / 3,424 missed),
새 batch 코어는 **91%**(159/14), marginal-risk 코어는 **92%**(39/3)다.
이전 log-quadrature 코어는94%(79/5)다. 숫자의 분모/미검증 statement를 함께 보존한다.
이는 branch 전수 증명이나 오류 부재의 보증이 아니다.

앞선 전체 테스트 시도는 최종 종료 결과를 확보하지 못하고 프로세스도 남지 않아 성공으로 세지 않았다.
파일 로그를 남기는 별도 전체 검증에서 위의 최종 종료 결과를 확보했다.
이는 새 연구 표본을 생성하는 재실험이 아니다.

CI 범위 Ruff 전체 통과, Mypy **216 source files 통과**, 새 두 runner의 별도 Mypy도 통과했다.
Python3.10.11 / NumPy1.26.2 / SciPy1.15.3 / Torch2.12.0.dev20260408+cu128 환경이다.
연구 실행은 CPUfloat64/1thread였고, 전체 pytest의 thread 설정은 각 test fixture에 따른다.
기존 RequestsDependencyWarning은 남아 있으며 의존성을 바꾸지는 않았다.
이번 미커밋 연구 코드의 원격 Linux/Python3.11 CI는 실행·확인하지 않았다.

Q0/Q1/실패 실행/복구 실행의 source archive와 현재 동결 파일은 모두 일치했다.
source별 파일 수는1,114/1,117/1,125/1,128이다.
최종 receipt 추가 전54개 파일/4개 archive의4,488 members를 확인하여,
검사한 credential 패턴·비정상 archive 경로·100MiB 이상 파일은 발견되지 않았다.
경계 없는 key 정규식이 기존 `risk-...` 라벨의 일부를 key로 오탐한 것은 문맥 확인 후
token 경계를 적용해 분류했고 source를 수정하지 않았다. 모든 비밀정보 부재의 보증은 아니다.
당시 local Markdown link163개와 trailing-whitespace 검사를 통과했다.

재생 감사와 별개인 최종 검증 자료:

- [무결성 검토 receipt](../../results/post_audit/analytic_residual_batch_v1/session_20261021_serialization_recovery/validation/integrity_review.json)
- [회귀/정적 검사 receipt](../../results/post_audit/analytic_residual_batch_v1/session_20261021_serialization_recovery/validation/regression_review.json)
- [전체 pytest 출력](../../results/post_audit/analytic_residual_batch_v1/session_20261021_serialization_recovery/validation/pytest-output.txt)
- [Ruff/Mypy 출력](../../results/post_audit/analytic_residual_batch_v1/session_20261021_serialization_recovery/validation/static_checks-output.txt)

최종 보고서·README·두 receipt·검증 로그를 반영한58개 파일에서 strict JSON31개와
local Markdown link167개를 확인했고 `git diff --check`도 통과했다.
네 anchor/source ZIP과1,134개 고유 동결 파일의 현재 hash, 검증 로그 hash가 일치했다.
staged 변경은 없고 HEAD는 `f556424360b8307284b23a6cfc0efcdee7380060` 그대로다.
연구/validation 결과의 커밋·푸시는 하지 않는다.
