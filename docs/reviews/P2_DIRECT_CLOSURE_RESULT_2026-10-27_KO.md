# P2 직접 해결 실행·결과·미해결 의무 보고서

2026-10-27 America/New_York. 기준 main `59f2809`. 커밋·푸시하지 않는다.

## 1. 이번 작업의 범위

이전 단계의 작은 N32 모델 후보/계산가속에서 벗어나, 원래 P2의 N4096
독립 기준, signed N4096/N2048 correction, 연속시간 remainder를 직접 다룬다.
기존 실패, 원래 threshold/BLP law, RSE1.5%와 equivalence10% 기준을 보존한다.
성공적인 구현과 P2 전체 자격 획득을 구분한다.

사전 고정한 [실행계획](../plans/P2_DIRECT_CLOSURE_EXECUTION_2026-10-27_KO.md)과
[이론 재검토](../theory/P2_DIRECT_REFERENCE_AND_CONTINUUM_REVIEW_2026-10-27_KO.md)를 참고한다.

## 2. 단계별 구현

1. 기존 N4096 frozen-ensemble IID의 .902%/.907% 관측 RSE 증거와 감사 hash를
   새 실행의 anchor에 묶었다. 그것을 다른 estimator가 반드시 따라야 할 참값으로 가정하지 않는다.
2. 기존 Gaussian augmentation을 유지한 joint fine/coarse family를 사용했다.
   raw dimension10240, H=(gf+gc)/2, full product normalizer와 endpoint factor로 복원한다.
3. gradient MH 대조군과 gradient+cyclic-pair MH를 구현했다. 이동은 표준 Gaussian
   prior에 reversible이고, 물리 target은 재평가/MH 보정한다. Volterra의 물리 대칭이나
   adapted continuous-time drift control이라고 주장하지 않는다.
4. 독립 whole를 통계 단위로 precision·LOO·최대기여·signed paired difference를 계산한다.
   fine reference의 precision이 미달이면 equivalence를 평가하지 않는다.
5. 기존 driver-MSE/CDF 절대오차 식을 log-domain 함수로 연결해 실제 수치 상계를 평가한다.
   값이 큰 것을 0으로 만들거나 관측 μ를 참μ의 인증된 lower bound로 사용하지 않는다.
6. 작업/seed/stage/복원 항등식을 전수 검사하고 대표4whole 재생과 전체 회귀를 수행한다.

구현 파일:

- `src/path_integral/cyclic_umbrella_smc.py`
- `src/path_integral/continuum_probability_bound.py`
- `experiments/post_audit_p2_direct_closure.py`
- 신규 이론·실행 계약 테스트21개.

## 3. 이론 검사에서 이미 확인된 범위

### 3.1 올바른 부분

- pair permutation의 역/orthogonality/Gaussian prior 보존과 MH 상세균형을 확인했다.
- permutation만으로 희귀 크기를 생성하지는 못한다. gradient-Gaussian 이동을 유지한다.
- 개별 이동의 invariant 성질에서 합성 kernel의 invariant 성질이 따른다.
- `(μhat_f+μhat_c)/2=Zhat`를 모든 whole에서 검사한다.
- 두 grid의 차이에 같은 whole의 covariance를 사용한다.
- drift norm cap은 proposal에만 적용하며 potential/weights/likelihood는 clipping하지 않는다.

### 3.2 전체 P2 인증과 구분해야 하는 부분

N4096에서 현재 보수적 absolute bias upper는 canonical 약68.83,high-eta 약5207.37이다.
확률 범위로 자르면 둘 다1이며 N8192/N16384에서도1이다.
현재 proof chain으로 희귀 확률의 상대 편향을 인증할 수 없다.

이것은 실제 편향이 큰 lower bound가 아니다. 관측 μ의10%를 이 식으로 맞추는
진단 격자 규모가 약10^217.59/10^218.10인 것도 실제 필요한 격자 수라는 주장이 아니다.
현재 상계가 실용적으로 지나치게 느슨하다는 뜻이다.

기존 weak-rate 논문들을 확인했으나 현재의 nonlinear digital probability,
원래 BLP scheme, 계산 가능한 유한-N 상대 상수를 모두 충족하는 적용 증명은 확보하지 못했다.
유한 adjacent 차이만으로 무한 remainder를 인증할 수 없는 논리적 이유도 이론 문서에 정리했다.

## 4. 최종 실행·감사·회귀 결과

### 4.1 고정 실험 32whole 완료: 새 cyclic 후보 미채택

2cells × 2arms × 8개의 독립 whole를 모두 실행했다. 각 whole는128particles,
64temperatures, stage당2mutation이며, 최종 stage에는 mutation을 하지 않는다.
preflight4개를 포함한 총량은 joint callback513,856 / physical forward1,027,712 /
time-step forward3,157,131,264다. 기록된 실행 wall은639.05초다.
gradient callback 요청은 joint 단위이며 backward를 physical forward와 같은 비용으로
바꾸지 않았다. 사전 고정한 RSS·작업량·whole/phase 시간 상한 위반은 없었다.

| 셀 | arm | N4096 추정 확률 | fine RSE | coarse RSE | 최대 whole 기여 | 최대 LOO 변화 |
|---|---|---:|---:|---:|---:|---:|
| canonical | gradient-control | 1.32573e-8 | 7.32% | 7.16% | 16.31% | 4.35% |
| canonical | gradient-cyclic | 1.13401e-8 | 32.16% | 31.31% | 39.32% | 30.65% |
| high-eta | gradient-control | 1.30565e-6 | 16.50% | 14.56% | 23.13% | 12.15% |
| high-eta | gradient-cyclic | 9.64146e-7 | 18.25% | 18.51% | 22.55% | 11.48% |

요구 fine RSE≤1.5%를 네 경우 모두 충족하지 못했다. 최대 whole 기여≤15%도
모두 미달이며, canonical control 이외에는 LOO≤5%도 미달이다.
따라서 기존 IID와의10% equivalence는 네 경우 모두 **평가하지 않았다**.
평균이 비슷해 보이는 경우가 있어도 정확성 자격이라고 부르지 않는다.

기존 N4096 frozen-ensemble IID는 canonical1.37653e-8/high-eta9.82276e-7,
관측 RSE .902%/.907% 및 기존 block/LOO 조건을 충족한 상태다. 기존 값이 참값임을
가정한 것이 아니며, 이번 SMC와의 차이만으로 어느 쪽의 bias라고 진단할 수 없다.

같은128particles에서 마지막 initial ancestors는 control10–15/13–18,
cyclic6–13/6–13이었다. 두 번째 이동의 평균 acceptance는 control83.79%/83.10%,
cyclic38.91%/41.33%다. acceptance와 stage ESS가 괜찮아 보이는 것만으로
normalizer의 independent-whole precision이 좋아지지는 않았다.

새 cyclic arm은 이번 고정 pilot에서 대조군보다 낮은 RSE를 얻지 못했다.
표본 수가8whole이므로 보편적인 열등성 정리를 주장하지 않지만, 채택·확장 근거도
없다. **이번 설정의 cyclic 후보는 중단하며 production 모델에 넣지 않는다.**

### 4.2 Signed adjacent correction: 개발 진단만 통과

차이의 정의는 `(μ4096−μ2048)/μ4096`다. 양쪽 확률을 같은 whole에서 복원하고
covariance를 포함했다. 네 비교에 조정된 z=2.4977054744를 사용한다.

| 셀 | arm | signed 상대 차이 | paired SE | `abs(diff)+z SE` | 10% 개발 gate |
|---|---|---:|---:|---:|---|
| canonical | control | -0.232% | 3.647% | 9.342% | 통과 |
| canonical | cyclic | 2.017% | 1.057% | 4.658% | 통과 |
| high-eta | control | 3.711% | 2.257% | 9.349% | 통과 |
| high-eta | cyclic | 0.935% | 2.053% | 6.062% | 통과 |

모든 판정은8개의 independent-whole delta-method 개발 통계다. fine와coarse가
함께 잘못 낮거나 같은 희귀 모드를 놓쳐도 비율 차이는 작을 수 있다. fine absolute
reference가 미자격인 상태에서는 이것을 joint accuracy의 독립 증명으로 쓰지 않는다.
특히 이 결과는 N4096 이후의 모든 격자 오차나 연속시간 상대 bias를 인증하지 않는다.

### 4.3 무작정 반복을 늘리지 않은 이유

사전 고정한 safety3, target RSE1.5%, minimum32/maximum256whole의 새 표본
예측은 다음과 같다. 기존8whole을 confirmation에 재사용하지 않는다.

| 셀 | arm | 예측 필요 새 whole | physical forward 예측 | pilot 평균 속도만 외삽한 시간 |
|---|---|---:|---:|---:|
| canonical | control | 576 | 18,432,000 | 약3.75시간 |
| canonical | cyclic | 11,040 | 353,280,000 | 약55.44시간 |
| high-eta | control | 2,904 | 92,928,000 | 약17.54시간 |
| high-eta | cyclic | 3,560 | 113,920,000 | 약15.95시간 |

네 경우 모두 고정한256whole 예산을 넘는다. 이는 pilot 기반 비용 예측일 뿐,
해당 반복 수를 실행하면 정밀도가 보장된다는 뜻이 아니다. 미관측 tail은 예측을
더 나쁘게 만들 수 있다. 사전 중단 기준을 지키고 새 대규모 실험은 시작하지 않았다.
그 반복이 성공하더라도 continuum bound의 이론 문제는 자동 해결되지 않는다.

### 4.4 재현 감사

- source/config/기존 reference anchor를 실행 전에 봉인했다. source ZIP SHA256은
  `0d5d7c2b948e73e30e8e0b130ffff1e13a51bb2b2fd4b28c2f8481778f4e9afb`다.
- 새로운44개 seed record와 범위를 명시한 과거 protocol 충돌 검사를 기록했다.
  서로 다른 seed가 PRNG의 수학적 독립성 증명이라는 주장은 하지 않는다.
- 32whole/4preflight의 seed·작업량·stage·dimension·저장 결과 binding 및
  full-normalizer 복원 항등식을 검사했다.
- cell×arm의 rep0 네 whole를 재생했다. wall time을 제외한 모든 기록이 일치했다.
  재생 추가 비용은 joint64,000 / physical128,000 / time-step393,216,000,
  wall99.74초다. 실험과 재생 총 physical forward는1,155,712다.
- 통계 판정과 continuum 계산을 다시 계산했고, 봉인한1,162개 파일의 현재 byte
  hash가 바뀌지 않았음을 확인했다.

결과 SHA256은 `3bbbafe35cc6bc96faf54a8caa6d79eb2ff8206c2ef8d80b4ca97281ce6a808d`다.
[결과](../../results/post_audit/p2_direct_closure_v1/session_20261027/result.json)와
[감사](../../results/post_audit/p2_direct_closure_v1/session_20261027/audit.json)를 보존했다.
재생은 동일 알고리즘의 소프트웨어 감사이며 새 독립 confirmation이 아니다.

## 5. P2 전체 판정과 지금 남은 본질적 과제

**P2는 전체 해결되지 않았다.** 구현·고정 실험·재현 감사의 완료와 연구 가설의
성공을 구분한다. 최종 decision은
`P2_UNRESOLVED_THEORETICAL_REMAINDER_AND_ANY_FAILED_EMPIRICAL_GATES`다.

| 의무 | 현재 상태 | 다음에 필요한 증거 |
|---|---|---|
| 기존 N4096 IID precision | 기존 관측 조건 충족 | tail coverage의 독립 상호 검증은 별도 |
| 다른 geometry의 fine reference | 이번 네 경우 미달 | 새로운 독립 reference의 정밀도·안정성 + 조건부 equivalence |
| signed N4096/N2048 difference | 이번 개발 gate 통과 | qualified reference 및 새 confirmation에서 재검증 |
| 연속시간 relative remainder | 미해결 | 현재 target/scheme에 유효하고 실제로 작은 유한-N 상계 |
| all-q M2/bank/P1 | 이번에 측정하지 않음 | 별도 독립 위험 기준·bank 자격 |
| production/대규모 학습 | 승인하지 않음 | 필요한 모든 조건의 실제 충족 |

### 5.1 채택/중단을 반영한 순서

1. **cyclic 후보의 단순 증량·확장은 중단한다.** 단순 time-slot 재배열은 latent
   magnitude 집합을 보존한다. 이번 결과로 성능 개선이 확인되지 않았고, critical
   rare magnitude 및 여러 시간의 상호작용을 새로 탐색하는 보장도 없다.
2. **continuum proof를 별도 핵심 과제로 둔다.** 전체 경로의 최악 CDF derivative와
   inverse moments를 곱하는 현재 bound에서 rare-event 크기가 사라진다.
   개선하려면 tail에 가중된 미분/오차를 정당하게 제어하고, localization 밖의
   remainder도 같은 상대 예산으로 지배해야 한다. 관측 tail quantile을 확정 상계로
   쓰거나 적은 adjacent level에서 무한 remainder를 extrapolate하지 않는다.
   Randomized multilevel debiasing도 원문과 single-term 비용/second-moment 조건을
   재검토했다. 현재 coupling에서 finite variance와 finite expected work를 동시에
   보장하는 decay 증명은 확보하지 못했다. truncated randomization을 continuous
   unbiased estimator라고 구현하지 않는다. 자세한 가정은 이론 검토8절에 있다.
   다음에 실제로 상계를 증명해야 할 양도 `C_I(N), C_J(N)` 두 tail-weighted
   미분 모멘트로 구체화했다(이론 검토9절). chain rule/모멘트 축약 자체는 성립하지만,
   두 상수가 충분히 작다는 모형별 인증이 남아 있다. 따라서 이 식을 새 certificate나
   새로운 수학 원리의 발견으로 과장하지 않는다.
3. **다른 geometry의 reference 재설계는 위 proof와 혼동하지 않는다.** 새 방법은
   독립 bank/새 seed/명시적 density 또는 올바른 full normalizer를 유지해야 한다.
   terminal mixing만 개선하고 과거 normalizer의 오류가 복구됐다고 주장하지 않는다.
   pilot을 성공할 때까지 반복해 좋은 seed만 남기는 방식도 사용하지 않는다.
4. **proof/precision 조건을 충족한 뒤에만** 새 confirmation allocation을 봉인하고
   all-q risk와 total-work 비교로 이동한다. N32에서 모델을 키우거나 GPU를 빌리는
   것은 이 두 의무를 대신하지 않는다.

정확한 finite-grid estimator 연구로 논문 범위를 좁히는 선택은 가능하지만,
그것은 원래 P2를 해결한 것과 다르다. 목표 변경은 사용자와 별도로 합의해야 하며,
이번 실행에서는 범위나 통과 기준을 바꾸지 않았다.

## 6. 소프트웨어 최종 검증

검사 범위 안에서 새 수치/재현 오류는 발견하지 못했다. 검사의 통과가 모든 수학적
오류의 부재, rare-tail coverage 또는 P2 자격을 증명하는 것은 아니다.

- 전체 `python -m pytest -q --cov=src --cov-report=term-missing`:
  **1,960 passed / 424.91초 / 관측 process exit0**.
- 전체 source statement coverage85%(23,427 statements,3,461 missed).
  새 cyclic core96%, continuum-bound core94%. branch coverage나100% 보증은 아니다.
- CI 범위 Ruff 통과, `mypy src main.py train_driftnet.py`226files 통과,
  새 runner의 별도 Mypy1file 통과. 로컬 Windows/Python3.10.11 결과이며 원격 CI가 아니다.
- 새 단위·실행 계약 테스트21개를 포함한다. analytic normalizer/상세균형/
  constant-family/작업량/미자격 equivalence 차단/P2 허위 승인 방지를 점검한다.
- tail-weighted chain-rule 보조 oracle: 두 threshold regime 각각512개 endpoint
  미분 검사, 4개 endpoint 적분 검사. 비극단 regime 최대 미분 차이8.88e-16,
  40,001점 quadrature의 최대 endpoint 차이1.65e-11. 모형별 모멘트 상계 검사가 아니다.
- 이전 단계까지5개 frozen source archive의5,755개 파일 byte 대조가 통과했다.
  새 JSON40개를 strict parse했고, 그 시점의179개 local document link가 존재했다.
  최종 receipt·문서 링크 추가 후에는 최종 무결성 로그에서 다시 검사한다.
- `git diff --check` 통과. HEAD/origin-main은 기준59f2809에서 바꾸지 않았다.

### 6.1 환경 제약도 숨기지 않는다

실행은 CPUfloat64/8threads였으며 runtime snapshot에 PyTorch
`2.12.0.dev20260408+cu128` 등 정확한 버전을 기록했다. 개발 build이고,
`deterministic_algorithms=False`였으므로 같은 기계에서 네 replay가 일치한 것을
모든 OS/하드웨어/버전의 bitwise 재현 보장으로 확대하지 않는다.

공유 전역 Python에 대해 별도 실행한 `pip check`는 **exit1**이었다.
기존 comfy-cli/click 및 gradio·gradio-client/aiofiles·websockets 호환성 충돌4건이다.
전체 회귀 시작 시 기존 requests/chardet 관련 dependency warning도 기록됐다.
이는 이번에 도입한 수치 코드의 실패나 높은 RSE의 입증된 원인이 아니다.
연구와 무관한 패키지를 임의로 업·다운그레이드하지 않았다. 따라서 환경 전체가
깨끗하다고 주장하지 않으며, 다음 생산 재현 단계에서는 격리 환경에서 dependency/CI를
별도로 다시 검증해야 한다. 초기 보조 oracle의 scalar detach 경고도 보존했고,
확장된 두-regime oracle에서는 detach 후 scalar 평가로 경고 없이 재검사했다.

### 6.2 보존 증거

- [최종 검증 receipt](../../results/post_audit/p2_direct_closure_v1/session_20261027/verification_receipt.json)
- [전체 회귀 stdout](../../results/post_audit/p2_direct_closure_v1/session_20261027/pytest.stdout.txt)
- [실행 stdout](../../results/post_audit/p2_direct_closure_v1/session_20261027/run.stdout.txt)
- [재생 감사 stdout](../../results/post_audit/p2_direct_closure_v1/session_20261027/audit.stdout.txt)
- [정적 검사 stdout](../../results/post_audit/p2_direct_closure_v1/session_20261027/static.stdout.txt)
- [환경 의존성 충돌 stdout](../../results/post_audit/p2_direct_closure_v1/session_20261027/environment.stdout.txt)
- [보조 미분·적분 oracle](../../results/post_audit/p2_direct_closure_v1/session_20261027/theory_chainrule.stdout.txt)
- [최종 무결성 stdout](../../results/post_audit/p2_direct_closure_v1/session_20261027/final_integrity.stdout.txt)

커밋·푸시는 수행하지 않았다. 기존 uncommitted 연구 결과를 보존했다.
