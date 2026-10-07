# Neural Path Integral 연구 현황 재검토 및 방향 재설정 보고서

작성일: 2026-09-27

검토 기준 소스: `da91604` — 검토 시작 시 작업 트리는 clean

대상: V16 및 `v16_hybrid_routing_v5`, 관련 V14/V15 비교기·이론·실험 기록

목적: 논문 가능성을 낙관적으로 재확인하는 것이 아니라, 지금의 증거로 무엇을 주장할 수 있는지와 다음 연구 투자를 결정한다.

## 0. 결론부터

**연구를 버릴 이유는 없다. 다만 현재 V16을 계속 복잡하게 만들어 “일부 셀의 성능 통과”를 늘리는 방향은 중단하는 편이 좋다.**

현재 가장 가치 있는 자산은 새로운 금융모형이나 대형 신경망이 아니라, **Gaussian Volterra 경로의 조건부 적분, 계산 가능한 proposal 밀도, 보정된 중요도표본추출(IS), 실패 실험까지 보존한 검증 기반**이다. 이를 유지하면서 연구 질문을 다음과 같이 좁힐 것을 권한다.

> 조건부 적분 후에도 남는 희귀사건 분산은 어떤 경로 방향에서 발생하는가? 그 방향을 식별하고 proposal을 개선하는 비용까지 포함했을 때, 기존의 강한 방법보다 실제로 더 싸고 신뢰할 수 있는 추정이 가능한가?

우선순위는 다음과 같다.

1. **비교기·감사·이론 상태표를 바로잡아 증거의 신뢰도를 회복한다.**
2. **성능 실패를 subspace 부족 / 목표함수 불일치 / mode 탐색 실패 / 계산비용 문제로 분해한다.**
3. **원인에 맞는 최소한의 구조를 구현하고, 미사용 task·독립 학습 반복·동일 실제 비용으로 검증한다.**
4. **그 결과를 설명하는 Volterra 특화 정리 또는 정량적 오차 분석을 주 기여로 만든다.**
5. 신경망 amortization, continuum 복잡도, 비금융 일반화는 앞 단계의 결과에 따라 확장한다.

현재 상태는 **박사 연구의 상당한 실험·구현 자산을 갖춘 연구 프로토타입**이다. 그러나 **완성된 박사학위 수준의 독창적 기여가 검증되었다거나, 최상위 저널 제출 준비가 끝났다고 평가할 근거는 부족하다.** 반대로 이번 검토가 게재 불가능성을 증명한 것도 아니다. 핵심 기여를 정립하는 단계가 남았다.

이번 검토에서 기존 해석을 특히 수정해야 할 사항:

- 기존 성능비는 재계산으로 확인됐다. 그러나 실제 시간의 배수나 학습 안정성의 보장은 아니다.
- 최종 7개 셀 중 4개는 work-proxy 점추정 우위, 3개는 우위가 없는 fallback이다.
- 32개 evaluation cluster는 **하나의 학습된 proposal**을 평가한 반복이지, 32번 독립적으로 학습한 결과가 아니다.
- 저장된 신경망 실험은 함수평가 횟수를 줄였지만, 기록된 총 보정시간은 세 seed 모두 cold solver보다 길었다.
- CEM 비교기의 학습은 표준 rare-event CE의 가중 목적함수와 다르고, 최종 평가에서도 동일한 조건부 적분을 사용하지 않는다.
- 자동 감사의 의미 검증과 양측 empirical-Bernstein 주장에 수정할 부분이 있다.
- 일부 연속시간 정리는 증명 연결고리가 충분히 문서화되지 않았다. `proved`라는 YAML 값이 이 공백을 메우지 않는다.

## 1. 검토 범위와 증거의 수준

### 1.1 실제로 한 일

최종 정책, 조건부 payoff, CM/tempered/V14 transport, CEM 비교기, confirmation 집계, risk selector, theorem ledger 및 핵심 증명 문서를 읽었다. 저장된 canonical 1개와 joint-regime 6개 셀의 정확도 z 및 성능비를 재계산했다. 별도의 in-memory 입력으로 최종 감사의 의미 검증도 점검했다. 관련 단위 테스트를 선택 실행했다.

선행연구는 저자 논문, arXiv 원문, 학술지 원문·공식 소개를 중심으로 대조했다. 이는 **목표형 문헌 검토**이며, 모든 최신 문헌을 망라한 체계적 문헌고찰이나 외부 동료심사를 대체하지 않는다.

재현 자료:

- [수치 재계산 및 감사 probe 결과](V16_REORIENTATION_EVIDENCE_WITH_OPERATOR_2026-09-27.json)
- [재현용 읽기 전용 진단 스크립트](reproduce_v16_review.py)
- [기존 최종 정책 감사](../audits/G11_V16_FINAL_IMPLEMENTATION_AND_CLAIM_AUDIT_2026-08-12.md)
- [기존 모델 설명](../G11_V16_CURRENT_MODEL_GUIDE_KO.md)

이 보고서는 **새 모델을 학습하거나 전체 실험을 새로 재현한 결과가 아니다.** 기존 수치의 산술 검증, 코드·증명 검토, 문헌 대조를 수행한 것이다. 기존 논문 주장에 대한 해석을 갱신하되, 역사적 결과와 theorem ledger를 덮어쓰지 않았다. 모델 코드 수정·커밋·푸시는 하지 않았다.

### 1.2 앞으로 구분해야 하는 다섯 가지

| 수준 | 의미 | 현재 해석 |
|---|---|---|
| 구현됨 | 코드가 존재함 | 상당 부분 충족 |
| 테스트됨 | 특정 입력에서 정해진 검사를 통과함 | 다수 충족, 검사의 누락 가능 |
| 이론적으로 유효함 | 명시된 가정에서 논리가 성립함 | 유한격자 핵심 항등식은 강함; 일부 확장 재검토 |
| 실험적으로 우수함 | 공정한 비교와 불확실성 평가에서 개선됨 | 제한된 셀의 유망한 기록, 최종 입증 미완 |
| 새로운 학술 기여임 | 선행연구로 환원되지 않는 중요한 내용 | 아직 중심 명제를 확정하지 못함 |

이 다섯 항목을 하나의 `passed`로 합치면 안 된다.

## 2. 지금 우리 모델은 정확히 무엇인가

### 2.1 금융모형과 계산방법을 분리하자

금융모형은 기존 rough Bergomi 계열이다. 우리가 만드는 것은 그 모형 아래의 희귀 확률 등을 계산하는 **추정 알고리즘**이다. 새로운 시장 동역학을 제안하거나 현실 시장을 더 정확히 예측했다는 증거는 아니다.

유한격자에서 원래 Gaussian 난수는 `3N`차원이다. Volterra driver를 나타내는 local 좌표 `2N`개와 독립 가격 driver `N`개가 있다. local 좌표 두 개는 동일한 Brownian driver의 서로 다른 셀 내 관측을 나타내며, 서로 다른 두 물리적 변동성 driver를 뜻하지 않는다.

독립 가격 driver 전체를 해석적으로 적분하면, terminal event의 0/1 indicator가 local 경로에 따른 조건부 확률 `g_N(z)`로 바뀐다. 평가된 주 실험에서는 `N=32`이므로 남는 난수 차원은 64이다.

```text
기존 rough Bergomi의 유한격자 법칙
  → 독립 가격 잡음의 조건부 적분
  → 남은 Volterra 경로에서 사건에 중요한 경로를 더 자주 생성
  → 실제 sampling proposal의 전체 밀도로 보정
  → ordinary IS 평균으로 확률 추정
```

쉽게 말하면 “매우 드문 폭락을 기다리는 대신 폭락에 중요한 경로를 많이 뽑고, 많이 뽑은 만큼 정확한 가중치로 보정하는 계산기”이다.

### 2.2 수학적으로 유지할 핵심

참조 밀도를 `φ_d`, proposal을 `q`, 목표를 `μ_N = E_P[g_N(Z)]`라 하면:

\[
\widehat\mu_N=\frac1M\sum_{i=1}^{M}g_N(Z_i)\frac{\phi_d(Z_i)}{q(Z_i)},\qquad Z_i\sim q.
\]

학습자료를 고정한 뒤 새 표본을 사용하고 support·적분가능성 조건을 지키면 `E[μ̂_N | training] = μ_N`이다. `q ≥ δφ_d`이면 `φ_d/q ≤ 1/δ`이며, 여러 proposal을 섞을 때는 **선택된 component만이 아니라 전체 mixture 밀도**를 분모에 사용한다.

이 구조는 좋다. 다만 “exact”는 조건부 공식·밀도비가 알려져 있다는 뜻이다. 연속시간 대상과 유한격자 대상의 차이, 부동소수점 오차, 희귀 꼬리의 수치 underflow까지 없다는 뜻은 아니다.

### 2.3 실제 V5 구성

V5는 하나의 새 신경망이 아니라 여러 정확한 proposal을 task 파라미터로 선택하는 정책이다.

- **Tempered target:** SMC로 조건부 target 쪽 입자를 모은 뒤 선택한 basis 좌표에서 Gaussian mixture를 적합한다.
- **CM action:** 희귀 경로의 action 최소화로 초기 proposal을 만들고 적응시킨다.
- **V14 residual:** 더 완만한 fractional potential 기반 SMC로 이동 성분을 학습한다.
- **Hybrid:** V14와 tempered proposal을 알려진 질량으로 결합한다.
- **Fallback:** joint extreme에서 V14로 돌아가며 성능 우위 주장을 하지 않는다.

중요한 코드상 구분: `tempered_target_transport.py`는 full local space에서 SMC를 하지만, **`basis.project(smc.final_particles)` 이후 clustering·평균·공분산 적합을 수행한다.** full-target 학습과 full-dimensional adaptive transport는 같은 말이 아니다. 현재 대표 rough route의 학습 basis는 16 drift + 8 bridge 방향이다. 별도 safety 성분은 더 넓은 공간을 덮는다.

주요 deep-tail 결과는 신경망이 핵심인 실험이 아니다. 신경망은 별도의 action 초기화 실험에 존재한다. 따라서 현재 성과 전체를 “새로운 neural operator의 승리”로 설명하면 안 된다.

## 3. 현재 성과: 확인된 수치와 한계

### 3.1 최종 성능표

아래 비율은 `가장 강한 자격 통과 비교기의 WNV / 후보의 WNV`이다. 1보다 크면 후보가 해당 **work proxy** 기준으로 유리하다. 학습비용을 포함하되 primary query count는 100이다. 모두 하나의 frozen proposal을 32 cluster × 65,536 = 2,097,152 표본으로 평가한 기록이다.

| 셀 | 후보 확률 추정 | WNV 비율 | 후보 robust RSE | 참조 RSE | 재해석 |
|---|---:|---:|---:|---:|---|
| H=.05, K=1 canonical | 4.838e-8 | 5.978 | 2.03% | 2.71% | 유망한 셀별 점추정 우위 |
| H=.05, K=2 | 1.976e-7 | 1.280 | 2.16% | 4.29% | 우위 폭 작음, 반복 검증 중요 |
| H=.05, K=.5 | 1.391e-8 | 5.023 | 1.82% | 3.74% | 유망한 셀별 점추정 우위 |
| H=.12, K=1 | 1.273e-6 | 1.815 | 0.67% | 4.82% | 조건부 성능 증거 |
| H=.05, η=2, K=1 | 4.370e-6 | .471 | 4.53% | 3.70% | fallback; 비교기보다 불리 |
| H=.05, ρ=-.9, K=1 | 9.222e-8 | .605 | 4.21% | 6.98% | fallback; 비교기보다 불리 |
| H=.12, η=2, ρ=-.9, K=1 | 3.612e-5 | .639 | 0.71% | 3.58% | fallback; 비교기보다 불리 |

기본값은 `S0=100, T=1, N=32, η=1.5, ρ=-.7, ξ=.04`이고 표에 표시된 값만 바뀐다. 여기서 K는 절대 threshold이다. `K=.5,1,2`는 **초기 가격의 0.5%, 1%, 2%까지 떨어지는 terminal event**이며, 일반적인 ATM/OTM 옵션 범위를 뜻하지 않는다.

수치 재계산에서 저장된 primary WNV 비율과 accuracy z는 모두 일치했다. 따라서 이 비율들을 계산 오류로 폐기할 이유는 발견하지 못했다. 하지만 “최대 5.98배 실제로 빠르다”는 결론은 별개이다.

### 3.2 100-query 가정의 영향

동일한 저장된 proposal과 분산을 두고 amortization 수식만 재계산하면:

| 셀 | 1 query | 10 queries | 100 queries |
|---|---:|---:|---:|
| canonical K1 | 5.327 | 5.911 | 5.978 |
| rough K2 | 1.164 | 1.268 | 1.280 |
| rough K.5 | 4.480 | 4.968 | 5.023 |
| regular K1 | 1.822 | 1.816 | 1.815 |

현재 점추정 우위가 100-query 가정만으로 만들어졌다고 볼 수는 없다. 반면 이 계산은 **동일 task의 반복 평가비용 상각**이지, 서로 다른 100개 parameter task에 학습 없이 일반화했다는 실험이 아니다. 실무의 parameter sweep·calibration workload와 구분해야 한다.

### 3.3 신경망 초기화: 함수평가 감소와 실행시간 개선은 다르다

operator 실험은 teacher 48개, 학습 seed 3개, 각 seed에서 동일한 holdout task 12개를 사용했다. teacher의 strike ratio는 .30–.60이고, 위 deep-tail 주 실험의 .005–.02와 다르다.

| 학습 seed | 함수평가 횟수 기준 중앙 speedup | 기록된 총 cold time / correction time | correction이 빨랐던 task |
|---|---:|---:|---:|
| 1618202 | 1.900 | .818 | 1/12 |
| 1618203 | 1.900 | .676 | 0/12 |
| 1618204 | 1.778 | .830 | 4/12 |

이 시간비에는 teacher 생성·네트워크 학습비용도 포함하지 않았다. 그 상태에서도 총 correction 시간이 더 길게 기록됐다. 다만 짧은 단일 timing이고 cold timing을 seed 간 재사용했으므로, 이것만으로 신경망이 항상 느리다고 확정할 수도 없다.

정확한 결론은 **좋은 초기값을 학습했다는 증거는 있지만, end-to-end 가속은 입증되지 않았고 현재 timing은 오히려 경고 신호**라는 것이다. 순서 무작위화·warm-up·동일 thread 수·반복 timing으로 재측정해야 한다.

### 3.4 실패 기록에서 얻는 정보

같은 rough K2 hybrid 계열은 별도 confirmation에서 3.556, 최종 V5에서 1.280을 보였다. η=2, ρ=-.9 hybrid confirmation은 .097로 실패했다. V3/V4의 joint regimes도 성공·실패가 섞여 있다.

이 차이는 제안 학습 seed, proposal 구성, 평가 잡음 등이 섞인 결과이다. 현재 기록만으로 원인을 하나로 확정할 수 없다. 그러나 **좋은 seed 하나의 승리를 구조적 성능 보장으로 해석할 수 없다는 점은 분명하다.** 실패 자료를 보존한 것은 연구의 강점이다.

## 4. 반드시 고쳐야 할 기술·통계 문제

### 4.1 P0 — 최종 감사는 독립적인 과학 검증기가 아니다

`experiments/g11_v16_final_policy_audit.py`는 artifact SHA-256을 확인하는 좋은 무결성 장치다. 그러나 핵심 의미 검증은 기존 `passed`, `claim_role`, 저장된 비율과 theorem status 문자열에 의존한다.

이번 재현 probe에서, **무결성 로딩 이후 의미 검증 단계만 분리하여** 다음 두 입력을 넣었다.

1. OOD record의 accuracy z·robust RSE·normalization z를 모두 1,000,000으로 변경하고 `passed=true`는 유지.
2. dominance record의 성능비를 NaN으로 변경하고 `passed=true`는 유지.

두 경우 모두 `passed=true`, `internal_failures=[]`를 반환했다. 원본 파일은 수정하지 않았으며, 이것은 SHA-256 우회가 아니라 **새로 binding된 내부 불일치 결과를 판별하지 못하는 문제**를 보여준다.

수정 요구:

- 필수 수치의 유한성, 부호, sample count, 셀 유일성, 누락·중복을 검사한다.
- 원시 sufficient statistics와 config에서 mean/SE/z/RSE/비용비를 재계산한다.
- 최상위 artifact뿐 아니라 그 안의 config·reference·baseline binding도 재검증한다.
- 정확도·성능·독창성·증명 상태를 서로 다른 판정으로 출력한다.
- 외부 reproduction은 “파일 존재”가 아니라 schema·내용·provenance·재현 대상 일치를 확인한다.
- NaN, stale-pass, 중복 셀, 바뀐 threshold, 잘못된 reference, malformed reproduction에 대한 실패 테스트를 추가한다.

### 4.2 P0 — CEM 비교기의 fidelity와 조건부 적분 통제

현재 `baselines/cem.py`는 `z_i ~ N(m,I)`를 생성한 후 elite 표본의 **무가중 평균**을 새 mean으로 사용한다. 참조 Gaussian의 희귀사건 조건부 분포를 KL 기준으로 맞추는 표준 CE-IS 갱신이라면, 현재 proposal로 뽑은 표본에는 `φ(z_i)/q_m(z_i)` 보정이 들어가야 한다.

\[
m_{new}=\frac{\sum_i 1_{A_\ell}(z_i)w_i z_i}{\sum_i1_{A_\ell}(z_i)w_i},\qquad w_i=\phi(z_i)/q_m(z_i).
\]

예를 들어 1차원에서 참조 `N(0,1)`의 `Z≥2` 조건부 평균과, sampling law `N(1,1)`의 같은 조건부 평균은 다르다. 무가중 elite 갱신은 후자를 학습한다. 이는 CEM형 최적화 휴리스틱으로 사용할 수는 있으나, 문헌의 CE rare-event 기준선을 충실히 구현했다고 말하려면 수정 또는 명확한 별도 명칭이 필요하다. 가중 CE 목적식은 [Uribe et al., §2.3–2.4](https://arxiv.org/pdf/2006.05496)에 직접 나타난다.

**이 문제는 최종 ordinary IS의 불편성을 자동으로 깨뜨리지는 않는다.** proposal이 덜 좋게 학습되는 문제와 밀도 보정이 틀리는 문제는 별개다.

또한 현재 비교 경로는 CEM/LD에 `evaluate_latent_is_units`를 사용하고, V16은 독립 가격 driver 전체를 적분한다. 따라서 “V16 대 CEM” 차이에는 **Rao–Blackwellization의 이득과 proposal의 이득이 함께 들어간다.**

반드시 추가할 통제군:

- 원래 hard-event CEM과 올바르게 가중한 CE-IS.
- **동일 `g_N`를 사용하는 2N차원 conditional CE**, mean-only와 covariance/mixture 버전.
- 동일 조건부 payoff를 사용하는 natural MC, RQMC, LD/FIS, V14.
- 동일 train 예산·tuning 예산·최종 평가비용에서 비교.

V14 비교가 있다는 점은 장점이나, 그것만으로 다른 강한 conditional 비교기를 대체하지 못한다.

### 4.3 P0 — 비교기 자격 판정이 두 단계에서 다르다

baseline confirmation은 robust SE 기반 accuracy와 RSE 조건을 함께 사용한다. 후보 runner의 `_qualify_comparator`는 pooled SE 기반 z와 양의 분산만 확인하며 RSE gate를 적용하지 않는다.

실제로 rough K2에서 baseline이 제외했던 conditional MC, LD, smoothing RQMC가 후보 평가에서는 다시 포함된다. smoothing RQMC는 매우 큰 상대오차를 갖는데도 z가 작아서 통과한다. **SE가 크면 잘못된 평균도 “통계적으로 유의하게 다르지 않다”고 나올 수 있다.**

최종 주요 셀의 가장 강한 비교기는 대부분 V14로 남으므로, 이 불일치만으로 모든 보고 비율이 무효라고 단정하지 않는다. 그러나 기준은 하나로 통일해야 한다. 성능이 불확실한 비교기를 결과를 본 뒤 빼서 우위를 선언해서도 안 된다. 비교기 실패를 별도 보고하고, 필요한 예산에서 다시 측정한다.

### 4.4 P1 — 비용 proxy는 실제 계산량과 같지 않다

V16 evaluation의 work는 대략 `samples × (dimension + N + components×dimension + sum(rank) + 1)`이다. 이 식은 basis projection·Gaussian density 행렬곱의 `dimension×rank`, simulator backend 비용, 추가 payoff 계산 등 실제 연산을 충실히 나타내지 않는다. tempered fitting의 proxy도 clustering 반복·projection·eigendecomposition 비용을 충분히 구분하지 않는다.

하위 함수는 wall/CPU 시간을 측정하지만 최종 candidate summary는 주로 proxy를 보존·판정한다. 따라서 지금의 WNV는 정의된 가상 단위에서의 지표이지 hardware speedup이 아니다.

개선은 두 축이다: (a) 실제 wall time을 주 지표로 저장·반복 측정, (b) FLOP/호출 수/메모리 traffic proxy를 보조 지표로 명확히 정의. GPU 전환 시에도 모든 비교기에 동일한 backend 최적화 기회를 준다.

### 4.5 P1 — reference, OOD, 반복의 독립성

현재 reference는 독립 seed·다른 추정 절차의 SMC라는 의미가 강하다. 외부 연구자가 별도 코드로 재현했다는 뜻은 아니다. 같은 simulator/payoff의 체계적 오류를 공유할 수 있다.

reference RSE는 표에서 약 2.7–7.0%이다. 후보 RSE가 .7%라고 해서 참조 대비 실제 오차가 .7% 이하로 검증된 것은 아니다. η=2, ρ=-.9 셀의 후보 평균은 reference보다 약 9.27% 높지만, z=2.53으로 기존 z≤4 gate를 통과한다. 이것은 편향의 확정 증거도, 높은 정확도의 확정 증거도 아니다.

또한 여러 버전이 동일한 joint-regime 셀을 보고 설계됐다. 새 seed confirmation은 가치가 있지만, 그 셀을 다시 **미사용 parameter OOD**라고 부를 수 없다. 기존 셀은 개발/회귀 검증군으로 격하하고, 새로운 task holdout을 따로 만들어야 한다.

독립 단위는 명확히 구분한다: task, proposal 학습 seed, frozen proposal의 evaluation cluster, RQMC scramble, SMC replicate. 같은 학습 proposal의 평가 표본을 늘려도 학습 실패확률은 측정되지 않는다.

## 5. 이론 검토: 유지할 것과 재증명할 것

### 5.1 현재 핵심 항등식의 평가

조건부 Gaussian terminal representation, frozen proposal의 ordinary-IS 항등식, natural defensive mass에 의한 밀도비 상한, mixture domination에 의한 second-moment 상한은 적절한 출발점이다. 하지만 대부분 알려진 일반 원리이며 **정확성의 기반과 새로운 논문 정리는 다르다.**

학습 proposal이 terminal event 전체를 이용하더라도 유한차원 알려진 Gaussian density로 보정하는 offline IS에서는 그 자체가 adaptedness 위반은 아니다. 이를 연속시간 Girsanov drift control이라고 다시 표현할 때는 별도의 adaptedness·절대연속성 조건이 필요하다. 유한격자 Gaussian transport와 실시간 인과적 제어를 혼동하지 않는다.

### 5.2 확인된 수정 사항 — T16-12B의 양측 신뢰구간

현재 문서와 selector 코드는

`log_factor = log(2 J / γ)`

를 사용하면서 모든 candidate 위험이 empirical mean의 **양측 반경** 안에 있다고 주장하고, 이를 excess-risk oracle bound에 사용한다.

[Maurer–Pontil Theorem 4 / Corollary 5](https://arxiv.org/pdf/0907.3740)의 해당 식은 한쪽 편차에 대한 것이다. 이 근거로 두 방향과 J개 후보를 union bound하면 `log(4J/γ)`가 되는 것이 안전한 수정이다. 다른 더 강한 양측 정리를 사용하려면 정확한 정리와 가정을 제시해야 한다.

이는 상계 UCB 한쪽만 필요한 부분과 양측·oracle 보장을 구분하지 않은 문제다. 위험 추정량의 기대값 항등식이나 독립 선택 후 최종 IS 불편성을 무효화하지는 않는다. 최종 V5 route는 이 selector를 쓰지 않으므로 위 성능표의 직접적인 원인은 아니다. 그러나 현재 상태에서 T16-12 전체를 무조건 `proved`로 읽으면 안 된다.

### 5.3 증명 공백 — T16-9 joint mesh/noise

현재 문서는 BLP kernel의 Hilbert–Schmidt 오차 `δ_N→0`에서 Gaussian error concentration을 거쳐, martingale quadratic variation이 `O(ε δ_N²)`라고 서술하고 **모든 `N(ε)→∞` schedule**의 exponential approximation을 결론낸다.

여기에는 다음 연결고리가 필요하다.

1. 사용한 Gaussian error의 norm이 시간 L2인지 sup norm인지 명시.
2. HS 오차로 제어되는 양과 필요한 path norm의 연결. sup norm이라면 추가 covariance/entropy 제어.
3. lognormal 비선형성의 localization 사건과 그 complement의 exponential bound.
4. kernel의 평균제곱 오차가 아니라 **확률적인 quadratic variation 자체**에 대한 tail 제어.
5. BLP cell projection, left-time discretization, stochastic integral을 함께 다루는 exponential approximation.
6. 고정 N의 수렴과 joint sequence의 균일성을 구분한 정리.

현재 짧은 논증만으로 이들을 모두 확인했다고 말하기 어렵다. 이는 정리가 거짓이라는 반례가 아니라 **완전한 증명으로 받아들이기에 근거가 부족하다는 판정**이다. T16-9 및 이에 의존하는 T16-11의 무조건적인 홍보는 보류한다.

### 5.4 증명 공백 — T16-5 continuous Laplace step

trace-class covariance의 Gaussian equivalence, density 표현, determinant scale의 dominated-convergence 논증은 유망하고 비교적 명확하다. 문제는 조건부 payoff에 포함된 Itô 적분과 ε-dependent integrand에 대해 “Gaussian LDP/Laplace upper bound를 적용”하는 단계다.

CM skeleton의 약연속성과 CM sublevel의 deterministic Mills ratio만으로 원래 확률적 functional의 Laplace upper bound가 자동으로 따라오지는 않는다. `(유한 Gaussian 좌표, integrated variance, stochastic integral)`의 joint LDP 또는 적절한 extended-contraction/exponential-approximation 정리를 명시하고 적용 조건을 확인해야 한다.

연속 Volterra LDP 자체는 [Gulisashvili의 원문 Theorem 13 및 digital option 결과](https://arxiv.org/pdf/1710.10711)가 중요한 출발점이다. 기존 확률 LDP의 적용과 새로운 proposal second-moment 정리는 서로 다른 작업이다. 여기서 noise scaling 지수와 roughness H를 별도 기호로 두는 기존 구분은 유지한다.

### 5.5 논문에서 반드시 분리할 보장

| 보장 | 현재 권고 |
|---|---|
| 유한격자 ordinary IS 불편성 | 조건과 수치 구현 범위를 붙여 유지 |
| fixed-grid small-noise exponent | 전체 가정 아래의 별도 결과로 검토·유지 |
| trace-class Gaussian equivalent law | 고전 이론을 인용한 구성 결과로 유지 |
| continuous second-moment / joint mesh-noise exponent | 연결 lemma를 채우기 전에는 재검증 중으로 취급 |
| logarithmic efficiency | bounded relative error나 finite-cost 우위로 번역 금지 |
| qualitative mesh convergence | N=32의 상대 bias가 작다는 근거로 사용 금지 |
| Newton local correction theorem | L-BFGS 구현의 같은 수렴률 보장으로 사용 금지 |
| positive-safety mixture theorem | safety mass가 0인 V14 fallback 전체에 확장 금지 |

참고로 rough volatility weak-error 연구도 payoff와 scheme에 따라 가정이 제한된다. [Gassiat의 원문](https://arxiv.org/abs/2203.09298)의 rate를 lognormal rare digital에 그대로 가져와서는 안 된다.

## 6. 독창성: 이미 알려진 것과 우리의 후보 기여

아래는 우선 대조해야 할 직접적인 선행연구다. abstract 수준의 유사성과 실제 theorem 중복을 구분하고, 투고 전에는 정리·알고리즘 단위 comparison을 추가해야 한다.

| 선행연구 | 이미 다룬 영역 | 우리 주장에 주는 제약 |
|---|---|---|
| [Bayer–Ben Hammouda–Tempone, rBergomi ASGQ/QMC](https://arxiv.org/abs/1812.08533) | rough volatility의 smoothing·차원 구조·효율적 적분 | 조건부 적분과 QMC 결합 자체를 최초라 주장 불가 |
| [Uribe et al., failure-informed CE](https://arxiv.org/abs/2006.05496) | Gaussian rare event의 사건 정보 기반 차원축소와 CE | 사건에 중요한 subspace라는 발상만으로 부족 |
| [El Masri–Morio–Simatos, optimal projection](https://computo-journal.org/published-202402-elmasri-optimal/) | Gaussian IS에서 KL-optimal 저차원 covariance 방향 | low-rank Gaussian covariance 자체는 새 기여 아님 |
| [Arandjelović et al., Finance and Stochastics](https://link.springer.com/article/10.1007/s00780-024-00549-x) | Cameron–Martin 공간과 신경망을 이용한 옵션 IS | CM+NN, path-dependent option이라는 조합만으로 부족 |
| [Cui–Dolgov–Scheichl, deep tensor-train IS](https://arxiv.org/abs/2209.01941) | transport를 통한 rare-event proposal 근사 | transport라는 용어만으로 flow/TT와 차별화 불가 |
| [Llorente et al., noisy IS optimality](https://arxiv.org/abs/2201.02432) | 조건부 잡음의 분산까지 고려한 optimal proposal | 아래 second-moment 원리 자체를 신발견으로 주장 금지 |

가장 가능성이 있는 새 기여는 다음의 **결합된 결과**이다.

> Gaussian Volterra의 조건부 rare-event 문제에서, 기존 KL/평균 적합이 놓치는 second-moment 경로 방향을 정량화하고, 그 정보를 이용한 mesh-compatible adaptive transport가 비용을 포함해 개선된다는 정리와 실험.

이 문장은 아직 **연구 가설**이다. 현재 구현이 이 결과를 달성했다는 뜻이 아니다. 선행연구의 단순 적용으로 귀결될 수도 있으므로 초기에 차별성을 검토한다.

기본 항등식을 여러 번호의 정리로 나누거나 새 acronym을 만드는 것은 기여의 깊이를 늘리지 않는다. 기존 VFO·DVDN·CAPT 등의 명칭보다, 실제로 입증한 수학적 질문을 논문의 중심에 둔다.

## 7. 가장 먼저 검증할 과학적 가설: 무엇이 성능을 막는가

### 7.1 구분해야 하는 네 가지 실패 원인

| 원인 가설 | 지금 보이는 근거 | 이를 구별할 실험 | 결과에 따른 조치 |
|---|---|---|---|
| 선택한 basis가 중요한 잔여 방향을 누락 | full-space SMC 후 low-rank projection; joint regimes 실패 | 같은 충분한 particle bank에서 rank/basis만 변경, complement conditional variance 측정 | 사건 기반 방향 추가 |
| target 평균·공분산 적합과 IS second moment의 목표 불일치 | target fitting은 기하/KL 친화적, 평가는 χ²/분산 | 동일 basis·family·budget에서 KL 대 second-moment fitting | variance-aware 학습 |
| SMC가 중요한 mode를 놓침 | 학습 seed 민감성과 극단 regime의 불안정 | 독립 SMC bank 간 mode mass, tail contribution, ancestry 비교 | bridge schedule/mutation/replicate 설계 개선 |
| 작은 분산 이득보다 계산비용이 큼 | 큰 mixture·safety·projection, operator timing | 동일 proposal의 raw/conditional 및 component별 timing | 단순화·vectorization·필요 성분만 사용 |

이 가설들은 아직 원인으로 입증되지 않았다. “roughness가 크니 고차원일 것” 같은 설명만으로 구현 방향을 결정하지 않는다.

특히 `|ρ|→1`이면 조건부 가격 variance `(1-ρ²)I`가 작아져 conditional CDF가 가파르게 바뀔 수 있다. 작은 H에서는 kernel의 가까운 과거 기여가 중요해질 수 있다. 큰 η는 lognormal volatility의 경로 민감도를 키운다. 이 구조적 관찰은 진단을 설계할 이유이지, 실패 원인이 확인됐다는 증거는 아니다.

### 7.2 재설계의 기준이 되는 분산 분해

참조 Gaussian 좌표를 직교 분해하여 `Z=(U,V)`, `U⊥V`라 하자. `g=g_N`는 이미 독립 가격 driver를 적분한 payoff이다. **U의 분포만 바꾸고 V는 참조 조건부 분포로 유지하는 proposal family**를 생각한다.

\[
q(u,v)=q_U(u)\phi_V(v),\quad
r(u)=q_U(u)/\phi_U(u),\quad E_{P_U}[r]=1.
\]

\[
m_1(u)=E[g\mid U=u],\quad m_2(u)=E[g^2\mid U=u],\quad \mu=E[g]>0.
\]

ordinary IS 한 표본의 second moment는

\[
M_2(r)=E_{P_U}\left[\frac{m_2(U)}{r(U)}\right].
\]

Cauchy–Schwarz에 의해 이 family의 이상적인 최솟값과 proposal은

\[
\inf_r M_2(r)=\left(E_{P_U}\sqrt{m_2(U)}\right)^2,\qquad
r^*(u)=\frac{\sqrt{m_2(u)}}{E\sqrt{m_2(U)}}.
\]

증명은 `E√m₂ = E[√(m₂/r)√r] ≤ √M₂(r)`이며 등호 조건에서 proposal을 얻는다. 0이 되는 집합은 적분에 기여하지 않도록 정의하고, 실제 구현에는 positive defensive mass를 추가한다.

반면 완벽하게 target의 U-marginal을 학습한 경우 `r_KL=m₁/μ`이고,

\[
M_2(r_{KL})-\mu^2
=\mu\,E_{P_U}\left[\frac{\operatorname{Var}(g\mid U)}{m_1(U)}\right].
\]

즉, **target 입자의 U 방향 평균·공분산을 아무리 잘 맞춰도, 생략한 V 방향의 조건부 변동 때문에 분산이 남을 수 있다.** 또 marginal target fitting과 최종 IS variance minimization의 최적해가 일반적으로 다르다.

U의 정보가 늘어나는 nested subspace에서는 이상적 floor `E√E[g²|U]`가 증가하지 않는다는 사실을 조건부 Jensen으로 확인할 수 있다. 그러나 실제 학습 오차와 비용은 rank와 함께 커질 수 있으므로 “차원 추가는 언제나 개선”은 성립하지 않는다.

중요한 제한:

- 위 식은 알려진 IS/noisy-IS 원리와 연결되는 **진단 항등식**이다. 그 자체를 새로운 수학적 발견이라 주장하지 않는다. [Llorente et al., §4](https://arxiv.org/pdf/2201.02432)의 조건부 잡음 분산을 고려한 optimal proposal과 직접 대조해야 한다.
- V16의 learned low-rank branch 분석에는 적합하지만, full-rank safety나 다른 subspace의 V14 성분을 포함한 **전체 V5 mixture에 이 floor를 그대로 적용할 수는 없다.**
- 실제 `m₁,m₂`는 미지이다. nested pilot 추정량의 불확실성과 추가 비용이 크면 이 방향도 실패할 수 있다.
- 희귀사건에서는 `m₁`이 매우 작다. 분모를 임의 clipping해 이론을 바꾸거나 log 값의 평균을 원래 second moment로 착각해서는 안 된다.

### 7.3 여기서 진짜 새로운 결과가 되려면

다음 중 하나 이상을 달성해야 한다.

1. **Volterra 특화 rank/오차 관계:** H, η, ρ, rarity와 격자 N에 따라 residual second-moment floor가 어떻게 달라지는지 정량화한다. 단순 절대 L2 오차가 아니라 상대 분산과 연결해야 한다.
2. **계산 가능한 방향 선택법:** 제한된 pilot으로 중요한 잔여 방향을 찾아, 기존 KL/FIS·고정 cosine·PCA보다 좋은 비용-정확도 tradeoff를 보인다.
3. **mesh-compatible 안정성:** 격자를 늘려도 학습 방향이 동일 연속 경로 구조를 추적한다는 결과와 실제 scaling을 제시한다.
4. **신뢰할 수 있는 선택·확장 기준:** 새 component나 rank를 추가할지 결정할 때 학습·선택 오류를 포함한 보장을 제시한다. 무의미하게 큰 absolute bound를 rare-event 상대 보장으로 포장하지 않는다.

단순히 `sqrt(m₂)`를 학습 목표로 바꾸고 한 셀에서 좋아지는 것은 좋은 실험이지만, 최상위 저널의 중심 기여로 충분하다고 보지 않는다.

## 8. 여러 factor를 고려한 연구 방향 우선순위

여기서의 순위는 게재 확률의 수치 예측이 아니라, **현재 자산 활용도·새 지식 가능성·검증 가능성·이론 위험·실무 가치·노트북 비용**을 고려한 연구 투자 순위다.

| 순위 | 방향 | 장점 | 주요 위험 | 현재 판단 |
|---|---|---|---|---|
| 1 | 조건부 second-moment 기하 기반 최소 transport | 현재 실패를 직접 설명; 기존 코드 활용; 작은 실험으로 반증 가능 | FIS/noisy IS 선행연구와의 차별성, pilot 비용 | 주 연구로 선택 |
| 2 | rare digital의 mesh bias 및 정확도-비용 정량 분석 | 수학·실무 모두 중요한 문제; 상위 수치해석/금융 저널과 정합 | proof 난도가 높고 느린 rough convergence | 범위를 좁힌 병행 이론 트랙 |
| 3 | 알려진 강한 sampler의 파라미터 간 neural amortization | 다수 task workload에서 가치 가능 | 현재 wall-time 가속 없음; teacher 비용; 새 architecture 기여 약함 | 단일 task sampler가 정리된 뒤 |
| 4 | full nonlinear/triangular conditional transport | low-rank floor를 넘어설 표현력 | density·학습 안정성·비용·선행 flow와 중복 | rank 진단이 필요성을 보일 때만 |
| 5 | 현 V5 route/mixture를 계속 세분화 | 단기 셀별 성능 개선 가능 | 사후 tuning, 불명확한 메커니즘, 복잡도 증가 | 주 논문 방향으로 비권고 |
| 보류 | 양자역학적 요소·임의 차원 가중·새 명칭 중심 모델 | 아이디어 탐색 가능 | 현재 병목과 인과관계·검증 가능한 이득이 없음 | 현재 연구 예산 배정 안 함 |

1위와 2위를 “둘 다 완벽하게 다 증명한 뒤에야 논문”으로 묶지는 않는다. **1위가 주 알고리즘 질문, 2위는 그 주장에 필요한 오차 범위부터 해결하는 지원 트랙**으로 둔다. 2위에서 독립적으로 중요한 정리가 나오면 별도 이론 논문으로 발전시킬 수 있다.

### 8.1 원래 목표와의 정합성

사용자의 원래 목표는 “경로적분을 활용한 의미 있는 새 방법 + 논문 기여 + 실무 활용”이었다. 확률 경로공간의 적분과 측도변환을 이용하는 현재 핵심은 그 목표와 맞는다. 다만 신경망·양자 용어가 반드시 포함되어야 목표에 맞는 것은 아니다.

현재의 양의 확률 측도에 대한 Wiener 경로적분을, 복소 진폭의 Feynman 경로적분과 동일시할 수 없다. 양자 요소를 도입하려면 어떤 계산량을 줄이고 어떤 classical baseline과 어떤 자원에서 비교하는지 새로운 연구 질문이 필요하다. 현재의 benchmark·bias·novelty 문제를 해결하는 장치로 볼 근거는 없다.

차원별 가중은 **proposal의 평균·공분산·basis 선택**으로 도입하고 정확한 likelihood를 계산하면 타당한 도구다. 그러나 참조 Brownian의 분산을 임의로 바꾸고 같은 금융모형이라고 부르면 목표 모형을 바꾼다. 우리의 제안은 전자이다.

## 9. 단계별 실행안과 판정 기준

아래는 다음 작업을 위한 구체적인 계획이다. 이번 보고서 작성 중에 모델 변경을 실행한 것은 아니다. 기간은 연구자의 작업량 추정이며, 성능·증명 완료를 보장하지 않는다.

### R0. 주장·검증 기반 재정비 — 최우선

작업:

1. 현 V16을 변경하지 않는 historical snapshot으로 보존한다.
2. 감사기를 raw statistics/config 기반으로 재설계하고 adversarial 실패 테스트를 넣는다.
3. T16-12B의 one-/two-sided 구분을 수정하고 T16-5/9/11은 근거별 검토 상태로 나눈다.
4. 올바른 weighted CE와 conditional CE를 추가한다. 기존 unweighted 버전은 별도 휴리스틱 이름으로 유지한다.
5. 비교기 qualification을 한 함수로 통합하되, unresolved method를 조용히 제외하지 않는다.
6. candidate와 comparator의 wall/CPU/memory 및 학습·선택·평가 비용을 동일 schema로 보존한다.
7. `exact`, `OOD`, `confirmed`, `robust`, `external reference`의 의미를 문서와 결과 schema에 고정한다.

검증:

- 1차원 Gaussian rare event에서 CE target mean을 analytic 값과 비교한다.
- Gaussian covariance·likelihood를 독립 closed form과 비교한다.
- 동일 local 경로의 raw independent-price 평균과 conditional CDF를 비교한다.
- 상수 payoff, 자연 proposal, mixture weight 경계, 극단 log tail, 잘못된 metadata를 테스트한다.

완료 기준: 허위 수치/NaN/stale-pass가 통과하지 않고, baseline 정의와 집계 기준이 일치해야 한다. 이 단계가 끝나기 전에는 새 모델의 “우월성”을 선언하지 않는다.

### R1. 작은 진단 실험 — 구조 추가보다 원인 확인

개발용 셀은 현재 성공·실패 셀 중 4–6개를 사용한다. 이들은 최종 holdout이 아니다.

작업:

- 같은 problem·payoff·train budget에서 weighted conditional CE, V14, V16 branch를 비교한다.
- 초기에는 독립 학습 seed 5개로 실패 패턴을 찾고, 동일 frozen fit의 evaluation 반복과 분리한다.
- basis rank를 예컨대 8/16/24/48/64로 비교하되 full-rank pilot은 작은 N에서 먼저 한다.
- 하나의 충분한 SMC bank를 공유한 fit 비교와, 각자 bank를 학습한 end-to-end 비교를 분리한다.
- fixed cosine, weighted-PCA/FIS, second-moment-informed 방향을 동일 rank에서 비교한다.
- 독립 bank 간 mode mass·기여도 상위 표본·cluster 분산·complement gradient/conditional variance를 기록한다.
- safety only / learned only / defensive+learned / defensive+safety+learned를 분해한다. learned-only가 support/분산 조건을 못 지키면 진단 용도로만 취급한다.

완료 기준: 성능을 가장 제한하는 요소를 실험적으로 구별해야 한다. rank를 늘려도 같은 bank에서 분산이 안 줄면 “표현력이 부족해서 실패했다”는 설명을 채택하지 않는다. 분산은 줄지만 시간이 더 늘면 차원 확대가 아니라 비용 절감 또는 중단을 선택한다.

### R2. 최소 구조의 candidate 하나로 좁히기

R1 결과에 따라 하나만 선택한다.

- **subspace가 문제:** pilot의 tail second moment에 민감한 방향을 기존 BLP-compatible basis에 추가.
- **fitting objective가 문제:** 같은 Gaussian family에서 KL fitting 대신/함께 second-moment 목적 사용.
- **mode 누락이 문제:** SMC replicate/mutation을 개선하고 미발견 mode 위험을 비교. 무조건 component 수만 늘리지 않음.
- **비용이 문제:** 전체 simulator·payoff·density evaluation의 중복을 제거하고 rank/mixture 단순화.

필수 불변조건:

- 최종 분모는 정규화 상수가 알려진 실제 sampling density이다.
- nested pilot으로 얻은 미지 `m₂`를 density 분모에 직접 끼워 넣지 않는다.
- adaptive fitting/selection은 final evaluation 전에 끝낸다.
- 기하를 바꾸더라도 참조 Gaussian 법칙과 좌표 Jacobian을 유지한다.
- nonnegative payoff 및 positive defensive mass 조건을 유지한다.
- variance reduction이 reference와 맞지 않는 평균, underflow, 누락 component 때문에 생긴 것은 아닌지 검사한다.

완료 기준: 복잡한 V5보다 설명이 간단한 frozen candidate를 만든다. 같은 학습 예산에서 conditional CE/FIS/V14보다 개선이 없으면 neural/flow를 덧붙이지 말고 R1 원인 판단을 재검토한다.

### R3. 과학적으로 방어 가능한 confirmation

설계:

1. 기존 개발 셀과 겹치지 않는 parameter task를 사전에 고정한다. ID interpolation, 경계, joint OOD를 분리한다.
2. 기존처럼 threshold 자체만 바꾸는 군 외에, 독립 pilot으로 비슷한 rarity 구간을 맞춘 군도 마련한다. difficulty와 parameter OOD를 구분한다.
3. 주 candidate·비교기·예산·metric·실패 처리·확장 규칙을 최종 결과 전에 고정한다.
4. 최소 10개 독립 학습 반복의 고정 예산 confirmation을 출발점으로 둔다. 20–30개가 필요할지는 pilot precision으로 사전에 정하고, 좋은 결과가 나올 때까지 반복하지 않는다.
5. 각 fit 아래 evaluation uncertainty와 fit 간 변동을 계층적으로 보고한다. median뿐 아니라 하위 성능 quantile·실패율을 포함한다.
6. ratio의 confidence interval, 평균 차이와 사전에 정한 equivalence margin, 실제 wall time 대비 RMSE를 보고한다. heavy tail에서 bootstrap/정규근사가 coverage를 보장한다고 가정하지 않는다.
7. baseline이 불확실하면 결과를 unresolved로 남긴다. “못 이긴 방법만 제외”하는 qualification을 금지한다.

주 지표:

- 고정 wall-time budget의 squared error/RMSE와 uncertainty.
- 고정 relative tolerance를 달성하는 총 wall time: fitting + selection + inference.
- 단일 query 및 **실제로 서로 다른 task들**의 batch workload. amortization break-even 포함.
- 각 method의 성공률, tail contribution concentration, 최대 memory, reference uncertainty.

내부 의사결정의 예시: 사전에 지정한 primary workload에서 총시간 speedup의 95% 하한이 1을 넘고, 실용적으로 충분한 개선 폭을 보일 것. 1.5배 같은 실용 기준은 연구 목표에 맞춰 미리 정하되 저널의 공식 합격선처럼 취급하지 않는다. 어려운 셀의 실패를 평균으로 숨기지 않는다.

### R4. 연속 대상과 실무 의미 연결

유한격자 확률 `μ_N`와 연속 확률 `μ`는 다르다. 불편한 유한격자 estimator라도

\[
E[(\widehat\mu_N-\mu)^2]
=\operatorname{Var}(\widehat\mu_N)+(\mu_N-\mu)^2
\]

가 된다. 수치 오차는 별도이다. stochastic SE만 줄여도 mesh bias가 크면 목적을 달성하지 못한다.

작업:

- N=32/64/128/256 등에서 coupled refinement를 수행하되, 계산비용에 따라 단계적으로 확장한다.
- BLP의 서로 다른 격자 좌표를 단순히 재사용하지 말고 공통 Brownian 관측에 맞는 coupling을 검증한다.
- 각 level difference의 오차막대와 독립적인 더 정밀한 scheme/reference를 함께 사용한다.
- 상대 tolerance를 주장하려면 reference uncertainty와 bias budget을 분리한다. 예컨대 목표 5%에서 bias 2%, sampling 3%처럼 보수적인 충분조건을 사전에 설계한다.
- Richardson/MLMC는 관측된 rate와 가정이 뒷받침될 때만 사용한다. 인접 두 grid의 근접은 rigorous bias certificate가 아니다.

실무 task:

- extreme terminal probability는 스트레스 테스트 군으로 유지한다.
- 별도로 시장에서 해석 가능한 strike/maturity와 forward variance curve의 pricing task를 정의한다.
- 시장 데이터로 calibration을 주장하려면 데이터 출처·기간·out-of-sample·오차를 별도 검증한다.
- risk-neutral 확률을 실제 시장의 물리적 폭락 확률이나 VaR로 바꾸어 설명하지 않는다.
- bounded digital의 defensive second-moment 분석을 unbounded call, Greeks, barrier에 자동 확장하지 않는다.

완료 기준: “어떤 대상의 어떤 오차를 얼마의 비용으로 계산하는가”가 명확해져야 한다. 연속 bias의 일반 theorem을 못 얻더라도 제한된 범위의 정직한 수치논문은 가능하지만, continuum complexity를 주장하면 그에 맞는 증명이 필수다.

### R5. 논문 중심 명제 고정과 외부 검토

알고리즘을 1페이지 pseudo-code로, 새 기여를 2–3개의 명제로 설명할 수 있어야 한다.

- 명제 A: 기존 원리와 분리되는 Volterra-specific variance geometry 또는 오류/복잡도 결과.
- 명제 B: 그것을 실제 계산 가능한 방법으로 바꾸는 알고리즘과 조건부 정확성.
- 명제 C: 비용·새 task·학습 반복까지 포함한 재현 가능한 이득 또는 명확한 유효 범위.

증명 검토자와 계산 재현자를 구분해 피드백을 받는 것이 좋다. 현재 repository의 “외부 검토 2인, 별도 hardware, G8 통과”는 유용할 수 있는 **내부 governance 조건**이지, 모든 저명 저널이 요구하는 공식 필수항목은 아니다. 증명의 완전성과 기여의 중요성을 대신하는 체크박스로 운영하지 않는다.

## 10. 자원·기간·중단 기준

### 10.1 지금은 GPU보다 실험 설계가 우선이다

검토한 핵심 conditional payoff와 risk selector에는 CPU float64 요구가 있다. 따라서 RunPod GPU를 빌린다고 현재 코드를 그대로 빠르게 돌릴 수 있다고 가정하면 안 된다. GPU 지원은 별도의 구현·정확도·공정 비교 검증 대상이다.

R0와 작은 N의 R1은 노트북으로 시작한다. CPU thread 수·power mode·온도·동시 작업을 기록하고, timed run은 다른 실험과 동시에 돌리지 않는다. 큰 학습 반복은 이후 multicore CPU 환경에서 병렬화할 수 있지만 비용 지출 전 실제 pilot 시간을 잰다.

예상 연구 순서는 R0 1–2주, R1 1–2주, R2–R3 3–6주 규모의 작업 묶음으로 잡을 수 있다. 이는 인력·실험 시간에 따른 계획 추정일 뿐이다. 연속시간 정리는 수주 이상의 별도 작업일 수 있고 해결을 보장할 수 없다. 먼저 작은 pilot의 wall time으로 예산을 다시 산정한다.

### 10.2 중단/전환 규칙

- **공정한 conditional CE가 대부분의 이득을 설명하면:** “새 transport 우월성” 주장은 내려놓고 conditioning+geometry의 기여를 다시 측정한다.
- **같은 basis의 second-moment fitting이 이득이 없으면:** 목표함수 mismatch 가설을 기각하고 mode exploration/cost를 조사한다.
- **rank 증가의 oracle 진단에도 gain이 없으면:** 단순 차원 추가를 중단한다.
- **훈련 반복 하위 tail에서 성능 붕괴가 지속되면:** 불안정성을 정량화하거나 더 단순한 안정적 모델을 선택한다. route 예외를 계속 늘리지 않는다.
- **실용 범위에서 강한 baseline보다 장점이 없으면:** 특정 극단 rare-event용 방법으로 scope를 축소하거나 부정 결과/한계 연구로 전환한다.
- **새 theorem이 고전 원리의 재서술이면:** 새 이름을 붙여 유지하지 말고 lemma/background로 내려놓는다.
- **연속 relative bias 정리가 장기 병목이면:** 제한된 finite-grid 논문과 별도 이론 연구를 분리한다. 미증명 주장을 성능 결과로 대체하지 않는다.

## 11. 논문 수준 및 투고 방향에 대한 객관적 판단

### 11.1 지금 당장

현재 결과를 그대로 묶어 “혁신적인 neural path-integral 모델이 기존 방법을 전반적으로 능가하며 연속시간 효율성까지 완전히 입증했다”고 제출하는 것은 권하지 않는다. 비교기 fidelity, 실험 독립성, 비용 측정, 증명 연결, novelty가 동시에 공격받을 수 있다.

반면 재현 가능한 구현, 수많은 실패 기록, 정확한 유한격자 inference 설계는 유용한 연구 기반이다. 수정 후 제한된 numerical-method 논문으로 발전할 여지는 충분히 있다. **코드 규모·테스트 수·정리 번호 수만으로 박사급 또는 최상위급을 판정할 수는 없다.**

### 11.2 성과가 어떻게 나오는지에 따른 저널 정합성

- **SIAM/ASA Journal on Uncertainty Quantification:** rare-event uncertainty와 차원축소·검증 가능한 새 알고리즘이 중심이면 주제 정합성이 높다. 공식 scope도 수학·통계·알고리즘·응용의 의미 있는 발전을 포함한다. 이는 적합성이지 게재 보장이 아니다. [공식 안내](https://epubs.siam.org/juq/about)
- **SIAM Journal on Financial Mathematics:** Volterra 금융 계산의 새 수학·수치 방법과 금융적 의미를 함께 확립할 경우 자연스러운 목표다. [공식 저널 안내](https://www.siam.org/publications/siam-journals/siam-journal-on-financial-mathematics/)
- **Mathematical Finance / Finance and Stochastics:** 단순한 tuned sampler 개선보다 일반성·깊이가 있는 금융확률/계산 이론이 핵심일 때 검토할 목표다. 현재 자료로 이러한 수준에 도달했다고 판단하지 않는다.
- **상위 ML 학회:** 별도 신경망 초기화와 좁은 금융 실험만으로는 중심 ML 기여가 약하다. task amortization의 일반적 원리와 폭넓은 검증이 생길 때 다시 검토한다.

“최상위”는 분야마다 다르다. 현재 질문과 가장 잘 맞는 분야는 scientific computing / uncertainty quantification / computational finance이다. 저널 명성에 맞춰 양자·신경망을 추가하기보다, 실제 새 결과가 가장 중요한 독자층을 선택한다.

## 12. 최종 의사결정

**유지할 것:** 조건부 적분, exact-density ordinary IS, defensive support, mesh-compatible 좌표, 실패 자료 보존, 훈련 비용을 포함하려는 원칙.

**당장 멈출 것:** 최종 셀을 반복 관찰하며 routing rule과 mixture를 늘리는 개발, 함수평가 감소를 실무 가속으로 표현하는 것, YAML `proved`를 증명 검토로 간주하는 것, 이름·분야 트렌드만으로 독창성을 판단하는 것.

**선택할 주 방향:**

> 조건부 rare-event 추정의 second-moment 경로 기하를 이해하고, 그 원리에 따라 가장 작은 유효 transport를 구성하며, 실제 비용과 연속대상 오차를 분리해 검증하는 연구.

작업용 제목 예시는 **“Conditional Risk Geometry and Reliable Importance Sampling for Gaussian Volterra Rare Events”**이다. 제목을 먼저 확정하지 말고, R1–R3 결과가 뒷받침하는 기여에 맞춰 수정한다.

다음 행동은 새 V17을 즉시 크게 구현하는 것이 아니라 **R0 → R1**이다. 이 두 단계에서 현 결과의 어느 부분이 conditioning 효과이고, 어느 부분이 새로운 geometry 효과인지 분리하면, 이후 수학과 구현에 투자할 방향이 훨씬 명확해진다.

이전의 “이미 완벽하다”, “박사급을 넘었다”, “저명 저널 가능성이 매우 높다” 같은 평가는 현재 증거만으로 정당화되지 않는다. 더 정확한 평가는 **좋은 기반을 확보했으나, 결정적인 독창성과 공정한 성능 입증을 앞으로 만들어야 한다**이다.

## 부록 A. 이번 검토의 재현 및 한계

repo root에서 다음을 실행하면 기존 artifact를 읽어 재계산한다. 원본 scientific 결과를 덮어쓰지 않는다.

```powershell
python -m docs.reviews.reproduce_v16_review
```

스크립트는 기존 수치의 산술 일치, query-horizon 민감도, qualifier 집합 차이, operator timing, 감사 의미 검증 probe를 출력한다. in-memory 변조는 무결성 로딩 이후의 테스트이며 디스크의 원본 artifact는 바꾸지 않는다.

이번 검토에서 실행한 테스트 묶음:

```powershell
python -m pytest -q tests/test_g11_v16_final_policy_audit.py tests/test_defensive_proposal_selection.py tests/test_g11_v16_deep_tail_comparator_confirmation.py tests/test_tempered_target_transport.py tests/test_v16_transport_policy.py
python -m pytest -q tests/test_volterra_conditional_payoffs.py tests/test_finite_rank_gaussian_transport.py
python -m ruff check docs/reviews/reproduce_v16_review.py
```

검증 결과: 첫 묶음 **15 passed (16.14초)**, 두 번째 묶음 **20 passed (27.10초)**, 진단 스크립트 Ruff 검사 **통과**. 저장된 7개 셀의 accuracy z와 primary 성능비 재계산도 모두 일치했다. 전체 저장소 test suite와 모든 장시간 실험을 이번에 재실행한 것은 아니다. 기존 단위 테스트가 통과해도 새로 발견한 감사·통계·비교기 설계 문제가 사라지는 것은 아니다.

환경에서 `requests` dependency compatibility warning이 관찰됐다. 선택한 테스트 실행을 막지는 않았지만 재현 환경은 lockfile/정확한 package 버전과 함께 정리할 필요가 있다.

보고서의 이론 공백 판정은 코드·증명 원문에 대한 비판적 검토이지 형식 증명 검증이나 외부 심사 결과가 아니다. 특히 T16-5/9는 반례가 발견된 것으로 표현하지 않았다. 비용비 confidence interval, 새 training 반복, 미사용 parameter holdout, 실무 calibration은 아직 수행하지 않았다.
