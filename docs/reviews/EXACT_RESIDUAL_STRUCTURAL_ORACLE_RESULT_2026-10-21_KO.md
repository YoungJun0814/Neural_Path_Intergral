# 다음 우선순위 결과 — 정확한 residual 구조 진단과 구현 재검토

2026-10-21. 결론은 **구조적 사실을 확인했으나 모델 성능 문제를 해결한 것은 아니다**.
기존 q/G, 373개 성분과 가중치는 변경하지 않았다. P1/P2와 D3 승인은 여전히 false다.

## 1. 무엇을 진행했는가

앞선 frozen-q 특징 검정은 독립 확인에 실패했다. 같은 불안정한 M2 비율을 다시 최적화하지 않고,
정답이 있는 Gaussian 예와 현재 finite-grid 코드의 조건부 항등식을 검증했다.

- [원래 실행계획](../plans/EXACT_RESIDUAL_STRUCTURAL_ORACLE_2026-10-21_KO.md)
- [수학적 계약과 증명](../theory/EXACT_RESIDUAL_STRUCTURAL_CONTRACT_2026-10-21_KO.md)
- [관측 후 작성한 수치 검토 부록](../plans/RESIDUAL_ORACLE_NUMERICAL_REVIEW_ADDENDUM_2026-10-21_KO.md)

새 core는 `src/path_integral/residual_structural_oracle.py`, 실행기는
`experiments/post_audit_exact_residual_structure.py`다. 과거 수치 코드는 수정하지 않았다.

## 2. 원래 실패와 별도 재검토를 구분한다

첫 실행의 판정은 **INVALID_STRUCTURAL_ORACLE**이다. 비트 단위 밀도 불변성을 검사한
`torch.equal` 조건이 nonzero context에서 실패했다. 원래 `--audit`도
`ValueError: saved scalar probe/density identity changed`로 중단되었다.
원본 result/source/사전 계획은 그대로 보존했다. 성공 결과로 덮어쓰지 않았다.

조사 결과 모든 mean의 해당 성분은 정확히 0이었고, dense log-density 계산 차이는
최대 **3.553e-15**였다. 이는 q/G의 수학적 좌표 의존과 별개인 GEMM/reduction의
float64 roundoff다. 정확한 항등식에 대해 서로 다른 행의 부동소수점 결과까지
bitwise 동일하다고 가정한 것이 검증 구현의 오류였다.

따라서 원래 코드를 사후 수정하지 않고 별도 검토 코드를 추가했다.

1. 모든 component mean의 해당 좌표가 정확히 0인지 검사한다.
2. probe 간 다른 모든 좌표가 정확히 같은지 검사한다.
3. identity covariance와 `logsumexp(log w + z·m - ||m||²/2)`에서 불변성이 대수적으로 따른다.
4. dense 계산의 작은 차이도 기록하고 scale-aware float64 진단 범위와 대조한다.
5. toy/여섯 고정 scalar context의 적분을 전수 재계산하고 양 셀 zero context를 full replay한다.

이 검토는 통과했다. 별도 판정은
**POSTHOC_STRUCTURAL_IDENTITY_VERIFIED_ORIGINAL_GATE_INVALID**다.
이는 관측 후의 구현 오류 설명/구조 검증이지, 원래 사전 gate를 통과한 독립 확인 실험이 아니다.

## 3. 실제 모델에서 확인된 사실

현재 N32의 64차원 Gaussian 좌표는 물리적으로 서로 다른 두 Brownian을 뜻하는 것이 아니라,
각 시간 구간의 correlated local Gaussian pair를 독립 표준정규 좌표로 표현한 것이다.

| 좌표, 0-based | 현재 dictionary mean | 모델에서 하는 일 | 이번 결론 |
|---|---|---|---|
| 57 | 전 성분에서 정확히 0 | 더 앞선 local integral residual | 불변 사실만 기록; 효과 별도 분석 안 함 |
| 61 = 2N−3 | 전 성분에서 정확히 0 | V31, 마지막 **사용** 분산에 영향 | q/G는 바꾸지 않지만 payoff는 변함 |
| 63 = 2N−1 | 전 성분에서 정확히 0 | V32, terminal payoff에서 **미사용** | payoff도 변하지 않음 |

q/G는 identity-covariance mean-shift mixture이므로 이 좌표의 conditional base Gaussian을
그대로 유지한다. mean이 0인 좌표를 모두 유용한 방향으로 추가하는 것은 옳지 않다.
특히 63번을 추가해도 해당 terminal payoff 자체는 개선할 수 없다.

전체 mean 행렬의 수치 rank는 양 셀 모두49다. 이는 기본 tolerance를 사용한 수치 결과이며,
정확 rank 정리, 실제 population M2 하한, 15개의 유효한 새 방향 발견으로 해석하지 않는다.

두 셀에서 zero/과거 discovery block0/U top0/과거 confirmation block0/U top0을 고정했다.
각 context의 61/63번에 [-4,-2,0,2,4]를 대입했다. 새 context를 선별하거나 새 MC bank를 만들지 않았다.
뒤의 두 context는 이미 관측한 경로이므로 새로운 독립 confirmation이 아니다.

별도 scalar식과 full evaluator의 log-payoff 최대 오차는 **1.365e-12 이하**였다.
61번은 여섯 context 모두 payoff 변화를 보였고, 63번의 full log-payoff 변화는 정확히0이었다.

## 4. 좋은 사건 q와 좋은 위험평가 G는 다르다

고정 2차원 Gaussian bump toy에서 q는 identity-covariance 단일 Gaussian family의 최적 평균이다.
N=65,536일 때의 **정확식 기반** 결과는 다음과 같다. MC 관측치가 아니다.

| 대상 | 정확 RSE |
|---|---:|
| 사건 확률 추정, 좋은 q | 0.2255% |
| 동일 q의 M2 추정, 의도적으로 나쁜 G | 125.0710% |
| 동일 q의 M2 추정, 이론적 matched G=r_q | 0% |

독립 scalar 적분과 analytic M2의 상대 차이는6.662e-16이었다.
이 예는 “위험 평가 실패만으로 q 자체의 나쁨을 결론 낼 수 없다”는 반례다.
실제 Volterra에서 좋은 q가 입증됐다는 뜻도, 미지의 r_q에서 실제로 표본을 만들 수 있다는 뜻도 아니다.
이 toy와 production defensive mixture는 같지 않다.

또한 q가 residual Y를 전혀 바꾸지 못할 때, A(X)=E[g²|X]에 대해 Cauchy–Schwarz로

```
M2(q_X p_Y) >= (E_X sqrt(A(X)))² >= mu²
```

가 성립한다. 분리가능한 toy의 residual 기여는 정확히 계산할 수 있지만,
실제 비분리 Volterra에 이 toy의 floor를 수치 그대로 적용하지 않았다.
이 일반 항등식을 새로운 수학적 발견이라고 주장하지 않는다.

## 5. 조건부 적분 결과와 해석 제한

U=z61을 고정 나머지 좌표 X에서 적분했다. m1=E[g|X], m2=E[g²|X]다.

| 셀 / 과거 context | m1 | m2/m1² | 상대 수렴 확인 |
|---|---:|---:|---|
| Canonical / discovery | 0.0822786 | 1.058958 | 예 |
| Canonical / confirmation | 0.0435558 | 1.039225 | 예 |
| High eta / discovery | 0.296145 | 1.0000468 | 예 |
| High eta / confirmation | 0.734890 | 1.0007060 | 예 |

48/96/192-order Gauss–Hermite와 별도 adaptive scalar 적분을 비교했다.
위 네 context의 두 모멘트는 사전 지정 상대 기준을 만족했다.
하지만 양 셀 **zero context 두 개는 absolute-only 기준으로만 통과**했다.
그곳의 작은 모멘트와 큰 m2/m1² 값은 상대정밀도가 확보된 값으로 인용하지 않는다.
원래 `all_context_quadratures_resolved=true`는 absolute OR relative 기준이므로
모든 context의 상대정밀도가 확보됐다는 뜻이 아니다.

g∈[0,1]이므로 [-12,12] 밖에서 빠진 모멘트의 absolute upper bound는3.553e-33이다.
이는 구간 내부 adaptive quadrature 오차까지 rigorous하게 bound한 것이 아니다.
서로 다른 수치 적분법의 일치 역시 production estimator 무편향성의 증명이 아니다.

중요한 구분:

- 기존 q의 M2를 평가할 때는 **m2/(D_G D_q)**가 필요하다.
- 조건부 평균 event estimator로 바꾼 뒤의 M2는 **m1²/(D_G D_q)**다.
- 이를 혼동하면 기존 q의 분산을 잘못 낮게 보고한다.
- 이번에 production event/risk estimator를 이들로 교체하지 않았다.

위 네 관측 context만 보면 특히 high eta에서 61번 단독 조건부 변동은 작다.
따라서 “61번을 처리하면 지금의 tail 문제가 해결된다”는 판단도 근거가 없다.
zero context의 큰 값으로 반대 결론을 내리는 것도 상대 적분 오차와 대표성 문제 때문에 부적절하다.

## 6. 비용과 재현성

| 단계 | Full Volterra payoff calls | Scalar calls |
|---|---:|---:|
| 원래 production | 66 | 4,704 |
| 실패한 원래 audit, control flow로 재구성 | 11 | 1,527 |
| 별도 수치 검토 | 22 | 5,688 |
| **합계** | **99 / cap100** | **11,919 / cap50,000** |

실패 audit의 비용은 완료 receipt가 아니라 실패 지점의 source/control flow와 결정론적
scalar 계산 횟수에서 재구성했다. 이를 확인하기 위해 physics를 다시 실행하지 않았다.
첫 production의 계측 계산 wall은6.053초, 별도 검토 계산 wall은0.308초였다.
source archive 작성·입출력·전체 테스트 비용까지 계측한 실용 속도 benchmark는 아니다.

원래 결과 SHA256:
`083c03374ee6d81062359db5ac63e669b3de10b84b87c1e1049144af11cf8f0a`

별도 수치 검토 receipt SHA256:
`9fa72501f8e74b60e3c7d47d7e750948a4c2778a4e24eea13054ff62cb8386b0`

별도 수치 검토 source ZIP SHA256:
`225a8672e1d5c1f1e098fbdaa96c0f309653b638e673a7af3467c320343ad4f1`

결과 디렉터리: `results/post_audit/exact_residual_structure_v1/session_20261021/`.
원본 source/protocol/result와 별도 source/protocol/receipt를 모두 보존했다.

## 7. 기술 검토

새 원래 진단 테스트15개와 사후 수치 검토 테스트4개가 통과했다.
정확식/독립 적분/두 mesh의 조건부식/미사용 좌표/비용 cap/잘못된 입력을 검사했다.
특히 mean 성분이1e-100처럼 작지만 **0이 아닌 경우에는 구조 불변성 검증을 거부**한다.
작은 관측 밀도 차이만 보고 좌표가 불필요하다고 판단하지 않도록 했다.

CI 범위 Ruff와 Mypy(213 source files), 새 두 runner의 별도 Mypy가 통과했다.
최종 전체 pytest는 **1,681개 통과 / 571.76초**였다.
source coverage는 **85%**(22,225 statements / 3,402 missed),
새 `residual_structural_oracle.py`는 **98%**(57 statements / 1 missed)다.
Python3.10.11 / Windows 로컬 결과이며 원격 Linux/Python3.11 CI 통과를 뜻하지 않는다.
기존 RequestsDependencyWarning은 남겨두고 환경 의존성 변경은 하지 않았다.

두 snapshot에 동결된1,112개/1,111개 파일의 현재 SHA와 source ZIP SHA는 모두 일치했다.
누적 완료 실험9개의 `result.json`도 기존 `audit.json`의 result SHA와 일치했다.
이는 기존 감사 receipt와의 무결성 검사이며 과거 전체 physics를 재실행했다는 뜻은 아니다.
source archive13개(14,237 members)와 공개할 text 파일을 점검했을 때 알려진 credential
패턴이나 비정상 archive 경로는 발견되지 않았다. 새 text source의 Git newline normalization
차이는 없었고, README/새 Markdown21개 파일의 local link 검사에서도 깨진 링크는 없었다.
이러한 제한된 패턴 검사가 모든 종류의 비밀정보 부재를 보증하지는 않는다.
최종 staged1,896개 파일은 working-file raw bytes와 Git index blob이 모두 일치했고,
13개 archive/14,237 members의 공개 파일 재검사 및 `git diff --cached --check`도 통과했다.
과거 실패·비용 기록은 삭제하지 않았다.

## 8. 다음 판단

채택할 것은 **실패 유형의 분리와 정확한 conditional-moment 계약**이다.
weight-only 재학습, 61번 단독 correction, 더 큰 모델이나 operator를 지금 채택하지 않는다.

다음 구현은 별도 예산/계획을 먼저 고정해야 한다.

1. 작은 모멘트에서 absolute-only 통과를 상대정밀도로 오인하지 않도록 log-scale/분할 적분과
   명시적인 오차 전파를 구현한다. 실패 context는 보존한다.
2. 기존 q의 평가용 m2와 Rao–Blackwell event용 m1을 분리한 소규모 matched-work 대조를 설계한다.
   단일 좌표로 충분한지 판단하려면 대표성 있는 risk-weighted 경로와 독립 평가가 필요하다.
3. 조건부 평균화가 줄일 수 있는 분산이 비용에 비해 작으면 이 좌표 단독 접근은 중단한다.
   57번/정확한 추가 residual subspace나 다른 G 구조는 별도 가설로 검증한다.
4. 적분 정확도·독립 reference·반복 학습·전체 비용을 통과하기 전 P2/대규모 학습을 재개하지 않는다.

새로운 확률모델의 우월성, 최상위 저널 기여, 통계적 실패의 완전 해결은 아직 확보하지 못했다.
