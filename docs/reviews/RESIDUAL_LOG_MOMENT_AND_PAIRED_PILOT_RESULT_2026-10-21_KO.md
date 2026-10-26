# §8 다음 판단 실행 결과: 적분 정확도 개선 완료, 단일 좌표 채택 보류

2026-10-21, America/New_York. 이전 보고서의 §8을 순서대로 실행한 결과다.

**성공한 것:** 작은 conditional moment를 log-scale에서 다루고, relative numerical resolution을
검사하는 구현·고정 context 확인·독립 fresh paired pilot·재생 감사를 완료했다.

**해결되지 않은 것:** fresh 표본의 event/M2 기여 집중과 정밀도 부족이다.
최종 판정은 **NO_ADOPTION_PRECISION_UNRESOLVED**다.
61번 residual 처리의 성능 우위, population M2 감소율 또는 실용 speedup을 입증하지 못했다.
기존 q/G와 원래 모델은 변경하지 않았다. P2/D3/대규모 학습은 승인하지 않았다.

실행계획: [RESIDUAL_LOG_MOMENT_AND_PAIRED_PILOT](../plans/RESIDUAL_LOG_MOMENT_AND_PAIRED_PILOT_2026-10-21_KO.md).
과거 INVALID 결과와 사후 구조 검토는 원본 그대로 보존했다.

## 1. 쉽게 설명하면

기존에는 아주 작은 확률을 적분하면서 “오차의 절대 크기가 작다”는 이유로 계산을 통과시킬 수 있었다.
그러나 답 자체가 훨씬 작으면 실제 상대오차는 클 수 있다.
이번에는 먼저 이 문제를 수정했다. 그 다음, 61번 좌표 하나를 평균내면 도움이 되는지
새로운 작은 표본에서 살펴봤다.

적분 계산 자체는 안정적으로 맞았지만, 표본에는 여전히 극소수의 큰 기여 경로가 지배적이었다.
따라서 **계산을 더 정확하게 한 것과, 모델의 성능을 확실하게 비교할 수 있게 된 것은 다르다.**

## 2. 새 구현과 수학적 검토

새 core `src/path_integral/residual_log_quadrature.py`를 추가했다.
기존 frozen source는 수정하지 않았다.

- 표준정규 residual U에 대해 m1(X)=E[g|X], m2(X)=E[g²|X]를 log-space에서 계산한다.
- [-16,16]을 폭1의32개 구간으로 나누고 각 구간 log integrand scale을 빼 adaptive 적분한다.
- 별도 composite Gauss–Legendre16/32/64-order 적분과 비교한다.
- QUADPACK reported error와 수치법 차이로 내부 오차 proxy를 만든다.
- g∈[0,1]의 가정 아래 생략한 Gaussian tail은 **1.27775e-57** 이하다.
- tail과 내부 proxy를 모멘트 크기로 나눠 **relative 기준1e-6**을 적용한다.
- underflow를0, zero variance, absolute-only 성공으로 바꾸지 않는다.
- m1/m2 구간을 `m2/m1²`의 구간으로 전달한다. Jensen/모멘트 관계와도 대조한다.

중요한 제한은 내부 quadrature 오차가 rigorous interval bound가 아니라는 점이다.
서로 다른 numerical rule의 일치는 실용적인 정확도 검사이지 무편향성·전체 함수족의 오류 부재를
증명하지 않는다. 매우 좁은 미검출 peak 같은 일반 callback의 한계를 완전히 없애지 못한다.
반면 Gaussian tail bound 자체는 bounded payoff 가정에서 성립한다.

## 3. Q0 — 고정 여섯 context 상대 수치 검사

새 runner `experiments/post_audit_residual_log_moments.py`로 다음을 검사했다.

- Gaussian bump known-answer3개: analytic moment와 log-scale numerical moment 대조.
- 기존 저장6개 profile 전수: canonical/high eta 각각 zero/discovery/confirmation.
- 별도 audit: known-answer와 두 zero profile 재생.

**여섯 profile 모두 relative numerical gate를 통과**했다.
이는 과거 경로의 계산 개선이지 새로운 population sample이나 독립 성능 confirmation이 아니다.

이전에 absolute-only였던 zero context의 값은 다음과 같다.

| 셀 / 모멘트 | 과거 adaptive 값 | 새 log-scaled 값 | 과거 값 대비 변화 |
|---|---:|---:|---:|
| Canonical / m1 | 1.568164e-19 | 1.541714e-19 | −1.69% |
| Canonical / m2 | 2.007257e-21 | 2.327913e-21 | +15.97% |
| High eta / m1 | 1.885833e-14 | 2.431408e-14 | +28.93% |
| High eta / m2 | 2.961525e-15 | 2.343140e-15 | −20.88% |

기존 보고서가 이들 값을 상대정밀도가 확보된 효과크기로 사용하지 않은 것은 타당했다.
새 zero profile의 relative error proxy 최댓값은 약3.25e-11 이하다.
이 작은 오차 proxy가 global M2나 희귀 기여 모드를 충분히 탐색했다는 뜻은 아니다.

### Q0 절차적 편차

계획은 audit에서 최대2개 known-answer 대조를 명시했으나, 구현은3개 전부를 재생했다.
예제나 threshold를 관측 후 바꾸지는 않았으며 **실제 audit49,042 scalar calls로 cap50,000 이내**였다.
이 추가 감사 작업은 사전 계획과의 차이로 기록한다. 계획/결과를 사후 수정해 숨기지 않았다.

## 4. Q1 — 새 표본의 paired effect-and-cost pilot

Q0를 통과한 후 새 source를 동결하고 실행했다.
N32의 **uniform373-component q=G**를 유지했다.
이 명칭은 natural 성분10%+나머지372개 shift에90%를 균등 배분하는 기존 U proposal을 뜻하며,
373개 성분 모두에1/373을 주는 분포가 아니다.
각 셀8개 독립 block×16경로, 총256개 새 IID context를 빠짐없이 사용했다.
이는 373개 label을 균등하게 한 개씩 넣은 bank가 아니라 **기존 mixture 확률에 따른 IID draw**다.

새32개 path/label seed를 할당했고, 지정한 과거 D2 및 특징 확인 범위의
13,198개 distinct seed와 충돌이 없었다. 저장소 전체의 모든 과거 난수를 전수 확인한 것은 아니다.

D=q/p=G/p는 U=z61과 무관하다. 같은 X,U에서 다음을 구분했다.

| estimator/양 | contribution | 목표 |
|---|---|---|
| 원래 event | g/D | μ |
| 조건부 평균 event | m1/D | μ, numerical approximation 별도 |
| 원래 event의 raw M2 평가 | g²/D² | 기존 event estimator의 M2 |
| 기존 q의 conditional M2 평가 | m2/D² | 위와 같은 기존 M2 |
| 조건부 event의 M2 | m1²/D² | 조건부 평균 추정기의 M2 |

`m2−m1²`의 차이는 조건부 변동에서 제거 가능한 부분이다.
기존 `g²/D²`의 표본평균과 `m2/D²`의 표본평균 차이는 estimator의 차이이며 모델 개선이 아니다.
실제로 canonical의 raw M2 점추정은2.50e-14, conditional 평가의 기존 M2는5.52e-13으로
큰 차이를 보였다. 이 정도로 불안정한 표본에서 낮게 관측된 수치를 성공으로 읽으면 안 된다.

모든256 profile의 두 모멘트는 numerical relative gate를 통과했다.
최대 conditional-moment relative error proxy는 **6.734e-11**이었다.
집계에 전달한 conditional event/두 종류 M2의 numerical proxy도 별도로 저장했고,
아래 표의 **sampling RSE와 혼합하지 않았다.**

## 5. 정밀도 gate는 양 셀 모두 실패

raw RSE는128개의 경로별 contribution으로 계산하고,
block RSE는8개의 독립 block 평균으로 별도 계산했다.

| 셀 / endpoint | raw RSE | block RSE | 최대 한 경로 기여 |
|---|---:|---:|---:|
| Canonical / 원래 event | 82.18% | 79.61% | 81.71% |
| Canonical / conditional event | 71.89% | 82.85% | 70.20% |
| Canonical / raw event M2 | 98.50% | 98.30% | 98.51% |
| Canonical / 기존 q conditional M2 | 66.55% | 62.92% | 61.91% |
| Canonical / conditional event M2 | 94.71% | 98.52% | 94.66% |
| High eta / 원래 event | 58.06% | 53.29% | 56.20% |
| High eta / conditional event | 57.60% | 52.78% | 55.66% |
| High eta / raw event M2 | 92.31% | 91.29% | 92.27% |
| High eta / 기존 q conditional M2 | 66.41% | 60.79% | 58.95% |
| High eta / conditional event M2 | 92.00% | 90.93% | 91.94% |

각 endpoint의 사전 기준은 raw/block RSE≤20%, 최대 경로 기여≤10%,
LOO 변동≤15%였다. **10개 endpoint 전부 실패**했다.

최종 `NO_ADOPTION_PRECISION_UNRESOLVED`는 따라서 정해진 판정이다.
눈에 보이는 분산 감소율을 성공으로 인정하거나, 이 파일럿의 표본 수를 관측 후 늘리지 않았다.
반대로 “61번은 쓸모없다” 또는 “conditional integration이 불가능하다”는 반증도 아니다.

### 비용 proxy를 왜 성능 결론으로 쓰지 않는가

실행기는 variance/cost의 plug-in 개발용 proxy를 저장했다.
수치적으로 canonical/high eta의 conditional 계산 비용 비율은 약253배/955배였고,
비용×분산 proxy는 약12.87/316.15였다. **둘 다 정밀도 미달 자료의 미자격 proxy다.**

따라서 이 값으로 population speedup, 정확한 slowdown, Pareto 우열을 주장하지 않는다.
원래 payoff는16개를 batch로 계산했고 조건부 적분은 경로별 adaptive Python 호출이므로,
현재의 비최적화된 구현 비용도 반영한다. 독립적인 timing 반복·matched-wall 대조가 아니다.
결론은 정밀도 실패에 따른 **채택 보류**이며, 단순 비용 수치만으로 이론적 가능성을 중단한 것은 아니다.

## 6. 기술·이론 재검토

원래 논리와 일치함을 다음 범위에서 확인했다.

1. 모든16개 block 파일이 complete result 내 record와 일치한다.
2. seed/전체 통계/정해진 gate/채택 보류 판정을 전수 재계산했다.
3. 양 셀 block0의32개 context를 전체 evaluator/profile/적분까지 재생했다.
4. 별도 zero-physics 무결성 검사에서 모든256개 score와 mixture density를 다시 계산했다.
5. 저장된 full log-payoff와 별도 scalar profile식의 차이는 전체256개에서
   **6.822e-13 이하**였다. profile 자체를256개 모두 full evaluator로 새로 만든 것은 아니다.
6. Q0/Q1 snapshot의1,114개/1,117개 파일 hash와 source ZIP, result/audit SHA가 일치했다.

알려진 제한도 남아 있다.

- 일반 callback의 수치 적분은 rigorous error certificate가 아니다.
- sampling 표준오차와 block sensitivity는 극희귀 tail coverage 보장이 아니다.
- 이 파일럿은 q=G=uniform dictionary 하나만 다뤘다. 다른10개 학습 q의 전수 우위는 검증하지 않았다.
- profile extraction과 full g를 각각 계산했으므로 실험 비교 비용과 최적 production 설계는 다르다.
- 반환된 variance/cost 점추정은 정밀도 gate를 통과하기 전 해석하지 않는다.
- audit는 대표32경로 full replay와 전체256경로 대수/저장 자료 재집계이지, 모든 적분을 다시 실행한 것은 아니다.

새 core14개 및 pilot7개 테스트가 먼저 통과했다.
테스트에는 known-answer/상수/underflow/tail-dominated 반례/잘못된 입력/예산 cap/
목표 M2 구분/비용 veto/정밀도 veto/누락 block/Jensen 위반이 포함된다.
최종 전체 regression은 **1,702개 통과 / 478.92초**였다.
전체 source coverage는 **85%**(22,304 statements / 3,407 missed),
새 log-quadrature core는 **94%**(79 statements / 5 missed)다.
CI 범위 Ruff 통과, Mypy **214 source files 통과**, 새 두 runner의 별도 Mypy도 통과했다.
Python3.10.11 / NumPy1.26.2 / SciPy1.15.3 / Torch2.12.0.dev20260408+cu128 환경에서
CPUfloat64로 계산했다. 문서 링크의 최신 SciPy 버전을 설치한 것이 아니며 의존성을 바꾸지 않았다.
기존 RequestsDependencyWarning은 남아 있다.

새 연구 변경의 원격 Linux/Python3.11 CI는 실행·검증하지 않았다.
공개 파일 점검 범위에서는 새32개 파일/2개 archive(2,233 members)에 알려진 credential
패턴/비정상 archive 경로/100MiB 이상 파일이 없었다. 이는 모든 비밀정보 부재의 보증은 아니다.
README/새 문서의 local link와 `git diff --check`도 통과했다.

## 7. 사용한 예산

| 단계 | Full Volterra evaluations | Scalar evaluations |
|---|---:|---:|
| Q0 production | 0 | 69,426 |
| Q0 audit | 0 | 49,042 |
| Q1 production | 512 | 1,420,244 |
| Q1 audit | 64 | 174,748 |
| 별도 전체 scalar식 무결성 대조 | 0 | 256 |
| **합계** | **576** | **1,713,716** |

생산용512회는 원래 g256회+profile256회다. audit64회는 원래g32회+profile32회다.
이는 Python 호출 횟수가 아니라 평가한 경로 수이며 test 내부 물리 계산은 별도다.
기존 구조 진단의99회를 이번 새 프로토콜 예산으로 다시 세거나 숨기지 않는다.

Q0 계산 wall은1.102초, Q1의 compute+block I/O는27.959초였다.
Q1 block I/O0.160초/집계0.043초를 따로 보존했다.
source ZIP 작성·준비·최종 I/O·전체 테스트를 포함한 end-to-end 속도 benchmark는 아니다.

## 8. 결과를 반영한 다음 판단

**채택:** log-scale 적분과 상대 numerical gate, tail/error propagation, 서로 다른 M2 목표를
분리하는 구현 계약. 수치 불안정과 sampling 불안정을 구분한 진단 체계.

**채택하지 않음:** 61번 단일 correction, 이 단일 좌표의 adaptive 적분을 곧바로 production
모델로 승격하는 것, 현재 점추정만으로 성능 승리 선언, 대규모 재학습.

현재 §8의 첫 번째 수치 구현은 완료했고, 두 번째 항목은 **동일 경로 수의 효과/비용 파일럿까지만**
수행했다. 요청된 matched-work 본 대조를 완료했다고 부르지 않는다.
이 파일럿의 정밀도 gate가 실패해 matched-wall/fixed-precision 후속 실험으로 넘어가지 않았다.
세 번째 항목은 불확실한 성능에 따른 채택 보류, 네 번째 항목은 대규모 학습 gate 유지로 반영했다.
다음 우선순위는 같은 고비용 적분을 더 큰 n으로 재실행하는 것이 아니다.

1. **Population inference 없이도 검증 가능한 계산 구조부터 개선:** scalar callback의
   해석적 형태·monotonicity/극값을 이용한 적분 분할과 벡터화, 전역적 tail/error envelope를
   검토한다. known-answer와 실패 반례를 먼저 고정한다.
2. **1차원만으로 충분하다는 가정은 하지 않기:** 57번 또는 정확한 residual subspace를
   고려하려면 density factorization, 영향 구간, 불필요한63번 배제 조건부터 증명한다.
   이것은 별도 모델 가설이며 이번 파일럿 결과로 자동 승인하지 않는다.
3. **Risk contribution 대표성 문제 별도 해결:** 기존 실패 bank를 성공 자료로 재활용하지 않고,
   기여 집중을 안정적으로 다룰 평가분포/독립 reference 설계를 먼저 한정한다.
4. 다음 새 실험은 계산법/효과크기/정밀도/비용을 각각 통과하도록 사전 예산을 다시 고정한다.
   이번에 실패한 표본 수를 임의로 늘리거나 허용폭을 완화하지 않는다.

이 상태는 최상위 저널용 성능·독창성 완성이 아니라, 올바른 검증 도구를 확보한 개발 단계다.

## 9. 재현 anchor와 GitHub 부가 작업

- Q0 result SHA256: `53c604c5faafa573ae380605342a130f68d54dcde6cefb30393a380bc482bb6b`
- Q0 source ZIP: `8ad244b40f754e3b1653dab47d43714786031fa3e106a3d498b20d553848209b`
- Q1 result SHA256: `8645b470b638ff9860ea831915c1bf10e2111d66ca1780f31640c2c6c008ade1`
- Q1 source ZIP: `c5dc299ce614bd1b3a48364b9d6068999cc96679dcbac5241fc9c05a29d8b693`

결과: `results/post_audit/residual_log_moments_v1/session_20261021/`,
`results/post_audit/residual_paired_pilot_v1/session_20261021/`.

Quickdraw는 과학적 gate와 무관하게 별도 실제 README clarification으로 처리했다.
[Issue3](https://github.com/YoungJun0814/Neural_Path_Intergral/issues/3)는
2026-10-21 05:56:46Z에 생성,05:56:59Z에 해결·종료됐다(13초).
[문서 수정 커밋 f556424](https://github.com/YoungJun0814/Neural_Path_Intergral/commit/f556424360b8307284b23a6cfc0efcdee7380060)은 main에 push했다.
프로필 HTML에 `Achievement: Quickdraw` 표시를 확인했다. 빈 Issue/가짜 공동 작성은 하지 않았다.
이 문서 정정 커밋의 [원격 CI](https://github.com/YoungJun0814/Neural_Path_Intergral/actions/runs/39552456787)도
통과했다. 이는 아직 미커밋인 새 numerical 구현의 원격 CI를 뜻하지 않는다.
새 연구 코드·결과의 커밋/푸시는 문서 정정 push와 별개의 작업이다.
