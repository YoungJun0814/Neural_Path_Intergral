# 다음 판단 실행: log-moment 정확도와 제한된 단일 residual 대조

2026-10-21. 선행 보고서 §8의 순서를 따른다. 새 q/G, operator, 대규모 학습은 도입하지 않는다.

## A. 해결할 문제와 주장 경계

기존 absolute-only 적분 결과를 상대정밀도 결과로 읽을 수 없었다.
기존 원본/INVALID 판정/사후 numerical review는 변경하지 않는다.
새 source를 첫 결과 계산 전에 보관하고 새 result 디렉터리에서 한 번만 실행한다.

1. log g를 직접 계산하고, 고정 [-16,16]을 폭1의32개 구간으로 나눈다.
2. 각 구간에서 log integrand scale을 빼고 adaptive QUADPACK 적분을 수행한다.
3. 별도의 composite Gauss–Legendre16/32/64-order와 비교한다. 두 모멘트의 nodes를 공유한다.
4. g∈[0,1]인 경우 생략한 Gaussian tail은 각 모멘트에 대해 `2*Phi(-16)` 이하이다.
5. 모멘트 크기가 tail bound보다 지나치게 작거나, 내부 수치법이 일치하지 않거나,
   adaptive convergence warning이 있으면 **UNRESOLVED**다. absolute-only 성공 경로는 없다.
6. adaptive reported error와 두 수치법 차이를 합한 내부 오차 proxy에 tail bound를 더해
   상대오차/모멘트 ratio 구간을 전파한다. 내부 오차 proxy는 rigorous interval bound가 아니다.
   따라서 상대 **numerical resolution**과 수학적 정확성 certificate를 구분한다.

고정 상대 gate는1e-6이다. `m2/m1²`의 log ratio는 `log m2 - 2 log m1`로 계산한다.
각 m의 하한/상한을 사용해 ratio의 하한/상한을 전달하고, Jensen/0≤g≤1의 모멘트 조건도 점검한다.
underflow를 관측0 또는 zero variance로 처리하지 않는다. log moment가 유지돼도 tail 대비정확도가
미확보이면 gate는 통과하지 못한다.

독립 Gaussian bump known-answer와 상수 payoff, 작은 모멘트, underflow/tail-dominated 반례를
unit test로 검사한다. 원래6개 context의 저장 scalar profile을 사용한다.
이 단계의 full Volterra calls는 **0**이다. 과거 context는 population 표본/독립확인이 아니다.

공식 구현 문서: [quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html),
[log_ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html).
QUADPACK abserr는 error estimate이지 rigorous guarantee가 아님을 유지한다. 설치된 환경은 변경하지 않는다.

## B. Q0 고정 budget와 다음 gate

source anchor:

- 원래 result SHA256: `083c03374ee6d81062359db5ac63e669b3de10b84b87c1e1049144af11cf8f0a`
- 수치 검토 SHA256: `9fa72501f8e74b60e3c7d47d7e750948a4c2778a4e24eea13054ff62cb8386b0`

저장6개 profile 전수 검사 + known-answer3개; scalar cap150,000, full physical cap0,
계산 wall120초, CPUfloat64/1thread. 감사는 저장6개 중 양 셀 zero profile 재계산,
cap50,000과wall60초를 별도 부여한다. 최대2개의 known-answer 적분은 exact식과 대조한다.
다른 source/계획으로 실패 결과를 덮어쓰지 않는다. 추가 허용폭/구간 확장 재시도는 이번에 없다.

여섯 profile 모두 상대 numerical gate를 통과하면 아래 Q1 실행기를 준비할 수 있다.
하나라도 실패하면 Q1 production은 실행하지 않고 원인·다음 설계만 작성한다.
Q0 통과 자체는 production 전체 정확도나 P2 승인이 아니다.

## C. Q1 paired/cost 대조의 사전 설계

Q0 후 source를 별도 동결한다. 변함없는 N32 uniform373-component proposal q=G에서
두 셀 각각8개의 독립 block, block당16개 fresh IID context를 뽑는다.
namespace `residual-log-moment-pilot-v1`, 셀/block/역할별 SHA-derived seed를 사용하고
256개의 새 context를 빠짐없이 보존한다. 기존 seed ledger와 충돌을 검사한다.
기존모델의density와새estimator의density를바꾸지않으며 residual61 mean이정확0임을요구한다.

동일 context에서 다음을 분리한다(D=q/p=G/p).

| 양 | G에서의 contribution | 목표 |
|---|---|---|
| 원래 event | g/D | μ |
| conditional event | m1/D | μ, 단 수치 적분 오차 별도 |
| 원래 event의 raw M2 | g²/D² | 기존 event estimator의 M2 |
| 기존 q의 conditional M2 평가 | m2/D² | 위와 같은 M2 |
| conditional event의 M2 | m1²/D² | 새 conditional event estimator의 M2 |
| analytic conditional variance loss | (m2−m1²)/D² | 두 M2의 비음수 차이 |

g는 원래64차원 표본의 U도 그대로 사용해 계산한다. scalar profile은 U를제외한X로만구성한다.
기존 g²와 m2의 차이를 모델 개선으로 주장하지 않는다. fitting/selection은0이고,
Q1은 **pairwise effect-and-cost pilot**이지 독립 confirmation/동일정밀도 speedup 검증이 아니다.

예산: production physics512(원래g256+profile256), replay64(양셀block0의동일32context×2),
합계576/cap600. scalar cap2,500,000 production+500,000 audit, wall900+180초.
unit test 내부 physics는 별도이며 연구 실험 call count와 혼합하지 않는다.
scalar budget 부족 또는 개별 unresolved도 삭제하지 않고 실패로 저장한다.
새 context에 맞춰 구간/허용폭을 바꾸지 않는다. 중단비용 보존, nonempty output overwrite금지.

I/O·추출·density·원래 fullg·profile·quadrature·집계 wall을 구분한다.
같은 n의 수치 비교를 matched-wall 성능으로 부르지 않는다.
동일정밀도 시간 비교는 `cost/path * (M2−mu²)`의 population quantity이므로,
pilot에서는 raw/block plug-in ratio를 **개발용 proxy**로만 기록한다.
상대오차 propagation은 표본 통계 SE와 구분해 별도 기록한다.

다음 gate: 모든 profile 상대resolved + 두셀각각 event/raw/block RSE≤20% + M2 raw/blockRSE≤20%,
최대기여≤10%, LOO변동≤15%일때만 event variance/cost proxy를 해석한다.
그때에도 비용보정20%이상의개선이없으면 single-coordinate 실용개선채택을중단한다.
정밀도실패면 **NO_ADOPTION_PRECISION_UNRESOLVED**, 좌표무용성의증명으로쓰지않는다.
관측 후 더 많은 표본을 추가하지 않는다. matched-wall 본 실험은 별도새budget/독립자료가필요하다.

## D. 인계와 안전 경계

알려진 수학식·별도 적분·전체 regression과 lint/types를 점검하고 결과/한계/다음 판단을 기록한다.
P2/D3/대규모학습은이번에승인하지않는다. 독립 reference/전체 q/반복학습/continuum문제는남아있다.
Quickdraw는별도README인덱스정정Issue3으로처리했으며과학적판정기준과무관하다.
