# 높은 η: 실패한 precision validation을 반영한 fresh 재검증

2026-10-10. 이전 fixed441units의 전체 완료와 numerical audit 이후, 아래532units
표본 생성 전에 고정한다. 최종표본에 추가하는 continuation이 아니라 새 개발 검증이다.

## 실패 보존 및 새 allocation

`high_eta_family_validation_v1/session_20261010/result.json`에서 guide-free69whole의
μ RSE .939%, risk q0..q4 RSE 1.6133/1.5718/1.5076/1.6053/1.4721%였다.
static/guided의 RSE·민감도와16개 primary equivalence는 통과했지만, 4risk의 사전1.5%
기준 실패로 scoped pass=false다. 원본·실패 flag를 보존하고 다른 표본과 평균내지 않는다.

새 free-full count는 max target의 observed CV²에 safety2를 적용한다:

    ceil(2 * 69 * (0.0161333894838604/0.015)^2) = 160 whole.

이는 안전배수이지 분포 독립적 보장이나 성공 보장은 아니다. static은 기존40×32768draw,
guided는52whole를 유지한다(이번 observed variance의 동일 safety2 allocation보다 크다).
weights/q/N32/physics/temperature/mutation은 모두 동결된 기존 설정을 유지한다.

## 새 전체 실행 및 판정

- namespace `high-family-revalidation-v1-`, 새 IID path/label 및 whole seeds.
- 원래 high-eta five-q family,6estimands, 세 방법 전부 새 표본.
- 320IID batches +160free-full +52envelope =532units, preparation128calls.
- 기대95,998,080forward evaluations. cap100m/1800초/RSS4GiB.
- whole30초 /IIDbatch5초 /preparation30초; source/anchor/input/weights/q/seed 동결.
- 이전69whole 결과를 final에 추가하거나 outlier/failed seed를 교체하지 않는다.
- 모든532units 완료 전 prefix 성공 판단 금지.
- 원래1.5% RSE, draw/whole maximum contribution, LOO, IID between-block 조건 유지.
- μ primary static-vs-free pair original2comparisons;
  5q의3risk-pairs에 original30comparisons; 10% simultaneous equivalence 유지.
- source/anchor/input/q/seed/moment/unnormalized identity/work/summary 재감사;
  first/lastIID firstbatch8192draw 및 각SMC firstwhole numerical replay.

계획 선택과 재시도는 모두 development 과정이다. 여러 번 실행한 개발 검증에서
하나가 통과했다고 단일 preregistered publication confirmation으로 보고하지 않는다.
확증자료는 독립적으로 봉인한 새 셀·whole-training 반복·외부 재현으로 별도 생산한다.

새 성공 flag 역시 high-eta five-q finite-N32 development scope만 허용한다.
canonical, 전체P1/P2, large training authorization은 모두false다.
이 재시도에서 실패해도 사후 sample 증가나 기준 변경 대신 새 설계와 실패분석으로 돌아간다.
