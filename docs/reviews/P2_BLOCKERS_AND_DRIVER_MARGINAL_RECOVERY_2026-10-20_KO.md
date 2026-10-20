# 왜 P2를 승인하지 못하는가, 그리고 지금 무엇을 고치는가

2026-10-20. 사용자 설명과 실제 실행 기록. 커밋·푸시하지 않는다.

## 1. 쉬운 설명

지금은 저울을 개선하는 연구인데, 저울을 검사할 기준 추 자체의 무게도 충분히 정확히 모르는 상태다.
P2 관련 진단은 이미 실행했지만, P2 검증 완료와 대규모 확장을 승인할 근거는 부족하다.
모델을 실행 못하는 소프트웨어 오류가 아니라, 통계적 정확도 자격의 미충족이 핵심이다.

| 막힌 부분 | 현재 증거 | 왜 그냥 통과시키면 안 되는가 |
|---|---|---|
| 기준 확률 계산의 흔들림 | 앞선 새 G1의 fine RSE 3.95%/3.03%, 목표 1.5% 미달 | 모델과 기준의 차이가 개선인지 잡음인지 구분하기 어려움 |
| 서로 다른 기준 계산의 일치 | 일부 새 canonical SMC가 이전 G1과 조정된 개발 비교에서 불일치 | 중요한 기여를 놓치면 RSE도 낮게 보일 수 있음 |
| 시간격자 오차 | 앞선 canonical G1 paired upper 10.0599% > 10%; 연속시간 상대 bias는 별도 미해결 | 격자를 잘게 하면 답이 안 변하는지 충분히 확인되지 않음 |

RSE는 관측 표준오차/관측 평균이다. 모든 오차(특히 미관측 tail/bias)의 확정 상한이 아니다.
독립 whole 8개의 정규근사 CI는 소수 반복/미관측 tail에서 취약할 수 있다.
10.0599%를 10%로, 1.5066%를 1.5%로 반올림해 통과시키지 않는다.
작은 paired difference도 연속시간 모델과의 오차 보증은 아니다.

## 2. 해결책을 세 갈래로 나눔

1. 잡음 감소: 변동성 경로를고정한뒤남은가격난수를더뽑지말고Gaussian조건부확률로적분한다.
2. 독립 기준: 새수식을기존denseQR/독립quadrature와먼저대조하고,최종적으로는guide-free
   등다른탐색메커니즘과새고정표본으로교차검증한다. 같은G1geometry만쓰는새IS는
   완전히독립적인blind-spot검증을대신하지못한다.
3. 격자 검증: 위두문제를먼저줄여같은격자쌍의차이를정밀하게측정한다. 연속시간주장이
   필요하면별도상대bias분석을하고, 불가능하면고정유한격자논문으로scope를명확히한다.

무작정표본수를늘리는것은비용이매우크고,같은누락문제가있으면낮은분산을반복할수있다.
차원/네트워크/새operator추가는이검증을통과시키는대체재가아니다.

## 3. 이번에 실제로 시작한 수정

새 `fft_driver_marginal_reference.py`는원래finite-grid driver공분산을
FFT로작용시키고,남은Gaussian가격잔차를적분한다. N4096대형denseQR를피한다.
**확률만적분하고기존likelihood를그대로쓰면틀릴수있으므로, mixture밀도도같이주변화한다.**
원래모델q의M2를다른integrand로바꾸지않는다.

고정proposalr에서새추정량은원래추정량을변동성driver로조건부평균낸것이다.
따라서해석적으로는분산이늘지않는다. 그러나처리시간이커질수있어실제효율은별도다.
수치계산은PCG이며actualresidual과log-contribution민감도를검사한다.
그bound역시float64이므로machine-verifiedintervalcertificate라고부르지않는다.

### 완료한 검증

- 초기독립oracle7개통과: FFT/densecovariance,PCG/densesolve,conditionalCDF/QR,
  주변mixture밀도/QRprojection,ρsign/zero,미수렴및autograd거부.
- 추가검증까지독립oracle10개통과(17.34초). proposal조건부혼합분포에대한별도
  Gauss–Hermite적분으로rawIS의조건부평균=주변화IS를직접대조했고,
  H=.2/.45에서도QR확률과일치했다. 현재H=.05셀에만맞춘수식검사가아니다.
- N4096고정preflight2cells×4fixture×16draw통과. 준비15.557/11.905초,
  16drawscore1.13–1.26초,최대RSS약0.858GB.
- fixture의최대log-contribution민감도추정상한3.69e-7로고정허용1e-4미만.
  이는통계적정밀도성공이아니며본평가표본으로사용하지않는다.
- 실제곱셈/PCGresidual검사비용을covariance row수로기록해원래physicalcall과섞지않는다.

### 고정 pilot 및 재현 감사 완료

새 seed로 2 cells × 8 독립 blocks × 1,024 draws = 총 16,384 draws를 완료했다.
셀당 8,192개이며, 두 추정량은 같은 z를 사용하므로 방법 간 차이의 SE는 paired로 계산했다.
fine N4096 확률만 비교하며 coarse/연속시간/P2를 자동 승인하지 않는다.

| 셀 | raw 평균 | 주변화 평균 | raw RSE | 주변화 RSE | 주변화 최대 단일 기여 비중 |
|---|---:|---:|---:|---:|---:|
| canonical | 1.23651e-8 | 1.34296e-8 | 11.58% | 13.75% | 10.43% |
| high-eta | 9.14562e-7 | 9.58507e-7 | 8.21% | 7.78% | 3.13% |

두 셀 모두 목표 RSE 1.5%에 미달했다. paired 방법 차이의 조정 불확실성 상한도
각각 약 41.44% / 17.30%로 10% 일치 자격을 얻지 못했다. 이는 편향이 검출됐다는
뜻이 아니라, 동등성을 입증하기에 정밀도가 부족하다는 뜻이다.

앞선 더 큰 G1 실험의 RSE와는 표본 수가 달라 직접 승패를 판정하지 않는다.
이 표의 raw/주변화만 동일 표본 비교다. canonical의 관측 RSE 악화는 모집단 분산
정리와 모순되지 않지만, 실용적인 개선이 입증됐다고 주장할 수도 없다.

runner phase는 약 1,160.16초(19.3분)였다. CLI 시작 시 import 등은 이 시간에
포함되지 않으며, 통제된 방법별 wall-time 벤치마크도 아니다. 준비 공분산 row 작용
107,424회, scoring 작용 4,908,304회를 기록했다. raw physical 평가와 조건부 CDF는
각각 16,384회, 밀도 평가는 합계 32,768회다. 최대 log-contribution 수치 민감도
추정 상한은 약 1.01e-6 미만으로 고정 허용값 1e-4보다 작았다.

대표 2 block을 다시 계산해 raw 2,048회/조건부 CDF 2,048회로 seed, moments,
solver iteration, 공분산 작용 수 및 최대 민감도를 재현했다. 전체 16 block의
membership, source snapshot, 고정 설정과 요약도 검사했다.

- 감사 상태: `fft_driver_marginal_source_seed_moments_solver_replay_pass`.
- 결과 SHA256: `95abfac2df32652b7aafc20f48ec913bd11416eb5b76a5911a45dd630d4f3819`.
- 결과: `results/post_audit/fft_driver_marginal_pilot_v1/session_20261020/result.json`.
- 감사: 같은 디렉터리의 `audit.json`.

감사는 계산 재현성 검사이며 새로운 독립 확인 실험이나 통계적 자격은 아니다.

## 4. 아직 해결했다고 하지 않는 것

새주변화IS가G1과같은proposalgeometry를쓰기때문에,이것만으로기존SMC불일치가
사라졌다고할수없다. 새로운수식/수치계산경로를검증하고잡음을줄이는첫구현이다.
새방법이비싸거나분산감소가작으면비용해결(검증된preconditioner등)이추가로필요하다.
조건부SMC같은guide-free기준은그뒤의별도고정실험이다.
고정r의variance정리는반복학습안정성/실무수익성/저널독창성까지보장하지않는다.

모든 qualification flags는 false다.

## 5. 실패 원인과 다음 해결 순서

### 이번 개선만으로 충분하지 않은 이유

1. 가격 잔차를 적분해도 **어떤 변동성 경로가 중요한지**에 따른 변동은 남는다.
   소수 표본에 희귀 기여가 집중되는 문제까지 해결하지는 못했다.
2. 같은 G1 proposal geometry를 사용했다. 계산 수식을 바꾼 것이지, 독립적인 탐색
   메커니즘으로 놓친 경로가 없음을 검증한 것이 아니다.
3. 반복 선형해법의 비용이 크다. 분산 감소 가능성과 총비용 효율은 별개다.
4. 단일 격자 확률만 검사했다. all-q M2, 학습 bank, 격자 및 연속시간 편향은 별도다.

이번 방법은 검증된 개발 후보로 보존하되 생산 기본값이나 P2 통과 근거로 채택하지 않는다.
같은 실험을 통과할 때까지 seed/count를 바꾸며 반복하지 않는다.

### 우선순위 A: 경로 탐색의 병목 분리

앞선 matched-window의 기여·경로 특성과 SMC 계보를 함께 분석한다. 적분 변동성,
가격-변동성 결합 항, 변동성 최대값/발생 시점 등의 특징을 이용하되, mixture label을
물리적 희귀 모드로 오해하지 않는다. 독립 block 사이의 기여 집중과 꼬리를 구분하고
관측되지 않은 모드가 없다고 단정하지 않는다. 분석용 표본은 최종 확인에 재사용하지 않는다.

### 우선순위 B: 다른 탐색 방식의 기준으로 교차 검증

물리적 window IS와 guide-free population/tempered 기준을 독립적으로 개발한다.
SMC는 acceptance가 아니라 독립 whole-run 변동·계보·resampling/mutation 대조를 본다.
양쪽 모두 동일 target의 RSE ≤ 1.5% 등 기존 자격을 얻은 뒤 조정 일치 기준을 적용한다.
현재 불일치하는 arm을 정답으로 채택하지 않는다. 필요한 예산이 노트북에서 비현실적이면
통과를 선언하지 않고 계산 자원 또는 논문 범위를 별도 결정한다.

### 우선순위 C: 계산 가속은 작은 별도 실험으로

새 preconditioner 등은 dense oracle, 실제 residual 및 최종 기여 민감도를 먼저
검사한다. solver 가속은 희귀 경로 누락의 해결책과 구분한다. 현재 감사된 runtime과
결과 snapshot은 수정하지 않는다. 수치 허용오차를 느슨하게 해 통과시키지 않는다.

### 그 뒤에만 P2/대규모 학습

동일 2N conditional target에서 bank ESS/ancestry를 개선하고, 새 bank·새 최종 표본으로
rank/basis/KL–M2 ablation을 다시 실행한다. 최소 두 방법이 같은 accuracy 자격을 얻은
셀에서만 offline/fit/selection/inference를 모두 포함한 fixed-precision total wall-time을
비교한다. 인접 격자와 연속시간 상대 편향은 별도 검사한다. 저널용 confirmation은
새로 봉인한 셀과 독립 학습 반복만 사용한다.

**판정:** 이번 구현·고정 pilot·수치 replay는 완료했다. 그러나 남은 본질적 문제를
해결했다고 선언할 수 없다. reference 개선을 원래 q의 성능 개선이나 논문 기여 확립으로
바꾸어 표현하지 않는다. P1/P2/large-training은 모두 false이며 커밋·푸시하지 않는다.

## 6. 전체 회귀검사

전체 pytest 회귀검사: **1,460 passed, 270.69초**. 추가한 핵심 oracle 10개와
실행 계약 검사 1개를 포함한다. 테스트 통과는 모든 입력·모든 이론적 주장에 대한
무오류 증명이나 P2 정밀도 자격을 뜻하지 않는다.

- CI와 같은 Ruff 범위 `src tests experiments main.py train_driftnet.py`: 통과.
- CI와 같은 Mypy 범위 `src main.py train_driftnet.py`: 202 source files 통과.
- 새 두 experiment의 직접 타입 검사(`--follow-imports=silent`): 통과.
- 제한 없이 imported experiments까지 추적하면 기존 네 파일에서 14개 타입 오류가
  보고된다. 새 구현 오류와 구분하며, 감사된 옛 실험 runtime을 변경하지 않았다.
- 저장소 루트 전체의 `ruff check .`는 기존 보조 스크립트 네 개에서 49개 문제를
  보고한다. 위 CI 범위 통과와 저장소 모든 파일의 무오류를 혼동하지 않는다.
  해당 네 파일의 Git blob hash가 HEAD와 동일함을 확인했다.
- 실행 시작 시 기존 requests/urllib3/charset 계열 의존성 경고가 나온다. 이번
  수치 실험 도중 환경을 업그레이드하지 않았으며 경고가 없다고 주장하지 않는다.
- 결과 파일 SHA와 false인 세 qualification flag를 별도 재확인했다.
- `git diff --check` 통과. HEAD 유지, staging 비어 있음. 커밋·푸시 없음.
