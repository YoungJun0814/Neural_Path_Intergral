# 조건부 risk-target SMC bank — 사전 bounded 실행계획

2026-10-22 America/New_York. 이전 marginal-G IID bank의 STOP을 보존하고,
bank를 만드는 방법만 변경한다. commit/push, q 변경, 실제 G fitting, P2/D3와 대규모 학습은 하지 않는다.
목표는 **target-aware bank의 제한된 개발 자격 판단**이다. 모든 관측 후 수정/예산 전용을 금지한다.

## A. 이론/알려진 답부터

[수학 계약](../theory/CONDITIONAL_RISK_TARGET_SMC_CONTRACT_2026-10-22_KO.md)을 먼저 고정한다.
bridge는 `q_X(m2/Dq²)^beta`, beta1 normalizer는 같은 고정q의 M2다.
Gaussian pCN, component-posterior Gibbs+centred pCN, q independence의 MH ratio를 각각 검증한다.
nontrivial mixture의 pairwise detailed balance, Gaussian known-answer normalizer/target moment,
stratified resampling/ancestry 구분, invalid potential/예산/직렬화/초기 I/O failure receipt를 검사한다.
원래 numeric source/result는 수정하지 않는다.

## B. 생산 실험 — 모델 차원/성분 수는 고정

N32 canonical/high-eta 두 셀. X62이며61/미사용63은 탐색하지 않는다.
두 arm, 각4개 독립 island. 총16개 island를 미리 고정한다.

- primary `component`: 매 stage centred-component pCN → Gaussian-prior pCN → q independence.
- matched-work control `prior`: 매 stage Gaussian-prior pCN → Gaussian-prior pCN → q independence.
- particles128, temperatures `(j/32)^4`, j=0..32.
- pCN scale0.35, 총32stage×3mutation. **마지막 beta1에서도 mutation**한다.
- 매4stage마다 stratified resampling, 단 마지막stage는하지않는다.
- 초기/독립 q draw, Gibbs label, Gaussian noise, acceptance/resampling용 seed를
  island별 namespace에서 분리한다. arm끼리도 seed가 다르므로 paired CRN 대조가 아니다.
- 실행순서: cell별 island0..3 안에서 prior/component. 좋은 arm만 남기지 않는다.

각 island work=128×(1+32×3)=12,416 full potential contexts.
모두 완료하면198,656 physical evaluations. 한 제안이라도 numerical unresolved이면 그 island만 실패로
보존하고 나머지 사전 job은 남은 cap 안에서 수행한다. 실패 island 재시도는 없다.

potential은 profile 생성에 context당1full, cutoff16/order16,32,64 batch 적분의 모든 padding/node/endpoint를 과금한다.
SMC mutation에는 **개별** m2 상대오차 proxy≤1e-6과 structural gate를 모두 요구한다.
불필요한 tiny proposal라고 clip하거나 나중에 aggregate해서 면제하지 않는다.

## C. 저장과 실패 복구 계약

실행 전 source/config/plan/theory/parent result+audit/seed mapping을 새 source ZIP에 봉인한다.
one-shot 디렉터리에서 atomic non-overwrite 저장을 쓴다.
setup/source freeze/protocol I/O부터 outer try/except로 감싸 failure receipt를 남긴다.
failure receipt 자체가 저장되지 못하면 stderr로만 실패를 알리고 성공으로 처리하지 않는다.
이 시작 단계 I/O failure를 unit test로 주입한다.

각 potential call의 X/logDq/logm2/relative proxy를 incremental SHA로 묶고 call count/digest를 기록한다.
모든 rejected proposal 좌표를 거대한 파일로 중복 저장하지 않는다.
각 stage의 weight/lineage/mutation/jump 요약과 trace digest,
최종X/weights/logm2/logDq/initial lineage/다섯 물리 feature를 보존한다.
full replay하지 않은 island의 모든 proposed payoff를 재검증했다고 주장하지 않는다.

## D. lineage와 mutation gate — nominal ESS와 독립성 구분

각 primary island에서 모두 요구:

1. 초기 ancestor≥16/128, 최종 weighted ancestor ESS≥8, 최대 ancestor 질량≤0.25.
2. 최종 nominal weight ESS≥64/128, 최대 particle weight≤0.05.
3. 정확한 좌표값 기준 distinct X≥96/128.
4. 마지막16stage 전체 mutation acceptance≥0.10.
5. 같은 구간에서 입자별 누적 accepted squared displacement의 중앙값≥0.5.
6. 모든 evaluated proposal의 개별 numerical/structural gate 통과.

initial ancestry는 accepted global move로도 reset하지 않는다.
이는 보수적 개발 기준이고 necessary mixing theorem이 아니다.
초기 ancestry가 낮지만 이동이 충분했다는 사후 이유로 gate를 완화하지 않는다.

## E. 독립 island와 희귀 기여 모드 진단

각 cell/arm의4개 독립 logZ를 ordinary mean으로 집계하고 RSE≤20%를 요구한다.
최대 island normalizer contribution≤0.5, 최대/최소logZ spread≤log4도 요구한다.
네 번뿐인 반복이 rare-mode coverage나 publication confirmation을 보장하지 않는다.
두 arm이 자격을 얻은 경우에만 3.291×combinedSE의 차이 경고를 적용한다.
미자격의 넓은 비반증을 일치 성공으로 세지 않는다.

primary endpoint마다 다음 5개 feature를 계산한다:
log(preceding integrated variance), preceding terminal, log(last base variance), ||X||², residual57.
모든 island에 동등한1/4 질량을 주고 final weight를 사용한다.
pooled weighted mean/std(floor1e-8)로 표준화 후 deterministic weighted K-means(K3,20iteration)를 실행한다.
pooled mass≥0.1인 cluster는 네 island 각각 mass≥0.01인지 확인한다.
이것은 development bank 자체의 기술적 모드 진단이며 held-out clustering 확인이나 누락 모드 부재 증명이 아니다.
mixture component 생성 label을 물리적 rare mode와 동일시하지 않는다.

## F. 사전 판정

- numerical failure가 있으면 `STOP_NUMERICAL_TARGET_UNRESOLVED`.
- resource cap/기술적 실행 오류는 각각 `STOP_RESOURCE_LIMIT`/`STOP_TECHNICAL_FAILURE`로 구분하고
  이를 수학적 실패 또는 모델 성능 반증으로 읽지 않는다. 전역 cap에 도달하면 남은 job을 강행하지 않는다.
- primary island의 lineage/mutation gate 하나라도 실패하면 `STOP_GENEALOGY_OR_MUTATION_UNQUALIFIED`.
- primary island-level normalizer/cluster gate 실패면 `NO_BANK_ADOPTION_ISLAND_OR_MODE_UNRESOLVED`.
- control normalizer 자격 미달 또는 자격 있는 두 arm 간 모순이면 `NO_BANK_ADOPTION_REFERENCE_UNRESOLVED`.
- 전부 통과해도 `BANK_CANDIDATE_FOR_SEPARATE_G_PIPELINE_ONLY`다.

어느 경우에도 이번 bank로 실제G를fit하거나 P2/D3/대규모 학습으로 자동 진입하지 않는다.
다음 G fitting은 성공 branch의 known-answer end-to-end, fresh selection/final/reference를 별도 동결한 뒤 진행한다.
현재 실험은 fixed-precision total-wall 우위 또는 새 모델의 독창성을 주장하지 않는다.

## G. 예산과 감사

production physical cap200,000 / scalar cap800,000,000 / compute wall2400초.
이론상 scalar 최대는 각batch에최대34panel×114nodes×198656≈770M으로 cap 안이다.
source/setup/I/O/end-to-end는 별도 기록하고 warmup을 별도 돌리지 않는다.
test의 물리 계산은 연구 예산과 별도다.

audit physical cap30,000 / scalar cap120,000,000 /wall600초.
모든 saved trace/stage/statistic/lineage/seed/hash/판정을 재집계한다.
16개 island의 final128contexts를 full 재평가해 저장값과 대조한다(2,048physical).
final logm2 오차≤1e-8, logDq 절대오차≤1e-10을 요구한다.
이 최종재평가의 batch padding/roundoff는 원래 제안 batch와 다를 수 있으므로 bitwise equality를 요구하지 않는다.
양 셀 **primary island0** 두 개를 처음부터 끝까지 통째로 replay한다(최대24,832physical).
source/seed가 같아도 wall 값은 같을 필요가 없다. numeric call digests/stages/finalarrays는 같아야 한다.
합계최대26,880physics로 cap 이내다. 실패 island는 동일 stopping point/오류/소비 work를 재생한다.
추가 seed, schedule, scale, particle 수, tolerance 탐색은 이번에 없다.

최종 source hash/full regression/Ruff/Mypy/JSON byte preservation/문서 링크/공개 파일 검토와
이론·기술 제한을 보고서에 남긴다. 성능 실패를 구현 테스트 통과로 덮어쓰지 않는다.

## H. Anchor와 seed 범위

이전 marginal-G result SHA256:
`ecd0a93de8dc90a24a3a3e91972c125178986c573c15a8f1ff6e1c33fcf4afbd`.
기존 IID STOP과모든 원본을보존한다.
namespace `conditional-risk-target-smc-bank-v1`,16island의모든외부stream을
서로및이전marginal-G의264개봉인seed와대조한다.
repository 전체 과거 seed의 망라 검사가 아니며 PRNG 이론적 독립성의 증명도 아니다.
