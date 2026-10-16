# 검증·main 반영 후 다음 실행 우선순위

2026-10-16. 사용자의 요청 순서: 기존 작업 완료 및 검토 → main 커밋·푸시 → 다음 작업.
기존 실패 결과와 accuracy gate는 보존한다. 새 작업은 로컬 노트북에서만 실행한다.

## 1. 현재 확인된 사실

high-η N32 원래 five-q reference 개발 검증은 통과했다. canonical도 all-five-q
risk와 모든 일치 검사는 통과했지만 μ guide-free RSE 1.5066%가 한계1.5%를 넘어
전체 실패했다. 새 finer paired 진단 네 경우는 adjacent difference를 통과했으나
미세 grid μ RSE 6.56–15.38%이고 continuum bias는 별도로 남아 있다.
새 driver-conditioning core의 작은 pilot는 두 셀 모두 reference 일치 한계 미달이다.

## 2. main 반영 전

세 완료 실험의 source/input/seed/whole replay/cost 감사를 수행한다. 전체 pytest,
CI Ruff/Mypy, diff 검사와 artifact 크기를 확인한다. 실패를 성공으로 바꾸지 않는
README·결과 보고서를 포함해 커밋한다. 원격 main의 별도 이력을 먼저 가져온 뒤
non-force merge로 반영하고 push한 remote SHA를 확인한다. 연구 성능 검증과
소프트웨어 CI 성공을 구분한다. 커밋 이름과 trailer에 AI 기여 표시를 추가하지 않는다.

## 3. 최우선: canonical μ의 독립 fresh 재검증

기존 71 whole의 실패를 보존하고 새 namespace에서 μ-only gradient SMC를 수행한다.
고정 할당식 ceil(3 ×71×(1.5066347721619342/1.5)^2)=215 whole.
관측 precision 부족에 대한 safety3 allocation이며 215개 전체 완료 전 판정하지 않는다.
old risk/static/guided 결과는 immutable anchor이고 새로운 μ와 pooling하지 않는다.
기존 μ 자료에 144개를 덧붙이는 방식도 금지한다.

원래 N32 law와 μ, 1024particles/64 quadratic temperature points/4 exact gradient-MH
moves(.35,cap20), every4 stratified, final mutation 없음을 유지한다.
총 forward/backward 각각54,819,840회. cap60m/60m, phase1800초, whole30초,
준비30초, RSS4GiB. 새 stream 이름 `canonical-mu-revalidation-v1`.
geometry 준비는 kernel math만으로 basis-path simulation 0회다.

최종 μ RSE<=1.5%, LOO<=5%, max whole share<=15%와 기존 static μ 대비
simultaneous equivalence10%(comparisons2)를 모두 확인한다. 새 μ의 actual precision과
anchor risk의 과거 통과를 구분한 development inventory만 작성한다.
옛 통합 budget/safety/manifest 또는 저널용 sealed confirmation 통과로 소급하지 않는다.
source/anchor/seed/cost 및 고정 rep0,last whole 재현 감사를 수행한다.

## 4. 두 번째: fine-grid reference의 자원 배분 재설계

목표는 μ와 paired difference 모두의 whole 변동을 줄일 수 있는 설계를 식별하는 것이다.
N4096 두 셀에서 original128particles/64temperatures 결과를 anchor로만 보존하고,
새 128particles/128temperatures 대 256particles/64temperatures를 각8whole 비교한다.
whole32개 전체를 완료한다. bridge power2,4 moves,.35,cap20,every4 stratified 유지.
namespace `paired-fine-resources-v1`.

설계당 whole joint work는64,640 또는63,744회. 총 joint forward/backward2,054,144회,
physical4,108,288회,time-step forward12,620,660,736회.
cap physical5m/backward2.5m, phase7200초, whole240초, 준비30초,RSS4GiB.
동시 실행/노트북 wall 값은 method speedup이 아니다. geometry 준비 simulation0회.

각 셀 μf/μc RSE와 paired SE·LOO·max share·ancestry·actual work를 보고하고,
safety3로 μ RSE1.5%에 필요한 fresh whole count와 예상 timestep/backward 비용을
계산한다. pilot의 작은 difference pass나 낮은 observed variance만으로 P2를 승인하지 않는다.
성능 선택은 새 confirmation의 source/design 동결 이전까지만 허용한다.

## 5. 결과에 따른 분기와 중단

canonical μ가 실패하면 결과를 보존하고 같은 설계에 표본을 무한 추가하지 않는다.
rare contribution/mode 및 variance 안정성을 분석한다. fine-grid 자원 대조도 정확도
allocation이 계산 예산 밖이면, exact all-driver joint conditioning이나 다른 독립 reference를
먼저 설계한다. single-grid driver-conditioning으로 pair가 구현됐다고 주장하지 않는다.

RMS moment-matched law는 별도 target이므로 이번 주요 경로에 끼워 넣지 않는다.
문헌 weak rate를 우리 digital target에 임의로 적용하거나 N32 q를 fine q로 이식해
문제를 해결한 것처럼 보고하지 않는다. continuum 주장을 유지하려면 적용 가능한
weak-error/tail 상계가 필요하며, 없으면 finite-grid 연구로 주장 범위를 제한해야 한다.

위 작업은 reference 및 numerical qualification 회복이다. learned q의 우월성,
fresh bank mode coverage, 총 학습 비용, 독립 학습 반복, sealed confirmation은
여전히 별도 관문이다. 대규모 학습의 무조건적 승인이나 저널 게재 보장은 하지 않는다.
