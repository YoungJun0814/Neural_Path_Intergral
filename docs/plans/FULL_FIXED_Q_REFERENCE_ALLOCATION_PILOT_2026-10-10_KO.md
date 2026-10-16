# 원래 10개 q 전체의 fresh reference allocation pilot

2026-10-10. 이 pilot의 결과를 보기 전에 고정한다. parent0 recovery 결과를
다른 q의 실제 정밀도라고 대신 사용하지 않는다. 다른 pilot/final과 합치지 않는다.

## 개발 anchor와 스케줄 선택

anchor: `results/post_audit/guide_free_block_recovery_v1/session_20261010/result.json`.
SHA256=d42c882bb7e11d0db9c8a3e0a61b6831bd7be9eae40102413451dd0e80e0a24f.
각 cell/target에서 anchor의 observed whole CV²×wall이 작은 guide-free arm을 선택한다.
canonical μ full / risk block, high-eta μ block / risk full. 이 선택은 개발 선택이며
새 방법의 우위 증명이 아니다. 선택을 새 pilot 결과에 맞추어 재변경하지 않는다.

## 전체 grid

- canonical/high-eta ×5개 원래 frozen q의 risk.
- 각 q static-guide IID 8×32768 draw,4096/batch. 640batch.
- 각 q guide-free SMC8whole, envelope proposal-start SMC8whole.160whole.
- μ guide-free SMC cell당8whole(16whole). μ는 q와 무관하므로 q별로 중복시행하지 않는다.
- static μ는 같은 cell의5q risk-IID에서 생성한 독립 draw 전체를 pool한다.
  각 risk는 자기 q의 표본262144개만 사용한다. μ와 M2의 공유 표본 역할을 기록한다.
- whole816units, preparation별도. expected71,876,864 potential calls incl256prep.
- guide-free 1024particles/64quadratic points/8mutations(full.35, block.7), every4stratified.
- guided envelope 1024/64points/4mutations(full/global/block/global), global every2,
  full.35, proposal base, every4stratified. 원래 risk scaledδq 후 마지막δq 나눔.
- final transition mutation없음. source/input/anchor/q/seed/work/checksum을 동결한다.
- fixed new namespace `full-fixed-q-allocation-v1`.
- 이 새로운 pilot의 cap85m potentials/1800초/RSSguard4GiB. 앞선40m pilot의 cap을
  사후 변경한 것이 아니라 원래 전체 q 범위를 측정하기 위한 별도 새 실험이다.
- 모든 whole/batch 완료 후에만 요약. IID moment streaming과 whole-run sensitivity.
- 전체 source/config/q/seed/moment/accounting 및 selected4IIDbatch와
  4SMCwhole(2cells×2mechanisms risk parent0 rep0)을 numerical replay한다.

## Allocation과 다음 단계

원래 allocation 목표 RSE1.5%, safety2를 유지한다. IIDminimum65536/max8388608,
SMCminimum32/max256. μ probability2 / risk30 비교 계약은 다음 production에 유지한다.
production forecast 기본cap160m potentials/3600초이며 새 결과 전에 count를 고정할 수
있는지 판정한다. 단순 quota 부족을 위해 목표/RSE/independence를 낮추지 않는다.
resource cap 확대가 필요하면 **새 생산 프로토콜에 사전 명시하고 별도로 판단**한다.
이번 pilot 자체의72m 비용을 이후 production 비용에서 숨기지 않는다.
warmup/전용 hardware가 아니므로 wall forecast는 practical budget 진단이지 저널 speedclaim이 아니다.

pilot에 근거한 per-q allocation feasible와 실제 fresh production precision은 다르다.
모든 결과에서 전체P1/P2/대규모학습 authorization=false. 성공된 q만 고르는 것이 아니라
전체10q completion/정밀도/guide-free corroboration을 검사해야 한다.
