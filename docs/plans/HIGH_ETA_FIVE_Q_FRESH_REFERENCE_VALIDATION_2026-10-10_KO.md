# 높은 η의 원래5q 전체: fresh reference validation

2026-10-10. 아래 validation 표본 생성 전에 고정한다. 전체2cell P1이나 P2를
대신 통과시키는 계획이 아니다. 개발 중 선택한 scope이므로 journal confirmation도 아니다.

## Anchor와 선택

`umbrella_reference_pilot_v1/session_20261010/result.json`의 full48whole 완료 및
numerical replay 감사 이후 선택했다. canonical은 free method의 forecast707/777whole로
256cap 초과; high-eta만 local budget 안에서 가능한 이유를 보존한다.
새 validation에서 umbrella mixture weights는 anchor에 동결된 weights를 그대로 쓴다.
설계효율이 참값이라는 가정은 없으며 actual M2를 pilot와 평균내지 않는다.

## 고정 실제 실행

- 원래 high-eta200-k1,N32,original frozen5q. 원래 event/μ/M2를 유지.
- common static-guide IID:40blocks×32768draw=1,310,720draw,4096/batch(320batch).
  같은 새 draw에서 μ와5M2를 계산한다. 모든 q의 density를 별도로 계산한다.
- guide-free full-pCN umbrella69 independent whole-run.
- proposal-start corrected-envelope umbrella52 independent whole-run.
- source/anchor/input/q/weights/seed/체크포인트 동결. 새 `high-family-precision-v1` namespace.
  pilot 및 앞선 parent0 final streams와 합치지 않는다.
- particles1024,64quadraticpoints,every4stratified,final mutation없음.
  free8full-pCN(.35) /guided4(full/global/rank2/global),global every2.
- 준비128path-law evaluations 포함 actual forecast49,685,632evaluations.
  441units+preparation. cap60m/1800초/RSSguard4GiB. IIDbatch5초/SMCwhole30초/prepare30초.
- umbrella endpoint productZhat를 유지하고 riskδj correction 후6vector를 저장한다.
  IID는 normalized full static density를 쓰는 ordinary mean. self-normalization 없음.
- 전체441units 끝나기 전 완료prefix로 pass/fail을 판단하지 않는다.

## 보수적 개발 판정

6targets×3methods 모두RSE<=1.5%, maximum IID draw share<=1%/SMCwhole share<=15%,
LOO block shift<=5%. IID between-block RSE<=1.5×IID draw RSE.
5q risk의3pair comparisons에 original30 Bonferroni count를 유지한다.
μ의 primary static vs guide-free comparison에는 original2count를 유지한다.
10% symmetric relative equivalence를 모두 충족해야 scoped pass다.
guided μ는정밀도/민감도에 포함하되 probability primary pair를 사후 선택하지 않는다.
normal-delta/bootstrap 기반 개발 판정이지 distribution-free tail certificate가 아니다.

source/anchor/input/q/seed/moment/endpoint identity/work/전체요약 재감사.
static first/last block의 firstbatch8192draw와 두 SMC method firstwhole를 replay한다.
replay 비용은 inference count에 합치지 않고 별도 기록한다.

성공 flag는 `high_eta_five_q_development_reference_pass` 하나다. 전체P1/P2/
large-training flags는false를 유지한다. canonical reference·fine-grid reference/relative bias,
learned added value, whole-training/confirmation이 남아 있다.
