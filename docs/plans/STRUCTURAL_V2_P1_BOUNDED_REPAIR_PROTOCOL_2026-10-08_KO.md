# V2-P1 단일 제한 repair 사전 프로토콜

작성일: 2026-10-08. 작성 시 새 repair 결과는 미관측이다. 이전 pilot은 [P1 보고서](../reviews/STRUCTURAL_V2_P1_EXECUTION_AND_P2_GATE_REPORT_2026-10-08_KO.md)의 development 근거이며 새 관측과 합치지 않는다.

## 질문과 고정 범위

고정 q 10개·동일 N=32·두 개발 셀을 유지한다. 가이드 없는 SMC에서 입자 수와 bridge/mutation 작업 배분을 바꾸면 whole-run 변동·기여 모드 탐색이 나아지는가? 원래 q, payoff, defensive floor, accuracy/비교 margin은 바꾸지 않는다.

| 설계 | 입자 | 온도 점 | mutation | pCN scale | bridge power | resample 간격 | whole당 SMC potential |
|---|---:|---:|---:|---:|---:|---:|---:|
| repair local-A | 1,024 | 48 | 4 | .35 | 2 | 4 | 189,440 |
| repair local-B | 512 | 64 | 6 | .35 | 2 | 4 | 190,976 |
| 기존 방식의 fresh global-C 대조 | 256 | 96 | 8, 매 2번째 global | .35 | 2 | 4 | 192,768 |

resampling은 모두 stratified. local-A/B는 static guide를 전혀 사용하지 않는다. 두 local SMC potential 비용은 약 0.81% 다르므로 정확한 동일 비용이라 부르지 않고 **near-matched work**로 공개한다. 입자·bridge·mutation이 함께 변하는 두 고정 설계의 대조이며 개별 factor의 causal ablation은 아니다.

## 실행량과 비용

- risk: 두 셀의 parent rep=0 × 세 SMC 설계 × whole run 16회 = 96회.
- μ: 동일 두 셀 × 세 SMC 설계 × 별도 whole run 16회 = 96회.
- IID: 고정 q 10개의 M₂ 및 두 셀의 μ 각각 새 262,144 draw = 3,145,728 draw.
- 전체 작업 24개. SMC whole 실행 192회.
- SMC 자체 potential 36,683,776회, 추가 종료 경로·conditional payoff 진단 114,688회.
- IID를 포함한 전체 평가 계수 **39,944,192회**.
- 상한: 평가 40,000,000회, wall 3,600초, process RSS min(4GiB, 물리 RAM 25%).
- 단일 sequential scientific run, torch/OMP/MKL threads=1. source/config와 이전 pilot SHA를 실행 전에 봉인한다.

종료 진단 비용도 예산에 포함한다. 완료·실패·partial 비용을 기록한다. warmup/test/audit 호출은 scientific production 0회라는 표현에 숨겨 합치지 않고 별도 검증 비용이다. 이전 결과는 삭제하지 않는다.

## 모드·기여 진단

종료 SMC particle의 left-point variance에서 peak-time quartile 4개와 largest-cell share 경계 [.5,.9,.99]를 사용해 16개 사전 고정 geometry 구간을 만든다. fitted clustering이 아니며 입자 수·실제 weighted mass를 저장한다.

각 whole run의 normalizer estimate × 구간의 terminal weighted mass로 unnormalized 기여를 계산한다. risk는 normalizer/δ_q를 사용하므로 μ와 M₂ 기여를 혼용하지 않는다. 구간별 평균·SE는 전체 SMC 실행 간에만 계산한다. 관측하지 못한 구간의 0 기여/0 SE를 missing-mode 배제 인증으로 쓰지 않는다.

leave-one-whole-run-out, 최대 whole 기여, whole/block bootstrap, terminal ESS 및 initial ancestry를 함께 보고한다. ancestry나 acceptance 개선만으로 채택하지 않는다. 전체 증분과 normalizer 산술, 모드 질량·count·seed·source binding을 재감사한다.

## 선택·allocation·출구

risk와 μ는 별도로 두 local 설계의 worst-cell empirical CV²×whole-run wall을 비교한다. 다른 reference 평균과 가까운 설계를 고르지 않는다. pilot은 production에 합치지 않는다.

production precision 목표 1.5%, 안전계수 3, whole 32–256회, IID 최대 8,388,608, total potential 최대 160,000,000·wall 7,200초를 유지한다. count는 새 결과를 보기 전에 고정하는 allocation으로만 결정한다. 한 q의 pass가 다른 q를 대신하지 않는다. μ와 risk의 두 comparison family·10% margin·기존 민감도 기준은 그대로 유지한다.

allocation이 불가능하면 production/G-REF/P2를 보류한다. 가능해도 production의 각 estimand·precision·agreement·sensitivity 통과 전 P2를 열지 않는다. μ reference SE≤method SE/5는 이후 특정 성능 비교에 별도로 적용한다.

이 repair 묶음 이후 같은 셀의 schedule 무한 탐색을 하지 않는다. 결과가 불충분하면 참값 메커니즘/연구 범위/compute 예산 재검토 보고서로 전환한다. 새로운 operator·차원 확대·paid cloud를 이번 실행에 추가하지 않는다.
