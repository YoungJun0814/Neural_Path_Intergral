# Fine-only SMC 초기 bridge 고정비용 대조: 결과와 재검토

2026-10-20. 상태: **48회 실행 완료. 부분적인 개발 신호는 있으나 독립 기준 자격은 실패.**
커밋·푸시는 하지 않았다. 원래 q나 신경망 구조를 바꾼 작업이 아니다.

## 1. 핵심 결론

초기 temperature 해상도를 높인 β³ schedule이 사전 고정한 후속 검토 후보
조건을 두 셀에서 충족했다. canonical의 관측 whole-run RSE는11.21%→5.93%로
낮아졌다. 그러나 high-η는11.53%→11.26%로 개선이 작다. **모든6개 설정이
1.5% 정밀도·민감도 결합 gate에 실패했고, 모든 cross-mechanism agreement도
불확실성으로 미해결이다. P1/P2/대규모 학습은 여전히 승인하지 않는다.**

β⁴는 계보와 minimum ESS가 개선돼도 canonical RSE가12.50%로 악화됐다.
따라서 “더 촘촘한 초기 β / 더 높은 ESS / 더 많은 계보 = 정확한 estimator”는
성립하지 않는다. 8개 whole만으로 모집단 분산 감소나 편향을 확정하지 않는다.
후반 β 간격 확대,mode 탐색 부족,작은 표본 변동은 가능한 설명이지만,
이번 대조만으로 어느 하나를 악화의 확정 원인으로 지정할 수 없다.

## 2. 실행 전 고정과 구현

- [실행 계획](../plans/FINE_SMC_EARLY_BRIDGE_CONTRAST_2026-10-20_KO.md)
- [수학적 계약](../theory/FINE_SMC_EARLY_BRIDGE_CONTRACT_2026-10-20_KO.md)
- runner: `experiments/post_audit_fine_smc_early_bridge.py`.
- 결과: `results/post_audit/fine_smc_early_bridge_v1/session_20261020/result.json`.

모든 arm은 N4096,8192차원,128 particles,64 temperature points,stage당2 moves,
scale.35,proposal drift cap20,4stage마다 stratified resampling을 유지했다.
변경은 β_k=(k/63)^p의 p∈{2,3,4}뿐이다. 실제 resampling β와 초·후반
mutation 배분도 함께 바뀌므로 분리된 mutation 효과로 해석하지 않는다.

2cells×8independent triplets×3powers=48whole을 모두 실행했다.
각 triplet 내 동일 난수는 의도적인 CRN이며 arm 간 독립을 가정하지 않았다.
과거 결과와 preflight는 새 final seed와 분리했다. 중간 결과를 보고 표본·seed·
threshold를 바꾸거나 실패 실행을 교체하지 않았다.

## 3. 확률 추정 결과

표의 RSE 단위는 resampling된 particles가 아니라 **8 independent whole**이다.
mean은 exp(mean(log Zhat))가 아닌 ordinary arithmetic mean(Zhat)이다.

| 셀 | power | 확률 추정 평균 | whole RSE | max whole share | max LOO shift | 결합 gate |
|---|---:|---:|---:|---:|---:|---|
| canonical | 2 | 1.4867813e-8 | 11.2096% | 19.8253% | 8.3717% | 실패 |
| canonical | 3 | 1.5123591e-8 | 5.9278% | 16.1059% | 4.1210% | 실패 |
| canonical | 4 | 1.0733621e-8 | 12.4990% | 17.9092% | 7.5346% | 실패 |
| high-η | 2 | 8.8651600e-7 | 11.5335% | 18.0113% | 6.8161% | 실패 |
| high-η | 3 | 9.6660681e-7 | 11.2558% | 18.3959% | 6.7382% | 실패 |
| high-η | 4 | 8.5732043e-7 | 8.7188% | 17.4157% | 5.6179% | 실패 |

결합 기준은 RSE≤1.5%,max whole share≤15%,max LOO≤5%다.
canonical/p3만 LOO 항목을 통과했다. 그것만으로 전체 통과로 표시하지 않았다.
관측 RSE는 작은 개발 표본의 통계량이며 미관측 rare contribution을 보장하지 않는다.

## 4. 계보·ESS 진단: 무엇이 개선됐고 무엇이 아닌가

각 milestone은 해당 β 이하 마지막 기록 stage를 사용한다. 실제 β는 primary
result에 보존했고 p마다 다르다. 아래는 whole8개의 median이다.

| 셀 / power | β≤.01 계보 | β≤.04 계보 | β≤.1 계보 | 최종 계보 | whole별 minimum ESS 범위 /128 |
|---|---:|---:|---:|---:|---:|
| canonical /2 | 71.5 | 20 | 15 | 9 | 33.08–53.78 |
| canonical /3 | 55.5 | 26.5 | 16 | 11 | 60.62–73.30 |
| canonical /4 | 63 | 26.5 | 20 | 12 | 64.87–76.92 |
| high-η /2 | 64.5 | 22.5 | 16.5 | 10.5 | 41.47–51.54 |
| high-η /3 | 52 | 28.5 | 19 | 14 | 64.80–77.94 |
| high-η /4 | 58 | 29 | 25.5 | 18.5 | 74.57–83.05 |

β³에서 β≤.04 및 final 계보는 baseline보다 늘었다. 하지만 β≤.01에서는
오히려 줄었다. 기록상 첫 계보 반감 β도 canonical/p2=.016125에서
p3=.006911로 앞당겨졌다. 이것은 작은 β에서 resampling을 더 자주 경험하는
calendar 효과와 분리되지 않는다. “최초 collapse를 늦췄다”는 주장은 틀리다.

최종 label은 전체 설정에서7–23/128에 불과하다. MH mutation은 새 label을
생성하지 않으므로 이 수는 독립 최종 표본 수와 같지 않다. 반대로 label이
많다고 원격 mode를 찾았다는 뜻도 아니다. whole별 평균 수락률은 약81.84–85.31%로
여전히 높지만 정확도 gate는 실패했다.

후속 후보 기준은 두 셀 모두 RSE 감소,β≤.04 median ancestry 비감소,
final median ancestry 비감소였다. **p3만 통과**했다. p4는 canonical RSE가
baseline보다 높아 탈락했다. 이는 후보 선택 규칙이지 p3의 통계적 우월성 증명이 아니다.

## 5. 독립 reference 비교

직전 fresh frozen-bank IID 기준은 canonical1.3765348e-8(RSE.902%),
high-η9.8227630e-7(RSE.907%)다. 새 SMC 난수와 분리돼 있다.
6회 Bonferroni-adjusted symmetric relative-equivalence(10% margin)의
관측 차이+불확실성 upper bound는 다음과 같다.

| 셀 | p2 | p3 | p4 |
|---|---:|---:|---:|
| canonical | 37.33% | 25.19% | 57.31% |
| high-η | 40.69% | 31.40% | 36.61% |

모두10%보다 높다. 모든 adjusted interval이 equivalence 영역과 겹치므로
상태는 `unresolved_equivalence_uncertainty`다. “편향을 입증했다” 또는
“동일 정확도를 확보했다”고 해석하지 않는다. 두 방법은 sampling mechanism이
다르지만 finite-grid physics/CDF 구현은 공유한다는 한계도 그대로다.

## 6. CRN paired 평균 비교

각 셀의 동일 replicate를 정렬해 (mean(Z2)−mean(Zp))/mean(Z2)를 계산했다.
4회 다중비교 보정한 delta-method 진단이다. 서로 다른 arm을 독립으로 처리하지 않았다.

| 셀 / 비교 | 상대 평균 차이 | paired SE | adjusted upper absolute difference |
|---|---:|---:|---:|
| canonical /2−3 | −1.7203% | 11.5696% | 30.6178% |
| canonical /2−4 | 27.8063% | 8.7532% | 49.6693% |
| high-η /2−3 | −9.0343% | 17.3524% | 52.3754% |
| high-η /2−4 | 3.2933% | 8.5757% | 24.7129% |

모두10% 평균 동등성을 확인하지 못했다. 이 검사는 평균 차이 진단이며
variance-ratio 유의성 검정이 아니다. CRN이 항상 좋은 coupling을 만든다고 주장하지 않는다.

## 7. 비용·재현 감사

- main48whole: forward/backward 각768,000.
- separate preflight6whole: 각24,000. inference에 포함하지 않음.
- 생산 총55journal units: **forward792,000 / backward792,000**.
- 생산 phase: **615.882초**. 준비·phase I/O를 포함하고 CLI import/source archive와
  사후 replay·회귀테스트는 제외한다. 정확도 자격이 다른 방법의 기록이므로
  fixed-precision total-wall speedup이 아니다. 기존 bank 학습 비용을 포함했다고 주장하지 않는다.
- 대표6whole(rep0,2cells×3powers) replay는 추가96,000forward/backward.
- result SHA256: `0d9b17a64656b95614055cbacbd69803e9d60b537cd509c0ea1c03562ecdfb2b`.

감사는 source archive/config/parent SHA,현재 numerical runtime hash,
55개 start/finish membership·cap·work·seed·CRN identity·8192차원·threads,
전체 summary 재계산,대표6개 whole의 normalizer와63stage 재현을 확인한다.
최종 감사·회귀테스트 상태는 아래 검증 기록에 기재한다.

## 8. 다음 우선순위와 중단선

1. **해결된 것으로 간주하지 않을 것:** 독립 fine-reference 정밀도·일치,
   bank ancestry/ESS,all-q M2,새 bank ablation,confirmation,mesh/continuum,P2.
2. β³는 다음 개발 대조의 baseline 후보로만 유지한다. 이번 결과로 최적
   schedule을 확정하거나 기존 실패를 대체하지 않는다.
3. 다음 작은 실험은 **β³ 고정 + resampling calendar만 분리**하는 대조가 우선이다.
   예: every4 대 every8. 같은128particles/64points/2moves로 physical work는
   동일하게 두고,skip stage의 weights를 운반하는 기존 core를 유지한다.
   resampling 횟수 감소가 계보에 주는 효과와 weighted ESS 악화를 함께 확인한다.
   난수 호출 수가 달라지므로 이번 CRN 위치 정렬을 그대로 주장해서는 안 된다.
   별도 plan/seed/analytic oracle/새 whole을 먼저 고정해야 한다. 아직 실행하지 않았다.
4. calendar 조정도 정확도를 개선하지 못하면 power를 계속 검색하지 않고
   mutation의 실제 이동·mode 탐색 geometry를 검사한다. 새로운 kernel은
   원래 πβ invariant성·역전이 보정·known-normalizer 테스트를 먼저 통과해야 한다.
5. 단순 반복 증설은 우선하지 않는다. 이번 p3 관측 RSE를 safety3로 외삽하면
   canonical375whole,high-η1352whole,총27,632,000forward/backward가 필요하다.
   작은 pilot 기반 예측일 뿐 보장이 아니며 **이 실행을 승인하거나 시작하지 않았다.**
6. reference 자격 이후에만 새 bank·새 final표본의 rank/basis/KL-M2 ablation,
   두 accuracy-qualified 방법의 offline+fit+selection+inference total-wall 비교를 진행한다.

## 9. 최종 검증 기록

실행 전 관련 단위·통합 테스트21개,CI 범위 Ruff,새 runner 직접 Mypy가 통과했다.
초기 analytic oracle의 unbounded potential은 확률형 callback 계약에 맞지 않아
bounded Gaussian closed-form oracle로 수정했다. 실제 finance target/core를 바꾸지 않았다.
최종 검증:

- CI와 동일한 pytest command(`-q --cov=src --cov-report=term-missing`):
  **1,498 passed,379.64초**,source line coverage84%.
- CI 범위 Ruff 통과. CI 범위 Mypy204source files 통과.
- 새 runner 직접 Mypy(`--follow-imports=silent`)1source file 통과.
- 감사 상태: `fine_smc_early_bridge_source_seed_crn_stage_work_replay_pass`.
  대표6whole의 normalizer 및63stage가 정확히 재현됐다.
- source snapshot1,076entries의 archive binding을 확인했고,봉인된 numerical
  `src/`·`experiments/` runtime의 현재 변경은0건이다.
- P1/P2/large-training flag 모두 false. `git diff --check` 통과.
- HEAD `9cafe59b689ce8a015a7e79fea5f8ba2a8dba00c` 유지,staging 비어 있음.
  커밋·푸시는 하지 않았다. 이 단계의 코드·문서·결과는 로컬 변경이다.
- 기존 requests 의존성 경고는 남아 있다. frozen 실험 중 버전 업그레이드나
  범위 밖 legacy helper 정리는 하지 않았고,그 문제까지 해결했다고 주장하지 않는다.

테스트와 replay 통과는 미관측 tail이나 모든 수학적·수치적 오류의 부재를
증명하지 않는다. 이번 결과로 논문 제출 준비가 끝났다는 주장도 하지 않는다.
