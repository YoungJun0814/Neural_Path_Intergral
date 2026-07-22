# G11 V6 구현·이론 감사 현황

작성일: 2026-07-23

상태: 공통 코어와 논문용 실행·감사 인프라 구현 완료, 대규모 qualification과
untouched confirmation은 아직 미실행

## 1. 현재 모델을 한 문장으로 설명하면

현재 모델은 하나의 신경망 예측기가 아니라, rough Bergomi의 희귀 하락 사건
확률을 목표 오차까지 계산하기 위해 `crude MC`, `DCS-SLIS`, `Hybrid DCS-MGI` 중
하나를 reference를 보지 않는 pilot으로 선택하고, 완전히 새로운 final random
stream으로 정확한 finite-grid 추정치를 만드는 **적응형 경로공간 중요도 샘플링
정책**이다.

핵심 수학은 양자역학적 Feynman 진폭이 아니라 Girsanov measure change,
defensive Gaussian mixture, Gaussian conditional integration, Rao--Blackwellization,
fine/coarse coupling과 telescoping이다. `path integral`은 경로공간 기대값을
제어된 확률측도 아래에서 계산한다는 의미로 사용한다.

## 2. 구현된 실행 흐름

```text
rare-cell calibration
        -> independent reference bank
        -> crude screening pilot
        -> reference-blind rarity/work router
        -> crude / DCS-SLIS / capped Hybrid selector
        -> frozen integer achieved-RMSE allocation
        -> independent final samples
        -> training-inclusive work artifact
        -> in-memory audit + offline JSON-only audit
        -> paired power plan
        -> outcome-blind freeze
        -> untouched confirmation
        -> independent Linux reproduction audit
```

Router와 selector가 본 pilot sample은 final mean에 재사용되지 않는다. 따라서
선택이 무작위여도 frozen pilot sigma-field에 조건부로 각 final estimator가 같은
`p_L`을 추정하며, tower property로 무편향성이 유지된다.

## 3. 이번 구현에서 닫힌 기술적 오류

### 3.1 Reference leakage

Router 입력 schema에는 reference probability가 존재하지 않는다. 저장된 routed
artifact는 성공 횟수, exact binomial interval, crude/DCS work interval, Hybrid
기회비용과 router config를 보존한다. 오프라인 감사기는 production router를
호출하지 않고 결정을 다시 계산한다.

### 3.2 Pilot/final 재사용

준비 객체에는 final seed가 들어갈 수 없다. final 실행은 별도의 `final` role을
추가하고, 오프라인 감사기는 준비 ledger가 final ledger의 부분집합인지, 새 key가
모두 `final` role인지 다시 검사한다.

### 3.3 Pure CEM의 거짓 bounded interval

Pure Gaussian-shift CEM likelihood는 경로별 deterministic upper bound가 없다.
따라서 pure CEM에는 asymptotic interval만 사용하며 Hoeffding bound를 붙이지
않는다. zero-variance pilot fallback은 allocation heuristic일 뿐이고, 실제
achieved-RMSE gate가 반드시 별도로 통과해야 한다.

### 3.4 Defensive bound의 잘못된 weight

Bound는 mixture의 최소 weight가 아니라 zero-shift expert의 실제 총 weight
`delta_0`에서 `1/delta_0`로 계산한다. 현재 V6 proposal은 zero expert를 index 0으로
명시한다.

### 3.5 비용 누락

Training, screening, routing, selector profiling, allocation pilot, final sample,
checkpoint, 실패와 retry를 서로 다른 work category로 보존한다. 비교 endpoint는
예측 비용이 아니라 실행된 총 operation work를 사용한다.

### 3.6 Confirmation hash의 순환 오류

실행 결과 hash는 실행 전에 알 수 없으므로 사전 동결 대상이 될 수 없다.
Confirmation freeze는 baseline/policy 실행 config, manifest, reference, power plan,
audit config hash를 사전에 고정한다. 결과 artifact는 실행 뒤 독립 audit가 source
hash로 결속한다. Linux reproduction도 같은 원칙으로 결과가 아니라 reproduction
실행 protocol을 사전에 고정한다.

### 3.7 Reference estimand drift

같은 `cell_id`만 확인하던 경로를 강화했다. Baseline과 policy는 reference artifact에
저장된 전체 cell 정의가 confirmation manifest의 `cell.to_dict()`와 정확히 같지
않으면 실행을 거부한다.

## 4. 독립 감사 범위

`g11_v6_result_audit.py`는 저장된 JSON만 읽고 다음을 production preparation 및
execution helper 없이 재계산한다.

- direct와 Hybrid의 continuous allocation과 정수 ceiling/floor;
- Hybrid design-variance target을 만족시키는 추가 integer allocation;
- 준비 hash, policy hash, result hash;
- operation-work 합계와 ledger hash;
- censoring condition;
- final term mean, variance, sampling variance, standard error와 asymptotic interval;
- 준비/final seed ledger와 role 분리; 그리고
- 저장된 router 결정.

테스트는 allocation count, work, estimate와 seed를 각각 변조하며 모든 변조가
거부되는지 확인한다.

## 5. Route A 통계 계약

Primary comparator는 task-tuned pure CEM이다. 동일한 `(cell, cluster)` pair가 하나라도
빠지면 power 또는 confirmation 분석을 거부한다. Primary effect는 cell별 log work
ratio를 cluster 안에서 동일 가중 평균한 값이다.

`log(total work_CEM / total work_V6)`

주요 claim에는 다음이 동시에 필요하다.

1. 모든 run complete;
2. resource censoring 없음;
3. independent audit 통과;
4. 각 method/cell의 target-attainment lower bound와 RMSE upper diagnostic 통과;
5. equal-cell-weighted geometric work ratio의 one-sided 95% lower bound가 1 초과;
6. qualification에서 계산한 powered cluster 수 이상 실행; 그리고
7. 사전 동결된 protocol hash 일치.

현재 development config의 RMSE upper gate는 prespecified cluster bootstrap이다.
이는 일반적인 통계적 근사이며 distribution-free finite-sample theorem은 아니다.
최종 protocol 전 blinded simulation으로 coverage를 확인하거나 더 보수적인 gate로
교체해야 한다.

## 6. Route B에서 새로 증명된 부분

Terminal affine slope는

`B_n=sqrt(1-rho^2)sqrt(dt) sum_i u_i sqrt(V_i)`

이다. 양의 unit direction에 대해 `c_n=sqrt(dt)sum_i u_i`라 놓으면 Jensen inequality와
Gaussian MGF로 모든 `q>0`에 대해

`E[B_n^(-q)] <= [sqrt(1-rho^2)sqrt(xi)c_n]^(-q)`

`* exp(eta^2 T^(2H)(q/4+q^2/8))`

를 얻는다. BLP historical kernel cell-average의 variance가 exact kernel variance보다
크지 않다는 Jensen bound를 사용한다. 현재 두 구간 control direction은
`inf_n c_n>0`를 명시적으로 만족하므로 terminal inverse-slope moment obligation은
해소된다.

Defensive mixture에서는 `L<=1/delta`이므로

`E_Q[L^2 B_n^(-q)] <= delta^(-1) E_P[B_n^(-q)]`

도 성립한다. 상세 증명은 `docs/theory/G11_V6_TERMINAL_SLOPE_THEOREM.md`에 있다.

## 7. 아직 증명되지 않은 부분

다음은 구현이나 경험적 plot으로 대체할 수 없는 실제 연구 과제다.

- exact BLP fine/coarse coupling에서 terminal intercept와 slope 차이의 `L^p` rate;
- ratio localization에 필요한 numerator envelope moment;
- 위 rate를 넣고 `kappa(h)`를 최적화한 terminal DCS correction theorem;
- continuous target을 주장할 경우 weak-bias exponent;
- cost exponent와 함께 도출하는 MLMC complexity;
- barrier active-time bad event와 fine-only monitoring mesh-enrichment rate.

따라서 현시점의 정직한 이론 claim은 “finite-grid exact estimator + terminal
inverse-slope negative moments”이다. “unconditional rBergomi complexity theorem”이나
“barrier theorem”은 아직 주장하면 안 된다.

## 8. 실행된 검증

- 연구 코드 범위 Ruff: 통과;
- 전체 pytest: `464 passed` (freeze/hardware 모듈 포함);
- 실제 rBergomi baseline smoke: 통과;
- 실제 rBergomi routed-policy smoke 및 offline audit: 통과;
- 실제 rBergomi Route B diagnostic smoke: 통과;
- constant-volatility inverse-moment equality oracle: 통과; 그리고
- 두 해상도 piecewise direction mass invariance: 통과.

저장소 루트 전체 Ruff는 이번 연구 변경과 무관한 기존 보조 visualization scripts의
49개 lint issue 때문에 실패한다. `src`, `experiments`, `tests` 범위는 clean하다.

## 9. 아직 실행하지 않은 계산

다음은 코드가 없어서가 아니라 실제 계산·통계 증거가 아직 없어서 남아 있다.

1. `H={0.05,0.12,0.30}`, terminal/barrier, `1e-2/1e-3/1e-4` full calibration;
2. 모든 accepted cell의 independent reference bank;
3. 여러 qualification cluster에서 세 baseline과 V6 policy의 실제 achieved-RMSE;
4. observed paired log-work variance와 power/resource forecast;
5. passing power result가 있을 때만 outcome-blind confirmation freeze;
6. untouched confirmation;
7. JSON-only independent audits;
8. 별도 protocol/seed를 사용한 clean Linux reproduction; 그리고
9. 남은 Route B proof obligations.

`1e-4` reference와 powered multi-cluster confirmation은 노트북에서 매우 오래 걸릴 수
있다. Laptop은 smoke, 한두 cell qualification, resource profiling과 theorem
diagnostic에 적합하다. External compute는 router/selector gate와 resource forecast가
통과한 뒤 cluster 병렬화 용도로만 사용한다.

## 10. 현재 논문 수준 판단

코드와 연구 설계는 박사과정 연구 인프라 수준이다. 그러나 논문 contribution은
실행된 결과와 완성된 theorem으로 판단되므로 아직 박사급 논문 “완성본”은 아니다.

- Route A가 통과하면 training-inclusive rare-event computational paper가 가능하다.
- terminal coefficient/rate theorem까지 완성되면 Route B 이론 논문이 가능하다.
- 두 경로가 모두 통과하고 confirmation/Linux reproduction까지 완료되면 SIFIN/SISC
  수준의 강한 후보가 된다.
- Mathematical Finance급 주장은 추가 금융적 insight와 model-level theorem의 깊이가
  필요하다.

실패 결과도 숨기지 않는다. V6가 CEM보다 느리면 superiority claim을 중단하고,
conditional smoothing의 유효 영역과 negative benchmark를 보고한다.
