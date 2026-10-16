# 위험 안정화·geometry 진단·독립 보조 IS 결과와 갱신 실행계획

작성일: 2026-10-07

범위: [기존 구조 개선계획](MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md)의 위험 평가·geometry 진단·다른 메커니즘 상호 검증. 새로운 model family나 operator를 임의로 추가하지 않았다.

## 1. 결론

**기존 proposal의 드문 영역에 second-moment 기여가 집중될 가능성은 더 구체적으로 관측됐지만, 신뢰할 수 있는 M₂ 기준값과 안정적인 모델 개선은 아직 확보하지 못했다.**

- risk-SMC 3개 설정·부모별 16회 전체 반복을 비교했다. canonical은 어떤 설정도 5개 부모 중 정밀도 통과가 없었고, 높은 η도 설정별 3/5, 2/5, 0/5였다.
- 독립 보정한 proposal geometry 밖으로 직접 q 표본은 약 5%만 나갔다. 반면 risk-SMC의 바깥 영역 M₂ 비중 중앙값은 canonical 91–93%, 높은 η 67–68%였다. 해당 risk-SMC 자체의 불확실성을 함께 공개한다.
- 다른 메커니즘인 auxiliary IS를 구현·실행했다. risk bank로 만든 r의 M₂ 추정 RSE 중앙값은 canonical 약 29%, 높은 η 약 35%로, q 직접 평가와 event-bank 대조보다 낮았다. 그러나 10개 부모 중 3개만 개발용 RSE≤20%를 통과했다.
- 원래 model q는 변경하지 않았다. 보정 mixture qα, 새 rank, 새 operator, 저널용 confirmation은 실행하지 않았다.

이번 개선은 **신뢰성 진단과 독립 위험 estimator 구현**이다. 학술 성능 개선의 완료나 참 위험값의 확정을 의미하지 않는다.

## 2. 실험 A — risk-SMC 안정화와 geometry

### 사전 고정 설계

S0/S1에서 학습해 고정한 `parent-as-is`를 두 셀 × 5개 부모 모두 사용했다. 성능이 좋은 부모만 고르지 않았다. 세 risk 설정은 모두 128 particles, 48 bridge 단계, mutation 2회, 독립 전체 반복 16회다.

| 설정 | bridge power | resampling 주기 | pCN scale |
|---|---:|---:|---:|
| original | 4 | 4 | 0.35 |
| linearized-bridge-frequent | 2 | 1 | 0.35 |
| lower-scale-bridge | 2 | 4 | 0.20 |

전부 stratified resampling이며 replicate당 potential 평가 12,160회다. 설정은 composite controls다. 따라서 변화의 효과를 bridge 또는 resampling 하나의 인과 효과라고 분해하지 않는다. `linearized`는 실제 선형 bridge가 아니라 기존 후보 ID이며, 사용한 함수는 β=(stage/48)²다.

각 설정에는 새 q 표본 65,536개 직접 평가도 포함했다. 전체 potential 평가 **7,802,880회**, 실제 실행시간 **227.24초**. 같은 q에서 schedule 간 차이는 참 M₂의 변화가 아니라 estimator 변동·탐색의 차이다.

### Geometry의 정확한 의미

원래 SMC bank의 원본 geometry를 과거 artifact에서 회수한 것이 아니다. **고정 q에서 독립 표본 8,192개를 생성해 만든 surrogate proposal geometry**다. 이 점을 결과 role에 명시했다.

1. training 표본에서 중심과 PCA rank 4를 계산한다. Gaussian 성분의 중심들을 projected center로 사용한다.
2. projected 최근접 거리와 PCA complement residual²/(d−rank)를 score로 쓴다.
3. 별도 IID q calibration 4,096개에서 score별 upper order statistic을 고정한다. α=0.05를 두 score에 α/2씩 배분한다.
4. 최종 직접 q 표본과 risk-SMC terminal 표본에 같은 고정 score/threshold를 적용한다. 결과를 보고 threshold를 다시 맞추지 않는다.

calibration rank는 `ceil((n+1)(1−α/2))`이며 interpolation하지 않는다. frozen q에서 calibration/test가 IID이고 training과 독립이면, exchangeability와 union bound로 **calibration을 포함한 marginal false-outside 확률 ≤α**다. 한 번 실현한 calibration의 conditional coverage, event/risk-tilted sample coverage, Gaussian support 탐지에 대한 보장은 아니다. 그래서 한 실행의 관측 fraction이 5%를 조금 넘는 것도 이 정리를 위반하지 않는다.

`geometry 밖`은 typical proposal geometry 밖이라는 뜻이다. Gaussian proposal은 전체 공간에 positive density를 가지므로 mathematical support 밖으로 해석하면 틀리다. 바깥 기여가 곧 새로운 mode라는 뜻도 아니다.

### 결과

표는 각 셀·설정의 5개 부모 중앙값이며, 서로 다른 열의 중앙값이 같은 부모에서 나온다는 뜻은 아니다.

| 셀 | 설정 | risk RSE | 직접 q 바깥 표본 | 직접 M₂ 바깥 비중 | risk M₂ 바깥 비중 |
|---|---|---:|---:|---:|---:|
| canonical | original | 43.11% | 5.26% | 5.34% | 92.14% |
| canonical | frequent | 35.04% | 5.35% | 19.43% | 93.25% |
| canonical | lower-scale | 71.98% | 5.49% | 37.97% | 91.34% |
| 높은 η | original | 18.71% | 5.13% | 38.96% | 68.27% |
| 높은 η | frequent | 21.90% | 5.13% | 58.41% | 66.61% |
| 높은 η | lower-scale | 47.72% | 5.01% | 13.13% | 68.47% |

risk의 바깥 영역은 각 whole replicate에서 `Ẑ × weighted_terminal_outside_fraction / δ`로 평가했다. 이는 particle fraction을 직접 M₂로 쓰는 방식이 아니다. fixed geometry에서 표준 SMC unnormalized functional 계약을 적용한다. 바깥/전체의 비율은 두 추정량의 ratio이므로 **unbiased share가 아니다**. 별도 ratio 신뢰구간을 입증하지 않았으며 탐색 지표로만 보고한다.

### 봉인된 판정

원래 q의 보정 실험으로 넘어갈 readiness를 다음과 같이 고정했다.

- 셀별 한 설정이 5개 부모 모두 risk RSE≤20%를 통과해야 한다.
- 그런 설정이 최소 두 개 있어야 한다.
- 각 부모에서 해당 설정들의 M₂ 평균이 `max/min−1≤25%`를 만족해야 한다.

이는 보수적 개발 관문이지 신뢰구간·oracle 인증이 아니다. **두 셀 모두 readiness=false**였다. α 보정은 이 관문을 우회하지 않았다.

## 3. 상황에 따른 전환 — 실험 B: 독립 auxiliary IS

세 risk-SMC 설정을 더 튜닝하는 대신, 기존 계획의 다른 메커니즘 상호 검증을 실제 구현했다. 이는 qα 모델 보정과 구분되는 **위험 estimator의 개발 대조**다.

고정 q의 위험을 정규화된 보조 분포 r에서 평가한다:

\[
Y_r(x)=g(x)^2\frac{p(x)^2}{q(x)r(x)},\qquad
E_r[Y_r]=M_2(q).
\]

log 계산은 `2log g−log(q/p)−log(r/p)`다. r=q이면 원래 squared ordinary IS contribution과 정확히 일치한다. 어떠한 self-normalization이나 estimated normalizer로 나누지 않는다.

### 대조·독립성

- `direct-q`: r=q.
- `event-auxiliary`: g-target SMC bank로 fit한 r.
- `risk-auxiliary`: h=δg²p/q-target SMC bank로 fit한 r.

event/risk bank는 동일하게 4 islands ×128 particles, 48단계·mutation 2회·기존 bridge/resampling/scale을 사용했다. 각 bank의 potential 평가 **48,640회**다. 두 fitting 모두 같은 K≤2 identity-covariance defensive mixture family다.

SMC의 상관된 terminal bank는 **r fitting 전용**이다. final IID r 표본과 SE 단위로 섞지 않는다. 세 r을 모두 고정한 뒤 각 estimator에서 새 131,072개 표본을 평가했다. 원래 q를 다시 fit하거나 변화시키지 않았다.

q≥0.1p, r≥0.1p, 0<g≤1이면 Y_r≤100이다. 이는 finite variance를 보장하지만, 희귀한 큰 기여의 누락이나 충분한 상대 정밀도를 보장하지 않는다. 독립 quadrature·r=q 항등식·full-mixture density와 sample-role 검사를 통과했다.

### 결과

10개 고정 부모, 위험 estimator 30개. 전체 potential 평가 **4,904,960회**, 실행시간 **92.45초**.

| 셀 | estimator | M₂ 추정 RSE 중앙값 | M₂ 추정 평균의 중앙값 | 개발 RSE≤20% |
|---|---|---:|---:|---:|
| canonical | direct-q | 41.99% | 1.14445e−12 | 0/5 |
| canonical | event-auxiliary | 63.01% | 1.03226e−11 | 0/5 |
| canonical | risk-auxiliary | 29.16% | 1.09919e−10 | 2/5 |
| 높은 η | direct-q | 55.17% | 2.27144e−8 | 0/5 |
| 높은 η | event-auxiliary | 47.92% | 4.56940e−8 | 0/5 |
| 높은 η | risk-auxiliary | 34.51% | 1.00563e−7 | 1/5 |

이 RSE는 **사건확률 mean(X)의 RSE가 아니라 M₂ estimator mean(Y)의 RSE**다. 이전 사건확률 RSE와 숫자만으로 비교하면 안 된다.

risk bank가 다른 대조보다 나은 탐색적 precision 신호를 보였지만, 3/10만 관문을 통과했다. 직접 q·event r에서 작은 M₂가 관측됐다고 참 위험이 작다고 결론 내리지 않는다. risk r도 큰 기여를 놓칠 수 있고 fitting 반복의 안정성은 미검증이다. 서로 다른 부모의 true M₂가 다르므로 표의 median M₂를 단일 공통 참값으로 해석하지 않는다.

**선택한 model winner는 없다.** 이번 실험은 위험 평가 방향의 개발 근거이며 사건확률 성능·training-inclusive 효율·새 학술 기여의 입증이 아니다.

## 4. 기술적·이론적 재검토

- calibration과 final 표본의 seed 역할을 분리했다. 감사기는 동일 seed로 training/calibration을 재생성해 PCA geometry와 threshold를 재검증한다.
- restricted SMC normalizer를 resampling/mutation 없는 oracle에서 raw prior MC restricted integral과 대조했다.
- auxiliary 위험 적분은 독립 Gaussian quadrature와 비교하고 r=q 항등식을 검증했다.
- source ZIP, 고정 q·r digest, source artifact SHA256, 전체 seed ledger, whole-replicate risk 통계, IID batch moment, matched training work, precision flag, 누락 비교군, readiness를 감사한다.
- geometry threshold 변조·cell decision 누락·auxiliary RSE 변조·budget 실패를 거부하는 검사를 추가했다.
- 실제 payoff 호출을 집계하므로 수치 실패로 중단된 호출도 계산량에서 빠지지 않는다. 실패 시간도 보존한다.
- 시간에는 q 표본 생성·geometry/fitting·potential/density 계산이 포함된다. import/source ZIP 생성/결과 I/O/과거 q 학습은 제외한다. controlled fixed-precision deployment benchmark가 아니다.
- 전역 무오류 보장, continuum bias bound, Volterra 특화 novelty는 이번 변경으로 해결되지 않았다.

## 5. 갱신된 개선 실행계획

기존 계획의 범위 안에서 우선순위를 다음과 같이 구체화한다. 같은 셀에 무한 튜닝하지 않는 원칙은 유지한다.

### D0 — 완료: 불확실성과 바깥 기여를 드러내기

이번 두 실험과 source/seed 감사로 완료했다. 신뢰도 부족은 숨기지 않고 artifact에 남겼다. SMC 스케줄의 우승 설정은 채택하지 않았다.

### D1 — 다음 제한된 묶음: auxiliary 위험 estimator의 안정성

**후속 갱신:** 전체 fresh-fit 안정성 부분은 7차 구조 기반 safeguard 실험에서 10/10 고정 개발 q가 통과했다. 그러나 static-only보다 learned addition의 효율 우위는 없었으며, 아래 7번의 독립 메커니즘 reference 관문은 여전히 미해결이다. 따라서 D1 전체 또는 D2 readiness를 완료로 바꾸지 않는다. 상세 실패·개선·검증은 [전체 보조 재학습 안정성 보고서](AUXILIARY_WHOLE_FIT_STABILITY_2026-10-07_KO.md)를 따른다. 아래 최초 설계와 음성 결과는 이력을 위해 보존한다.

1. model q는 계속 고정한다. 현재 개발 셀을 confirmation으로 승격하지 않는다.
2. risk/event 보조 bank 전체 학습을 각 q에서 새 seed로 반복해, 한 r에서의 final 변동과 r 학습 변동을 구분한다.
3. 최초 대조는 baseline/event-r/risk-r 세 방법으로 유지한다. 새 covariance rank/operator를 동시에 추가하지 않는다.
4. fitting·selection·final stream을 분리한다. 현재 final을 재사용해 r이나 sample count를 선택하지 않는다.
5. throughput pilot에서 bank/final의 비용을 따로 측정해 총 상한을 봉인한다. 상한을 넘으면 unresolved로 남긴다.
6. 원래 계획의 training-inclusive 개선 신호와 precision을 함께 평가한다. 낮은 sample RSE나 bank ESS만으로 승리 판정하지 않는다.
7. 최소 두 독립 위험 메커니즘의 충분한 정밀도·상호 일치가 확보되지 않으면 M₂ oracle 또는 저널용 sample-complexity 기준으로 쓰지 않는다.

현재 risk-r RSE 29–35% 중앙값은 필요 표본 수의 확정 forecast가 아니다. 단순 n 비례로 5% 목표 비용을 확정하지 않는다. 추가 budget은 새 pilot과 전체 r 학습 반복의 나쁜 경우까지 포함해 산정한다.

### D2 — 조건부: risk-bank 부분 보정 qα

D1의 독립 위험 평가와 별도 selection/final이 준비된 경우에만 원래 계획의 qα=(1−α)q0+αr을 실행한다.

- α=0 baseline을 보존한다. α 후보는 사전 고정 집합에서만 비교한다.
- event-bank correction을 동등 training work·capacity·selection budget으로 대조한다.
- normalized r와 full qα density를 사용한다. natural 질량을 유지한다.
- `M₂(qα)≤M₂(q0)/(1−α)`는 악화 상한일 뿐 개선 보장이 아니다.
- bank 기하에서 개선된 것과 실제 최종 사건확률 정확도·위험·총비용 개선을 구분한다.
- 보정에 개선 신호가 없으면 새 차원·operator로 덮지 않고 제한된 family/문제 범위를 재검토한다.

**현재 D2 readiness는 미확보다. 이번 turn에서 qα는 실행하지 않았다.**

### D3 — 기존 reference·whole-fit·confirmation 관문

사건확률 reference의 SE 비율, 두 방법의 동일 accuracy qualification, 전체 학습 5회→추후 20회, 미사용 task, controlled fixed-precision 총비용, novelty/theory gate를 그대로 유지한다. 위험 estimator를 개선했다고 이 조건을 자동 통과한 것으로 처리하지 않는다.

## 6. 파일과 검증

- [geometry 구현](../../src/path_integral/path_geometry_diagnostics.py), [위험 통계·auxiliary 적분](../../src/path_integral/conditional_second_moment.py)
- [risk/geometry 실행·감사기](../../experiments/post_audit_r2_risk_geometry.py), [설정](../../configs/post_audit/r2_risk_geometry_stability_v1.yaml), [결과](../../results/post_audit/r2_risk_geometry_stability_v1.json)
- [auxiliary 실행·감사기](../../experiments/post_audit_r2_auxiliary_risk.py), [설정](../../configs/post_audit/r2_auxiliary_risk_crosscheck_v1.yaml), [결과](../../results/post_audit/r2_auxiliary_risk_crosscheck_v1.json)
- [geometry 검사](../../tests/test_r2_path_geometry.py), [auxiliary 검사](../../tests/test_r2_auxiliary_risk.py)

두 결과의 source·seed·산술 감사는 통과했다. geometry training/calibration replay도 통과했다. 전체 회귀의 최종 확인은 아래에 기록한다.

- 전체 pytest: **1,111개 통과**, 172.44초.
- 전체 Ruff: 통과. CI 범위 mypy: **169개 소스 파일 통과**.
- 커밋·푸시: 하지 않음.
