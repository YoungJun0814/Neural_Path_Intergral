# FR0 실행 계약: 유한격자 경로 기하의 제한된 검증

2026-10-28. 기준 `main 51e665d`. 기존 P2와 별개인 새 개발 트랙이다.
본 계약은 새 알고리즘의 독창성·효율성 인증이 아니라, **제한된 반증 실험의 착수 계약**이다.

## 1. 주 질문과 현재 허용 범위

> Volterra의 사용 잔차–직후 Brownian 기하를 활용한 제한된 proposal 보정이,
> 같은 정보·표현능력의 대조보다 전체 학습 비용을 포함한 오차–비용을 개선하는가?

정답은 아직 모른다. 일반 IS·M2 항등식, Gaussian covariance 변경, conditional MC는 기존 원리다.
새 기여가 될 수 있는 것은 Volterra 특유의 **정량 구조·실패 조건·반복 가능한 실용 이득**이다.
그중 어느 것도 없으면 모델 조합을 새 이름으로 포장하지 않고 중단한다.

FR0 판정은 `GO_BOUNDED_ORACLE_AND_REFERENCE_FEASIBILITY_ONLY`다.
FR1 기존 검증과 FR2 reference-only pilot까지만 현재 실행 수치를 고정한다.
bank×family 원인 실험은 이 pilot을 본 뒤 별도의 job matrix를 먼저 봉인해야 한다.
FR3/대규모 학습/confirmation을 자동 승인하는 계약이 아니다.

## 2. 고정 estimand와 데이터 역할

- 원래 BLP hybrid, epsilon=1, 금리·배당0, CPU float64.
- digital의 `mu_N=E[g_N]`, put의 `mu_N=E[conditional_put/K]`를 구분한다.
- put/K는 [0,1]이며 가격 복원은 K를 곱한다. call·Greeks·시장 예측은 제외한다.
- `configs/finite_grid_geometry/reset_v1.json`의 D01–D04는 개발 task다.
- C01–C04는 봉인 확인 task다. FR1–FR4의 reference·학습·평가 어디에도 쓰지 않는다.
- confirmation은 후보·allocation digest가 동결되고 별도 예산·분석 계약이 마련된 뒤에만 가능하다.
- 새 코드의 접근 guard는 임의의 다른 스크립트나 과거 모든 task 평가 이력을 통제하는 보안 경계가 아니다.

FR2 pilot은 셀당 MC32×8192, RQMC32×8192, 두 방법·네 셀 총 **2,097,152** payoff 평가다.
FR2 전체 cap8,000,000/3600초에서 차감한다. 대표8unit replay65,536회도 별도 감사 비용으로 더한다.
따라서 pilot 후 FR2의 남은 원인 실험 예산은 감사 포함 시 최대 **5,837,312**회다.
다음 실행기는 이전 receipt를 읽어 누적 잔액을 확인해야 하며, 새 runner마다8백만 cap을 새로 받지 않는다.
FR0–FR4 전체 재설계 기회는 합계1회이며 이번 문서 작성·정확성 bugfix를 과학적 재설계로 세지는 않는다.

## 3. 기준값 pilot의 통계 단위와 판정

- MC는 IID path가 기본 단위이고 32block SE도 별도 진단한다.
- RQMC는 독립 scramble 평균32개가 단위다. Sobol point를 독립으로 세지 않는다.
- 각 stream은 전체 role/task/N/replicate/method를 포함한 기존 SeedKey로 파생한다.
- 같은 g_N과 simulator를 공유하므로 MC/RQMC 일치는 독립 물리 구현 검증이나 모든 blind spot의 배제가 아니다.
- 역정규 CDF의 0/1 endpoint는 에러로 기록한다. clipping·선별 rescrambling은 하지 않는다.
- 각 방법 관측 RSE<=1.25%와, 두 mean 차이의 근사 radius를 포함한20% 상대 margin을 feasibility에 사용한다.
- radius는 4셀 Bonferroni와 df31의 t critical을 쓰는 **근사 진단**이다. rigorous finite-sample CI가 아니다.
- reference-only 셀 선택은 후보 성능과 무관하다. 실패 셀을 유리한 strike로 바꾸지 않는다.
- pilot 수치는 향후 accuracy 확인 reference와 합치지 않는다. 그 단계에서는 새 자료가 필요하다.
- 통과해도 `REFERENCE_FEASIBLE_DEVELOPMENT_ONLY`다. mechanism/accuracy/P2/continuum gate는 그대로 false다.

기본 source/seed/계산 수/결정과 method당 첫 unit을 재생한다. 이는 같은 난수의 감사이며 새 독립 표본이 아니다.
운영 중단·예산 초과·수치 실패는 별도 failure receipt로 남긴다. incomplete prefix를 성공으로 승격하지 않는다.

## 4. 이후 오차·총비용 검증의 의무

단일 whole는 새 부모→bank→fit→selection→freeze→새 final이다.
여러 whole 평균의 SE가 작다는 사실은 한 번 배포한 workflow의 precision이 아니다.
FR4에서는 독립 pilot의 고정 allocation rule과 whole별 자격·실패·fallback 비용을 평가한다.
성공한 whole만 골라 속도비를 계산하지 않는다. 연구용 반복 불확실성·실패율과 단일 실행 오차를 따로 낸다.

final `gp/q`는 full normalized q와 독립 표본을 쓰는 ordinary IS다.
fitting의 weight 정규화와 final self-normalization을 구분한다. q>=.1p는 finite moments 보호이지 정밀도 보장이 아니다.
SMC endpoint의 주변밀도를 알려진 G로 취급하지 않는다. IID bank는 적합·동결한 밀도에서 새로 뽑는다.
event·risk 2×2에서 한 factor만 바꾸며 같은 rank/spectrum/성분 수와 탐색 기회를 유지한다.
frozen-q0 risk-target fitting을 direct M2 최소화라고 부르지 않는다.

## 5. FR0 선행연구 경계 검토

아래는 초록만이 아니라 관련 본문 수식·알고리즘을 확인한 경계다. 완전한 최신 문헌고찰이나 novelty 인증은 아니다.

| 원문·확인 부분 | 기존 내용 | 새 연구에서 추가로 입증할 것 |
|---|---|---|
| [McCrickerd–Pakkanen §2, 식2.3·Table1](https://arxiv.org/html/1708.02563v3) | 조건부 lognormal pricing과 control/antithetic 조합 | 동일 조건부 integrand 위에서 구조 보정의 추가 비용 이득 |
| [Uribe 등 §3.1·3.2, Algorithm1](https://arxiv.org/html/2006.05496v1) | smoothed failure target의 저차원 profile·KL 제어·iCEred | 특정 Volterra 잔차 기하와 M2·총비용의 관계; iCEred를 재발견했다고 하지 않음 |
| [Agapiou 등 §2.1–2.3](https://arxiv.org/html/1511.06196v3) | 정규화 IS의 상대 second moment·오차·ESS 연결 | ordinary normalizing-integral IS 계약을 직접 증명; 그 정규화 estimator 정리를 그대로 이식하지 않음 |
| [Llorente 등 §4.1, 식11–13](https://arxiv.org/html/2201.02432v1) | noisy integrand second moment를 반영한 최적 proposal | conditional residual floor의 일반 원리를 새 정리로 주장하지 않고 구체 Volterra 정량 분석을 추가 |

현재 구조 후보는 알려진 성분의 조합이다. “새 기하의 실질적 효과”는 앞으로 검증할 가설이고,
탑 저널 기여가 이미 확보됐다는 판정은 아니다. 제한된 원인 실험에 들어갈 질문은 있으나,
FR2에서 원인·FR4에서 실용 이득이 없으면 원고 방향을 재선정한다.

## 6. 현재 구현·검토 범위

FR0 schema는 task/law/role/digest, confirmation guard, first-cycle cap, seed namespace를 구현한다.
2N conditional MC/RQMC adapter는 같은 cached terminal 물리량을 사용하고, full evaluator와 oracle로 대조한다.
pilot 자체는 proposal 학습·M2 기하 비교를 하지 않는다. 테스트 통과가 주 가설의 성공을 뜻하지 않는다.
후속 schema는 누적 FR2 비용·모든 job·단일 workflow 자격·multiplicity·최종 실패 정책을 추가 봉인해야 한다.
