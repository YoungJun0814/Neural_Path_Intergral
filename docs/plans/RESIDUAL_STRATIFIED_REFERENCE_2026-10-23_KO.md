# 다음 단계: 잔차57 stratified-IS로 독립 reference geometry 점검

2026-10-23 America/New_York. 이는 **아직 실행/자격 획득하지 않은 후속 설계**다.
계보 calibration과 과거 STOP을 소급 수정하지 않는다.
실행 시 새 source/config/seed를 동결하고 known-answer 테스트부터 통과해야 한다.
현재G 학습/P2/새 proposal 채택을 이 문서로 승인하지 않는다.

## 1. 우선 선택 이유와 대안

가장 먼저 고칠 것은 모델 표현력이 아니라 같은 M2(q)에 대한 독립 기준의 부재다.
추가 SMC/새 schedule은 계보와 common missed mode 문제를 공유한다.
AIS/no-resampling은 유용한 대조이지만 같은 annealing geometry를 공유하므로 첫 독립 기준으로는 후순위다.
RQMC만 적용하면 smoothness/effective dimension/희귀 기여 포착 문제가 남는다.
raw p/q IID는 이전에 정확도가 부족했으므로 단순 재실행하지 않는다.

현재 고정373성분 q는 **모든 성분의57/61/63 mean이 정확히0이고 covariance가identity**다.
61은 이미 적분하며63은 payoff 미사용이다.57은 실제 used variance/payoff에 영향을 준다.
따라서57의 Gaussian tails를 구간별로 직접 보는 방법을 첫 bounded reference로 선택한다.
이는 새로운 모델이나 완전한 다차원 rare-mode 탐색법이 아니다.

## 2. 같은 estimand의 두 identity

`Dq=q_X/p_X`, `m2=E[g²|X]`, `M2(q)=∫ p_X m2/Dq`다.
X는61/63을 제외한62좌표이며57은 포함된다.

사전 구간 A_b는57의 값에 대해
`(-∞,-6],(-6,-4],(-4,-2],(-2,0],(0,2],(2,4],(4,6],(6,∞)`다.
경계점 확률은0이다. π_b는 표준정규의 구간 확률이며,8개 구간이 전체 실수선을 덮는다.

q의 모든57 mean0/identity covariance를 먼저 검증해야 q57=phi이고 다른좌표와 독립이다.
그때 다음 두 estimator가 **동일한 고정 q의 M2**를 추정한다.

```
q-strat: M2 = Σ_b π_b E_q[m2/Dq² | X57∈A_b]
p-strat: M2 = Σ_b π_b E_p[m2/Dq  | X57∈A_b]
```

제안분포가 다르므로 Dq의 제곱 유무를 혼동하지 않는다.
각 구간n개 IID를 뽑고 π_b*(ordinary mean contribution)를 합한다.
구간들을 같은 확률로 샘플링한 것처럼 단순 flatten 평균내거나 전체 contribution을 self-normalize하지 않는다.
normalized bank moment가 아닌 **unnormalized M2 estimator**다.
q-strat는 q의full draw에서57만 independent truncated-N draw로 교체할 수 있다.
p-strat는 나머지좌표도standard normal에서 뽑는다.
61/63은 profile 생성용0으로 둔 뒤 m2적분/미사용법칙을 그대로 적용한다.

### 2.1. 유계성은 있지만 상대 정확도가 자동으로 인증되지는 않는다

현재 q의 natural defensive mass가0.1이므로 이론적인 정확한 m2에 대해
`Dq≥0.1`, `0≤m2/Dq≤10`, `0≤m2/Dq²≤100`이다.
따라서 고정 sample size의 절대오차 concentration bound를 만들 수는 있다.
하지만 매우 작은 M2에 대한 유용한 상대오차 bound가 그 sample size에서 얻어지는 것은 아니다.
또한 numerical quadrature proxy는 certified integration-bias bound가 아니므로
이 유계성만으로 실제 수치 estimator의 전체 오차를 인증할 수 없다.

예를 들어 |X57|>6 전체의 실제 M2 기여는 `10*P(|N(0,1)|>6)≈1.973e-8`로 상계된다.
작아 보이는 Gaussian tail 확률도 canonical M2의 과거 개발 점추정 크기보다 이 상계가 클 수 있다.
그래서 tail을 버리거나 작다고 확정하지 않는다. 반대로 이 느슨한 상계가 실제 tail 기여가 크다는
증거도 아니다. 64개/구간 표본이나 두 estimator의 비반증만으로 상대 정확도가 인증됐다고 하지 않는다.

## 3. truncated normal 구현에서 먼저 막을 오류

- 양쪽무한구간/±6tail을 drop/clip하지 않는다.
- tail probability는 생존함수/log-CDF 대칭을 사용한다. `1-Phi(6)`의 cancellation에 의존하지 않는다.
- U가0/1endpoint가 되지 않는 난수 mapping 또는 검증된 truncated-normal 알고리즘을 사용한다.
  무한값이 나왔을 때 임의 clip해통과시키지 않고 오류로 보존한다.
- π_b와 actual truncated sampling density의 일치, 합1, 구간안샘플인지 확인한다.
- variance는 구간간고정allocation을 반영한 `Σ π_b² s_b²/n`로 계산한다.
  서로dependent SMC particle variance를 IID variance처럼 사용하지 않는다.
- reference 자체는16개 독립 replicate를 inferential unit으로 두고 RSE를 계산한다.

## 4. 실행 전 known-answer

현재 calibration의3개 analytic target을 재사용하되 새 random stream으로 다음을 검사한다.

1. truncated-N 구간확률/구간0·1·2차 moment를 analytic 값과 대조한다.
2. 각구간constant integrand에서 전체합이1이 되는지 확인한다.
3. **q≠p인2D mixture**에서 q-strat/p-strat 각각 같은 `∫p²m2/q` 정답을 재현하는지 검사한다.
   mixture shift는첫좌표에만 두고, 구간분할은 unshifted 두번째좌표에 적용한다.
   1D shifted-mixture의 그 shift좌표에 표준정규 π_b를 그대로 쓰면 q(A_b)≠p(A_b)여서 틀린다.
   1D rare-target 검사는q=p로 유지하고, nontrivial-q 검사는 이2D 구조로 분리한다.
4. p/q density보정이하나씩다른반례, π_b누락/두번곱함/구간누락을 검출한다.
5. extreme-tail scalar숫자가작아도 상대기여/전체합의 회귀검증을 수행한다.
6. seed/예산/수치 gate/atomic artifact/저장실패/전체재생을검사한다.

known-answer Monte Carlo tolerance,seed수,오차gate는 해당실행계획에서 **실행 전에** 구체값을 봉인한다.
이 문서는 그 구현/검증이 완료됐다고 주장하지 않는다.

## 5. 실제 Volterra reference pilot — 제안하는 bounded 크기

실행 전 완결 config로 동결할 기준안:

- canonical/high-eta 두셀, q fixed373component/N32.
- q-strat/p-strat 두방법, 방법당16독립replicate.
- 각replicate8구간×64sample=512context.
- 총2×2×16×512=**32,768 Volterra physical**.
- analytic m2는기존개별relative proxy≤1e-6/structuralgate를 모두 요구한다.
- proposed scalar cap130,000,000/physical33,000/compute wall900초.
  실제 plan freeze 전에 worst-case panel/node 비용이 이 cap 안인지 다시 산정한다.
- 별도audit: 모든64replicate통계/해시, 셀×방법×구간의첫replicate를 fullreplay.
  예상2,048physical, proposed cap3,000/scalar12,000,000/wall180초.
- 모든구간의Z기여,구간별sample variance/maximum contribution,replicate별Z를 저장한다.
  마지막구간이중요하면예산을사후그구간에몰지않고별도설계를요구한다.

이 크기는 성공 보장이 없는 개발 pilot다.57의 모든 구간에 표본을 배치하더라도
각 tail 구간 내부의 모든 희귀 기여나 나머지61좌표의 rare mode를 포착한다는 보장은 없다.
이전 raw IID 실패를 이 작은 pilot가 반드시 해결한다고 주장하지 않는다.

## 6. 자격·비교·실패 시 처리

16개 independent-replicate RSE≤20%, 최대replicate질량≤25%를 개발조건으로 제안한다.
둘 다 개발 자격을 얻은 셀에서만3.291 combined-SE의 차이 경고를 계산한다.
이 기준은16회/희귀 꼬리에서의 명목 유의수준 보장이나 정확도 인증이 아닌 휴리스틱이다.
비반증은 equivalence/정답 인증이 아니다. 두 방법 모두 같은 numerical m2를 쓰므로
공통quadrature bias에대한독립검증은별도로남는다.

위조건은최종32,768실행전에plan/config로봉인해야하며,실행후수치에맞춰바꾸지않는다.
한방법이라도미자격이면 reference unresolved로STOP하고 P2/대규모 fitting을하지않는다.
둘이모순되면sign/density/stratum/tailcontribution을감사하고평균내서해결하지않는다.

둘이자격을얻어도과거SMC값을현confirmation으로소급승격하지않는다.
다음별도단계에서freshSMC/더많은독립반복과교차검증해야한다.
기존bankgenealogy실패도별개로보존한다.
같은M2평가의정밀도·coverage기반을확립한뒤에만modelranking/실제Gfitting/P2를재논의한다.

## 7. 범위와 중단 판단

이번calibration은diagnostic의필요충분성을부정하는근거다. productiongate를낮추는허가가아니다.
독립referencepilot도안정성을못얻으면새operator를반복추가하지않고,
연구의주목표를성능우위모델에서조건부분해/희귀기여오차진단으로축소할지명시적으로재평가한다.
top-journal게재또는오류0을보장하는계획은아니다.
