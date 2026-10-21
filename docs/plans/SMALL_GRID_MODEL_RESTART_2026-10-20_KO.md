# 연구 진행방식 재설정: 원래 모델 q의 작은 격자 판별 실험

2026-10-20. 사용자 요청: 검증기 튜닝에 머무르지 말고 파악한 문제를 실제로 개선한다.
커밋·푸시 없음. 과거 결과·gate·봉인 소스는 변경하지 않는다.

## 1. 변경하는 운영 원칙

기존의 “fine reference 완전 자격 → 모든 q 개발” 직렬 순서를 바꾼다.

1. **모델 개발 트랙:** 검증 가능한 N32 유한격자에서 실제 사건 estimator q를
   바꾸고, 강한 static baseline과 비교한다. 탐색적 모델 개발이지 P2 개방이 아니다.
2. **제출용 검증 트랙:** N4096 독립 reference, all-q M2, mesh/continuum,
   미사용 task/독립 학습 confirmation은 계속 미완료로 둔다. 이번 턴에는
   추가 SMC schedule 검색이나 대규모 학습을 하지 않는다.

검증 기준을 소급 완화하지 않는다. 여기서 새로 정의한 개발 신호는 이전 P1/P2와
다른 상태이며, 기존 1.5% 실패를 통과로 바꾸지 않는다.

## 2. 이번 연구 질문

같은 finite-grid g,p에서 **기존 q의 희귀 위험 기여를 겨냥한 작은 보정이
실제 사건 추정기의 위험과 비용을 개선하는가?** 그리고 학습은 물리 기반 분포만
사용하는 것보다 추가 가치가 있는가?

N32 두 기존 stress 셀(canonical/high-η), 원래 spot100/threshold1을 유지한다.
이는 99% 하락 stress probability이며 일반 pricing/calibration 성능으로 확대하지 않는다.
N32와 N4096은 다른 finite-grid target이므로 이전 N4096 reference를 참값으로 쓰지 않는다.

## 3. 고정된 네 arm

각 셀에서 부모를 처음부터 새로 학습하는5회 반복, 총10부모를 사용한다.

| arm | 실제 사건 추정용 q | 포함 비용 |
|---|---|---|
| parent | 새 SMC 부모 q0 | simulator 준비+부모 탐색/fit+final |
| static | fixed mixed-window physical G | simulator+geometry 준비+final; 부모 비용 없음 |
| event | (1−α)q0+αr_event | 부모+G준비+train+event fit+selection+final |
| risk | (1−α)q0+αr_risk | 부모+G준비+train+risk fit+selection+final |

부모는 과거 S0/S1와 같은3islands×256particles, β=(k/48)^4,
49points,nonfinal stage당2moves,scale.35,resample4,stratified,K≤2 shifted
components,identity covariance,natural mass.1이다. 이는 새 학습이지 과거 좋은 부모 재사용이 아니다.

G는 기존 physical mixed-window 생성기를 N32에 적용한다. amplitudes[2,4,6,8],
price amplitudes[0,2,4],natural.1,총373components다. static은 실무 best-available
baseline이지3component 부모와 동일 capacity가 아니다. event/risk fit끼리는 같은
bank·K≤2·identity·알고리즘/예산을 사용한다. blend의 실제 성분 수는 별도 기록한다.

## 4. train/selection/final의 분리

부모별로 다음 순서를 고정한다. 모든 physics는 원래 finite-grid conditional CDF다.

1. 새 SMC 부모72960payoff calls. terminal particles는 fitting에만 사용.
2. 독립 IID G train8192. event weights는 g/(G/p),risk weights는
   g²/[(q0/p)(G/p)]. 동일 bank로 matched fitting한다. 가중치 정규화는 fit 전용.
3. K≤2 identity Gaussian correction 두 개를 동결한다.
4. 별도 IID G selection16384. α∈{0,.25,.5} 각각 ordinary empirical M2를
   계산해 각 종류의 최저 α를 선택한다. 동점은 작은 α. α0은 q0와 정확히 같은 분포다.
   선택 목적값을 독립 성능 추정치로 보고하지 않는다.
5. 네 arm 각각 새 IID final8blocks×32768=262144. 각 arm/부모/블록의 seed는 다르다.
6. 별도 새 IID G131072를8blocks×16384로 생성해 네 동결 q의 M2를 **공통 표본**에서
   평가한다. parent/event/risk/static의 각 쌍을 covariance-aware로 대조한다.
   이 heldout은 선택이나 재학습에 사용하지 않는다.

같은 G bank라는 이유로 서로 다른 q의 위험 추정값을 독립이라고 취급하지 않는다.
static geometry는 모든 반복에 공통이지만 final/heldout 난수는 새것이다.
static의5회는 독립 학습5회가 아니라 fixed-static sampling5회라고 표시한다.

## 5. 이론 및 판정 계약

- full-mixture ordinary IS `g/(q/p)`를 사용한다. component-only denominator,
  empirical normalizer 삽입, contribution clipping 금지.
- 모든 실제 q,G≥.1p이므로 event contribution≤10,heldout risk contribution≤100.
  이 bound는 작은 relative error나 미관측 tail 부재를 보장하지 않는다.
- 같은 target에서 variance 순서는 M2 순서다. μ를 exact하게 몰라도 독립
  paired M2 차이를 개발 진단할 수 있다. 이것이 final accuracy 인증은 아니다.
- exact risk tilt의 이론적 성질은 별도 계약에서 증명한다. 유한 bank/K2 근사는
  그 보장을 자동 상속하지 않는다. 위험 표적은 최적 q*=gp/μ와도 다르다.
- heldout 6pairs×10부모=60comparisons에 Bonferroni normal/delta diagnostic을 쓴다.
  raw IID path 및8block 양쪽에서 signed absolute difference와 relative reduction의
  adjusted lower bound가 모두0보다 클 때만 감소 신호로 분류.
  작은 block 수·미관측 tail 한계 때문에 confirmation으로 부르지 않는다.
- final의 새 개발 precision 표시는 rawRSE≤5%,blockRSE≤2rawRSE이다.
  기존1.5%gate와 별개이며 reference accuracy/equivalence 자격을 뜻하지 않는다.
- α0 선택 등 정확한 no-op은 분포 digest로 표시한다. 우연히 좋아진 새 final RSE를
  모델 개선으로 세지 않는다.
- 양 셀 각각5회 중4회 이상에서 q0대비 paired M2 감소+새 final 개발 precision을
  보인 방법만 후속 후보로 남긴다. learned/static 비교도 별도로 공개한다.
  이 반복 수로 population 실패율·최상위 저널 적합성을 보장하지 않는다.
- fixed-final-budget workflow 시간만 보고한다. accuracy-qualified fixed-precision
  total-wall 우월성,보편적 speedup은 주장하지 않는다. G bank 공유의 실제 물리 비용과
  각 단독 배포 workflow에 전액 부과한 비용을 구분한다.

## 6. 고정 예산과 운영

torch CPU float64,1thread. batch2048. RSS4GiB; OS/배터리 정책 변경 없음.
훈련72960+train8192+selection16384=97536calls/부모.
final1048576+heldout131072=1179648calls/부모.
총10부모 **12771840calls**, 별도 preflight2×256=512를 더해
**12772352payoff calls**. 평가용 heldout은 배포 inference 비용에 넣지 않되
연구의 실제 물리 비용에서 제외하지 않는다.

준비120초,preflight30초,fit180초,final block60초,heldout180초,
phase7200초. 초과/수치 오류/불완전 family는 기록을 보존하고 실패 처리한다.
seed 교체·optional stopping·최종 결과를 본 표본 연장 없음.
343units(prepare1+preflight2+fit10+final320+heldout10)를 모두 완료해야 집계한다.

## 7. 감사와 다음 결정

실행 전 core/runner/oracle 테스트 및 lint/type → 계획·이론·설정·source 봉인
→ preflight →10부모 및320final blocks+10heldout → 전체 summary/seed/work/digest검사
→ 대표 부모2개·final8blocks·heldout2개 numerical replay → 전체 회귀테스트.

static만 강하면 이를 실제 사건 proposal의 새로운 개발 기준선으로 삼는다.
학습의 추가 가치를 숨기거나 신경망을 억지로 유지하지 않는다. event/risk 보정이
강한 static을 반복해서 개선하면 그 원리를 후속 연구로 좁힌다. 어느 쪽도 충분한
신호가 없으면 같은 셀의 schedule/power 튜닝을 확대하지 않고 geometry/family와
workload를 재검토한다. 결과와 상관없이 P1/P2/large-training flag는 false다.
