# P2 직접 해결 실행계획: 독립 기준·signed mesh·연속시간 잔여오차

2026-10-27 America/New_York. 기존 실패·봉인 소스·통과 기준을 보존한다.
커밋·푸시, 클라우드 비용 발생, 논문 범위의 임의 축소는 하지 않는다.
이 문서의 구현 완료와 P2의 통계·이론 자격 획득은 서로 다르다.

## 0. 고정된 목표와 현재 증거

원래 terminal downside K=1, H=.05, eta=1.5/2, rho=-.7, xi=.04, S0=100,T=1.
N4096 원래 BLP mean-cell Gaussian law를 유지한다. frozen-ensemble IID의
관측 RSE .902%/.907%는 확보했다. 이를 과거 guide-free SMC의 정답으로 강제하지 않는다.

P2의 전체 해결에는 다음이 모두 필요하다.

1. 두 셀의 N4096 확률에 대한 독립 탐색 기준의 RSE≤1.5%, whole 안정성,
   그리고 기존 IID와 조정된 10% equivalence. 불일치는 어느 쪽의 bias인지 단정하지 않는다.
2. 같은 Gaussian augmentation law로 coupled N4096/N2048의 signed correction을
   충분히 정밀하게 평가한다. 작은 signed 차이를 absolute-path 차이와 혼동하지 않는다.
3. 연속시간 원래 모형과의 잔여 상대 bias를 정당한 유한-N 상계로 통제한다.
   2개 또는 몇 개의 adjacent 차이나 fitted weak rate를 infinite remainder로 바꾸지 않는다.
4. 원래 P1 all-q risk/bank 자격이 필요하면 별도 충족해야 한다. μ만 맞았다고
   M2(q)까지 해결한 것으로 판정하지 않는다. 전체 large-training 승인은 별도다.

## 1. 이론적 진입 검사

기존 original BLP integrated-driver MSE 상한과 conditional-CDF absolute bound를
새 실행 가능한 log-domain 함수로 연결한다. 극단 상수의 overflow를 숨기지 않는다.
N4096/8192/16384에서 절대 상계와 관측 μ 대비 크기를 기록한다.
관측 μ는 참확률 lower bound가 아니므로 상대 certificate라고 부르지 않는다.

Friz–Salkeld–Wagenhofer, Theorem1.1은 polynomial test function 및 해당 left-point
scheme에 대한 정리다. 현재 BLP digital probability의 명시적 유한-N relative bound로
직접 대체할 수 없다. 정리의 가정/대상/상수를 확인하지 않고 3H+.5를 대입하지 않는다.
원문: https://arxiv.org/pdf/2212.01591 , https://doi.org/10.1214/24-AAP2109 .

상계가 크면 실제 bias가 크다는 증거가 아니라 현재 proof chain이 불충분하다는 뜻이다.
이론 조건이 미해결이어도 아래 제한된 독립 기준/mesh 개발 실험은 별도로 수행한다.
속도 1.2배 gate와 연결하지 않는다.

## 2. 새 guide-free 비국소 mutation

기존 gradient pCN MH에 **Gaussian pair-slot cyclic permutation MH**를 합성한다.
augmentation을 포함한 표준 Gaussian 좌표를2개씩 묶고, row별 독립 uniform cyclic
shift로 pair 위치를 바꾼다. 변환은 orthogonal이고 역 shift가 같은 확률을 가져
prior-reversible이다. target은 바꾸지 않으며 acceptance는 beta*(logH(new)-logH(old)).
이는 Volterra path를 물리적으로 시간 이동하는 대칭이 아니다. raw noise의 prior
대칭을 쓴 MH proposal일 뿐이며, target potential을 재평가해 보정해야 한다.
permutation만으로 새로운 크기가 생기지는 않으므로 gradient-Gaussian mutation을 유지한다.
마지막 미사용 좌표/augmentation도 움직일 수 있으며 전체 Gaussian law에서 유효하다.

H=(gf+gc)/2의 fixed-temperature SMC를 사용한다. endpoint의 full-family factor로
각 unnormalized μf/μc를 복원한다. normalized weighted endpoint mean 자체를
확률 추정량으로 사용하지 않는다. gradient norm20 cap은 proposal에만 적용한다.

단위검사: permutation norm/inverse·prior covariance, MH detailed balance dense oracle,
constant/analytic normalizer, full-family identity, resampling/cost/invalid 입력.
낮은 ancestry는 진단으로 남기되 좋은 ancestry만으로 precision을 승인하지 않는다.

## 3. 사전 고정 fine-only paired pilot

새 namespace `p2-direct-cyclic-reference-v1`.
N4096/N2048 augmentation dimension10240, CPUfloat64/8threads/RSS4GiB.
2cells×2arms(control gradient, gradient+cyclic)×8independent wholes=32whole.

- 128particles,64temperatures beta=(k/63)^3,resample every4,stratified.
- control: 각 nonfinal stage gradient2회.
- cyclic: 각 nonfinal stage gradient1회+cyclic1회.
- gradient scale.35,drift norm cap20,양arm 모두총125joint callbacks/particle.
- one callback=원래fine+coarse physical2회. whole joint16,000/physical32,000.
- production joint512,000/physical1,024,000. preflight각cell/arm16particles,
  16temperatures로 별도 새 seed,physical928×4=3,712.
- preflight는 비용·수치 검사만 사용한다. total physical cap1,050,000,
  whole wall300초,phase7200초. unit 중단 시 부분 normalizer를 결과로 쓰지 않는다.
- 32whole 전체가 끝나기 전 prefix 자격 판정 금지. 좋은 seed/arm 선택 재실행 금지.
- 새/과거protocolscope seed 충돌 검사와 source/config/anchor 봉인.

## 4. 판정과 후속 실행

기존 precision gate: RSE≤.015, maximum whole share≤.15,LOO≤.05.
fine/coarse 둘 다 확인한다. 원래 IID와 fineμ의 2cells×2arms=4comparison,
10% symmetric equivalence는 **양쪽 precision 자격을 얻은 경우에만** 판정한다.
raw와 whole 통계가 다를 때 whole 독립 반복을 기준으로 한다.

signed correction은 independent whole의 μf,μc에 paired delta covariance를 사용한다.
4comparison upper |delta|+zSE≤.1은 개발 진단이지 continuum certificate가 아니다.

사전 primary는 cyclic이다. primary두셀 fineprecision과crosscheck가 통과하면
새 production protocol의 sample allocation을 작성한다. 같은 pilot을 production으로
승격하지 않는다. 미달하면 observed worst RSE에서 safety3/target.015,
minimum32/maximum256whole의 fresh forecast를 계산해 비용 및 불충족 원인을 기록한다.
예산 밖 계산을 자동으로 무한 실행하지 않는다. 이론 잔여오차 미해결 상태를 감추지 않는다.

## 5. 감사와 완료의 정의

대표cell×arm block0 네 whole을 새로 재생(physical128,000 cap130,000/wall1800).
모든 stage/whole의 source/seed/moment/count/decision을 검사한다.
이론 상계 함수를 독립 직접 식·고정값·scale검사로 대조한다.
전체 regression/Ruff/CI범위 Mypy/새 runner typing을 실행하고 결과보고서를 작성한다.

P2 true는 필요한 세 축의 실제 증거가 모두 있을 때만 가능하다. 마지막 theoretical
obligation이 남으면 `P2_UNRESOLVED_THEORETICAL_REMAINDER`를 명시하고 코드 완성과
문제 전체 해결을 혼동하지 않는다. user의 요청은 통과 기준 변경이나 새 수학 정리를
증거 없이 존재한다고 가정할 권한을 주지 않는다.
