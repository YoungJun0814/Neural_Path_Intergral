# 정확한 밀도 압축과 다중 구간 결합: 다음 우선순위 실행계획

2026-10-27 America/New_York. main59f2809 이후 새 작업. 커밋·푸시하지 않는다.
과거 결과·source·gate는 바꾸지 않는다. 목표는 계산비 개선과 실제proposal의 제한된 새 가설 검증이다.

## A. 동일 분포의 exact local-mean grouping

기존 local covariance correction은 한시점의component mean 두값에만 의존한다.
그 두값이 **정확히 같은** component들의 원래 log mixture항을 먼저합치면 같은밀도가 된다.
float rounding/mean 양자화/작은component 제거는 금지한다.
group별 max와logsumexp를사용해작은항을전역max 아래에서0으로버리는지름길을피한다.
base/개별arm/전체arm을각각평가할수있게하되sampling은원래family를그대로쓴다.

- 사전small dense Gaussian oracle,원래함수,극단좌표,α0,원본parameter변경/invalid입력 검사.
- 이후새namespace의셀×(base/localcoupled/R)×512=3,072좌표로old/grouped log밀도대조.
- old/grouped all-arm AB/BA/AB고정타이밍과개별arm타이밍을별도로측정한다.
- 최대log차≤1e-9와median all-arm kernel speed ratio≥1.2일때만kernel후보채택.
  speed 조건은개발가치판단이며population model개선/동일정확도total-wall우위를뜻하지않는다.
- source/config를실행전에봉인한다.실패하면원래kernel로되돌려production을강행하지않고STOP한다.

## B. 기존 큰 기여 경로의 사후 구조 감사

지난96unit에저장된모든witness 슬롯을읽고정확히동일한좌표는중복제거한다.
전체witness의g와old/grouped밀도를재평가해저장값과대조한다.
사용분산의peak시각/HHI/effective duration/상위1·4구간질량,
Brownian 결합항,local versus window 방향 projection을기록한다.
이들은이미선택된극단경로이며모집단mode빈도/인과성/새confirmation으로쓰지않는다.
큰M2비율만으로window/시점/부호/variance를선택하지않는다.

## C. trace를맞춘다중구간proposal

기존localcoupled는각sample에서한시점j의
`w_j=(e_(2j−1)+s e_(2j))/sqrt2`,eigenvalue9인rank1 covariance를쓴다.
새window는연속j구간W에서
`w_W=Σ_(j∈W)(e_(2j−1)+s e_(2j))/sqrt(2|W|)`를쓴다.
모든window가norm1이므로각covariance는I+8wwᵀ,trace증가8로local과동일하다.
서로다른좌표를독립적으로모두넓히는rank증가가아니다.

N32에서used j=1,...,31을폭4의고정비중복window로분할한다.
W=(1..4),(5..8),...,(25..28),(29..31),총8개,질량1/8.
마지막window는3개이며길이를4라고취급하지않는다.미사용63은제외한다.
둘이여러시점의같은부호변동을공유하는새geometry지만Volterra memory전체나
continuous-time Wiener law/mesh-invariance를증명한것은아니다.

원래physical373means/weights와natural.1을유지하고shiftedmass의절반만windowcovariance로바꾼다.
`qW=.1p+.5Σshiftw_k N(m_k,I)+.5/8Σ_WΣshiftw_k N(m_k,I+8w_Ww_Wᵀ)`.
정규화상수는1이고qW≥.1p,qW≥.5q0,M2(qW)≤2M2(q0)이다.개선보장은아니다.
명목Gaussian항수3,349이며base373/local11,905와capacity가다르다.
새density는`.5(1−1/9)*((z−m)·w)²−.5log9`를원래component log항에더한다.
sampling은whiteGaussian에정확한rank1square-root변환을적용한다.
shiftmean을변환하거나단일label만분모로쓰지않는다.

## D. 사전고정작은실험

A의수치·speed조건과windowdenseoracle를통과한경우에만진행한다.
CPUfloat64/1thread/RSS4GiB/로컬노트북,새namespace `grouped-window-coupling-v1`.
시점/폭/α/variance/seed를관측후바꾸지않는다.학습/선택없음.
두셀canonical/high-eta에서사전주후보window,base와oldlocalcoupled대조.

- direct:2cells×3arms×8blocks×4096=196,608physics.
- heldoutH:각cell×8blocks×3strata×1024=49,152physics,
  H=(q0+qL+qW)/3,각stratum knownmass1/3의mean/covariance를사용한다.
- priorwindowreferenceR:각cell×8blocks×2048=32,768physics,
  R=.1p+.9/8Σ_W N(0,I+8w_Ww_Wᵀ).shiftmean없지만candidate와window구조는공유한다.
- 총80unit(48direct+16heldout+16reference),
  production278,528physics.표본수가작은개발실험이며성공을보장하지않는다.
- pairedM2는g²/(D_H D_q)또는g²/(D_R D_q)로같은3target을평가한다.
- eventwindow rawRSE≤5%,blockRSE≤2raw,maxpath≤10%,LOO≤15%.
- H/R각3M2모두raw/blockRSE≤20%,maxblock≤25%,LOO≤15%를요구한다.
- window/base20%감소및window/local0%감소의pairedabsolute/relative lower를raw/block에서요구한다.
  2cells×2geometry×2contrast=8family Bonferroni normal/delta개발진단.
- 양reference가자격을얻은경우에만2cells×3target=6family,10%equivalence를검사한다.
  작은반복의근사CI는tail/accuracy보증이아니며R의공통windowblindspot도남는다.
- 둘다실패하면성공arm/seed를골라계속하지않고종료한다.추가학습/P2는승인하지않는다.

## E. 비용·감사·검토

A:physics0(밀도만),3,072fresh좌표및기존witness.최대old-call50,000coordinate-pass,
wall600초. 추정량의새성능표본으로쓰지않는다.
B:중복제거witness수만큼physics,cap500.
D:preflight2cells×3arms×64=384physics,총278,912. cap280,000/wall1800초/unit60초.
새seed는실행전할당하고과거protocolscope와충돌검사한다.
대표각cell의windowdirect block0(8192),Hblock0(6144),Rblock0(4096)재생=18,432physics.
전수array/moment/판정/source/hash를검사하고auditcap20,000/wall600초.
기존 결과를 재계산해 덮어쓰지 않고 새 source/config/namespace에 저장한다.
전체pytest/Ruff/Mypy 및기존source불변·완료로그·공개파일검토후README/결과보고서를갱신한다.
검증통과와연구효율입증을구분하고계산가속만으로P2가해결됐다고하지않는다.

## F. 실행 전 이론 검토와 주장의 한계

1. **Grouping의 등식.** component의 원래 log항을 `a_k(z)`, 시점 j의
   공분산 보정항을 `c_j(z,m_k[2j-1:2j+1])`라고 하자. 동일 local mean의
   집합 G에서는 `Σ_(k∈G) exp(a_k+c_j)=exp(c_j)Σ_(k∈G)exp(a_k)`이다.
   다른 좌표의 mean은 원래 a_k에 남는다. 전체 mean이나 weight를 평균내는 것이 아니다.
   연산 순서에 따른 부동소수 오차는 별도 수치 gate로 검사한다.
2. **공분산 밀도와 생성의 일치.** 단위벡터 w에 대해
   `Σ=I+(λ-1)wwᵀ`, `Σ⁻¹=I-(1-1/λ)wwᵀ`, `detΣ=λ`이다.
   따라서 log-density correction은 계획의 식과 같고, 생성 변환은
   `ε↦ε+(sqrtλ-1)(ε·w)w`이다. mean은 변환 후 더한다.
3. **불편성과 안전장치.** 모든 density는 같은 표준 Gaussian p에 대한
   `D=q/p`이며, event 기여는 `g/Dq`, 독립 보조 표본 h의 M₂ 기여는
   `g²/(Dh Dq)`이다. self-normalization하지 않는다. 0≤g≤1과
   Dq,Dh≥.1로 event 기여≤10, risk 기여≤100이다. 유한 모멘트가 있지만
   희귀 평균에 대한 상대오차는 매우 클 수 있다. 무한분산 문제라고 해석하지 않는다.
4. **안전한 상대 bound.** qW≥.5q0이므로 M₂(qW)≤2M₂(q0)일 뿐,
   M₂ 감소나 동일 정확도 비용 개선을 보장하지 않는다. marginalize한
   terminal probability g와 N32의 원래 모형을 바꾸지 않는다.
5. **검증의 경계.** selected witness는 사후 사례, 8 block의 normal/delta
   구간은 개발 진단이다. guide-free R도 window와 simulator를 공유하므로
   독립 난수 사용만으로 독립적인 구조 검증이 완성되지 않는다.
   본 작업의 성공만으로 N4096의 다른 geometry 기준, signed mesh correction,
   continuum bias 검증 또는 저널 투고 자격을 획득하지 않는다.
