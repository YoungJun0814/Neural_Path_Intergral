# Canonical 원래5q: 새 full reference validation

2026-10-10. 관련pilot의전체완료·source/seed/numerical감사후아래580units생성전에동결.
canonical scope 개발검증이며publicationconfirmation/P2/large-training 승인이아니다.

## 1. 선택 근거와 실패 보존

- 원래pCN 전체allocation,6component umbrella,μ/risk분리pcn은실패했다.
- target-gradient resource sweep의128temperature후보는5risk최악RSE3.730%,
  safety2 allocation99whole,forecast51,194,880forward+같은backward였다.
- separate probability-gradient는RSE3.141%,safety2 allocation71whole.
- guided-envelope umbrella는최악μRSE4.644%,safety2 allocation154whole.
- 층화는μallocation13,094,400으로실패했다.낮은새IIDvariance만으로그전budget실패를지우지않는다.
- 마지막Gaussianpair를정확히적분한probabilityreference는pairedpilot에서
  fullμRSE5.028%→RB4.688%,allocation5,144,576draw였다.
  final은낙관pilotcount가아니라원래IIDcap8,388,608draw전체로동결한다.
  원래N32사건μ/M2와q는변경하지않는다. 이전성공/실패pilot/final은어느것도pool하지않는다.

## 2. 세 독립 경로와 고정 실제 실행

canonical-h005-k1,N32,original frozen5q.

1. Static-guide IID:256blocks×32768=8,388,608draw.각block은8×4096cachebatches.
   last-pairzero path의cached a,b,c로fullg를정확히재구성하고raw5M2를계산한다.
   μ는Φ(a/sqrt(c²+b²))/(rU/pU)를ordinarymean한다.
   risk는gbar²가아니라fullg²/[(q/p)(r/p)]다. full/prefixdensity각자의mean norm사용.
2. Guide-free gradient-MH risk-only family:99independentwhole.
   1024particles,128quadratictemperaturepoints,4mutations(.35),every4stratified,
   fixeddriftnormcap20,정확한forward/reverseMH. 5riskweights는동결anchor에서재정규화.
3. Guide-free gradient-MH probability-only:71independentwhole.
   1024particles,64quadraticpoints,4mutations(.35),나머지동일. risk와seed분리.
4. Guided normalized-envelope μ+5risk family:154independentwhole.
   기존64quadraticpoints/4mutations/full/global/rank2/global/every4stratified를유지.
   최초normalizedr시작과정확한bridgeincrement/MH/endpointδcorrection.

마지막mutation없음. 전체256+99+71+154=580units, preparation128(guide64+envelope64).
total116,953,216forward evaluations /69,298,176actualautogradbackward.
cap160mforward/80mbackward/3600초/RSS4GiB,unitstatic30초/SMC30초/prepare30초.
source/3pilotanchors/input/q/weights/seed동결. namespace `canonical-family-precision-v1`.
32oldpilotwhole이나high-eta자료와합치지않는다. full580units전prefixpass판단금지.
interrupted/failedwhole은보존하고교체하지않는다. budget실패도통계실패와구분해보존한다.
wallforecast는공유노트북측정에근거한자원계획이지exclusive publication speed비교가아니다.

## 3. 실제 판정

μ+5M2×3methods모두RSE<=1.5%,IIDmaxdrawshare<=1%/SMCmaxwholeshare<=15%,
LOOblockshift<=5%;IIDbetweenblockRSE<=1.5×drawRSE.
μprimary(static-vs-guidefree)는original2comparisons,
5risk×3methodpairs는original30comparisons,10%symmetricrelativeequivalence를유지한다.
guidedμ는precision/sensitivity에포함하되primarypair를사후선택하지않는다.
normal/delta/empiricalSE의개발관문이며distribution-free tail certificate는아니다.

재감사: source/anchors/input/q/seed/moments/unnormalizedfamilyidentity/δcorrection/work;
staticfirst/lastblock65536draw와3SMC firstwhole numericalreplay.
cache-fullg의독립1024paths비교(2048path-lawcalls)도추가한다.
replaycost는finalestimate의계산량과별도기록한다.

## 4. 승인 범위

새flag `canonical_five_q_development_reference_pass`만actual결과로갱신한다.
high-eta별도scopedpass와합쳐도originalP1protocol의schedule/independence 요건을
모두대체했다고자동해석하지않는다. 원래gate문서와새mechanismcoverage를최종대조해야한다.
P2fine-grid uncertainty/continuum bias,bank quality,learnedaddedvalue는여전히별도관문이다.
`p1_qualified`, `p2_qualified`, `large_training_authorized`는false다.
두finite-gridreference를확보하면그다음fresh mesh-reference및bankquality회복설계를동결한다.
