# 가격 방향만 분리한 새3arm 메커니즘 pilot

2026-10-19. 이전G0/G1결과로생긴가설을새seed로점검하는개발실험이다.
다른실험과무거운계산을겹치지않고 bridge/mutation32whole+감사 이후 실행한다.
그뒤N4096 B의고정4chunk(각512draw)로새feature분기와원래paired분기를대조한다.
seed26900..26903에서path=10seed/label=10seed+1/augmentation=10seed+2다.
8192physical/4096density/time-step25,165,824는별도preflight비용이다.
prep60초/RSS4GiB/maximumfeaturechunkwall×64<180초(2배시간여유)를통과해야본실행한다.
이검사는whole시간/precision보증이아니며표본을본실험에섞거나결과로arm을선택하지않는다.
A=기존G0, B=G0의31point-volatility방향/시점/가중치를완전히유지하되futureprice만
T/8(만기clip)로변경, C=기존G1. 동일373성분/identitycov/natural.1/amplitude2,4,6,8/
price0,2,4/energy최대40. A→B만가격방향변경이며B→C는volatilitygeometry와시점결합변경이다.

2cells×3arms×8blocks=48blocks,각16384fine IIDdraw,chunk512/32chunks.
namespace matched-window-mechanism-v1,rolespath/label/augmentation/bootstrap분리.
fine4096/coarse2048원래target·보정compensator·독립augmentation유지,
ordinary mean g/(r/p),정규화된전체balance-mixture density1회.
physical1,572,864/density786,432/time-step4,831,838,208. prep186linearrows,
nonlinearbasis-simulation0. CPUfloat64/8threads. phase7200초/block180초/prep60초/
RSS4GiB/physical1.6m.배터리조회/OS설정변경/커밋/푸시없음.

fine양을기존cache.quantities에서한번계산한뒤같은CDF로변환한다.그출력으로I,J,
maximumleftvariance와시점·샘플링componentnoiseRMS·volfeatureprojection을계산한다.
extra nonlinear simulation0;추가linear연산/진단저장wall은실제block시간에포함된다.
모든성분의labelcount/logcontributionsum및전체top16경로의6feature만저장한다.
label은sampler의mixturelabel이지물리modecluster가아니다.고위험top16은분포대표표본이
아니며thresholdcalibration/outsidegeometry탐지/누락mode배제근거로쓰지않는다.
feature의observed contribution-weightedmean은서술적ratio이며unbiasedclaim없음.
확률추정의logmoments는어떤selfnormalization도하지않는다.

보고:6cell-arm의μf/μcRSE/maxshare/blockLOO,paired차이6comparison,
셀별A-B/B-C/A-C의fine평균동등성총6comparisonBonferroni10%,관측RSE제곱비율
(확정variance배수/시간가속아님),성분별관측기여와top16feature.선택규칙변경없음.
이실험의저정밀G0를추가sample로구제하지않는다.모든P1/P2/trainingflags=false.
각cell/armrep0의6block재현감사physical196608/density98304별도.
source/geometry/seed/diagnostic/moment/cost를결합해재현한다.후속precision/독립reference/
새taskconfirmation은별도다.분포/학술적독창성주장하지않는다.
