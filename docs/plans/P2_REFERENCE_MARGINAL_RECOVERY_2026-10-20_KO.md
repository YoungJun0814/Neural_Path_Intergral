# P2 병목을 풀기 위한 조건부 주변화 reference 실행계획

2026-10-20. 커밋·푸시없음/배터리조회없음/로컬CPUfloat648threads.
현재P2미승인은(1)fineμRSE1.5%미달,(2)독립SMC불일치경고,(3)pairedgrid불확실성 및
continuumrelativebias미해결때문이다. 단순threshold완화/반올림/좋은seed선택을하지않는다.

1. FFTdriver covariance/marginalIS를새모듈로구현한다. 작은grid의독립denseQR/solve/
   density와대조하고solver실패/수치범위도검사한다. 원래물리/과거source는보존한다.
2. 결과확인전고정한preflight:2cells,N4096,G1dictionary그대로.각373projectedmeans를
   준비하고독립4fixtureseed의각16draw를raw와주변화로평가한다.
   namespace fft-driver-marginal-calibration-v1의hashseed/rolepath,label분리.
   prep<=180초/cell,score<=10초/16drawchunk,RSS4GiB,최대PCG2048,
   residualrtol2e-13/atol2e-14/log-contribution sensitivity<=1e-4.
   모든fixture는성능추정에사용하지않는다. posterior/densityenergy/residual을보고한다.
   이점검은speedup/precision/whole실행시간보증이아니다.
3. preflight가통과할때만새namespace fft-driver-marginal-pilot-v1의고정
   2cells×8blocks×1024draw=16384draw pilot을수행한다. chunk16,
   같은z에서rawg/fullD와gbar/marginalD를paired로비교한다.
   fineN4096만비교하며coarse/continuumgate를통과시켰다고하지않는다.
   prep180초/unit600초/phase10800초/RSS4GiB.1024draw의시간예측이600초를넘으면
   실제producer를시작하지않고boundedpreflight실패/예산한계로기록한다.
4. 원래ordinarymean·logmoments·rawIIDSE/8block감도·paired차이2comparison·실제비용/
   수치오차/PCGrowcount를보고한다.고정r에서분산이늘지않는정리와관측variance비를구분한다.
   작은pilot에서1.5%정밀도를예상하거나성공할때까지연장하지않는다.
5. 각cellrep0의2block재현감사뒤전체회귀와결과보고서를작성한다. 실패도보존한다.
   독립SMC경고를이pilot만으로없어진것으로취급하지않는다.

preflight가실패하면원인을분리해다음version계획을만들며같은namespace를조건완화해
재실행하지않는다. 수식/likelihood/solver가틀리면해당구현부터수정하고성능주장중단.
실제해결성과가확인되기전P1/P2/large-trainingflags는false로유지한다.
