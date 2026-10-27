# 다중 시점 잔차–Brownian covariance proposal: 고정 개발 실험

2026-10-26 America/New_York. 커밋·푸시하지 않는다. 기존 source/result/STOP은 보존한다.
P2 및 production 승인이 아니라 실제 사건 proposal q를 바꾸는 N32 개발 단계다.

## 1. 가설과 기존 방법과의 차이

기존 prior-centered local/collective scale reference는 물리 경로의 shifted mean을 사용하지 않았다.
이번에는 uniform373-component physical dictionary의 mean을 그대로 유지하고,
shifted component 내부에서 모든31개 사용 시점의 residual/직후Brownian 쌍을 넓힌다.
특정 최대기여 경로·시점·seed를 mean에 넣거나57에만 예산을 몰지 않는다.

좌표쌍은 `(2j−1,2j), j=1,...,N−1`이다. 첫 좌표는 V_j를 만드는 local residual,
둘째는 그 분산이 적용되는 다음 가격 구간의 Brownian white 좌표다.
unused terminal residual2N−1은 포함하지 않는다. N32 밖의 성능·continuum 동등성은 주장하지 않는다.

## 2. 네 가지 실제 사건 proposal

base `q0=w0 p+Σ_{k>0}w_k N(m_k,I)`, w0=.1, Σshiftw=.9다.
모든 후보에서 mean과 원래 mixture weights는 고정한다. 자연성분은 항상 그대로 보존한다.
shifted mass 안에서 α=.5만 covariance mixture로 바꾼다.

`q_a = .1p + .5Σshift w_k N(m_k,I) + .5/(N−1) Σ_jΣshift w_k N(m_k,Σ_j,a)`.

사전 covariance parameter v=5, price sign s=−sign(rho), rho=0이면s=1이다.

| arm | residual/price 쌍 covariance | 목적 |
|---|---|---|
| base | I | 강한 static baseline |
| residual | diag(v,1) | 잔차 크기 확대만 |
| independent | diag(v,v) | 독립적 쌍 확대 |
| coupled | [[v,s(v−1)],[s(v−1),v]] | 같은 marginal variance에서 coupling 추가 |

주후보는 coupled다. 나머지는 ablation이며 결과를 본 뒤 주후보를 바꾸지 않는다.
coupled의 eigenvalues는2v−1=9 및1이다. positivity/정규화가 정확히 알려져 있다.
세 후보 모두 q_a≥.1p 및 q_a≥.5q0이므로 support와 `M2(q_a)≤2M2(q0)` 안전상계가 있다.
이는 개선 보장이 아니며, Gaussian covariance mixture 적용의 독창성도 주장하지 않는다.

## 3. 정확한 density와 sampling

각 k,j의 whitened 차이 a=z_odd−m_k,odd, b=z_even−m_k,even에 대해
identity Gaussian 대비 covariance log-ratio는 다음과 같다.

- residual: `.5(1−1/v)a²−.5log v`
- independent: `.5(1−1/v)(a²+b²)−log v`
- coupled: `.25(1−1/(2v−1))(a+s b)²−.5log(2v−1)`

component mean likelihood `z·m_k−||m_k||²/2`와 더하고 k,j 전체를 logsumexp한다.
단일 생성 label의 denominator로 바꾸지 않는다. matrix chunk로 memory를 제한한다.
sampling은 base component label, covariance 여부, 시점 label, 독립 standard Gaussian을 뽑은 뒤
정확한 square-root covariance transform을 적용한다. 자연 component는 확대하지 않는다.
유한격자 Gaussian change-of-measure이며 미래를 보는 SDE drift controller로 해석하지 않는다.
원래 adapted Itô simulator는 변경하지 않고 전체 경로 밀도로 보정한다.

## 4. 다른 geometry의 보조 reference

guide means 없이 `.1p+.9/(N−1)Σ_j N(0,Σ_j,coupled)`인 reference R을 별도로 둔다.
shift dictionary에 의존하지 않는 sampling geometry지만 물리 simulator/CDF는 공유한다.
따라서 독립 simulator 검증 또는 tail coverage 인증이라고 부르지 않는다.
기존 local pair(2j,2j+1) 대신 인접 구간의(residual,직후Brownian) 쌍을 쓰는 차이도 명시한다.

## 5. known-answer 및 실행 전 검사

1. 작은2N=4/6 mixture의 전체 component를 dense covariance Gaussian으로 열거하여 density 대조.
2. source/sample의 mean/covariance를 analytic mixture와 대조하고, 자연성분 보존·α0 identity 검사.
3. reference의 정규화와 density를 독립 dense mixture로 확인한다.
4. invalid rho/sign/v/α/nonfinite/잘못된 shape·중복seed·missing unit·overwrite·budget를 검사한다.
5. shifted conditional scale를 z 자체가 아니라 z−m에 적용하는 반례를 검사한다.
6. 불안정한ratio·낮은RSE가 채택을 우회하지 못하는 계약 검사를 한다.

## 6. 고정 실제 실험

N32 canonical/high-eta 두 셀,CPUfloat64/1thread,노트북만 사용한다.
모델 학습/선택은 없고 사전 주후보를 그대로 평가한다. 따라서 independent training 반복의
성과로 세지 않는다. 새 namespace `multitime-residual-brownian-covariance-v1`를 사용한다.

- preflight: 셀×arm 각각128개=1024 physical; all-density 계산과 full mean/cov parameter 검사.
  첫 고정batch의 비용예측이 phase1800초를 넘으면 production을 시작하지 않고 비용 실패로 남긴다.
- direct final: 각 셀×4arm×8block×8192=524,288 physical. 각 arm 자기 q에서 g/Dq.
- heldout H: 각 셀×8block×4strata×2048=131,072 physical.
  `H=(q0+qres+qind+qcoup)/4`를 사용하고 각 stratum mean을 known mass1/4로 합친다.
  g²/(D_H D_qarm)를 모든4arm에 공통 평가한다. 층별 covariance와 block sensitivity를 계산한다.
- guide-free R: 각 셀×8block×4096=65,536 physical. g²/(D_R D_qarm)로 같은4M2를 평가한다.
- 총production **720,896 physical**,preflight 포함 **721,920**.
- source/config/plan/기존intervention anchor를 sampling 전에 봉인하고 모든seed를 먼저 할당한다.
  full cap730,000,phase1800초,unit60초,RSS4GiB,density chunk256.
  actual covariance-component score 수와 density/physics/sampling wall을 별도로 기록한다.

## 7. 개발 판단: 채택 우회 금지

- 주후보 direct event rawRSE≤5%,8blockRSE≤2rawRSE,maxpath≤10%,LOO≤15%.
- H/R 각4M2에 raw 및blockRSE≤20%,maxblock≤25%,LOO≤15%를 요구한다.
- coupled/base paired absolute 및relative 감소의 raw/block lower가 모두 양수이고,
  relative lower>20%여야 한다.2cells×2geometry=4comparisons에Bonferroni normal/delta 진단.
- H/R 비교는 각geometry가 위precision을 얻은 뒤에만 수행한다.
  2cells×4M2=8comparisons,10% symmetric equivalence margin을 개발 진단으로 쓴다.
- raw/block 작은 표본 CI는 tail 인증이 아니며 같은geometry 공통누락은 남는다.
- 위가 모두 통과해도 별도confirmation 후보일 뿐 P2/production 승인이 아니다.
- fixed-count의 실측비만 보고한다. 같은accuracy의fixed-precision total-wall 우위는 주장하지 않는다.
- coupled가 실패했는데ablation만 유리하면 사후 주후보 교체 대신 별도 새 계획이 필요하다.

## 8. 감사와 종료

실제 unit 수는 direct64+heldout16+reference16=**96units**다.
heldout 각unit은4strata×2048을 모두 완료해야 한다. partial unit은집계하지 않는다.
전수array/hash/moment/통계/seed/source·기존anchor를 재검사한다.
대표각셀 direct coupled block0(16384),heldout block0(16384),R block0(8192)를 재생한다.
audit physical40,960, cap45,000/wall600초. 새 확인 자료로 pooling하지 않는다.

전체pytest/Ruff/Mypy 후 README와 결과보고서를 갱신한다. 실패도 보존하며 count/seed/variance/
coupling sign을 같은실행에서 바꿔 반복하지 않는다. 실패하면 어떤geometry에서도risk가
불안정했는지/실제event만 나아졌는지/밀도계산비가 문제인지 분리해 다음 판단을 남긴다.
기존P2 fine reference/mesh/continuum gate는 별도 미완료로 유지한다.
