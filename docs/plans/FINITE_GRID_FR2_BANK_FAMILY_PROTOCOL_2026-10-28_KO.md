# FR2 bank × family 원인 분리: 실행 전 고정 계약

2026-10-28. FR2 reference-only pilot 이후, 후보 결과를 관측하기 전에 작성했다.
기존 P2는 미해결이다. 새 기하/큰 모델/confirmation은 실행하지 않는다.

## 1. 고정 대상과 예산

기본 로드맵의 D02(digital)와 D04(put/K)를 사용한다. 두 payoff 유형을 사전에 포함하며,
모든 D01–D04가 reference feasibility를 통과했으므로 후보 성능에 따른 task 선택은 아니다.
각 task에서 부모 생성부터 새로 시작하는 whole3회, 총6whole를 고정한다.

- parent: 3회×2048, event-weighted K2 identity Gaussian mixture fitting.
- bank: parent q0와 static physical G에서 각4096 새 IID 경로.
- family: 같은 bank에서 K2 유지와 K1 압축. 동일 refit loss/Adam40step/step size .04.
- final: q0/static 및 bank×family4개, 총6arm×32768 새 IID 표본.
- risk H: .5q0+.5static을 각32768씩 known-mass 층화.
- risk P: 자연 p에서65536 새 IID 표본. H와 독립이며 모두 raw ordinary IS 위험 기여를 사용한다.
- whole당342,016회, 총 **2,052,096회** 새 payoff 평가.
- 대표 재생 감사: task별 whole0, 총 **684,032회**, 새 독립 표본으로 더하지 않는다.
- 직전 pilot+감사2,162,688회와 합계 **4,898,816회**가 FR2 cap8백만 안에 든다.

FR2 계산 wall cap3600초 안에서 새 실행은1800초로 제한한다. 앞 pilot의 계측24.14초 외
startup/archive/audit의 완전한 공통 wall 계측은 없으므로, 이전 사용분에 보수적으로600초를 예약한다.
600초는 실측값이 아니다. 실패/테스트/감사 연구비와 실제 child wall은 따로 보고한다.
예산 장부를 runner마다 초기화하지 않으며, 재생 시에도 소비를 누적한다.

## 2. 정확성·공정성

parent는 이 실험의 event-weighted clustered CE 방식이다. 기존 문헌 iCEred 전체 구현이 아니다.
put에서는 put/K를 학습하고 digital payoff로 잘못 대신하지 않는다. covariance는 모두 I다.
parent K2가 실제로 유지되지 않으면 해당 whole은 실패로 남긴다. 좋은 parent만 다시 뽑지 않는다.

static은 mixed-window guide의 amplitude[2,4,6], price amplitude[0,1,2,3]와 .1 natural mass다.
373개 성분이므로 q0와 같은 capacity baseline이 아니라 강한 practical guide/자료 원천이다.
K2/K1은 동일 train bank와 optimizer 기회를 사용한다. 초기값 차이와 optimizer 제한도 결과 해석에 포함한다.
압축 arm의 초기 learned mean은 parent learned mixture의 평균이다. 두 family 모두 독립 final로 평가한다.

H의 stratum별 covariance를 알려진 질량으로 합산한다. pooled-IID 분산은 쓰지 않는다.
H와 P에 같은 q를 평가하여 paired difference와 두 reference의 차이를 구분한다.
bank ESS·parent ESS는 보조 진단이며 독립 표본 수나 coverage 인증이 아니다.

## 3. 사전 대비와 계속/중단 기준

네 방향을 고정한다.

1. CE-bank K2 vs CE-bank K1: 표현 손실.
2. static-bank K2 vs static-bank K1: 표현 손실의 자료 의존성.
3. CE-bank K2 vs static-bank K2: 같은 family의 자료 효과.
4. CE-bank K1 vs static-bank K1: 같은 family의 자료 효과.

앞쪽이 더 나은 방향을 검정한다. 반대 방향의 관측은 개발 결과로 보고하되,
사후 방향을 뒤집어 같은 확인 성공으로 세지 않는다.
4대비×6whole×2risk reference에 Bonferroni-normal 근사 diagnostic을 사용한다.
각 대비의 최소 관심 효과는 M2 10% 감소다. `.9J_baseline-J_candidate`의 lower>0을 검사한다.
두 J의 관측 risk RSE<=10%, H/P 평균 차이와 근사 radius가20% margin 안에 있어야 signal이다.
이 기준도 유한표본 tail coverage나 population 우열의 rigorous 증명이 아니다.

동일한 대비가 **두 task에서 각각3whole 중2회 이상** 지지될 때만 후속 작은 FR3 가설을 선택한다.
새 hypothesis 확인은 FR3의 별도 bank/final에서 해야 한다. direct event precision5%와 비용도 같이 보고하되,
이번 실험은 fixed-precision speedup이나 학습 일반화의 최종 입증이 아니다.

모든 대비가 미달하면 현재 자료·표현 원인으로 FR3를 정당화하지 못한 것으로 판정한다.
실제 차이가 작음과 reference 부족을 분리하고, 결과에 맞춰 새로운 kernel을 자동 추가하지 않는다.
모든 실패·미자격 whole를 분모3에 남긴다. 표본/seed/step/cap 변경 없이 종료한다.
