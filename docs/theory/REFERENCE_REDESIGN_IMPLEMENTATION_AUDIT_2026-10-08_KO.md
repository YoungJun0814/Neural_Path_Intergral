# Reference 재설계 구현의 수학·수치 계약 감사

작성일: 2026-10-08. 성능 입증 보고서가 아니라 구현 계약 검토다.

## 1. 변경 범위

원래 rBergomi finite-grid simulator, frozen parent q, payoff target은 변경하지 않는다. 추가한 것은 exact shift-mixture marginal, last-pair cache, unbiased nested raw-M₂ evaluator, fixed-block full-path evaluator, Gaussian-prior elliptical-slice mutation, role/SE 계약 및 bounded 실행기다.

`weighted_tempered_smc`의 기본 kernel은 계속 pCN이다. 새 ellipse kernel은 명시적 opt-in이며 static independence mutation과 동시에 사용할 수 없다. 기존 resampling calendar·temperature normalizer·마지막 stage 무변이 규칙은 유지한다.

## 2. μ identity

마지막 pair가 zero인 경로에서 a=logK−logS_T, b=ρ√(v_(N−1)dt), c²=(1−ρ²)I를 캐시한다. Cholesky 첫 행과 left-point variance에서 g=Φ((a−bx)/c)가 나온다. 마지막 y는 가격 payoff에 영향을 주지 않지만 terminal volatility는 바뀔 수 있다.

독립 Gaussian x 및 price noise를 합하면 E[g|A]=Φ(a/√(c²+b²)). mean의 부호, 마지막 left variance index, N=1의 empty prefix를 별도 검사한다. Φ를 log_ndtr로 계산한다. 현재 공개 evaluator는 ε 인자를 받아 target을 바꾸는 기능이 없으며 ε=1만 지원한다.

## 3. original M₂의 denominator 보존

    Y(A,B)=g(A,B)² / (q(A,B)/p(A,B))

outer r_A와 inner φ₂의 곱분포에서 (p_A/r_A)Y의 평균은 ∫g²p²/q다. μ의 조건부 평균을 제곱하는 연산은 수행하지 않는다. cached density에는 **full mean norm**을 유지하므로 마지막 pair 평균을 잘라 전체 q의 normalization을 변경하지 않는다.

q의 마지막 y 평균도 그대로 남긴다. r_A는 prefix mean의 norm으로 정확한 marginal ratio를 계산한다. 이 두 norm은 다르며 혼용할 수 없다. non-rank-zero covariance는 명시적으로 거부한다.

## 4. 분산과 독립 단위

L개의 inner draw를 평균낸 Y_i가 하나의 outer unit이다. 각 outer vector와 그에 속하는 inner draw들이 다른 outer unit과 독립이면 ordinary sample variance/n으로 SE를 추정한다. nL을 표본 수로 사용하지 않는다.

서로 조건부 독립인 두 inner 평균 X_i,Y_i를 같은 A_i에서 만들면

    E[(X−Y)²]/2 = E[Var(X|A)]
    Var((X+Y)/2) − E[(X−Y)²]/4 = Var(E[X|A]).

scaled coordinates에서 분산 진단을 수행한다. 이 차이가 표본오차로 음수가 나오면 clipping하지 않고 음수 진단을 보존한다. 같은 outer를 공유한 서로 다른 L의 결과는 독립 estimator가 아니다. L 비교는 개발 screening이며 independent-equivalence CI의 입력으로 사용하지 않는다.

## 5. nonterminal block의 미래 기억

고정 block 좌표를 변경하면 해당 local pair를 full coordinate tensor에 삽입하고 원래 simulator로 전 경로를 다시 계산한다. I/J/미래 variance를 고정한 affine-CDF 식은 last pair에만 사용한다. path-dependent peak 좌표 선택은 구현하지 않았다.

## 6. ellipse 이동의 정합성

Gaussian prior p와 deterministic potential h에 대해 bridge likelihood는 h^β다. 구현은 [Murray–Adams–MacKay 원 논문 Figure 2](https://proceedings.mlr.press/v9/murray10a/murray10a.pdf)의 Gaussian auxiliary draw, uniform initial angle, 전체 2π bracket, log slice threshold, zero-angle 기준 bracket shrinking을 따른다. 배치 구현은 particle마다 별도의 noise/angle/threshold를 사용한다.

이 invariant transition을 고정 bridge SMC에 넣는다. slice 내부의 likelihood 호출 수는 가변이므로 모든 active-particle 평가를 센다. mutation acceptance=1은 slice의 최종 이동 정의에 따른 값이지 pCN의 MH acceptance와 같은 효율 지표가 아니다. 평가 수와 wall-time을 같이 보고해야 한다.

finite attempt cap의 잘린 이동은 반환하지 않는다. cap/budget에 닿으면 전체 실험을 protocol failure로 표시하며 이후 job도 실행하지 않는다. 따라서 일부 빠른 whole-runs만 골라 reference를 만들지 않는다. finite cap이 존재하는 실행에서 무조건적 unbiasedness를 증명했다고 주장하지 않는다. 정상 완료 조건의 경험적 안정성을 별도 확인해야 한다.

## 7. 생산 실행과 예산

final sample을 생성하기 전에 pilot file SHA, q artifact SHA, complete allocation, total budget를 봉인한다. production config를 allocation report의 내용과 완전히 비교한다. 불일치하면 실행하지 않는다. 각 run의 source ZIP에 당시 source/config/tests/계획을 넣고 runtime digest 불변을 검사한다.

이번 실행기의 `potential_equivalent_evaluations`는 **FFT path 평가 수 + CDF 평가 수**다. 한 full potential 호출은 보통 두 단위로 센다. 과거 P1의 `potential_evaluations`와 그대로 숫자 비교하면 안 된다. 두 원시 count, density-component count, guide probe, actual wall/RSS를 별도로 기록한다. 기존 cap의 숫자는 유지하므로 더 보수적인 실행 한도다.

고정 block 묶음은 4batch×256outer×3block×2cell, L=1/4/16/64, 두 내부 반복으로 사전 고정한다. 경로/CDF 합계는 대략 209만이며 guide probe가 추가된다. 4batch×512outer로 세 block을 모두 돌리면 새 count 계약의 300만 cap을 예측상 초과하므로 실행 전에 줄였다. 마지막-pair microstudy는 4×512를 유지한다.

## 8. 검증 한계

- analytic Gaussian posterior stationarity·known normalizer 검사는 우리 희귀 tail에서 mixing을 보장하지 않는다.
- bounded contribution과 finite variance는 유용한 relative precision 보장이 아니다.
- empirical SE·bootstrap·equivalence는 distribution-free coverage 증명이 아니다.
- exact algebra와 floating-point 구현은 구분한다. log-domain을 사용해도 극단적 overflow·nonfinite는 실패로 처리한다.
- oracle 통과와 실제 성능·최상위 저널 novelty는 별개다.

기술적으로 발견한 seed 누락 경계 오류는 명시적 ValueError로 수정했다. 수정 이전 회귀 실패를 신규 방법의 성능 실패로 혼동하지 않으며, 최종 회귀 결과와 실제 실험 상태는 별도 실행 보고서에 기록한다.

실행 후 추가 점검에서는 source binding 재구성에도 실험의 1-thread convention이 필요함을 확인했다. 감사에서 이를 강제했고, 저장된 표본·guide·판정 기준을 바꾸지 않은 채 원자료 일치 검사가 통과했다. 최종 현재 source의 전체 1,233개 회귀와 Ruff/Mypy가 통과했다. [실험 결과 및 production 잠금 사유](../reviews/REFERENCE_REDESIGN_IMPLEMENTATION_AND_EXPERIMENT_REPORT_2026-10-08_KO.md)는 이 theory 계약의 수치적 성능 입증과 별개로 읽어야 한다.
