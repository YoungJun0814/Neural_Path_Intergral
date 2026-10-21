# Fine-only reference와 새 정밀도 검증의 수학 계약

2026-10-20. 기존 구현·이론을 현재 target에 적용한 검토. 새 정리라는 주장이 아니다.

원래 terminal downside의 독립 가격 driver를 해석적으로 적분한 g_f(z)∈[0,1],
z∼N(0,I_2N), μ_f=E_p g_f를 두 방법 모두 동일하게 계산한다.

## 앙상블 IID

과거 bank로 정한 weights를 고정하고 새 r_e=p D_e에서 X=g_f/D_e를 계산한다.
bank 조건부 E X=μ_f다. natural mass .1에서 X≤10이지만 상대 오차 보장은 아니다.
개별 sample이 bank 조건부 IID이므로 raw SE를 계산할 수 있고, equal-size block
평균의 변동도 별도로 본다. 독립 새 block을 과거 pilot에 덧붙이지 않는다.

pilot에 의존한 사전 sample-count 선택은 새 독립 final의 평균을 바꾸지 않는다.
하지만 목표 정밀도, CI coverage, 누락 tail의 부재를 보장하지는 않는다.
같은 frozen weights의 새 final은 새 bank 학습 반복이나 저널용 confirmation과 다르다.

## Fine-only guide-free SMC

β∈[0,1]에 대해 π_β∝p g_f^β, Z_0=1, Z_1=μ_f다. 고정 β schedule에서
normalized 입자 가중치와 각 increment normalizer의 곱을 함께 유지한다.
최종 단일 component normalizer는 전체 SMC normalizer와 일치해야 한다.
μhat_SMC는 독립 whole의 Zhat를 ordinary mean한 값이다. log Zhat를 평균내는
geometric mean 또는 최종 normalized particle 평균으로 대체하지 않는다.

coarse target과 독립 augmentation을 제거해도 μ_f는 변하지 않는다. 다만 과거
umbrella H=(g_f+g_c)/2와 bridge 분포·mutation dynamics·estimator variance는 달라진다.
fine-only가 효율적이거나 덜 퇴화한다는 결론은 실험 전에 가정하지 않는다.

사용하는 pCN-L 후보는 a z+(1−a)β d(z)+s ξ, a=sqrt(1−s²)이며 d는 cap을 적용한
∇log g_f다. full reverse/forward Gaussian density correction을 유지한다.
cap은 제안 drift에만 적용하며 target/likelihood/SMC weights는 자르지 않는다.
row-separable 원래 cached CDF에 대한 autograd를 사용한다. PCG 주변화의 미검증
implicit gradient를 쓰지 않는다. kernel에는 G1 dictionary나 ensemble weights를 입력하지 않는다.

고정 schedule의 정확한 MH 불변성과 unbiased resampling은 정규화 상수 추정의
기본 정합성을 지지하지만, 유한 입자의 혼합·tail coverage·상대 오차를 보장하지 않는다.
ancestry는 initial lineage를 mutation 뒤 리셋하지 않고 추적한다.

## 비교와 통계 범위

I의 seed/stream과 G의 whole seed는 분리한다. frozen bank 조건부로 두 메커니즘은
독립이고, 그 평균은 같은 μ_f다. 따라서 독립 SE 기반 개발용 delta-method 비교를 한다.
sample의 낮은 RSE만으로 정확성을 증명하지 않는다. 소수 whole와 tail에서 정규근사
구간은 취약할 수 있다. 정밀도 자격과 동등성 검사를 구분하고 다중비교를 조정한다.

양쪽 정밀도가 충분하지 않은 쌍에서 일치 구간이 넓으면 unresolved다. 차이 구간이
±10% margin 밖에 완전히 있으면 detected development disagreement로 표시하지만,
어느 쪽이 참값인지 확정하거나 영구적 bias가 있다고 단정하지 않는다.

두 방법이 개발 기준을 통과해도 fine μ에 국한된다. q의 M2를 조건부 g의 제곱으로
바꾸지 않고, mesh/continuum bias와 all-q P1/P2·논문 독창성은 별도다.
