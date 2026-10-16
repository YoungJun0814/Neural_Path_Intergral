# 미세 격자 정적 shift reference: exact implicit mixture 진단

2026-10-16. 생성 전 고정. N32에서 검증했던 정적 Gaussian shift 가이드를
N4096에서 거대한 평균 행렬 없이 구현한다. 원래 BLP law를 바꾸지 않는다.

## 식과 검증

left standardized driver Y_t=u_tᵀz, 다음 가격 white coordinate P_t=e_(2t)ᵀz,
u_tᵀe_(2t)=0. Gaussian component mean은 a u_t+b sign e_(2t)이며
component/p=exp(aY_t+b sign P_t-(a²+b²)/2)다. t,a,b가 균등한 product grid이므로
합은 Σt[Σa exp(aY_t-a²/2)][Σb exp(b sign P_t-b²/2)]로 factorize한다.
natural massδ=.1을 logaddexp로 포함해 full normalized mixture density를 계산한다.
price sign=-1 if rho>0, else+1. threshold conditioning이나 empirical normalizer 없음.

FFT로 Y_t를 계산하고 sample 시 선택한 u_t만 analytical kernel로 생성한다.
준비 O(N), score O(B*N*(|A|+|B|)+B*N logN), sample memory O(B*N).
N32 dense 원본과 모든 component mean/full density를 비교하고, sampler PRNG
재구성, natural component exact 보존, N4096 저메모리 준비를 검사한다.
동일한 알려진 mixture의 계산 개선이며 새로운 일반 원리/learned model은 아니다.

## 고정 pilot

원래 두 셀, fine4096/coarse2048. 각8 independent blocks, block32768fine samples,
chunk512. amplitudes2/4/6/8, price0/2/4, δ.1. 총 fine524288draw와 동일한 independent
coarse augmentation. raw physical g evaluation1,048,576, density scoring524288회,
time-step physical forward3,221,225,472회. density FFT 비용은 별도로 발생한다.
준비 basis-path simulation0회. namespace `fft-static-fine-reference-v1`.

fine에서 q를 생성하고 ε는 prior로 생성한다. coarse exact marginal을 만드는 coupling을
쓰되 likelihood는 qfine/pfine 하나다. coarse density를 별도로 곱하지 않는다.
두 g/Dr의 ordinary means와 signed paired difference를 그대로 평가한다.

cap physical1.2m, density0.6m, phase7200초, block240초, prep30초,RSS4GiB.
중간 성공 판정/seed 교체/기존 데이터 pooling 없음. 16 blocks 전체를 완료한다.
source/model/anchor/seed/moment/cost와 각 셀 첫 block의 numerical replay를 감사한다.

pooled μ RSE, maximum single contribution와 block sensitivity, 2개 비교의 paired
10% small-difference upper를 보고한다. 8block의 bootstrap/normal SE는 tail certificate가
아니다. 새 static reference는 기존 deterministic geometry와 관련되므로 다른 guided
reference와의 일치만으로 guide-independent blind spot이 해결됐다고 주장하지 않는다.

성공적인 pilot여도 final precision allocation, 독립 gradient SMC 대조, continuum
bias/주장 범위를 따로 검토해야 한다. P1/P2/large-training flag는 false로 유지한다.
실행은 canonical μ 이후 또는 unit/phase 예산이 지켜지는 범위의 가벼운 동시 진단만
허용한다. 이후 resource-v2 대형 실행과는 겹치지 않는다.
