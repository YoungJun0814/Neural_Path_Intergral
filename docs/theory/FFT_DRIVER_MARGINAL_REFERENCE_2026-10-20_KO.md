# FFT driver marginal reference: 정확한 수식과 수치해법의 구분

2026-10-20. Gaussian conditioning/조건부중요도샘플링의 알려진 원리를 현재 original BLP에
적용한 구현 검토다. 독창성/속도 우위/논문 수준을 인증하지 않는다.

## 1. 원래 유한격자 법칙

N개cell의독립white좌표(z0,z1), n=N−1개left-volatility driver Y를 쓴다.
Y=B z0[:n]+σ z1[:n]. B는lower triangular Toeplitz이며
b0=sqrt(2H)L10, b_l=sqrt(2H)L00*k_l(l>=1), σ=sqrt(2H)L11이다.
L은원래local Gaussian Cholesky, k는원래 historical **cell-mean** kernel이다.
Σ=BBᵀ+σ²I>=σ²I>0. diagonal은원래driver variance와일치한다.
FFTlinearconvolution과reverse-convolution으로 B,Bᵀ,Σ를작용시켜N²matrix를저장하지않는다.
localcell적분/compensator/leftItô target은바꾸지않는다.

## 2. Gaussian 가격 잔차 적분

Y가주어지면v=(ξ,ξexp(ηY−η²diagΣ/2)), f=sqrt(v), I=hΣ_k v_k,
a=B f[:n]. ΔW의조건부평균을사용하면

    m=logS0−I/2+ρ sqrt(h) aᵀΣ^(-1)Y,
    V=I−ρ² h aᵀΣ^(-1)a >= (1−ρ²)I>0,
    g_bar(Y)=Φ((logK−m)/sqrt(V)).

마지막ΔW와독립가격Brownian을포함한모든남은Gaussian가격잡음을적분한다.
미래Y를조건으로쓰는것은평가기법이며원래SDE에미래정보control을넣는것이아니다.
작은grid의기존denseQR core와같은수식이다.

## 3. likelihood도 함께 주변화해야 함

원래r(z)=Σ_l w_l φ(z−m_l), p=N(0,I), a_l=A m_l라하면

    D_Y(Y)=r_Y/p_Y=Σ_l w_l exp(YᵀΣ^(-1)a_l−a_lᵀΣ^(-1)a_l/2).

새추정량은r에서z를뽑아Y=Az를만든뒤 **g_bar(Y)/D_Y(Y)** 의ordinarymean이다.
g만바꾸고기존full D(z)로나누면일반적으로틀리므로금지한다.
자연성분질량δ=.1이있으면D_Y>=δ, g_bar/D_Y<=10.
각projectedmean의energy<=원래meanenergy는Gaussianprojectionidentity검사다.

원래X=g(z)/D(z),새U=g_bar(Y)/D_Y(Y)라하면

    E_r[X|Y]=U, E_r U=μ, Var_r U<=Var_r X.

증명은r(z|Y)=r(z)/r_Y(Y)에p/r를곱해적분하고조건부Jensen을적용하면된다.
0<=X<=10이므로L²조건도충족한다. 이는**고정r의μ estimator**에대한관계이지
원래q의M₂를g_bar²로대체하는규칙이아니다. 계산시간까지포함한효율을보장하지않는다.

## 4. PCG와수치민감도

Σx=b의근사xhat와actual recomputed residual e=b−Σxhat에서
||x−xhat||<=||e||/σ². solve는고정rtol2e-13/atol2e-14/max2048을사용하고
실제residual을검사한다. 미수렴표본을제외하거나더작은weight로바꾸지않는다.
row별독립선형계이며대형denseQR/Cholesky를N4096에사용하지않는다.

이bound를m,V 및각densitylogit의dotproduct오차로전파하고, m±error,V±error의
네모서리CDF와logsumexp의supnorm Lipschitz성질로log contribution 민감도를계산한다.
모든표본에서추정상한<=1e-4(약0.01%로그오차)를요구한다. 기준실패시전체단계를실패로기록한다.
PCG covariance **row evaluation 수**는실제반복과residual재검사까지포함한다.
이는FFT/CG알고리즘비용이지기존physical simulator call과동일단위로섞지않는다.

이오차계산도float64이며FFT반올림/BLP입력형성의모든오차를interval arithmetic으로
감싼것은아니다. directed-roundingcertificate라거나bitwise정확한해라고주장하지않는다.
허용오차는통계RSE목표보다작은수치민감도점검이다. 작은denseoracle·큰grididentity검사·
독립μ평가가추가로필요하다. exact라고부를수있는것은**선언한finite-grid의해석적식**이다.

## 5. 검증순서

작은grid에서independentdriverrows/covariance/dense solve/QRconditionalCDF/marginalmixture
를대조한다. N4096성능은별도고정preflight에서시간·메모리·residualbound를통과해야한다.
그뒤에만같은표본의raw/reference pairedpilot을시작한다. 아직precision/독립reference/
continuumrelativebias/P2/large-training을승인하는작업이아니다.
