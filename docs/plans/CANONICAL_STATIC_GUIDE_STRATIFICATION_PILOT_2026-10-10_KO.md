# Canonical static guide: 알려진 component 고정 층화 pilot

2026-10-10. 새pilot 생성 전 동결. reference의 normalized density나 q/g를바꾸지않는다.
canonical staticμ의이전allocation은8,986,624>8,388,608cap이었다.
cap/safety를바꾸는대신불필요한mixture-label변동을제거할수있는지검사한다.

## 정확한 추정과 SE

r=Σwj rj는원래373component Gaussian shift guide다. K=372,
w0=.1, shifted wj=.9/K. Hμ=g/(r/p), Hrisk=g²/[(q/p)(r/p)].
각rj에서정해진nj개를뽑고 μhat=Σwj mean_j(H)를계산한다.
모든draw의denominator는각component density가아니라 **full r mixture density**다.
E μhat=Σwj E_rj H=원래μ 또는M2다. empirical/self-normalized weights없음.
Var μhat=Σwj² Var_rj(H)/nj; SE도이stratum variance를사용한다.
pooleddraw를IID라고취급한samplevariance는금지한다.

고정비례allocation n0=K*m, nj=9*m, total10*K*m은원래wj와일치한다.
이때전체variance는IIDmixture의law-of-total-variance에서nonnegative label-mean term을
제거한것이다. 그렇다고실제rare-tail precision이충족된다는보장은없다.

## fresh실험

canonical,N32,frozen5q,μ+5M2를같은새draw에서계산한다.
ordinaryIID와fixedstratified 두arms,각8blocks. m=8,block당29760draw.
각block의mean/stratum moments/seed를저장하고최종stratum별8blocks를merge한다.
총16blocks476160draw +staticguidegeometry64=476224potential calls.
cap600k/600초/RSS4GiB,block30초. namespace `canonical-static-strata-v1`.
wholeblock가sampleallocation unit이며final에새block를추가하지않는다.
기존pilot/final과합치지않는다. 원래density373Gaussian의identitycovariance계약을검사한다.

## 판정

두arm의μ/5M2 RSE,block민감도및tailshare를비교한다.
staticIID와층화는같은physicalguide이므로두독립메커니즘의corroboration으로세지않는다.
RSE1.5%/safety2,wholeblock multiples,min32blocks,max8,388,608draw의새allocationforecast.
forecast가능해도정밀도자격아님. selectedstaticmethod는새finalnamespace에서검증해야한다.
가능한후보중최악estimand의CV²×actualwall이가장작은설계를고른다.
두방법모두실패하면cap을무시하지않고coverage/stratificationallocation새설계로돌아간다.
gradient-SMC와guided-SMC를포함한canonicalfresh전체reference는다음별도프로토콜이다.
