# Guide-free gradient umbrella: 수학·기술 재검토

2026-10-10. 신규 reference kernel의 finite-grid 계약 검토다. 새 일반 정리·성능 우위
또는 continuum convergence 주장이 아니다.

## 1. 정확한 양방향 MH

πβ(z) ∝ exp(-||z||²/2) H(z)^β. a=sqrt(1-s²), c=1-a.
고정 deterministic v(z)에 대해 Qβ(z,·)=N(a z+cβv(z),s²I).
v는 ∇logH 또는 그 고정norm cap20 버전이다. 따라서 Gaussian log density를 직접 쓰면

    logα = -.5(||z'||²-||z||²)+β(h'-h)
           +[||z'-a z-cβv||²-||z-a z'-cβv'||²]/(2s²).

prior와 zero-drift proposal의 quadratic terms가 상쇄된다. c/s²=1/(1+a)이므로

    logα = β(h'-h)
           +β/(1+a)[(z-a z')·v'-(z'-a z)·v]
           +cβ²/[2(1+a)](||v||²-||v'||²).

코드는 두 번째 식을 사용하고 첫 식과 numerical agreement를 검사한다.
β=0 또는v=v'=0에서 prior-reversible pCN으로 환원한다. plainβΔh만 사용하는 것은 오류다.
clip한v를 proposal에 넣었다면 reverse density에도 실제 clip한v'를 넣는다.
proposal drift 제한은 target truncation이 아니다. g/H/endpoint weight를 clipping하지 않는다.
known Gaussian stationary law, 양방향 logratio antisymmetry와 min(1,α) 상세균형도 검사했다.
이는 pCN/pCNL 계열의 finite-dimensional MH 활용이다.
[원 방법 계열](https://arxiv.org/abs/1202.0709)의 일반 robustness 결과를 이 rare-event
target에서 증명한 것으로 외삽하지 않는다.

## 2. Derivative와 계산 법칙

callback은 각row의component를 그row의latent만으로 계산해야 한다.
sum_row logH에 대한 autograd는 이 조건에서 row별 ∇logH를 반환한다.
batch간 정규화·batch statistics를 target callback에 넣으면 이 조건이 깨질 수 있다.
현재 FFT rBergomi/conditional Gaussian CDF/frozen shift-mixture density는 row-separable이며,
singleton-vs-batch와 directional finite difference를 실제N32에서 검사했다.

각callback의 log potential과 derivative는 CPUfloat64/finite를 검사한다.
nonfinite derivative나 variance overflow는 whole failure로 기록한다. failed draw만 제거하거나
다른 seed로 교체하지 않는다. norm cap은finite derivative 이후 적용한다.
constant callback은 derivative0이며 실제 backward pass를 실행하지 않는다.
core는 forward evaluations와 실제 autograd backward evaluations를 따로 센다.
실제물리callback은requires_grad이므로 이번gradient pilot에서는 두 수가 같다.
두 count를 하나의 기계종속 비용으로 합쳐 pCN 대비efficiency라고 해석하지 않는다.
실제 wall에 precomputation/I/O/fit/selection을 포함한 matched accuracy 비교가 추가로 필요하다.

## 3. SMC normalizer와 adaptedness

mutation은 각β에서πβ-invariant다. 고정schedule과unbiasedresampling을 유지하므로
unnormalized-measure induction과 component estimator ZhatΣWiHj/H는 그대로 성립한다.
같은 whole의 여러component는 상관된다. SE는 independent whole 단위다.
normalizer와endpoint ratio의sample correlation은진단이고variance bound가아니다.
mixture weights는core진입때clone/freeze한다. callback도frozenq만사용한다.

reference는complete latent path를 MH로 갱신할 수 있다. 이는 future 정보를 사용한
adapted Brownian pricing drift를 구현한 것이 아니다. 가격법칙의 Itô left integration은
변경하지 않았으며 final q도동결했다. reference의gradient와causal finalcontroller를혼동하지않는다.
fine-grid에도동일MH산술을쓸수있지만, 현재pilot는N32에서만수행했다.

## 4. 발견한 준비 비용 표기 보정

새canonical-only gradient pilot/resource sweep는공용prepare를재사용했다.
그prepare는cell당CoverageEnvelopeReference의geometry를64basis paths로한번만 만든다.
guide-free kernel에서는그reference를사용하지않는다. static guide를추가하지않으므로
실제준비path-law count는64인데, runner의준비기록은two-cell pilot의128을그대로썼다.
따라서두run의`work`는64회씩 **과다 계상**했다. 수치추정/gradient/seed는영향없고
workcap도더보수적이었다. 저장된manifest/checkpoint/result/source를소급수정하지않는다.

- target-separation pilot: recorded12,222,592, corrected path-law12,222,528.
- resource sweep: scheduled record14,327,936, corrected path-law14,327,872.
- replay audit도첫pilot1,527,936→1,527,872; sweep는동일하게64차감한다.
- high-eta validation/revalidation은envelopegeometry64+staticguidegeometry64=128이므로
  이보정대상이아니다. 원래two-cellumbrella의64×2=128도정확하다.

이차이는precisionpass여부나선택에쓰이는wholecostproxy를바꾸지않는다.
새프로토콜에서는사용하지않는guidegeometry를아예준비하지않고실제준비count를명시한다.
현재두canonicalartifact의companionerratum에원기록과보정값을함께남긴다.
추정량검증과정확한workaccounting을별개로검토해야한다는실제발견이다.

## 5. 남은 위험

gradient의local정보가희귀mode전체발견을보장하지않는다. acceptance나ancestry만으로
mixing을인증하지않는다. 작은pilotforecast는새고정validation에서깨질수있다.
독립staticIID/guidedSMC와상호검증및tail/mode분해가계속필요하다.
P1의전체two-cell/five-q, P2의mesh/continuum, learned added value는별도관문이다.
