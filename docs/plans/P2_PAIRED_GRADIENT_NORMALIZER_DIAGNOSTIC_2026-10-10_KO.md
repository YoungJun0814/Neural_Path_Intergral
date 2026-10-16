# P2 병목: exact joint paired-gradient normalizer 진단

2026-10-10. 새pilot이전동결. P2confirmation이나large-training을승인하지않는다.
canonicalfullreferencevalidation과는독립namespace의작은method진단이다.

## 기존 실패와 새 구조

FFT envelope의N1024/2048/4096 IIDdiagnostic은fineμRSE9–32%,highηN4096의
차이점추정1.29%/uncertaintyupper20.66%로small-difference실패였다.
같은envelope에서차원만늘리는대신두grid의rarecontribution를함께탐색하는SMC를검사한다.

joint x=(zf,ε)~N(0,I5N/2), zf∈R2N, ε∈RN/2.
zc=AdjacentBlockLatentAdapter(zf,ε)로정확한coarse표준Gaussianmarginal을유지한다.
coarseL=C1+L2와기존BLP/Itôleft법칙을유지한다. naivefuturetrapezoid없음.
Hf=gf(zf), Hc=gc(zc), H=(Hf+Hc)/2,0<H<=1.
πβ ∝ pjoint H^β의guide-freegradient-MH를실행한다.
endpointvector Zhatf=ZhatΣWiHf/H, Zhatc=ZhatΣWiHc/H.
각각원래finite-gridμf/μc이고signeddifference Zhatf-Zhatc를그대로사용한다.
Hf/H,Hc/H<=2지만common Zhat의변동이나mode누락을배제하는bound는아니다.

normalizingratio의추정으로정확한likelihood를대체하는것이아니다. SMCproductZhat가필수다.
ε독립변수를제외하고coarse/fine밀도를중복곱하지않는다.
twooutput의covariance를포함한independentwhole delta-methodSE로차이를평가한다.
smallobservedadjacentdifference는remaininginfinite-meshbias의상계가아니다.

## 사전 software review

공용gradientMH의Gaussian역밀도/상세균형/oracletests를유지한다.
fine/coarseexactCDF재구성,constantvolatilitycoupling동일성,
jointaugmentation을통과하는FFTgradientfinite-difference/batch-singleton일치검사.
jointdimension은5N/2이며power-of-two를요구하는oldblockcore에강제로넣지않는다.
새gradientcore는arbitrarypositivefinite-dimensionalGaussian좌표를지원한다.

## 작은 고정 진단

2originalcells ×fineN512/1024 ×8independentwhole=32whole.
128particles,64quadratictemperatures,4gradientmutations(.35),cap20,
every4stratified,no finalmutation,no guide/bank/independenceproposal.
namespace `paired-mesh-gradient-v1`,동결source/model/seed/grid.
whole당31872joint forward/actualgradient evaluations.
total1,019,904joint/2,039,808individualgridpath-law evaluations.
준비는FFTkernel와2×5Gaussianmap뿐이며basis-path simulation0회.
time-stepforwardevaluation은각joint에N+N/2를곱해별도기록한다.
cap3mindividual-path/1.2mjointbackward/1800초/RSS4GiB,whole60초.
futureCPUwall은동시에실행하는development작업의sharedlaptop값이지speedbenchmark아니다.

## 판단 및 후속

wholeμf/μcRSE,relative signed difference,pairedSE,2cells×2grids의4comparisons
normalquantile/10%small-differenceupper,LOO/ancestry/ESS/acceptance를기록한다.
32whole완료전prefix를보고pass를고르지않는다. pilot는P2gate자격이아니다.
source/input/seed/endpointweightedidentity/whole-numericalreplay/cost감사.
개선되면freshfinergrids의allocation및독립reference를별도동결한다.
불변함수/누락mode/variancecollapse가해결되지않으면실패로보존하고kernel/target를재설계한다.
fineN32q/bank를N512/1024자산으로재사용하지않는다.
