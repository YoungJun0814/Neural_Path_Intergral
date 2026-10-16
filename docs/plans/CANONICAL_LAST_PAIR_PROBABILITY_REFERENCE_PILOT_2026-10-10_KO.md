# Canonical probability reference: 마지막 Gaussian pair의 정확한 적분

2026-10-10. fixed-stratum pilot 전체완료·감사 후, 새pilot 생성 전에 동결한다.
층화의μforecast는13,094,400draw로cap실패였고새IIDforecast는6,011,520이었다.
작은새IIDpilot의낮은분산만으로이전8,986,624 allocation실패를해결했다고하지않는다.
현재목표는기존exact last-pair core를static probability reference에적용하는것이다.

## 수학 계약

z=(u,v), v=last2standardGaussian. finite-gridItô left law에서마지막가격innovation만
남기면 g(u,v)=Φ((a(u)-b(u)v0)/c(u)), v1은terminalpayoff에사용되지않는다.
따라서 gbar(u)=E_pv g=Φ(a/sqrt(c²+b²))가정확하다.
staticr의prefixmarginal rU는기존shiftmeans의prefix와원래mixtureweights로정확히계산한다.
fullr denominator의−.5||mean||²를그대로prefix에쓰면틀리므로prefixnorm을다시계산한다.

full draw z~r에서Yfull=g/(r/p), YRB=gbar/(rU/pU).
E_r[Yfull|u]=YRB이므로동일μ와Var(YRB)<=Var(Yfull)이다(현재bounded floor로finitevariance).
이는새normalizer이론이아니라Rao–Blackwell원리의정확한finite-grid적용이다.
fullq와원래M2는변경하지않는다. M2는같은fullz에서2logg-logDq-logDr로계산한다.
gbar²를써원래M2로표기하지않는다. analyticalμ와risk의target를분리한다.

## fresh paired pilot

canonicalN32,staticr/frozen5q.8blocks×32768=262144draw.
namespace `canonical-last-pair-mu-v1`; 기존pilot/final과합치지않는다.
한zero-last-pairpath cache로fullg와gbar를둘다계산한다. fulldensity/prefixdensity는각정확히계산.
preparation64guidegeometry 포함262208forward evaluations;cap300k/600초/RSS4GiB.
source/input/q/seed 동결,block30초. 완전8blocks 후판단한다.
knownCDFconvolution/cachefullg동일성/prefixdensity계약은기존tests와새regression으로검토.
numericalaudit의새1024fullpath와cache재구성을비교하고각first/lastblock를재생한다.

## 판단

μfull/RB와5rawM2의moments/SE/maximumshare/LOO/pairedμequivalence를기록한다.
RSE1.5%/safety2,drawcap8,388,608의새estimatorallocationforecast를계산한다.
pilot에서보이는variance ratio는populationratio증명이아니며작은표본의선택불확실성을보존한다.
새RB방식이budget가능할때만다른mechanism(guide-freegradient/guidedenvelope)과freshfull
canonicalvalidation을별도로동결한다. 개발중여러재시도는publicationconfirmation과분리한다.
P1전체/P2/large-training flags는false다.
