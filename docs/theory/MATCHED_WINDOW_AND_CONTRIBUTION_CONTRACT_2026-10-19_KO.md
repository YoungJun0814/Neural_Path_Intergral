# Matched price window 및 기여 진단의 수학적 계약

2026-10-19. 기존 exact Gaussian-mixture IS와 원래 유한격자법칙의 적용 검토다.
새로운 general theorem/최적성/미관측tailcoverage를 주장하지 않는다.

## 최소변경과 해석

G0의u_i는시점t_i이전Volterra feature의normalizedGaussianrepresenter다.
B는u_i,t_i,a,b,weights를유지하고v_i만futurephysicalwindow로변경한다.
과거/미래disjointsupport이므로<u_i,v_i>=0,||a u_i+b v_i||²=a²+b².
동일componentcount와energy만으로동일한비선형payoff효율은보장하지않는다.
서로다른component간innerproduct/correlation구조는변할수있어effective표현능력까지
동일하다고주장하지않는다. 전체density계산과실제준비/추론비용을따로기록한다.
특히v_i는미래driver를바꾸므로가격직접forcing효과만의실험은아니다.
A→B는**Gaussianprice-controlrepresenter의전체동역학효과**를분리한다.

C와B는다른voltemplate/시점을쓰므로B→C는그결합변경의효과다.
단일window-volfeature효과라고표현하지않는다.

## 정확한likelihood와label기여

z=m_L+ε,L~Categorical(w),ε~N(0,I),r=Σw_l N(m_l,I).
D(z)=r/p=Σw_l exp(z·m_l−||m_l||²/2),Y=g(z)/D(z).
E_r Y=μ. 어떤label이선택됐든분모는항상전체D다.
label별관측sum S_l=Σ_i1_{L_i=l}Y_i는Σ_lS_l=Σ_iY_i를만족한다.
단, E[1_{L=l}Y]=∫w_l φ(z−m_l)g(z)/D(z)dz이며사건의고유물리mode질량이아니다.
같은경로가다른label로샘플될수있다. labelESS나label다양성을경로modecoverage로읽지않는다.

feature ratioΣYF/ΣY는관측기여의서술적self-normalizedratio다.
이는일반적으로unbiasedfeaturemean이아니며μ의ordinarymean추정식을바꾸지않는다.
현재의self-normalization금지는μ/M2추정량에적용된다.진단ratio와확률추정을명확히분리한다.
독립calibrationthreshold/외부거리/누락modecoverage는아직추가하지않았다.
top16은큰Y표본의요약이며대표표본/독립confirmation이아니다.

## 동일fine/coarse법칙

finez와독립ε_aug로coarsez를정확한adapter로만든다.
두estimator의분모는fineD(z)하나이며augmentationprior는상쇄된다.
E[g_c(Az+Bε_aug)/D(z)]도선언한coarseμ다.
새feature경로는finecache.quantities한번에서terminal,I,variance를얻고
기존과같은log_ndtr((logK−terminal)/sqrt((1−ρ²)I))를쓴다.
J=Σsqrt(v_left)ΔW,I=hΣv_left,maxvariance는마지막after값을제외한left값이다.
미래정보를이용하는새adaptedcontrol을만든것이아니라평가된경로의진단이다.

## 독립검증과한계

N32에서B의volrow를독립denseoperator와비교하고fullGaussianlogdensity식과
samplerlabel재구성을검사한다.기존pairedcore와새feature분기의두logpayoff를직접대조한다.
ledger의labelsum과rawmomentlogsum이일치하고chunk분할merge가전체집계와일치해야한다.
source/seed/geometry/moments/features6blockreplay는numericalreproducibility만보여준다.
RSE제곱비율은불확실한평균으로정규화된pilotproxy라서확정variance감소비가아니다.
같은development셀에서만든방법의topjournalnovelty/다른task성능은별도검증이필요하다.
