# 다중 시점 잔차–Brownian proposal: 구현·실험·검토

2026-10-26 America/New_York. 새 작업은 커밋·푸시하지 않는다.
이 문서는 결과 해석 보고서이며 실행 전 계획을 소급 수정하지 않는다.

## 1. 이번에 실제로 바꾼 것

부호 평균 후보와 잔차57 단독 가설의 한계를 반영해 **실제 사건 proposal q**를 바꿨다.
기존 uniform373-component physical dictionary의 mean을 보존하고 shifted component 내부의
국소 residual–직후 Brownian covariance를 바꾼다. 특정 극단 경로를 새mean에 넣지 않았다.
모든31개 사용 시점을 사전 균등하게 포함한다. 자연성분10%는 그대로 남겼다.

사전 주후보는 coupled이고 residual-only/independent는 ablation이다.
이는 학습된 operator나 새로운 일반 수학 원리가 아니라, 검증 가능한 finite-grid proposal 가설이다.
혼합 importance sampling의 가중치 최적화 자체도 기존 연구가 있으므로, 이 구성만으로
독창성을 주장하지 않는다. [He–Owen 원 논문](https://arxiv.org/abs/1411.3954).
이번 실험은 가중치 학습을 하지 않는다.

[실행 전 고정 계획](../plans/MULTITIME_RESIDUAL_BROWNIAN_PROPOSAL_2026-10-26_KO.md).

## 2. 구조와 수학적 검토

N32에서 각 j=1,...,31의 좌표쌍은 `(2j−1,2j)`다.
첫 좌표는 V_j의 local variance residual이고 둘째는 그 분산이 적용되는 직후 가격 구간의
Brownian 좌표다. 미사용 마지막 residual63은 제외한다.

base는 `.1p+Σshift w_k N(m_k,I)`이고 shifted mass는.9다.
후보는 이 shifted mass의 절반을 동일mean의 covariance mixture로 바꾼다.

| arm | 쌍의 covariance | 역할 |
|---|---|---|
| base | I | 강한 static baseline |
| residual | diag(5,1) | 잔차 절댓값 확대 |
| independent | diag(5,5) | 잔차·직후Brownian 독립 확대 |
| coupled | [[5,s4],[s4,5]] | 같은 marginal variance에서 coupling 추가 |

s=−sign(rho),현재 두 셀에서는+1이다.coupled eigenvalues는9,1이므로 양의 정부호다.
원래mean/가중치를 유지한다는 것은 원래 label의 전체mass를 보존한다는 뜻이다.
실제 Gaussian 성분 수가373으로 유지된다는 뜻은 아니다.
각 후보는 명목상 `1+372+31×372=11,905`성분에 해당한다.
base 대비 capacity가 다르며, 세 covariance ablation끼리는 같은 명목 성분 수다.

### 정규화·안전 상계

모든 항은 정규화된 Gaussian이고 mixture mass 합은
`.1+.5×.9+.5×.9=1`이다. 따라서 추정 normalizer가 필요 없다.
`q_a≥.1p` 및 `q_a≥.5q0`이므로 원래 bounded g에 대해
`g/(q_a/p)≤10`, `M2(q_a)≤2M2(q0)`가 성립한다.
이 안전 상계는 개선 또는 작은 상대오차를 보장하지 않는다.

component mean을 m, 해당쌍의 z−m을(a,b)라 하면 coupled log covariance 보정은
`(1−1/9)(a+s b)²/4−log(9)/2`다.
전체밀도는 원래 mean likelihood와 합쳐 모든 component/time을 logsumexp한다.
생성 label 하나만 분모로 쓰거나, z−m 대신 z 자체를 covariance 보정에 넣지 않는다.

sampling은 독립 Gaussian을 정확한 square-root covariance로 변환한다.
finite-grid Gaussian change-of-measure이며 미래정보를 사용하는 SDE drift로 주장하지 않는다.
원래 adapted Itô simulator는 바꾸지 않았다. 연속시간 Wiener measure 정합성·mesh 일관성은
이 finite-grid 검증으로 자동 확보되지 않는다.

특히 고정variance의 한 local white-coordinate 확대는격자마다 같은물리시간 범위를
확대하는것과 다르다.local residual의Volterra scale은시간간격에도 의존한다.
따라서N32의v=5를N4096에그대로 이식해mesh-invariant개선이라고 주장하지 않는다.
성공할경우에도physical-window/trace-class구조와별도의mesh실험이 필요하다.

## 3. 실험 분리와 추정량

N32의 기존 canonical/high-eta stress 셀,spot100/threshold1을 유지한다.
N4096의 확률이나 일반 pricing/calibration 문제와 직접 비교하지 않는다.
train/selection이 없는 고정구조 실험이므로 독립 학습 반복의 성과로 부르지 않는다.

- direct: 각arm×셀마다8×8192새 IID표본에서 `g/(q_arm/p)`를 계산한다.
- heldout H: 네proposal을각1/4비중으로 섞은분포를4strata로 고정배분한다.
  `g²/[(H/p)(q_arm/p)]`를 모든arm에 공통 계산한다.
- reference R: guide mean없이 `.1p+.9/31 Σ_j N(0,Σ_j,coupled)`에서 독립 표본을 만든다.
  `g²/[(R/p)(q_arm/p)]`로 같은M2를 계산한다.

H는4strata의 mean을 알려진mass1/4로 합치고 분산은층별 sample covariance를 사용한다.
정확히 같은 표본 수라도 strata 간 평균 차이를 pooled-IID 분산으로 잘못 넣지 않는다.
paired M2 비교는 공통 표본의 covariance를 반영한다.8block 변동/LOO도 별도로 검사한다.
낮은ratio SE만으로 reference precision 실패를 우회하지 않는다.

R은 shifted guide geometry와 다르지만 simulator/CDF를 공유하고,그 역시mode를 놓칠 수 있다.
이 개발 대조가 qualified한 fine-grid 독립 기준을 이미 만들었다는 뜻은 아니다.

## 4. 실행 전 소프트웨어 검증

전용26개 테스트가 통과했다.독립 dense covariance Gaussian 열거와 log density를 비교했고,
sampling mean/covariance,정확한natural floor,base retention,α0 identity,guide-free reference,
invalid input/seed/budget,known-stratum covariance,missing unit/잘못된 채택 우회를 검사했다.
새core/runner의Ruff와Mypy도 실행 전에 통과했다.

이후 source/config/계획을 sampling 전에 봉인했고 304개seed role을 먼저 할당했다.
기존11개protocol의seed-context 정수2,518개와 충돌이 없었다.
이는 저장소 모든 과거난수/PRNG 독립성의 증명이 아니다.

실행 전source1,149파일의 archive/current hash 일치를 확인했다.
사전1024경로 비용점검은1.213초였고 safety1.5를 적용한production 예측은1280.52초로
1800초 예산 안에 들어왔다.이 예측은 실행시간 또는 통계적 성공 보장이 아니다.

## 5. 전체 고정 실행 결과

96개 unit을 전부 완료했고,preflight포함721,920physical evaluations를 사용했다.
부분 결과로 중단·표본 추가·seed 교체·coupling sign 변경을 하지 않았다.
최종 판정은 **`NO_ADOPTION_COVARIANCE_PRECISION_OR_COVERAGE_UNRESOLVED`**다.
실제proposal 구현은 완료했지만production winner로 채택하지 않는다.

### 같은 표본 수에서 실제 사건 추정의 관측 precision

각arm×셀마다65,536개새IID표본이다.8block은 독립학습8회가 아니라
고정proposal의독립sampling block이다.사건확률표준오차를M2의표준오차로 재사용하지 않는다.

| N32 셀 | base | residual-only | independent pair | coupled 주후보 |
|---|---:|---:|---:|---:|
| canonical raw RSE | 36.38% | 12.02% | 6.32% | 6.92% |
| high-eta raw RSE | 13.38% | 8.22% | 9.03% | 3.90% |

주후보의canonical blockRSE6.41%,maxpath3.09%,LOO4.94%였지만rawRSE가5%를 넘었다.
high-eta는blockRSE4.88%,maxpath1.25%,LOO2.67%로 사전 **사건 개발precision**을 통과했다.
이는기준확률의accuracy 인증이나population 효율우위,학습일반화,저널confirmation이 아니다.
다른arm/seed의유한표본RSE비를모집단분산감소율로단정하지 않는다.

base canonical의maxpath가34.04%였다는점도 중요하다.이번base 표본이불안정한것을
숨기거나이한실행의큰RSE를분모로삼아거대한speedup을선언하지 않는다.
canonical에서는independent가coupled보다낮은관측RSE였지만주후보를사후교체하지 않았다.

### 독립 위험 평가가 채택을 막았다

아래의모든숫자는 **M2 reference의관측RSE**다.사건평균RSE와다르다.

| 셀/geometry | base | residual | independent | coupled |
|---|---:|---:|---:|---:|
| canonical/H heldout | 46.64% | 44.06% | 46.08% | 48.30% |
| canonical/R guide-free | 64.18% | 73.60% | 72.09% | 82.07% |
| high-eta/H heldout | 68.79% | 38.78% | 46.82% | 62.66% |
| high-eta/R guide-free | 89.74% | 90.25% | 88.21% | 79.02% |

양geometry 모두20%정밀도기준을충족하지못했다.block집중과LOO도 별도미달했다.
H/R의equivalence 비교는계약대로 **실행하지않고 미자격**으로기록했다.
넓은불확실성속에서비슷해보이는평균을독립일치성공으로부르지 않았다.

주후보의관측M2감소점추정은canonical/H25.09%,canonical/R95.45%,
high-eta/H53.07%,high-eta/R76.51%였다.
그러나모든경우paired **absolute 감소lower가0이하**였고referenceprecision도실패했다.
일부relative ratio lower가양수라는이유만으로절대감소·정밀도실패를우회하지 않았다.
최대risk 단일기여는H에서약28–66%,R에서약52–90%로여전히집중됐다.

R의최대risk개별비중은coupled canonical79.88%,high-eta76.15%였다.
즉guide를쓰지않는local covariance sampling도이번budget에서안정적인oracle이되지못했다.
이관측이영구적bias나특정mode누락의발생확률을증명하는것은아니다.

## 6. 비용과 자원 검토

| 범위 | physical evaluations | Gaussian component/time score 수 |
|---|---:|---:|
| preflight+production | 721,920 | 25,267,200,000 |
| 대표6unit replay | 40,960 | 1,433,600,000 |
| **합계** | **762,880** | **26,700,800,000** |

production phasewall724.82초는준비/import/source봉인을제외하고preflight/저장/집계를포함한다.
완료unit의wall합719.64초중density709.31초,physics6.38초,sampling2.63초였다.
최대unitwall10.01초로60초cap 이내였다.실제score 수는FLOP 또는총CPU instruction 수가아니다.
단위/전체회귀테스트의물리계산은위연구장부와별도다.

중요한제한:실행기는진단을위해모든표본에서 **4arm과R의밀도**를계산했다.
따라서709초를coupled하나의배포inference비라고부르면안된다.
반대로이공통진단비를무시하거나base와같은밀도계산복잡도라고표현해서도안된다.
모든accuracy조건이실패했으므로fixed-precision total-wall speedup을계산하지않는다.
감사wall54.92초는새확인실험이나속도benchmark가아니다.

별도추가physics0회의구조검사에서각시점의372개shift 중같은local pair mean을
공유하는그룹은5–139개,전시점합2,321개였다.명목11,532component/time항보다작다.
정확한항통합으로density를최적화할가능성은있지만 **구현/측정하지않았다**.
이것이통계적coverage실패를해결한다고주장하지않는다.

## 7. 채택·중단과 다음 판단

**유지할것:** full normalized density를가진실제proposal 후보,자연성분과baseline retention,
mean/variance/coupling 분리,새표본의direct/두risk geometry 대조,실제비용과실패보존.

**채택하지않을것:** coupled를완성모델·생산기본값으로등록,high-eta3.90%를독립정확도인증으로
확대,canonical의더좋은ablation으로주후보교체,v/α/sign을같은실행에서재튜닝,
또는N4096/대규모학습으로자동진입.

다음은순차표본증설보다먼저검토할사항이다.

1. **성능주장과도구개발분리:** high-eta 직접precision신호는보존하되population M2감소는미입증으로둔다.
   다음별도confirmation을설계하려면rare-contribution bank의신뢰성과독립geometry가필요하다.
2. **한시점rank2의한계점검:** 모든시점을넣어도한sample에서확대되는시점은하나다.
   여러구간에걸친volatility memory와결합경로를표현한것과같지않다.
   이에대한정규화가능한physical-window/joint-family를검토하되이번불안정한통계만으로선택하지않는다.
3. **밀도계산비검토:** exact local-mean grouping 또는기존low-rank matrix식을활용하는계산최적화는후보다.
   같은분포의최적화인지별도oracle로검증해야하며,이를새모델혁신이나coverage개선으로부르지않는다.
4. **독립R의한계인정:** prior-centered local covariance만으로안정적인기준을얻지못했다.
   같은R에유리한seed/더큰v를찾는반복보다어떤joint path기여가불안정한지설계용으로살펴야한다.
5. **P2분리유지:** fineensemble약0.9%RSE와별개로different-geometry accuracy,
   signedmesh correction 및continuum 상대bias가남아있다.이번N32결과로대체하지않는다.

이번사이클은고정범위에서닫는다.좋은개발신호가있다는것과논문핵심기여가확립됐다는것은다르다.
계속조건을늘리기만하는대신,통계적coverage와실용비용을함께해결할가설이있는지다음단계에서판단한다.

## 8. 감사·저장·회귀검사

[결과](../../results/post_audit/residual_brownian_covariance_v1/session_20261026/result.json),
[실행전protocol](../../results/post_audit/residual_brownian_covariance_v1/session_20261026/protocol.json),
[전체통계·대표재현감사](../../results/post_audit/residual_brownian_covariance_v1/session_20261026/audit.json).

- 96개array의hash/원래unit/result binding·전체moment·최종decision을전수재계산했다.
- 각셀coupled direct block0,H heldout block0,R block0의총6unit을40,960physics로재생했다.
  전체log contribution arrays와저장된극단경로정보가정확히일치했다.
- source1,149개파일의archive/current해시불변도확인했다.주실험후source를수정하지않았다.
- result SHA256: `5dc431315907c8c645cd2eff6ea59ee995caec5c912aa57f3820b5881e666994`.
- source SHA256: `7af9b67e71b781792866a3ba015853b03fbb12ea9b72869e0260f3872ce2de95`.
- audit SHA256: `c101fe08776e66869ddbfcebfc0f8ba1140fc5a5eab324f8d017b176792ed3eb`.
- CI범위Ruff,Mypy222source files,새runner1개별도Mypy가통과했다.
  전체회귀검사완료로그에서 **1,898passed/579.66초**를확인했다.
  source statement coverage85%(23,157statements/3,452missed),새core96%(107/4)다.
  원래terminal session은재개시회수할수없었으나로그에100%·coverage표·최종passing summary가
  모두있다.관측하지못한process exit code를추정값으로기록하지않았다.
  회귀검사일부는대표재현감사및다른로컬process와겹쳤으므로독점실행시간benchmark가아니다.
  [검증receipt](../../results/post_audit/residual_brownian_covariance_v1/session_20261026/validation/regression_review.json),
  [완료pytest로그](../../results/post_audit/residual_brownian_covariance_v1/session_20261026/validation/pytest-output.txt),
  [정적검사로그](../../results/post_audit/residual_brownian_covariance_v1/session_20261026/validation/static-output.txt).
  소프트웨어검사성공과연구채택실패는구분한다.
- 기존RequestsDependencyWarning은환경변경없이보존했다.이번새변경의remote CI는미확인이다.
  커밋·푸시는하지않았다.
- 완료후공개파일387개,strict JSON189개,archive/NPZ179개(3,793members),local link184개,
  완료artifact372개의Git filter이후byte보존을확인했다.알려진credential pattern/위험archive경로/
  100MiB이상파일/trailing whitespace검출은0이었다.비밀정보부재의전수증명은아니다.
  이전intervention/극단감사/이번proposal의동결source와결과해시도불변이었다.
  [최종무결성receipt](../../results/post_audit/residual_brownian_covariance_v1/session_20261026/validation/integrity_review.json).
