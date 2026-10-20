# Window pilot 이후 우선순위 실행 보고

2026-10-19 시작,2026-10-20 후속 실행. 이전 [후속 계획](../plans/POST_WINDOW_PILOT_PRECISION_AND_MECHANISM_2026-10-19_KO.md)에
따른 로컬 개발이다. 커밋·푸시하지 않는다. 배터리/AC 조회나 OS 설정 변경은 하지 않는다.

## 실행 중인 작업과 완료한 구현

1. [SMC bridge/mutation 배분 pilot](../plans/FINE_BRIDGE_MUTATION_ALLOCATION_2026-10-19_KO.md):
   고정32whole 완료,4whole 재현 감사 통과. 두 arm의 particles는64로 동일하고 bridge/mutation/
   resampling 주기를 함께 조절한다. 거의 같은 potential 평가량의 개발 비교이며,
   이전128particles 결과보다 더 우수하다고 가정하지 않는다.
2. [3arm matched window pilot](../plans/MATCHED_WINDOW_MECHANISM_PILOT_2026-10-19_KO.md):
   sampler/diagnostics/runner를 구현했고 SMC 감사 및preflight 통과 후48blocks 완료했다.
   대표6blocks의moment/geometry/seed/diagnostic 재현 감사 통과.
3. [수학적 계약](../theory/MATCHED_WINDOW_AND_CONTRIBUTION_CONTRACT_2026-10-19_KO.md):
   price-only 변경의 의미, 전체 mixture likelihood, label과 물리 mode의 차이,
   관측 feature ratio와 ordinary probability mean의 차이를 검토했다.
4. 대조군/진단/3arm runner/label 분석의 국소10tests가 통과했다(89.47초).
   SMC 배분 runner의 별도2tests도 통과했다. 이 국소검사는 실제N4096 성능 검증이 아니다.
5. SMC audit 뒤 N4096 고정4chunk preflight 통과. 같은입력의 기존/새분기
   logpayoff 최대차이는4개 모두0이었다. prep8.340초,featurechunk0.431–0.496초,
   최대RSS약1.256GB. 2배 시간여유/RSS/prep 제한을 통과했다.
   별도physical8,192/density4,096/time-step25,165,824,본평가에포함하지않는다.
   이4chunk는wholeruntime/accuracy 보증이 아니다.

## 보류한 항목과 이유

- 새 G1 precision forecast 약542만fine draw를 바로 실행하지 않는다. 독립 reference
  정밀도가 나쁜 상태에서 G1만 더 계산해도 교차 검증은 해결되지 않기 때문이다.
- N4096 dense all-driver conditioning은 실행하지 않는다. 빠르고 안정적인 대안 solve가
  아직 입증되지 않아 두 번째 reference 후보를 완성된 알고리즘처럼 표시하지 않는다.
- 원래 q를 재학습하지 않는다. 이번 작업은 reference와 원인 진단에 한정된다.
- mode clustering/독립 calibration threshold는 아직 구현하지 않았다. 현재 label 진단과
  top16 feature는 관측 기여를 설명할 뿐 unseen mode를 배제하지 못한다.
- 모든 P1/P2/large-training flags는false다. 고정N 정확성과 연속시간 relative bias,
  실무 성능 및 학술적 독창성은 별도 검증 대상이다.

## 최종 결과

### 독립 SMC 배분 pilot: 개선 성공으로 판정하지 않음

32whole/physical1,028,096/joint-backward514,048/time-step3,158,310,912 완료.
4whole numerical replay도통과했다. 별도physical128,512/backward64,256이며resultSHA는
`11cdd6d189b157280d42fda6a2cb0b38cb4e5fc540fc1e78639d1bba672d5c61`이다.
runner phase wall1471.08초이며 import/source 준비 제외, 초기 국소테스트와 일부 동시 실행한
개발 계산 기록이지 비교 wall benchmark가 아니다.

| 셀 | temperature points / moves | fine RSE | coarse RSE | initial ancestors | paired 차이 upper |
|---|---|---:|---:|---:|---:|
| canonical | 64 /4 | 14.510% | 12.897% | 7–11 | 12.546% |
| canonical | 128 /2 | 5.240% | 5.109% | 6–11 | 12.672% |
| high-η | 64 /4 | 14.653% | 17.950% | 8–10 | 18.487% |
| high-η | 128 /2 | 18.336% | 19.622% | 10–13 | 13.297% |

4개 paired endpoint는 모두10% 기준을 충족하지 못했고,reference RSE도 모두1.5%를 초과했다.
mutation acceptance는약82–85%지만 좋은 mixing/독립표본 수를 보장하지 않는다.
ancestry 개선도 입증하지 못했다. 현재 자료는 reference SMC이며 training bank 성과가 아니다.

더 중요한 점은 canonical128/2의 fine logmean=-18.36540이다. 이전 G1과의 symmetric
relative difference=-34.909%,4comparison 조정구간[-51.434%,-18.384%]로, 구간 전체가
±10% 범위 밖이다. 나머지3개 교차 비교는 불확실성이 margin을 걸친다.
이는 근사8whole 구간의 **불일치 경고**이지 영구적 estimator bias의 증명은 아니다.
비교 G1도 qualified oracle가 아니다. 그러나 낮게 보이는 관측RSE로 이arm을 채택할 수는 없다.

사전 cost-forecast selector는 canonical128/2(새293whole),high-η64/4(새3437whole)를
반환한다. 이는 계산된 allocation 예측에 불과하며, 특히 canonical의 불일치 때문에
reference 채택/확인실험 승인으로 읽지 않는다. 이count를 지금 자동 실행하지 않는다.
희귀 기여 누락/작은 입자수/whole분포의 tail과 다른 메커니즘의 오차를 더 검토해야 한다.
seed나 설정을 바꾸어 현재 실패를 성공으로 덮어쓰지 않는다.

검토 중 helper의서술적metadata 문제를 발견했다. `interval_diagnosis.scope`문구는
원래8comparison이라는문자열을고정해서새4개/6개family에서도같은문구가나온다.
실제z와CI는각각4개/6개Bonferroni로계산돼수치는영향없다. 보존된producer/archive를
수정하지않고최종postprocessing에서실제z를family별로다시검사하며이불일치를명시한다.
문자열을보고새검사를8개로조정했다고인용하지않는다.

### 새3arm 방향-메커니즘 실험

48blocks/fine786,432draw/physical1,572,864/density786,432/time-step4,831,838,208 완료.
phase wall1782.26초는runner clock부터이며 CLI import/source archive 준비 이전시간을
포함하지 않는다. hardware/전력 상태 통제된 벽시계benchmark가 아니다.

| 셀 | arm | fine RSE | coarse RSE | 최대 단일 기여 | 최대 block LOO | paired upper |
|---|---|---:|---:|---:|---:|---:|
| canonical | A 기존G0 | 63.830% | 52.733% | 62.512% | 66.105% | 46.165% |
| canonical | B price만window | 13.453% | 14.002% | 6.002% | 5.118% | 19.329% |
| canonical | C 기존G1 | 3.954% | 3.597% | 1.618% | 0.958% | 10.060% |
| high-η | A 기존G0 | 45.230% | 41.358% | 43.546% | 53.107% | 44.489% |
| high-η | B price만window | 12.254% | 8.669% | 9.897% | 8.009% | 32.281% |
| high-η | C 기존G1 | 3.031% | 2.886% | 0.830% | 0.896% | 4.827% |

같은volfeature/시점의A→B에서기여집중과RSE가낮아지는개발신호를관측했다.
그러나 B만으로는현재reference정밀도를달성하지못한다. B→C에서도관측RSE가더낮으므로
price방향하나만으로기존G1의모든차이를설명하지않는다. 이것은두작은development셀의
관측이며확정variance감소비/인과적최적성/새task일반화를주장하지않는다.

모든fineRSE가1.5%를초과했고6개within-studymean동등성은모두불확실성때문에미확인이다.
paired기준도high-η C만통과했다. canonical C upper10.0599%를10%로반올림하여
통과시키지않는다. small-differencepass조차continuumrelative-biascertificate가아니다.
rawIIDRSE와blockRSE가다를수있으므로둘을함께보고, 더낮은blockRSE만선택하지않는다.

6block감사physical196,608/density98,304회도별도계산해통과했다. resultSHA는
`d65891685c543988e68b476064177a8a570b77c03537a6b6e84a936ec9084296`이다.

### 관측 기여 진단

source/receipt/SHA를결합한후처리는새physics0회다. 실제4개/6개comparison의z도
재검사해일치했다. label은sampler성분이지경로의물리적mode가아니다.

| 셀 | arm | 상위16개표본의 관측 기여 | price amplitude0 성분의 관측 기여 | price2 | price4 |
|---|---|---:|---:|---:|---:|
| canonical | A | 95.986% | 95.497% | 4.235% | 0.265% |
| canonical | B | 44.022% | 8.982% | 31.556% | 59.462% |
| canonical | C | 10.781% | 20.156% | 43.496% | 36.340% |
| high-η | A | 79.422% | 80.381% | 15.674% | 3.928% |
| high-η | B | 31.594% | 11.545% | 44.182% | 44.271% |
| high-η | C | 7.869% | 23.280% | 41.610% | 35.110% |

price성분표는shift성분만이고자연성분의작은별도질량이남는다. A의한-cellprice방향에
큰amplitude를둔성분은관측확률기여가작았고, physicalwindow B/C에서는가격이동성분의
관측기여가커졌다. 이는현재고정dictionary의표본진단이며이성분들의실제mode질량이나
최적weight가아니다. 자연성분의관측기여가작다는이유로defensive질량을제거하지않는다.

기여가분산되는신호는C에서가장강하지만, 관측top16밖의tail을모두찾았다는뜻은아니다.
저장한I/J/maximumvariance/time/featureprojection은후속가설생성에사용할수있다.
이를독립calibrationthreshold나clustering분류기로이미검증했다고주장하지않는다.

새C fineμ는canonical1.32155e-8/high-η1.00164e-6이다. 이전pilot의값을이번표본과
합쳐새독립확인으로부르거나, 더유리한평균만선택하지않는다.

## 이번 결과에 따른 다음 우선순위

1. C의fixed-physical-window구조를후보로유지하되아직referenceoracle로채택하지않는다.
   가격방향만바꾸는것은부분개선이고volatilitygeometry와함께검토해야한다.
2. canonical의새SMC불일치경고를먼저다룬다. 낮은RSEarm을선택해수백회연장하지않는다.
   whole분포/작은particle수/독립메커니즘오차를분리하는새고정설계가필요하다.
3. 독립Rao–Blackwellreference의확장가능성을작은grid oracle부터검토한다.
   조건부Gaussian잔차를정확히적분하되원래q의M2를다른integrand로대체하지않는다.
   N4096denseQR나검증안된fastsolve를현재production방법처럼표시하지않는다.
4. pathgeometry의독립calibration및새표본진단을설계한다. 현재관측top표본으로threshold를
   맞춘뒤같은표본에서검증하지않는다. label을cluster로바꾸어탐색완료라고하지않는다.
5. μreference정밀도/독립agreement,continuumrelativebias,originalqbank/전체학습반복의
   자격이없는동안대규모학습·효율우위·저널제출준비완료주장은보류한다.

## 소프트웨어 최종 검증

Ruff 전체범위와정규Mypy201source 통과. 전체pytest **1,449개 통과(276.75초)**.
새4개experiment는`--follow-imports=silent`명시Mypy도통과했다.
두producer/result와최종분석의sourcearchive3개를검사하고,최종analysis를정확히재계산해
일치함을확인했다. 추가physics0회이며앞선4whole/6blockreplay와별도검사다.
로컬Python3.10.11만실행했고이번uncommittedtree의원격CI/Python3.11은실행하지않았다.
시작시기존requests/urllib3/charsetdependency warning이표시됐으나동결된환경을실험중
임의업그레이드하지않았다. 기존다른experiment의확대Mypy오류를해결한것으로표시하지않는다.
scope문자열metadata의문제와실제numeric보정의일치는위에구분하여기록했다.
이검사는유효범위내구현/재현성의근거이지오류부재·unseen-tailcoverage·학술독창성의보증이아니다.
원래 고정 계획과 수치 source를 변경하지 않고 새 module/result를 추가했다.
커밋·푸시·staging·원격변경은하지않는다.
최종HEAD는`ace4bca4ae298fe438fa8b74d0b6013f238c0afe`이며staging은비어있다.
`git diff --check`도통과했다. 이실행의새코드/결과/문서는모두로컬workingtree에남긴다.
