# P2 남은 두 핵심 의무의 실행·증명·실패 경계 검토

2026-10-28 America/New_York. 기준 main59f2809. 커밋·푸시·클라우드 비용 없음.
기존 결과와 실패·threshold·original BLP 법칙·통과 기준을 보존한다.

## 1. 무엇을 실제로 바꿨는가

직전 cyclic 후보는 기존 Gaussian 값의 집합을 보존하고, 이번 개발 표본에서
정밀도를 개선하지 못했다. 같은 후보의 seed/count만 반복하지 않고 다음을 구현했다.

1. 독립 momentum을 새로 생성해 Gaussian-split Hamiltonian trajectory를 만드는
   guide-free reference. 기존 dictionary/learned bank/ensemble weight는 입력하지 않는다.
2. 같은 physical/gradient callback 수의 gradient4move 대조군. HMC4step을4개의
   독립 표본으로 세지 않는다.
3. full Hamiltonian correction의 안정적인 energy-change 누적. target을 clipping하지 않는다.
4. original N4096 finite-grid probability SMC product normalizer, carried weights,
   independent-whole 통계, 조건부 fresh precision allocation과 실패 시 중단.
5. tail-weighted CDF derivative에서 explicit Gaussian envelope와 Hölder를 통해
   확률의 거듭제곱 차이를 제어하는 새 분석식과 log-domain 계산.

[사전 실행계획](../plans/P2_HAMILTONIAN_AND_TAIL_BOUND_2026-10-28_KO.md),
[상세 수학 유도](../theory/TAIL_WEIGHTED_ROOT_BOUND_AND_SPLIT_HMC_2026-10-28_KO.md).

구현은 `gaussian_split_hmc.py`, `hamiltonian_probability_smc.py`,
`tail_weighted_continuum_bound.py`, `experiments/post_audit_p2_hamiltonian_tail.py`다.
신규 단위·계약 검사23개가 실제 실험 전 통과했다.

## 2. 이론적 타당성과 주장 범위

### 2.1 Reference 이동

Hamiltonian은 `Hβ=(||z||²+||p||²)/2−βlogg_N(z)`다. momentum kick / harmonic
rotation / kick의 symmetric composition에 momentum flip을 합치면 involution이
되고 absolute Jacobian1이다. 정확한 Hamiltonian MH로 target을 보존한다.
힘의 norm cap도 deterministic shear라 유한차원 MH의 이 논증을 유지한다.
cap한 힘이 어떤 새 potential의 gradient라고 가정하거나 energy가 자동 보존된다고
하지 않는다. energy correction에는 원래 target을 사용한다.

HMC는 기존 방법이며 신규 모델 원리라는 주장을 하지 않는다. Gaussian prior를
정확히 처리하는 rotation도 기존 문헌의 원리다. 현재 rough payoff의 Hilbert-space
조건이나 dimension-independent mixing theorem을 얻은 것이 아니다.

단위 검사는 force cap 유무의 역변환·dense Hamiltonian ratio·phase Jacobian,
8192차원 prior rotation, constant 및 analytic probability normalizer를 포함한다.
이 검사들이 실제 희귀 모드 탐색·정밀도 성공을 보장하지는 않는다.

### 2.2 새 continuum proof step

기존 L² driver error와 inverse moments의 가정 아래, r>1,q=r/(r−1),
b=1−1/(2q)에 대해 `|μ^b−μ_N^b|≤bD_N(r)`를 유도했다.
D는 tail-weighted derivative moments를 explicit envelope로 지배해 계산한다.
극단 확률의 크기를 식에서 완전히 버리지 않는 것이 분석상의 진전이다.

이는 모든 μ/N에서 이전 absolute bound보다 반드시 작은 수치 구간이라는
주장이 아니다. Gaussian envelope와 Hölder, inverse moments가 다시 느슨할 수 있다.
관측 μ_N을 certified true grid mean으로 쓰거나 float64를 interval arithmetic으로
부르지 않는다. 유도 자체는 기본 원리의 조합이며 학술적 독창성이 검증된 새 정리가 아니다.

## 3. continuum 수치 평가: 유용한 상대 인증은 미해결

고정 r8개×N3개×2cells=48개 조합을 전부 계산했다. 가장 작은 log root-budget도
canonical4.61263909/high-eta9.07205289로 양수다(N16384,r1.05).
root difference의 범위로 자르면 budget1이며, 관측 mean을 넣어도 두 셀 모두
구간은[0,1]이다. N4096/8192에서도 유용한 구간을 얻지 못했다.

따라서 **새 부등식의 유도와 실용적인 상대오차 상계의 확보를 구분한다**.
actual bias가 크다는 lower bound도, 모델이 틀렸다는 증거도 아니다.
확률 관측값/몇 개 adjacent correction만으로 이 인증을 만들어내지는 않았다.

상수 폭증의 원인은 기존 inverse moment의 lognormal exponential 상수와
global envelope다. 이 단계 이후에는 이를 다시 반복 최적화하는 대신, 실제 tail
tilt 아래 localization remainder까지 포함한 작은 상계를 증명하거나 다른 정당한
continuum estimator의 finite variance/work를 입증해야 한다. 현재 완결 증명은 없다.

### 3.1 시간평균 Gaussian 분산을 정확히 쓰는 추가 보강

첫 scan 이후, `Var(T⁻¹∫Y_tdt)`를 최대 시점 분산으로 대체하던 부분을
정확한 Gaussian 적분으로 계산했다. H=.05,T1에서 연속시간 값은
.1574183392이고 original BLP N4096 평균의 값은.1574491076다.
finite 평균이 항상 continuous보다 작다고 가정하지 않고 각각 계산해 bound를 취했다.
또한 convexity로 θ interpolation inverse moments를 제어해 불필요한 factor2를 제거했다.

원래 HMC/root-bound source·설정·seed는 수정하지 않았다. 별도 deterministic
source archive에서48개 조합을 다시 계산하고 전수 재생 감사를 통과했다.
작은 grid의 dense driver-row 직접합, 연속 평균 kernel 적분식과 대조한5개 검사도 통과했다.
새 물리 표본은0개이며 reference precision의 독립 증거로 세지 않는다.

| 환경 | N4096 기존 절대 상계 | N4096 보강 절대 상계 | N16384 보강 절대 상계 |
|---|---:|---:|---:|
| canonical | 68.83 | 8.125 | 7.581 |
| high-eta | 5207.37 | 118.116 | 110.206 |

보강 root-budget도 best(N16384,r1.05)에서 canonical9.620/high-eta153.835다.
모든 경우 clipped probability interval은 여전히[0,1]이다. 실제 상수 감소는 확인했지만
**유용한 relative remainder를 얻지 못했다**. 이 계산을 성공 인증으로 바꾸지 않는다.

[보강 유도](../theory/SHARP_JENSEN_INVERSE_MOMENT_SUPPLEMENT_2026-10-28_KO.md),
[48개 보강 결과](../../results/post_audit/sharp_jensen_supplement_v1/session_20261028/result.json),
[전수 재생 감사](../../results/post_audit/sharp_jensen_supplement_v1/session_20261028/audit.json).

### 3.2 이 proof 계열을 무한 반복하지 않아야 하는 수학적 이유

현재 global-CDF-envelope/Hölder 공식의 미절단 root-budget 자체에
`Broot≥sqrt(Cv Aupper_N)/(4sqrt(π(1−ρ²)ξ))`라는 하한이 생긴다.
inverse moments와 envelope의 값을 최소로 가정해도 남는 양의 항이다.
N4096에서 canonical.50635/high-eta2.95206이다. 확률 범위로 자르면 floor도
각각.50635/1로 해석해야 하며, 이 경우에도 아래 필요조건보다 훨씬 크다.

μ_N을 관측 mean과 같다고 정확히 아는 낙관적 경우에도10% relative band를 위해
필요한 root-budget의 느슨한 상한 `.1sqrt(μ_N)`보다 각각 약43,158/29,786배 크다.
따라서 같은 driver-L²/Cauchy–Schwarz/global-envelope 구조를 그대로 두고
r이나 inverse-moment 상수만 더 조정하는 것으로 현재 격자의 인증을 얻을 수 없다.

이것은 **상계 공식의 하한이지 실제 bias의 하한이 아니다**. 모든 증명법이나
모형의 불가능성도 아니다. 더 날카로운 ΔI 오차 통제, weak cancellation,
tail/error correlation을 보존하는 논증은 이 proof 계열을 바꿀 수 있다.
[유도와 가정](../theory/GLOBAL_HOLDER_BOUND_FAMILY_LIMIT_2026-10-28_KO.md)에 구분했다.
추가 r12개 numerical sanity check도 만족했지만 수치 scan이 전 r에 대한 증명을 대신하지 않는다.

## 4. 고정 reference 실험과 조건부 fresh precision 결과

### 4.1 첫 실행의 운영 실패를 보존

첫 실행은20/32pilot,4preflight 완료 후 다음 whole에서 TimeoutError로 종료했다.
failure에 기록된 phasewall은31,403.44초(약8.72시간)로 사전7200초를 넘었다.
완료20whole의 기록 시간은 그보다 훨씬 짧다. 장시간 정지/외부 자원 대기 가능성이
있지만 OS suspend나 특정 프로그램이 원인이라고 입증한 것은 아니다.
미완료 whole의 부분 normalizer는 버렸고 새 정밀도 단계는 실행하지 않았다.

이는 numerical correctness 실패나 HMC의 통계적 실패로 판정하지 않는다.
동시에 원래 운영 gate를 통과했다고 주장하지 않는다. 기존 failure.json/source/
완료20개와4preflight의 byte를 보존한다.

별도 [운영 재생 계약](../plans/P2_HAMILTONIAN_OPERATIONAL_REPLAY_2026-10-28_KO.md)에서
같은 원래 설정·seed로4preflight와32pilot **전체**를 다시 실행한다. 기존24개 완료기록은
wall제외 전부 일치해야 하며, 두 실행을 독립 표본으로 더하지 않는다. 새 운영 attempt
예산은 별도로 명시하고 원래 실패 비용도 연구 총비용에 포함한다.
원래 고정된 조건부 precision allocation 및 통과 기준은 그대로다.

**봉인 계획서 정정:** 복구 계획1절의 완료20whole 시간 합계 "약1483초"는
작성 중 산술 오타다. 원래20개 JSON의 정확한 합계는1,393.67677초이며 preflight4개는
3.32021초다. 원래 failure wall31,403.44384초와 큰 차이가 있다는 결론과 운영 상한,
seed·수치·통계 판정에는 영향이 없다. 실행 전 봉인한 계획서 byte는 변경하지 않고
이 정정을 남긴다. 시간 차이의 정확한 OS 원인을 특정한 것은 아니다.

### 4.2 전체 동일-seed 재생 결과

4preflight+32pilot을 모두 완료했다. 기존24개 완료기록은 wall제외 모든 값이 일치했다.
원래20pilot은 새로운 독립20개로 더하지 않는다. 통계 단위는 arm/cell당8whole다.

| 셀 | 방법 | N4096 확률 평균 | Whole RSE | 최대 whole 기여 | 최대 LOO 변화 |
|---|---|---:|---:|---:|---:|
| canonical | gradient4move control | 1.2216373e-8 | 8.88% | 18.83% | 7.24% |
| canonical | Gaussian-split HMC | 6.6270355e-9 | 24.76% | 28.13% | 17.86% |
| high-eta | gradient4move control | 1.0478278e-6 | 5.15% | 14.58% | 3.97% |
| high-eta | Gaussian-split HMC | 8.2529123e-7 | 20.95% | 26.78% | 16.32% |

네 경우 모두 RSE≤1.5%에 미달했다. high-eta control은 최대기여·LOO 조건을
충족했지만 RSE는 미달이므로 정확도 자격을 주지 않는다. 다른 세 경우는 세 조건
모두 미달이다. 기존 IID와 equivalence는 평가하지 않았다. 서로 다른 평균만으로
어느 estimator의 bias인지 진단하거나 모집단 불일치가 확정됐다고 주장하지 않는다.

HMC의 평균 coordinate당 제곱 이동거리는 canonical 약.983/high-eta.959로,
control약.390/.388보다 컸다. 평균 acceptance도 약59.19%/57.75%였다.
하지만 HMC의 마지막 initial ancestry 범위9–15/8–21(control19–24/20–26)와
whole normalizer 안정성은 개선되지 않았다. 큰 ambient 이동·적당한 acceptance는
희귀 기여 모드 탐색이나 과거 normalizer 오류 회복을 보장하지 않는다.

이는8whole의 개발 표본에서의 결과이지 모든 HMC 설정의 보편적 열등성 증명은 아니다.
그러나 현재 fixed HMC 후보의 채택·확장 근거는 없으므로 **미채택·중단**한다.

사후 고정 stage 진단에서는 HMC의 log-normalizer 변동에 대한 covariance share가
β≤.04/.04<β≤.25/β>.25에서 canonical43.42%/57.21%/−.62%,
high-eta45.35%/51.28%/3.37%였다. 최종 logZ가 stage log-increment의 합이라는
항등식에서 각 구간 increment와 최종 logZ의 covariance를 계산한 값이며 합은100%다.
음수 share도 가능하다. 이는 **unbiased 확률 estimator의 variance decomposition이나
인과적 오류 원인 분석이 아니다**. 새 튜닝이나 좋은 seed 선택에도 사용하지 않았다.
다만 late endpoint 이동거리만 크게 만드는 것이 충분하다는 가설은 지지하지 않는다.
β≤.04 마지막 단계의 initial ancestry median도 HMC34.5/33,control39/39였다.
다음 reference 설계는 최종 kernel만이 아니라 초기·중간 bridge의 기여 모드 탐색을
함께 검증해야 하며, 이 진단만으로 특정 새 방법의 성공을 주장할 수는 없다.

### 4.3 추가 표본을 실행하지 않은 이유

원래 사전 primary인 HMC의 safety3/RSE1.5% 새 whole forecast는
canonical6,544개/high-eta4,688개로 모두 상한128을 넘었다. pilot평균 실행시간으로
외삽하면 두 셀 합계 약207,222초(57.56시간)이며 정확도 달성 보장이 아니다.
고정 조건에 따라 `STOP_PRIMARY_PRECISION_FORECAST_BUDGET`로 판정했다.
더 좋은 control 값을 사후 primary로 바꾸지 않고, 새 precision0개로 종료했다.
pilot/preflight를 final/저널 confirmation으로 승격하지 않았다.

### 4.4 비용과 운영 성공/실패를 모두 공개

- 최초 실패 시 physical forward646,592회. 이 중 완료20pilot+4preflight641,088회,
  미완료 whole에서 이미 사용한5,504회다. 부분 normalizer는 폐기했다.
- 동일-seed 전체 재생: physical forward1,023,552회, gradient 요청도1,023,552회,
  time-step forward4,192,468,992,phasewall776.11초. whole/RSS/phase/work 상한 위반 없음.
- 최초 운영 실패의31,403.44초 경과시간은 숨기지 않는다. OS 정지/자원 대기 등의
  정확한 원인을 확정하지 못했으며 useful CPU시간이나 deployment benchmark가 아니다.
- 별도 deterministic 상계 보강의 물리 시뮬레이션 표본은0개다.
- 최종 대표 whole 재생 비용은 아래 감사 기록에 추가한다. 테스트/문헌/분석 비용을
  추론 비용0으로 간주하지 않으며 fixed-precision 속도 우위를 주장하지 않는다.

## 5. 연구 판정: 두 핵심 의무 모두 아직 미해결

**코드·고정 실험의 완료와 남은 두 연구 문제의 해결은 다르다.**

1. 독립 reference 정밀도: 이번 HMC와 control 모두 미달. 새 HMC는 채택하지 않는다.
   기존 ensemble IID의 약.9% 관측 RSE는 유지되지만 독립 상호 검증의 공백은 남아 있다.
2. 연속시간 상대 remainder: 새 root부등식과 sharper Jensen을 유도·구현했고 상수는
   줄였지만 유용한 relative certificate는 확보하지 못했다.

P2/production/large-training 모두 false다. all-q M2/bank qualification도 이번에 해결한 것이 아니다.
이 결과를 박사급/최상위 저널 준비 완료나 새로운 수학 원리의 검증 성공으로 포장하지 않는다.

### 5.1 반복 확장 대신 다음 판단이 필요한 이유

- **이 fixed HMC 및 cyclic 후보의 증량/production 통합은 중단한다.** 정합성은
  확인했으나 원래 확률의 정밀도 개선을 보이지 못했다.
- **같은 global-envelope/Hölder 상수 조정도 중단한다.** 현재 계열의 구조적 floor를
  확인했으므로 r scan이나 inverse-moment 상수 최적화만 반복할 근거가 없다.
- 원래 목표를 유지하려면 tail/error correlation·weak cancellation을 보존하는
  실제 새 수학적 증명과, 그 구조에 맞는 다른 reference 탐색 설계가 필요하다.
  가능한 연구 방향이지 성공을 보장한 구현 계획은 아니다. 새 operator/GPU만 붙이지 않는다.
- finite-grid estimator 자체를 핵심 논문으로 좁히는 선택은 별도 가능하지만 원래 P2
  해결과 다르다. 논문 범위 변경을 사용자 동의 없이 실행하지 않았다.

같은 실험을 성공할 때까지 seed/count만 바꾸는 방식으로 두 의무를 해결했다고 하지 않는다.
현재 상태에서 "완벽하게 마무리했다"고 말할 근거는 없다.

## 6. 감사·회귀·최종 검증

### 6.1 완료된 재현·무결성 검사

- 운영 복구4preflight+32pilot의 source·seed·전체 unit 파일·작업량·판정을 대조했다.
- 원래 완료24개와 새 기록은 wall외 모두 일치했다. 대표4whole를 다시 재생한 값과
  모든 stage 진단도 일치했다. 새 독립 confirmation으로 세지 않는다.
- 대표 재생은 physical127,488 / time-step522,190,848 / wall111.11초를 추가 사용했다.
- 최초 실패+전체복구+대표감사의 총 physical forward는 **1,797,632**회다.
  순서대로646,592+1,023,552+127,488이며 중복 seed 작업도 실제 연구 비용에 포함한다.
- 별도 sharper Jensen48개 조합도 전수 재계산과 source 감사를 통과했다.
- 이전 단계 포함8개 frozen archive,9,289개 파일 byte 대조가 일치했다.
  snapshot members9,297개,새 strict JSON70개,원래 실패/prefix anchor28개를 확인했다.
- CI 범위 Ruff 및 Mypy230source/main files, 새 runner3개의 별도 Mypy가 통과했다.
  전체 회귀 결과는 종료를 확인한 뒤 아래 기록에 추가한다.

복구 결과 SHA256:
`8b61fd67337dec8abf22bd5bc4ab084261f287362a39e7157f3037f59868001e`.
복구 source SHA256:
`496063469dd64083d006efb07c83444194454d31ee4bb5612dcb56818b7d4849`.
보강 결과 SHA256:
`ebf40e7eec82c8150add153f5e3e25b7703d43f764175f800d042036bd7a8ba0`.

[복구 결과](../../results/post_audit/p2_hamiltonian_operational_replay_v1/session_20261028/result.json),
[복구 감사](../../results/post_audit/p2_hamiltonian_operational_replay_v1/session_20261028/audit.json).

### 6.2 오류와 환경 제약을 숨기지 않는다

위 검사는 모든 이론적/기술적 오류의 부재를 증명하지 않는다. 신규 검증은
HMC/root-bound23개,sharper Jensen5개,운영 재생 계약3개다. plan의 시간합계 오타는
4.1절에 정정했고,원래 운영 timeout은 failure로 보존했다.

전역 Python의 기존 pip check는 exit1(무관한 comfy-cli/click 및 gradio 계열4개 충돌)이다.
전체 테스트 시작의 requests dependency warning도 보존한다. 임의 전역 패키지 변경을
하지 않았으며 환경 전체가 깨끗하다고 하지 않는다. 현재 Windows/Python3.10.11,
CPUfloat64/8threads 및 development PyTorch에서의 결과다. 동일기계 replay 일치를
다른 OS/버전의 bitwise 재현 보증이나 remote CI 통과로 확대하지 않는다.

### 6.3 전체 회귀와 최종 receipt

전체 `python -m pytest -q --cov=src --cov-report=term-missing`는
**1,991 passed,1warning /486.46초 /관측 process exit0**로 종료했다.
source statement coverage85%(23,645statements,3,473missed)다.
새 core coverage는 Gaussian-split95%,HamiltonianSMC97%,tail-root92%,sharpJensen93%다.
branch coverage나 모든 오류 부재의 보증은 아니다.

경고1건은 제한된 권한에서 기존 `.pytest_cache/v/cache/nodeids`를 저장하지 못한
PytestCacheWarning이다. 캐시가 아닌 테스트 자체는 모두 통과했다. 원래 캐시/ACL을
수정하거나 보안 제한을 우회하지 않았다. 별도 신규31개 검사에 `-p no:cacheprovider`를
사용한 재검사는 **31passed/20.03초/관측exit0**이며 cache warning은 발생하지 않았다.
requests의 기존 startup
dependency warning은 별도 stdout에 남아 있다.

[전체 회귀 stdout](../../results/post_audit/p2_hamiltonian_operational_replay_v1/session_20261028/pytest.stdout.txt),
[정적 검사 stdout](../../results/post_audit/p2_hamiltonian_operational_replay_v1/session_20261028/static.stdout.txt),
[최종 검증 receipt](../../results/post_audit/p2_hamiltonian_operational_replay_v1/session_20261028/verification_receipt.json).

로컬 Windows/Python3.10.11 결과이며 remote CI가 아니다. HEAD/origin-main59f2809는
그대로다. 커밋·푸시는 수행하지 않았고 이전 uncommitted 연구 작업도 보존했다.
