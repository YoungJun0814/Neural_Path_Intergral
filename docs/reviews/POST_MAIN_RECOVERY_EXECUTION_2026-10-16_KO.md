# Main 반영 후 reference 회복 실행 기록

2026-10-16 작성,2026-10-19 상태 갱신. 진행 상태와 완료 증거를 구분하는 보고서.
원래 q의 개선이나 전체 P2 통과 보고가 아니다.

## 1. main 반영과 검증 완료

검증 commit `bec0f11`, 원격 main merge commit
`f025b67b072db86a7573eac41779c044a2ea7d18`로 push했고 원격 SHA 일치를 확인했다.
forced push나 원격 이력 삭제는 하지 않았다. commit author는 기존 사용자 설정을 유지했다.

로컬 전체 pytest1376개, CI Ruff, production Mypy197files 및 git connectivity 검사 통과.
[원격 CI](https://github.com/YoungJun0814/Neural_Path_Intergral/actions/runs/38952464612)도
Python3.10/3.11 두 job 모두 성공했다. 이것은 당시 tree의 소프트웨어 검증이며
아래 후속 source와 통계 accuracy의 자동 인증은 아니다.

## 2. Canonical μ: 새 검증 완료·감사 통과

기존580-unit validation은 guide-free μ RSE1.5066% 때문에 실패한 채 보존했다.
새 namespace215 independent whole에서 original N32 μ만 검증했다. 이전71whole과
pooling하지 않고 source/anchor/seed/count를 고정했다.

| 항목 | 실제 값 |
|---|---:|
| 새 whole count | 215 |
| μ log mean | -16.8135296658 |
| RSE | 0.756787% |
| maximum whole contribution | 0.619271% |
| maximum LOO relative shift | 0.154875% |
| static μ와 symmetric relative difference | -0.223264% |
| 2-comparison equivalence upper | 3.399004% (<10%) |
| forward / actual backward | 각각54,819,840 |
| phase wall | 1065.97초 |

source/anchor/seed/cost 및 first,last whole numerical replay 통과.
감사 forward/backward 각각509,952회는 본 실험과 별도다.

`fresh_mu_development_reference_pass=true`와
`canonical_componentwise_development_recovery_pass=true`를 얻었다.
뒤의 값은 이전의 변하지 않은 risk/static/guided 성분과 이번 μ를 조합한 개발 inventory다.
원래 full-trial flag를 바꾸거나 한 번의 새로운 full confirmation으로 표시하지 않는다.
반복 개발 선택을 거쳤으므로 저널용 sealed confirmation과도 다르다.
P1/P2/large-training 전체 승인 flag는 false다.

데이터: `results/post_audit/canonical_mu_revalidation_v1/session_20261016/result.json`.

## 3. 운영 실패 보존과 후속 대조

N4096 resource-v1은 첫 more-temperatures whole가240초 cap을 넘겨 실패했다.
actual observed wall271.92초, conservative physical reservation129,280회.
finish에 normalizer result가 없으므로 이 prefix를 통계 분석에 이용하지 않는다.
source/manifest/failed finish를 모두 보존한다. backward64,640회는 설계상 예약/예상 값이며
완료 result의 독립 count로 잘못 보고하지 않는다.

v2는 같은 수학적 설계와 전체32whole를 새 namespace로 다시 생성하도록 사전 고정했다.
whole600초/phase6시간, 원래 physical/backward 상한은 유지한다. 실패 SHA를 새 manifest에
묶고 다른 대형 로컬 실험과 겹치지 않는다. 이후 v2를 실행했으나,20whole 완료 후
배터리 절전으로 high-η/more-temperatures/rep4가1745.756초를 기록해600초 cap에
실패했다. 원래 v2 result.json은 없으며 prefix를 성공으로 집계하지 않는다.
Windows 전원 기록의 저전원 구간은 UTC21:28:28.651166–21:53:29.669933이다.
별도 [전체 고정-seed 복구 계획](../plans/RESOURCE_V2_SLEEP_RECOVERY_2026-10-16_KO.md)에
따라20개 immutable complete와 원래 seed의 미완료12개를 함께 복구하도록 실행했다.
2개를 완료했으나 세 번째 복구 whole는 배터리20% guard에 실패했다.
전체22/32개만 complete이고 두 프로토콜 모두 result.json이 없다. 새 대형 계산은 중단했다.
원래 운영 실패와 새 계산비용은 보존하며, fresh32개 confirmation이나 wall 벤치마크가 아니다.
근거: [운영 재시작 계획](../plans/P2_FINE_RESOURCE_OPERATIONAL_RESTART_2026-10-16_KO.md).

## 4. 저메모리 static reference

`FFTStaticShiftReference`는 원래 uniform time/amplitude Gaussian mixture를 implicit하게
계산한다. N4096에서49,141개 full mean 벡터를 저장하지 않고 analytical row sampling과
FFT/full factorized density를 사용한다. N16 dense 모든 component mean/density(ρ±.7),
sampler 재구성, natural exact 보존 및 큰 grid 준비 테스트4개를 통과했다.

두 셀×8 IID blocks, 각32768fine samples의 fine4096/coarse2048 paired pilot는
16blocks 표본 생성을 완료했다. 마지막 recursive aggregation에서 `log_maximum`
metadata 누락으로 예외가 나서 원래 result.json은 없다. 원본 finish/source를
보존한 별도 recovered-result.json을 만들었고 고정2block 재현 감사를 통과했다.
추가 표본이나 seed 교체는 없다. 설명: [집계 복구 기록](FFT_STATIC_SUMMARY_METADATA_RECOVERY_2026-10-16_KO.md).

| 셀 | fine μ RSE | coarse μ RSE | 최대 단일 기여 | paired relative 차이 | paired SE | 차이 상한 |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 12.364% | 11.326% | 4.871% | +8.832% | 8.734% | 28.408%: 실패 |
| high-η | 4.353% | 4.600% | .939% | -1.738% | 3.337% | 9.218%: 작은 차이 진단만 통과 |

canonical block RSE14.755%/LOO10.920%, high-η block RSE3.003%/LOO2.216%다.
두 셀 모두 fine μ의1.5% reference precision을 입증하지 못했다. 모든 forward1,048,576회,
density524,288회,time-step3,221,225,472회. 복구 시점까지 original phase wall2808.88초다.
이는 정상 종료한 원래 실행으로 덮어쓰지 않은 recovered development 결과다.
감사 physical131,072/density65,536회는 본 실험과 별도이며, 기존 저장 moments는
동일하게 재현되고 새 복구 log-maximum metadata도 허용 roundoff 이내였다.
이는 원래 P3의373성분 matched 모델 비교가 아니며 learned proposal의 개선 증거도 아니다.
데이터 경로: `results/post_audit/fft_static_fine_reference_v1/session_20261016`.
근거: [고정 pilot 계획](../plans/FFT_STATIC_FINE_REFERENCE_PILOT_2026-10-16_KO.md).

## 5. 기술 검토와 남은 범위

후속 runner의 명시적 Mypy 검사에서 annotation 문제3개를 발견했다. v1 resource가
중단된 후 그 파일의 두 자료형을 보강했고, canonical μ가 완료된 후 lambda를 명시적
typed operation으로 바꿨다. 수식·RNG·target은 바꾸지 않았고 source archive는 덮어쓰지 않았다.
canonical first/last numerical replay가 일치했다. 신규4runner Mypy와 CI Ruff,
후속 targeted tests10개 통과. 최종 전체 suite는 해당 phase 종료 후 다시 검사한다.

원래 q의 실용적 second moment는 별도 문제다.
[reference와 모델 준비도의 구분](REFERENCE_VERSUS_MODEL_READINESS_2026-10-16_KO.md) 및
[grid에 따른 price guide scaling](../theory/GRID_STABLE_PRICE_GUIDE_SCALING_2026-10-16_KO.md)을 참조한다.
fine-grid 독립 accuracy, bank/tail coverage, 실제 q의 개선, 학습 포함 비용, 새 confirmation,
continuum 주장 범위는 남아 있다. 소프트웨어 green이나 개발 reference 회복만으로
최상위 저널 제출 가능 수준이라고 결론 내리지 않는다.

## 6. 시작 전 backend 교정과 수학적 재검토

N4096 callback의 같은128개 입력을 threads1/2/4/8에서 각각1warmup+5timed 평가했다.
components/family potential/capped gradient를 rtol5e-10/atol1e-8 안에서 대조했고 모두 통과했다.
median seconds는 .39658/.24938/.16541/.11345로8threads를 고정 선택했다.
관측 callback ratio3.495배는 full SMC/다른 모델의 speedup 증거가 아니다.
물리 평가6144회/actual backward3072회, 전체wall15.87초였다.
8thread full kernel의 deterministic replay 및 알려진4차원 Gaussian normalizer
oracle도 통과했다. calibration source/result의 hash를 v2 manifest에 묶도록 구현했다.
당시 실행 중이던 static pilot는 원래1thread 설정을 그대로 유지했다.

추정식·구현을 다시 대조한 핵심 사항:

- full Gaussian reverse-density MH 보정은 capped drift에도 필요하다. βΔlogH만
  사용하는 plain pCN acceptance로 바꾸지 않았다.
- component normalizers는 SMC product Zhat와 endpoint ratio를 함께 사용한다.
  ratio만으로 μ를 추정하거나 empirical self-normalization하지 않았다.
- paired ratio diagnostic의 influence는 `(F-C)-relative*F`다. 두 output의 covariance를
  유지하고 whole/equal-size-block을 SE 단위로 사용한다. 작은 차이는 continuum bound가 아니다.
- implicit static density는 time/amplitude product grid 전체를 정확히 합산한다.
  sampler와 scorer 모두 explicitfloat64이고 price direction은 left-driver row와 직교한다.
- 새 μ conditioning을 원래 q의 raw M2로 대체하지 않는다. fixed seed/replay 및
  code 검사는 미관측 tail이나 모든 floating-point 입력의 정식 검증을 보장하지 않는다.

추가로 Gaussian lognormal coupling의 정확한 second-moment 식, integrated variance
inverse-moment Jensen bound 및 conditional CDF derivative를 결합한 절대 bias bound를
검토했다. 3개 독립 oracle를 통과했다. 유효한 driver MSE 상한이 따로 필요하고,
μ≈10^-8에서 유용한 상대 bias 보증이 되지는 않는다. 임의의 weak rate를 넣어
P2를 통과시키는 근거로 쓰지 않는다.
[상세 유도](../theory/CONDITIONAL_CDF_ABSOLUTE_BIAS_BOUND_2026-10-16_KO.md).

## 7. 이 구간 종료 검증

원래 16block source/finish와 recovery source를 모두 감사했고 μ215회 검증의
first/last replay도 통과했다. 최종 전체 pytest **1392개 통과(335.79초)**,
CI Ruff 통과, production Mypy **198 source files 통과**, diff 공백 검사 통과.
신규 runner의 명시적 Mypy와 recursive-moment/8thread kernel/수학 oracle 회귀도 통과했다.
이는 위 실패한 reference precision을 성공으로 바꾸는 결과가 아니다.

다음 장시간 단계로 v2의 원래32whole 대조를 실행했으나 운영 실패했다(3절).
별도 complete-grid recovery도 전원 guard로 중단됐다. 아직 fine-grid qualification을
얻지 못했으므로 large-training은 시작하지 않는다. raw q의 실용성 문제도 별도로 유지한다.

## 8. 후속 physical-window 설계: 실제 표본 생성 전 검증

373성분 G0/G1 reference core를 기존 normalized Gaussian mixture sampler/density 위에
구현했다. G0는31point-volatility/one-cell-price, G1은15point-volatility+16window-volatility이며
**모든31price 방향**이T/8 physical future window(만기 인접 마지막은T/16)를 쓴다.
초기 geometry-only prototype에 남아 있던15개 one-cell price 방향은 strong grid
consistency 가정을 만족하지 않아 실제 sampling 전에 수정했다. 원래 q/BLP/payoff는 변경하지 않았다.

adjoint FFT/dense identity, exact mixture reconstruction, causal support/energy,
Brownian price embedding, recursive paired summary, 새 복구 ledger/power guard 및 Gaussian
toy를 합한 targeted14tests와 explicit Mypy3files를 통과했다. 전체 suite는 phase 종료 후 재검사한다.
G0/G1 실제 성능 표본은 아직 생성하지 않았다. 별도 고정32block pilot는 complete-grid
recovery와4whole 감사 후에만 시작하도록 구현되어 있고 현재 실행 조건을 만족하지 않는다.

[조건부 density consistency 검토](../theory/PHYSICAL_WINDOW_DENSITY_CONSISTENCY_2026-10-16_KO.md)는
고정 finite dictionary/강수렴 representer/payoff L1 수렴이라는 가정 아래 second-moment
연속성을 유도한다. 한-cell 방향의 stable-noise/Jensen penalty는 별도 극한 결과다.
Gaussian toy에서 priorM2=.117871, shrinking-cell noise limit=.228066,
fixedT/4windowM2=.120742를 확인했다. 이는 실제 Volterra 효율/독창성/저널 수준의 증거가 아니다.

## 9. 전원 중단 후 최종 상태

2026-10-19 복귀 후에도 원래 파일/HEAD 변화가 없음을 확인했다. battery13%/AC 연결 표시를
관측해 새 장시간 계산과 전체 로컬 suite 재실행은 보류했다. 새 변경분의 targeted14tests,
전체 CI Ruff, production Mypy199files 및 신규3files 명시적 Mypy는 중단 전에 통과했다.
원래20complete+새2complete+failed records의 source/archive/checksum/count 무결성 검사를
통과했다. 새로운 complete-grid statistical summary나 numerical replay 인증은 아니다.
[비용·중단 원인·재개 조건 상세](RESOURCE_POWER_INTERRUPTION_REVIEW_2026-10-19_KO.md).

## 10. 전원 안정 확인 후 후속 실행

사용자 확인 뒤 별도 protocol로 원래 seed1개의 numerical whole를 더 저장해 기록은23/32개다.
그러나30→26% 감소로 새 power guard가 중단시켰다. 이어2thread 고정 입력 교정의
수치 비교는 통과했지만, 충전 표시와 RemainingCapacity 감소가 서로 맞지 않아 전원
안정성 gate에 실패했다. 새 full-grid result는 없고 G0/G1 실제 표본은 아직0개다.
[상세 수치·비용·가정·검토](STABLE_POWER_RESTART_AND_CALIBRATION_2026-10-19_KO.md)를 참조한다.
