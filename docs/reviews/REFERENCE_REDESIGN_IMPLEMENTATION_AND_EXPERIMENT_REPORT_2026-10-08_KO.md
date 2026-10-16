# Reference 재설계: 생산용 구현·실험·오류검토 결과

작성일: 2026-10-08.

## 1. 결론부터

**생산용 계산 코어·실행기·봉인된 production allocation/qualification 경로를 구현하고, 계획의 마지막-pair·guide-free kernel·고정 block 실험을 실행했다. 그러나 성능·정밀도 관문은 통과하지 못했다.**

사전 한도 안에서 실제 수행한 합계는 7,798,132 potential-equivalent 단위, FFT 경로 3,553,274개, CDF 4,244,858개다. 세 실행기의 내부 wall-time 합은 약 177.57초다. Python import·source archive 생성·별도 감사·테스트 시간은 이 합에 포함하지 않는다.

최종 fresh reference production은 **미실행**이다. 34개 예정 job 중 33개만 forecast할 수 있는 불완전한 상태에서도 약 172.07억 단위·185.54시간으로 전망됐다. 이는 실제 실행시간이나 정확도 인증이 아니며, 누락된 high-η probability ellipse job을 포함한 전체 forecast도 아니다. 한도는 1.6억 단위·2시간이다.

P2는 잠금 유지. 모델 q 변경, 새로운 학습, confirmation, 논문 성능 주장, 커밋·푸시는 하지 않았다.

## 2. 구현한 기능

| 기능 | 실제 구현 | 보호 조건 |
|---|---|---|
| exact marginal | [gaussian_mixture_marginal.py](../../src/path_integral/gaussian_mixture_marginal.py) | identity-covariance shift mixture만 지원, 다른 rank 거부 |
| μ 해석적 조건부 평균 | [structural_v2_conditional_reference.py](../../src/path_integral/structural_v2_conditional_reference.py) | terminal downside·ε=1·현재 left-point grid |
| cached original-M₂ 내부 평가 | 같은 파일 | full-q mean norm·마지막 y 의존성 보존 |
| 고정 block 적분 | 같은 파일 | 원래 전체 미래 경로 재계산, path-dependent block 금지 |
| nested 분산 분해 | [nested_reference_statistics.py](../../src/path_integral/nested_reference_statistics.py) | outer unit SE, 음수 noise-subtraction 결과 비절단 |
| guide-free ellipse mutation | [elliptical_slice_kernel.py](../../src/path_integral/elliptical_slice_kernel.py) | finite rejection cap은 전체 실패, 잘린 transition 반환 금지 |
| SMC adapter | [weighted_tempered_smc.py](../../src/path_integral/weighted_tempered_smc.py) | 기본 pCN 유지, fixed bridge/resampling 유지 |
| 실행·봉인·감사 | [post_audit_v2_reference_redesign.py](../../experiments/post_audit_v2_reference_redesign.py) | pilot/final 분리, exact q binding, seed ledger, work/RSS/time cap |
| 최종 production 경로 | 같은 실행기 | 34개 grid, allocation SHA/config 완전 일치, forecast 통과 전에 실행 금지 |

production에서는 모든 최종 표본을 Python list로 누적하지 않는다. 배치별 sufficient moments·최대 기여·block means를 저장하고 전체 moments를 merge한다. final을 본 뒤 표본 수를 늘리는 기능은 없다.

수학·수치 검토의 자세한 내용은 [구현 계약 감사](../theory/REFERENCE_REDESIGN_IMPLEMENTATION_AUDIT_2026-10-08_KO.md)를 참조한다. 신규 theory가 무오류로 완전히 증명됐다는 선언은 아니다.

## 3. 실제 실행과 source/seed 기록

| 실행 | 상태 | 단위 합계 | FFT | CDF | wall초 | peak RSS bytes | seed |
|---|---|---:|---:|---:|---:|---:|---:|
| 마지막 pair micro | 개발 실험 완료, gain 불통과 | 708,864 | 8,448 | 700,416 | 1.576 | 596,004,864 | 98 |
| guide-free kernel | 전체 묶음 protocol failure | 4,999,924 | 2,499,962 | 2,499,962 | 164.453 | 578,523,136 | 67 |
| 고정 block 묶음 | 개발 실험 완료, gain 불통과 | 2,089,344 | 1,044,864 | 1,044,480 | 11.541 | 703,918,080 | 240 |

단위 합계는 **FFT+CDF**다. 과거 P1의 단일 potential-call count와 직접 비교하지 않는다. 전체 density-component 계산 수는 14,929,490이다. 최대 메모리는 약 671.31 MiB였다.

산출물:

- [micro 원자료](../../results/post_audit/reference_redesign_micro_v1.json), [당시 source ZIP](../../results/post_audit/reference_redesign_micro_v1.source.zip).
- [kernel 원자료](../../results/post_audit/reference_redesign_kernel_v1.json), [당시 source ZIP](../../results/post_audit/reference_redesign_kernel_v1.source.zip).
- [block 원자료](../../results/post_audit/reference_redesign_block_v1.json), [당시 source ZIP](../../results/post_audit/reference_redesign_block_v1.source.zip).
- [production allocation 실패 보고](../../results/post_audit/reference_redesign_allocation_v1.json).

source ZIP은 각 실행 당시의 dirty source/config/tests/계획을 봉인한다. micro 이후 감사의 thread convention을 수정했으므로 micro ZIP과 후속 ZIP의 실행기 소스는 다르다. 측정에 사용한 conditional evaluator·μ/M₂ target은 바꾸지 않았고, micro 표본도 재생성·교체하지 않았다.

## 4. 마지막-pair 결과

각 셀 2,048 independent outer units. 서로 다른 L은 같은 outer를 공유하며 아래 수치 비교는 **개발 point estimate**다. independent final 비교나 significance 결과가 아니다.

| 셀 | L | raw M₂ 점추정 | outer-unit RSE |
|---|---:|---:|---:|
| canonical | 1 | 1.4376e−9 | 20.92% |
| canonical | 4 | 1.5199e−9 | 21.38% |
| canonical | 16 | 1.5172e−9 | 21.56% |
| canonical | 64 | 1.5302e−9 | 21.57% |
| high-η | 1 | 6.1089e−7 | 21.88% |
| high-η | 4 | 6.1531e−7 | 22.10% |
| high-η | 16 | 6.1898e−7 | 21.92% |
| high-η | 64 | 6.1530e−7 | 21.91% |

paired-inner 진단의 canonical L=64 inner-mean CV²는 0.0322, outer CV²는 95.4740이었다. high-η는 0.00527 대 98.6516이었다. 작은 pilot에서 추정한 분해이며 population guarantee는 아니지만, **지금 제거한 마지막 pair가 주된 잔여 분산 원인은 아니라는 증거**다.

L=1 대비 worst-cell cost proxy 비율은 L=4 약 1.082, L=16 약 1.249, L=64 약 1.883이었다. 사전 기준은 ≤0.8이므로 모두 실패했다. 추가 내부 표본은 이미 작은 내부 오차를 줄이고 계산량을 늘렸다.

μ의 해석적 평균 결과는 canonical 7.2451e−8, RSE 47.97%; high-η 4.3412e−6, RSE 15.17%였다. μ identity가 틀렸다는 뜻이 아니라 작은 outer 표본에서 여전히 희귀한 경로를 찾는 문제가 크다는 뜻이다. 이 평균을 참값으로 사용하지 않는다.

## 5. guide-free kernel 결과

128 particles, 64 fixed temperature points, mutation 2, fixed resampling calendar. 각 완료 방법은 8 independent whole-runs. pCN과 ellipse는 같은 schedule이지만 **실제 work는 동일하지 않다**.

| 셀·대상 | 방법 | 점추정 | whole-run RSE | wall초 |
|---|---|---:|---:|---:|
| canonical M₂ | pCN | 3.8570e−10 | 39.97% | 4.581 |
| canonical M₂ | ellipse | 1.3763e−9 | 26.18% | 46.514 |
| canonical μ | pCN | 2.8189e−8 | 22.49% | 3.989 |
| canonical μ | ellipse | 4.9258e−8 | 14.63% | 37.699 |
| high-η M₂ | pCN | 4.1048e−7 | 28.33% | 4.528 |
| high-η M₂ | ellipse | 6.0136e−7 | 26.74% | 46.305 |
| high-η μ | pCN | 4.1737e−6 | 10.24% | 4.051 |
| high-η μ | ellipse | **미완료** | 평가 불가 | 16.288까지 |

high-η μ ellipse는 3개의 whole-run만 완료한 뒤 다음 run 내부에서 전체 500만 단위 cap에 도달했다. 3개만 골라 평균·RSE를 만들어 pass시키지 않았다. 묶음 전체 status는 protocol failure다.

canonical risk에서 ellipse의 관측 mean은 이전 guide-based reference에 더 가까웠지만 RSE가 매우 높다. 이를 참값 확인·bias 제거·우위 증명으로 해석하지 않는다. pCN의 낮은 8-run 평균 역시 무조건적 estimator가 biased라는 증명이 아니다. rare whole-run 누락/분포의 skew와 높은 Monte Carlo 변동 가능성을 구분해야 한다.

관측 RSE²×wall proxy에서 ellipse/pCN 비율은 canonical risk 약 4.36, canonical μ 약 4.00, high-η risk 약 9.11이었다. 이 pilot에서는 **RSE 일부 감소보다 비용 증가가 컸다**. 동일 accuracy 자격이 없으므로 논문용 fixed-precision 승패 비교가 아니다.

canonical risk의 최종 initial ancestor 수는 pCN 4–7, ellipse 3–9였다. high-η risk는 4–11 대 8–14로 일부 개선됐다. descendant들이 MCMC로 이동하므로 ancestor 수가 independent path 수와 같지는 않지만, genealogy 문제가 해소됐다고 주장할 수도 없다.

## 6. 고정 block 결과

0.25T/0.5T/0.75T 근처 pair를 사전 고정하고 셀·block당 1,024 outer units를 사용했다. 각 inner 좌표를 바꾼 뒤 미래 경로 전체를 원래 simulator로 재계산했다.

worst-cell cost proxy의 L=1 대비 비율:

| block | L=4 | L=16 | L=64 |
|---|---:|---:|---:|
| 0.25T | 1.804 | 3.496 | 13.685 |
| 0.5T | 2.347 | 5.856 | 20.960 |
| 0.75T | 1.885 | 4.351 | 16.110 |

모두 ≤0.8 기준을 충족하지 못했다. 일부 셀/block에서는 RSE가 감소했지만 미래 경로 재계산 비용을 상쇄하지 못했다. 초기·중기 한 pair로는 다른 시간대·기억 방향의 outer 변동이 남았다. 제한된 block 후보가 실패한 것이며 모든 가능한 조건부 적분의 불가능성 증명이 아니다.

## 7. final production 판정

실패 사유는 중복 없이 다음과 같다.

1. kernel pilot 전체가 완료되지 않았다.
2. 두 셀 모두에서 20% cost gain을 얻은 nested 후보가 없다.
3. ellipse의 whole-run 수 forecast가 최대 256회를 넘었다.
4. high-η μ ellipse의 완료 pilot이 없어 full 34-job allocation이 불가능하다.
5. 구성 가능한 33-job partial forecast도 총 work/time cap을 크게 넘었다.

rep0 risk ellipse 필요 whole-run forecast는 canonical 7,314회, high-η 7,627회였다. canonical μ ellipse는 2,283회였다. 모두 pilot 기반·safety factor 3의 예측이며 필요한 표본 수의 확정값이 아니다.

반면 nested L=1의 raw M₂ 필요 outer count forecast는 canonical 1,245,184, high-η 1,310,720였다. 여기서 L=1은 **진단용 fallback**일 뿐 후보 통과나 production 승인으로 쓰지 않았다. reference 정확도를 인증할 guide-free 경로가 병목이다.

생산 config는 생성하지 않았다. allocation 보고서의 `production_config=null`, `p2_authorized=false`가 기계 판정이다. 사용자가 큰 실험을 요청했어도 사전 계획의 실패·예산 관문을 성공으로 변경하지 않는다.

## 8. 이론적·기술적 오류 재검토

### 이론적 계약

- μ 조건부 평균과 raw M₂의 조건부 내부 평균을 분리했다.
- full q의 분모·mean norm·마지막 y 의존성을 유지했다.
- 정확한 r marginal을 사용했고 full r(A,0)을 marginal로 쓰지 않았다.
- nested inner 수를 IID 표본 수로 잘못 계산하지 않았다.
- noisy inner potential을 SMC의 log-likelihood로 넣지 않았다.
- 고정 block에서는 미래 volatility/I/J를 재계산했다.
- fixed temperatures/resampling과 Gaussian-invariant mutation 구조를 유지했다.
- truncated ellipse와 incomplete whole-run을 estimator로 반환하지 않았다.

### 발견·수정한 경계 문제

1. 누락된 bootstrap stream에서 StopIteration이 나던 경계를 명시적 ValueError로 수정했다. 누락 자료를 받아들이는 문제는 아니었지만 오류 유형과 감사 계약이 불명확했다.
2. 별도 감사 process의 기본 CPU thread 수가 실험의 1-thread와 달라 guide를 bitwise 재구성하지 못했다. 같은 thread convention으로 재검사하면 원자료 audit가 통과했고, 감사 실행기에 해당 설정을 강제했다. tolerance를 넓히거나 저장된 guide를 교체하지 않았다.
3. full block bundle의 work 사전 합계를 계산해 batch를 최대 512 범위 내 256으로 동결했다. 실행 후 결과를 보고 표본 수를 바꾼 것이 아니다.
4. production 전체 raw sample list 보관으로 메모리가 커질 위험을 streaming batch moments로 바꿨다.

### 검증 범위

신규 core/runner 검사 26개와 기존 SMC/role/identity 회귀를 수행했다. 생산용 source 변경 전 전체 1,233개 회귀가 통과했고, thread-convention 수정 뒤 targeted 26개와 세 실제 artifact 감사가 통과했다. 마지막 전체 회귀 결과는 아래 검증 상태에 기록한다.

초기 전체 회귀에는 수정 전 bootstrap 경계 오류로 1개 실패가 있었으며, 최종 결과로 숨기지 않는다. 그 run은 이미 import된 수정 전 코드를 사용했고, 수정 후 새 process 전체 회귀가 통과했다. Requests dependency 호환성 warning은 기존 환경 경고이며 이번 변경에서 패키지를 설치·업데이트하지 않았다.

**최종 검증 상태:** thread-convention 수정까지 포함한 현재 source 기준 전체 1,233개 테스트 통과(203.67초). Ruff 전체 범위 통과, Mypy 178개 source file 통과. micro/kernel/block artifact의 source·q·role·arithmetic 감사 모두 통과했다. protocol failure인 kernel artifact의 감사 통과는 실패 기록이 일관된다는 뜻이지 scientific gate 통과가 아니다.

테스트·artifact 감사 통과는 무오류 보증·unconditional tail certificate·continuous-time theorem·논문 novelty 증명이 아니다.

## 9. 왜 실패했는가, 무엇이 다음인가

이번 실패는 단순히 GPU가 없거나 신경망이 작아서 생긴 문제라고 보기 어렵다.

- cheap last-pair 적분은 정확하지만 **잔여 outer path 변동**을 제거하지 못했다.
- nonterminal 한 pair 적분은 미래 재계산 비용이 커서 작은 분산 감소와 교환관계가 나빴다.
- ellipse는 다른 geometry를 탐색하지만 rejection/짧은 active batch 비용과 초기 bridge의 희귀 population 문제를 동시에 해결하지 못했다.
- guide-free reference의 낮은 whole-run 정밀도가 전체 검증을 지배했다. guide-based 방법만 크게 돌리면 공통 blind spot 문제가 남는다.

후속 후보를 검토한다면 아래는 **새 사전 등록과 fresh 표본이 필요한 가설**이다. 이번 개발 결과를 새 조건으로 재채점해 성공이라고 부르지 않는다.

1. **L=1 marginal-reference 자체의 가치:** 마지막 pair 내부 평균 확대는 실패했지만 r_Aφ₂라는 다른 full reference proposal이 기존 full-r static IS보다 저렴할 가능성이 있다. pilot 시대·표본 수·timing이 다르므로 이전 artifact와의 단순 score 비교는 확인이 아니다. 별도 fresh matched-work 대조가 필요하다. 공통 guide 독립성 문제는 그대로 남는다.
2. **초기 bridge coverage를 직접 검증하는 별도 방법:** invariant mutation을 다른 이름으로 교체하는 것보다 초기 population에 희귀 기여가 들어오는 확률·whole-run skew·ancestry를 분리할 수 있는 대조를 먼저 설계한다. temperature sweep을 추가 무제한 반복하지 않는다.
3. **여러 시간대 기여를 joint하게 평균내는 방법:** 한 pair가 아닌 구조적 방향의 conditional integration은 가능한 연구 방향이지만, conditional Gaussian density·future-memory 변화·cost bound를 먼저 유도해야 한다. path-dependent spike 선택을 그대로 쓰지 않는다.
4. **계산 예산 재설정:** 현재 pilot을 그대로 대규모 확장하는 185시간급 실행은 승인 없이 시작하지 않는다. 비용 최적화만으로 whole-run count cap과 coverage 문제가 해결된다고 가정하지 않는다.

현 단계의 권고는 **새 kernel 추가나 inner L 확대를 중단하고, 독립 reference의 초기 rare-mode coverage 문제를 다시 설계하는 것**이다. 주모델 구조 확대·P2 이후 성능 주장·최상위 저널 submission 준비는 아직 이르다.

## 10. 재현 명령

기존 결과 파일은 덮어쓰지 않는다. 재실행하려면 fresh output/source 이름과 config identity를 먼저 봉인해야 한다.

```powershell
python -m pytest -q
python -m ruff check src tests experiments main.py train_driftnet.py
python -m mypy src main.py train_driftnet.py
python -m experiments.post_audit_v2_reference_redesign --audit results/post_audit/reference_redesign_micro_v1.json
python -m experiments.post_audit_v2_reference_redesign --audit results/post_audit/reference_redesign_kernel_v1.json
python -m experiments.post_audit_v2_reference_redesign --audit results/post_audit/reference_redesign_block_v1.json
```

실험 configs는 [micro](../../configs/post_audit/reference_redesign_micro_v1.yaml), [kernel](../../configs/post_audit/reference_redesign_kernel_v1.yaml), [block](../../configs/post_audit/reference_redesign_block_v1.yaml)에 보존했다. 결과 보고서와 production allocation의 상태를 구분해서 읽어야 한다.
