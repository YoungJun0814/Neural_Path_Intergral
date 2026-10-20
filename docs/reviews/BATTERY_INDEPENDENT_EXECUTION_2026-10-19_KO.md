# 고정-grid 복구와 physical-window reference 실행 기록

2026-10-19. 실행 중인 단계와 검증 완료를 구분한다. 커밋·푸시하지 않는다.
사용자의 명시적 요청에 따라 새 실행의 battery/AC 조회·중단 조건을 제외했다.
OS 전원 설정이나 과거의 실패 기록, 통계 qualification 기준은 변경하지 않았다.

## 실행 순서와 현재 상태

1. 원래32개 seed grid: 23개 검증 재사용+9개 새 계산으로32/32개 complete.
   신규 physical1,149,184회, backward574,592회,time-step3,530,293,248회.
   이번9개 runner phase wall2634.50초이며 CLI import/source 준비 이전 시간은 포함하지 않는다.
2. 원래 cell/design rep0의4whole 재현 감사: 통과. physical513,536/backward256,768회를
   별도로 계산해 normalizer/component/stages를 동일 seed로 대조했다. result SHA는
   `7db917c24bbae6034b48b53f4a838e7f38cc3471b9c0ecfcd23d3cbb1de27ee1`이다.
   과거 operational protocol 실패는 그대로false다.
3. G0/G1의 사전 고정32block 실제 pilot:31개 complete 뒤 마지막 block이 start-only로 중단됐다.
   원래 result/protocol pass는 없다. [별도 복구 계획](../plans/PHYSICAL_WINDOW_INTERRUPTION_RECOVERY_2026-10-19_KO.md)에
   따라31개를 membership으로 재사용하고 마지막1개를 같은 seed로 재계산 완료했다.
   원래 interrupted physical32,768회 예약 비용은 별도 보존한다. 종료 원인은 확정하지 않는다.
4. G0/G1 각cell/method rep0의4block 재현 감사: 통과. 별도 physical131,072/density65,536회.
   복구 result SHA: `46b8a9291e2aff81dd55be0a008c2556bd447a2372cfa506a4790af6b2274c54`.
5. SMC2designs×IID2guides×2cells의8개 fine μ 비교: 완료.8/8 uncertainty-crossing,
   동등성 확인0/8, interval 전체가 margin 밖인 경우0/8. 새로운 physics 계산0회.
6. Ruff/정규 Mypy200source 통과. 전체 pytest **1,437개 통과**,374.31초.
   HEAD `ace4bca4ae298fe438fa8b74d0b6013f238c0afe`와 빈 staging 상태를 재확인했다.
   커밋·푸시·원격 변경은 하지 않았다.

현재 HEAD는 `ace4bca`다. 이번 변경과 결과는 로컬 working tree에 남긴다.

## 구현·검증된 제한

- 새 completion loader는 원래23개 numerical record와37개 evidence binding을 검사한다.
  기존8threads/float64/seed/SMC kernel/normalizer identity를 유지한다.
- G0/G1 실행 profile은 `battery_independent`를 명시적으로 선택해야 한다.
  해당 profile에서 battery query가 호출되지 않는 회귀 검사를 추가했다.
- source/seed/member/count/finite/memory/wall 검사는 유지한다. 새로운32개 confirmation으로
  부르거나 작은 adjacent difference를 continuum relative-bias 보증으로 취급하지 않는다.
- 교차 검증은 잘 맞는 method만 골라 비교하지 않고8개 조합 전체를 보고한다.
  Bonferroni 조정은 이8개 개발 비교만의 범위이며 finite-sample/tail coverage 보증이 아니다.
  관측 RSE1.5% 여부와 formal accuracy qualification을 구분하고 모든 training flags는false다.
- G0/G1 전체 요약 열람 전, 기존8개 adjusted CI에 대한 서술적 분류를 comparator에 추가했다.
  interval 전체가10% margin 밖인 경우와 interval이 margin을 걸쳐 동등성을 확인하지 못한
  경우를 구분한다. 새 hypothesis test/유리한 셀 선택/alpha 또는 pass 기준 변경은 아니다.
  특히 정밀도가 낮아 동등성 확인에 실패한 것을 편향의 검출로 표현하지 않는다.

## 추가 이론 점검

[Price-window coupling 검토](../theory/FINITE_GRID_PRICE_WINDOW_COUPLING_2026-10-19_KO.md)의
독립 oracle6개가 통과했다. price 방향은 과거 volatility feature와 직교하지만 미래의
rough driver도 바꾼다. 따라서 상수 volatility의 직접가격 forcing∝sqrt(h)만으로
실제 G0의 효과를 설명하면 불충분하다. driver peak는h^H로 줄어들며 H=.05에서 매우 느리다.
integrated norm과 peak scaling을 구분하고 BLP의 grid-aligned mean shift를 정확한
kernel 적분 및 독립 dense operator와 대조했다. 전체 연속시간 covariance/효율 보증은 아니다.

G0→G1에는 price window 이외의 volatility feature/시점 변경도 포함된다. 이후 G1이
좋아져도 원인을 price window 하나로 단정하지 않는다. 같은 centres/volatility feature에서
price 방향만 바꾸는 matched ablation 등이 필요하다. 원래32block 결과를 본 뒤 arm을
추가해 같은 confirmation처럼 해석하지 않는다.

### 추정량·구현 경계의 재검토

- 자연 law는 whitened Gaussian p이며, g는 독립 가격 Brownian을 해석적으로 적분한
  같은 terminal-downside conditional CDF다. 두 guide는 물리 simulator와 target을 바꾸지 않는다.
- IID reference는 E_r[g/(r/p)]의 ordinary mean이다. label별 density나 선택한 component만의
  density가 아니라 natural component를 포함한373성분 전체 balance-mixture를 쓴다.
- coarse latent는 fine latent와 독립 Gaussian augmentation의 정확한 선형 coupling으로 만든다.
  paired output에도 fine proposal likelihood를 한 번만 적용한다. coarse proposal density를
  추가로 곱하거나 같은 white coordinate를 두 grid의 Brownian coupling으로 오인하지 않는다.
- SMC SE는 독립 whole 사이에서 계산한다. IID raw SE와8개 equal-block sensitivity는 별도로
  보고한다. SMC particle나 paired fine/coarse output을 독립 replicate로 세지 않는다.
- G0/G1의 reference 성과와 원래 모델 proposal q의 μ/M2/훈련 bank 성과는 다른 대상이다.
  이번에는 q를 재학습하거나 bank/차원/새 operator를 변경하지 않는다.
- 작은 adjacent difference는 연속시간 bias의 certificate가 아니며, source/seed replay도
  미관측 tail coverage·수학적 최적성·저널 독창성을 인증하지 않는다.

## 최종 수치·검토

### 완료한 resource grid (4whole numerical replay 감사 통과)

각행8개 독립 whole로 SE를 계산했다. particle들을 독립 표본으로 세지 않았다.

| 셀 | 고정 설계 | fine μ RSE | coarse μ RSE | paired 차이 상한 | initial ancestors | 새 precision whole 예측 |
|---|---|---:|---:|---:|---:|---:|
| canonical | 128particles/128temperature points | 8.260% | 8.348% | 7.405% | 16–26 | 744 |
| canonical | 256particles/64temperature points | 11.974% | 11.732% | 5.628% | 27–36 | 1530 |
| high-η | 128particles/128temperature points | 6.526% | 6.582% | 9.260% | 21–25 | 463 |
| high-η | 256particles/64temperature points | 6.693% | 5.347% | 7.510% | 31–41 | 478 |

4개 paired small-difference endpoint는10% 기준에서 통과했지만, μ reference의1.5%
precision은 어느 행도 충족하지 못했다. canonical/more-particles fine의 maximum whole
share22.139%/LOO11.016%도 별도로 보고한다.8whole bootstrap sensitivity는 미관측 tail
coverage 보증이 아니다. ancestry는 reference SMC의 값이지 CE/SMC training bank 개선 인증이 아니다.

사전 고정 safety3 timestep-cost forecast rule은 두 셀 모두 more-temperatures를 선택했다.
새 whole744/463개, physical96,184,320/59,856,640회를 예측한다. 이는 실제 precision
달성/우위 검증/실행 승인이 아니다. 선택 및 전원 상태가 섞인 wall forecast를 속도 비교로 쓰지 않는다.
따라서 많은 SMC whole를 즉시 추가하는 대신 사전 고정 G0/G1 reference variance 비교가 우선이다.

### 추가로 명시한 driver MSE bound

[Original BLP integrated MSE 상한](../theory/ORIGINAL_BLP_INTEGRATED_DRIVER_MSE_BOUND_2026-10-19_KO.md)을
구현했고 독립 oracle6개가 통과했다. original cell-mean κ=1 coupling의 projection error와
continuous-driver increment를 구분해 A_N<=T h^(2H)[2C_inc/(2H+1)+2C_proj]를 유도했다.
기존 conditional-CDF absolute-error 유도와 결합하면 고정 파라미터에서 보수적 strong-type
O(h^H) absolute upper rate를 얻는다. empirical weak rate를 가정한 것이 아니다.

H=.05,N4096,T=1의 A_N 상한은 약1.59350으로 느슨하다. 이를 희귀 확률의 상대 bias가
작다는 증거로 쓰거나 P2를 통과시키지 않는다. 반대로 이 큰 upper bound로 실제 bias가
크다는 lower-bound 주장을 할 수도 없다. float64 평가는 directed-rounding certificate가 아니다.

### G0/G1 실제 고정-grid 결과

각cell/guide131,072개의 fine IID draw, 총32blocks/524,288draw다. fine+coarse physical
1,048,576회이며 interruption reservation32,768회와 replay131,072회는 별도다.
복구의 새1block은18.845초 phase wall로 완료했다. 이 숫자는 전체실험 시간이나 효율 비교가 아니다.

| 셀 | guide | fine μ RSE | coarse μ RSE | fine 최대 단일 기여 | fine 최대 block LOO | paired 차이 상한 |
|---|---|---:|---:|---:|---:|---:|
| canonical | G0 point | 48.286% | 41.731% | 38.985% | 36.613% | 59.409% |
| canonical | G1 mixed-window | 4.371% | 4.029% | 1.751% | 2.074% | 7.946% |
| high-η | G0 point | 29.439% | 27.735% | 23.046% | 16.196% | 32.172% |
| high-η | G1 mixed-window | 3.051% | 3.451% | 1.004% | 0.637% | 6.891% |

G1 fine μ는 약1.50378e-8 /9.63389e-7이다. 동일draw에서 관측한 RSE제곱 비율 G0/G1은
약122.05/93.11이다. **평균도 불확실한 pilot의 normalized variance proxy**이며 확정된
분산감소 배수, 실제 시간 가속, 원래q의 성능 개선으로 쓰지 않는다. G1 paired endpoint는
두셀 통과했지만 μ RSE1.5%는 미달이다. G0/G1끼리의 동등성도 두셀 모두 미확인이다.

사전 고정 safety3/더 나쁜 fine·coarse RSE 규칙의 G1 새draw 예측은 canonical3,338,752/
high-η2,081,280이다. fine+coarse physical 합계10,840,064회다. 실제 precision 보증이나
실행 완료가 아니다. G0에 같은 방식으로 수억draw를 쓰는 것은 현재 우선순위가 아니다.

### 서로 다른 메커니즘의8개 비교

| 셀 | SMC design | IID guide | symmetric 상대차이 | 조정 absolute upper |
|---|---|---|---:|---:|
| canonical | more-temperatures | G0 | 65.730% | 185.212% |
| canonical | more-temperatures | G1 | -5.680% | 31.214% |
| canonical | more-particles | G0 | 70.857% | 189.813% |
| canonical | more-particles | G1 | 0.122% | 34.977% |
| high-η | more-temperatures | G0 | -27.062% | 108.004% |
| high-η | more-temperatures | G1 | 7.768% | 27.437% |
| high-η | more-particles | G0 | -40.301% | 119.502% |
| high-η | more-particles | G1 | -5.858% | 25.953% |

전체8개 adjusted interval은0을 포함하며10% margin을 걸친다. 따라서 disagreement가
검출됐다고도, agreement가 입증됐다고도 하지 않는다. 특히 G1만 정밀하게 만들어도
SMC의6.5–12% RSE 때문에 이 교차 검증은 자동 해결되지 않는다.

### 다음 결정

1. 정확성 검토 및 소프트웨어 회귀를 마친 뒤, G1은 **유망한 reference 개발 후보**로 유지한다.
2. 독립적인 새 fixed-count precision 설계를 만든다. G1 count forecast와 독립 SMC 또는
   다른 메커니즘의 precision을 함께 예산화하고, 같은관측을 재사용한 confirmation으로 부르지 않는다.
3. 같은volatility templates/centres에서 price 방향만 바꾸는 matched ablation을 별도 개발
   연구로 사전 고정한다. 지금의G0/G1 차이를 price window 단독 원인으로 단정하지 않는다.
4. rare contribution/label/path-geometry 진단을 위한 새 기록 schema를 먼저 검증한다.
   현재 moments만으로 미저장 path나 mode를 복원했다고 주장하지 않는다.
5. μ reference·continuum relative bias·originalq bank/M2·전체훈련 반복이 해결되기 전에는
   large training/새operator/차원확장/저널용 속도 비교를 열지 않는다.

구체적인 후속 count·독립 기준·matched ablation·기여 기록과 gate는
[다음 정밀도·메커니즘 계획](../plans/POST_WINDOW_PILOT_PRECISION_AND_MECHANISM_2026-10-19_KO.md)에
정리했다. 거기에 적힌 새 실험은 아직 실행하지 않았다.

### 소프트웨어·증거 검토 범위

- Ruff 전체 정규 CI 범위 통과; Mypy `src main.py train_driftnet.py`의200source 통과.
- 새/변경 experiment4개는 `--follow-imports=silent` 명시 검사 통과.
- imports까지 확대하는 별도 Mypy 시도에서는 역사적4experiment의14개 타입 오류가
  나왔다(r1_diagnostics, r15_reference_design, r15_reference_crosscheck,
  v2_independent_reference). 이번 파일의 오류가 아니며 정규CI 범위 밖이다.
  frozen runtime을 바꾸며 숨기지 않았고, 전체 저장소의 모든 experiment가 type-clean이라고
  주장하지 않는다. 정규 Mypy 통과와 이확대검사 실패를 구분한다.
- 복구·교차비교·BLP bound 국소11tests 통과. 최종전체pytest1,437개 통과(374.31초).
  git diff --check도 통과했다.
- 독립 재산술 검사에서 source archive3개를 검증하고 교차비교8개를 정확히 재계산했다.
  이 검사는 추가physics0회이며 위numericalreplay와 별도다.
- Python3.10.11 로컬 실행이다. 이번 uncommitted tree의 Python3.11/원격CI는 미실행이다.
- pytest 시작 시 requests/urllib3/charset 계열의 dependency warning이 표시됐다.
  동결된 numerical dependency를 이실행 중 임의 업그레이드하지 않았다.
- 검사들은 발견 가능한 오류에 대한 근거이며 오류부재·tail coverage·독창성의 증명은 아니다.

현재 이 문서는 P2 통과/저널 제출 준비 완료 보고가 아니다.
