# G2 v3 Confirmatory Review

Date: 2026-07-13<br>
Protocol: `g2-heston-affine-confirmatory-v3-sealed`<br>
Protocol SHA-256: `f9efcc5fcc43963dc3bfd8721df40be4507c85550a7a484f1034757a4cfa9b63`<br>
Result: `results/g2_v3_confirmatory_1e-4_2026-07-13.json`

## 결론

Heston \(10^{-4}\) terminal-left-tail에서 compact affine feedback은 CEM
constant control과 **통계적·계산적으로 거의 동급**이다. 현재 실행의
형식적 gate 값은 1.016으로 통과했지만, 동일 모델 재실행에서 wall-clock
변동만으로 0.982가 나왔고 audit seed별 paired VRF도 1.106과 0.936으로
방향이 갈렸다. 따라서 “neural/feedback이 CEM보다 빠르다”는 주장은 아직
허용하지 않는다. v3 evaluation seed 11101–11120은 접근하지 않았으며 계속
봉인한다.

## 이번 단계에서 수정한 기술적 문제

1. v1 smoke에서 사용한 evaluation seed는 publication evaluation에서 폐기했다.
2. v2에서 validation을 selection 3개와 독립 audit 2개로 분리했다.
3. 매 epoch 동일 root seed를 재설정해 같은 Brownian batch를 반복하던 문제를
   발견했다. v3는 root별로 서로 다른 deterministic substream을 생성한다.
4. affine 입력의 \(S/S_0\)와 \(S/K\)가 선형적으로 중복됨을 확인하고
   3-feature controller로 축소했다.
5. affine inference를 frozen 3계수 함수로 변환해 `nn.Linear`와 feature stack
   overhead를 제거했다.
6. bias-z를 두 종류로 분리했다. `bias_z_score`는 seed 간 empirical dispersion,
   `reported_bias_z_score`는 각 run의 estimator standard error를 합성한다.
   두 seed audit gate에는 후자를 사용한다.

## v3 데이터 분리

| 용도 | Seed roots | 사용 여부 |
|---|---|---|
| Train | 6101–6105 | 사용 |
| Checkpoint selection | 7101–7104 | 사용 |
| Independent audit | 7105–7106 | 한 번 사용 |
| Final evaluation | 11101–11120 | 미사용·봉인 |

각 epoch의 실제 train seed는 root에 stream index를 더해 생성되므로 50 epoch
동안 동일 Brownian batch가 반복되지 않는다.

## Audit 결과

기준 확률은 Heston characteristic-function CDF에서 역산한 \(10^{-4}\)이다.
각 audit seed는 50,000경로를 사용했다.

| 지표 | Compact affine | CEM constant |
|---|---:|---:|
| Mean estimate | 9.6769e-5 | 9.8235e-5 |
| Relative bias | -3.23% | -1.76% |
| Reported bias-z | -2.42 | -1.29 |
| CI coverage | 1.00 | 1.00 |
| Single-path variance | 1.7891e-7 | 1.8694e-7 |
| Contribution ESS, seed 7105 | 2632 | 2422 |
| Contribution ESS, seed 7106 | 2357 | 2487 |
| Top-1% contribution share | 17.5–17.8% | comparable |

Affine은 분산을 약 4.3% 낮췄다. 추론비는 CEM보다 약 6% 높아서 aggregate
work-normalized VRF는 약 1 부근이다. seed 7105에서는 1.106, seed 7106에서는
0.936이므로 우월성 방향이 안정적이지 않다.

## 이론 검토

### Measure change

Spot Brownian을 \(u\)로 이동하면 correlated variance drift에
\(\rho\xi\sqrt v\,u\)가 반드시 추가된다. 현재 natural/controlled simulator,
likelihood, Gaussian reconstruction test가 이 convention과 일치한다.

### Hard-indicator gradient

hard event를 단순 pathwise derivative로 학습하지 않는다. 현재 second-moment
objective는

$$
\nabla_\phi J=-E_{Q_\phi}[1_A L_\phi^2\nabla_\phi\log q_\phi]
$$

의 score-function gradient를 사용한다. 동일 trajectory의 Q-Brownian
increment와 detached state를 사용하며 Gaussian closed-form gradient test를
통과한다.

### Selection validity

v3 controller는 train 및 selection seed로만 선택됐고 audit은 선택 이후 한 번
사용됐다. audit 결과를 보고 v3 parameter를 다시 조정하면 leakage가 되므로
v3에서 추가 튜닝하지 않는다.

## 알려진 한계와 남은 위험

- audit seed가 2개뿐이어서 coverage는 0, 0.5, 1 중 하나만 가능하다.
- CPU wall-clock 차이가 1–2% 수준이면 OS scheduling과 allocator noise보다 작다.
- \(dt=1/256\) discretization과 continuous-time Heston reference 차이가 남는다.
- weight normalization z-score는 강한 tilt에서 높은 분산을 갖는다. event
  contribution ESS와 함께 해석해야 한다.
- 현재 결과는 \(10^{-4}\) 하나뿐이며 \(10^{-5},10^{-6}\) confirmatory audit은
  실행하지 않았다.

## 연구 판단

현재 Heston terminal event에서는 CEM constant가 매우 강한 baseline이다.
feedback controller의 추가 이점은 작다. 따라서 논문의 차별화는 Heston에서
억지로 superiority를 만드는 방향보다, constant control이 구조적으로 부족한
rough-volatility memory 및 multi-task amortization에서 검증하는 편이 타당하다.
Heston 결과는 correctness/reference benchmark와 negative result로 유지한다.
