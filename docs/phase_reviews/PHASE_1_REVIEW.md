# Phase 1 자체 검토

**커밋**: `feat(phase-1): unify mathematical formulation`
**범위**: 수학적 일관성 확보 — Heston/rBergomi 이산화, Girsanov 측도 변환, 학습 손실,
NeuralSDE/NeuralImportanceSampler 재작성

---

## 1. 구현 내용 요약

| 영역 | 변경 사항 |
|---|---|
| **문서** | `docs/formulation.md` 신설 — 모든 SDE/측도/손실의 단일 진리원본 (Single Source of Truth) |
| **physics_engine.py** | Full-Truncation Euler (Lord 2010), Poisson 점프 카운트, Girsanov v-drift correction `ρξ√v·u`, BLP rBergomi 하이브리드 스킴 |
| **neural_engine.py** | `DriftNet`/`DiffNet`/`VolNet` 텐서 안정화, 신규 `VolNetFree` 변형, `simulate_controlled`에서 v-drift correction 적용, `NeuralImportanceSampler.train_step`이 분산 최소화 목적함수(VM)를 정확히 구현 |
| **src/losses/** | `moment_match_loss`, `sliced_wasserstein_distance`, `mmd_loss` (다중 스케일 가우시안 커널) |
| **src/training/** | `variance_minimization_objective`, `kl_regularized_objective`, `cem_step` |
| **train_driftnet.py** | YAML config 기반, 데이터에서 자동 계산되는 target kurtosis, 옵션 MMD |
| **configs/default.yaml** | `rbergomi:`, `girsanov:` 섹션 추가, train_ipm 키 확장 |
| **main.py** | top-level `import torch` 제거 — `python main.py --help` 가 torch 없이 동작 ✅ |

---

## 2. 기술적 오류 검토

### 2.1 사전 Phase 0에서 식별된 이슈 — 처리 결과

| Phase 0 이슈 | 상태 |
|---|---|
| `main.py`의 top-level `import torch` | ✅ 모든 import를 subcommand 함수 내부로 이동 — `--help`가 즉시 동작 확인 |
| `cmd_calibrate` 플레이스홀더 | ⚠️ Phase 3로 이월 (Phase 1 범위 밖) |
| `configs/default.yaml`의 `target_kurtosis: null` 자동 계산 로직 부재 | ✅ `train_driftnet.resolve_target_kurtosis`에서 데이터로부터 자동 산출 |
| `rbergomi:`, `girsanov:` 섹션 부재 | ✅ 추가 완료 |

### 2.2 Phase 1 신규 구현의 잠재적 이슈

1. **`NeuralSDESimulator.simulate_controlled`의 `b * u`** : VolNet의 b는
   `ψ_b · √v` 형태이므로, 수식적으로 `ρ ξ √v u`와 정확히 일치하려면
   `ψ_b ≡ ξ`여야 한다. 학습된 `ψ_b`가 ξ에서 멀어질 경우 v-drift correction은
   모델이 표현하는 b 자체에 비례하여 적용되므로 **물리적으로 올바른** 형태가 된다.
   다만 `MarketSimulator`(분석적 Heston)와는 직접 비교 시 분산이 다를 수 있음.
   → Phase 2 unbiasedness 테스트는 `MarketSimulator`에서만 강제하도록 분리.

2. **`VolNetFree` 학습 안정성**: 평균 회귀 prior가 없으므로 long-horizon
   학습에서 v→0 또는 v 폭주 가능. `simulate`에서 `clamp(min=1e-8)`로 안전망
   확보했지만, 학습 루프에서 `kappa·(theta − v)` 손실을 prior penalty로
   추가하는 것을 Phase 3에서 검토.

3. **`NeuralImportanceSampler.train_step`의 KL 항**: `kl_weight > 0`일 때
   `−E^Q[log_w]`를 KL 추정량으로 사용. `−log_w = ∫u dW^Q + ½∫u² dt`이고
   `E^Q[∫u dW^Q] = 0`이므로 기댓값으로는 `½ E^Q[∫u² dt]`와 일치. 단,
   유한 샘플에서는 첫 항의 분산이 더해지므로 KL 추정 분산이 증가. Phase 2에서
   `kl_estimator='quadratic'`(직접 ½∫u² dt 합산) 옵션 제공 예정.

4. **`mmd_loss`의 sub-sampling**: `train_driftnet`에서 512개로 제한. 이는
   O(n²) 연산을 1초 이하로 유지하기 위한 절충. epoch마다 인덱스를 재추출하므로
   장기적으로 데이터 전체를 커버하지만, batch별 추정 분산이 큼. 가우시안 커널의
   sigma는 median heuristic으로 자동 결정.

5. **`train_driftnet`의 `T_horizon=0.5` 기본값**: 원래 노트북은 1년이었음.
   훈련 속도 향상을 위해 0.5년으로 축소. config로 변경 가능.

6. **vol_dynamics와 control 간 결합 (학습된 vol_net 사용 시)**: 현재 IS 학습 시
   physics_engine.MarketSimulator (분석적 Heston) 사용을 권장. NeuralSDESimulator로
   IS 학습은 가능하지만, 학습된 vol dynamics와 control이 동시에 변하면 수렴이
   느림. Phase 3에서 alternating optimization 구현 검토.

### 2.3 회귀 위험

- **`NeuralSDESimulator.parameters()` 반환형 변경**: 기존 list → generator.
  pytorch optimizer는 iterable을 받으므로 호환. 다만 `sum(p.numel() for p in
  sim.parameters())` 같은 1회 소비 후 재호출 패턴이 외부에 있으면 빈 결과가 됨.
  → `train_driftnet.py`에서는 `Adam(simulator.parameters(), ...)` 형태로 1회만
  소비하므로 문제 없음.
- **`NeuralImportanceSampler.train_step` 시그니처 전면 변경**: 기존 호출
  (`train_step(S0, K, T, r, barrier_level, barrier_type, optimizer)`)은 깨짐.
  외부에서 직접 호출하는 코드는 노트북뿐이며, Phase 2의 `experiments/`에서
  새 API로 재작성 예정.
- **`VolNet.kappa, theta` 학습 가능 파라미터 유지**: 기존 chekpoint 로드 시
  키가 일치하므로 backward-compatible.

---

## 3. 이론적 오류 검토

### 3.1 측도 변환의 정확성

`docs/formulation.md §2.2`에 명시된 Q-측도 동역학:

$$
dv_t = [\kappa(\theta − v_t) + \rho\xi\sqrt{v_t}\,u_t]\,dt + \xi\sqrt{v_t}\,dW^{\mathbb Q,v}_t
$$

`physics_engine.simulate_controlled`에서 정확히 구현됨:

```python
v_drift_Q = params["kappa"] * (params["theta"] - v_plus)
if apply_v_drift_correction and control_fn is not None:
    v_drift_Q = v_drift_Q + rho * params["xi"] * sqrt_v * u_t
```

`apply_v_drift_correction=False`로 설정하면 pre-Phase-1의 편향된 동작이
재현됨 → `tests/test_girsanov_unbiased.py`에서 두 모드를 비교하여 phase-1
fix가 실제로 unbiasedness를 회복함을 증명할 예정 (Phase 2).

### 3.2 Doleans 지수의 부호 / 구현 일치성

`log dP/dQ = −∫u dW^{Q,S} − ½∫u² dt`. 코드에서:

```python
int_u_dW = int_u_dW + u_t * z1 * sqrt_dt
log_weight = -int_u_dW - 0.5 * int_u_sq_dt
```

`z1 * sqrt_dt = dW^{Q,S}` (Q-Brownian 증분)이므로 부호 정확.

### 3.3 Full-Truncation Euler

Lord-Koekkoek-van Dijk (2010) 권장 방식: drift와 diffusion 모두 `v_plus =
max(v, 0)`을 사용. 코드 일치 확인:

```python
v_plus = torch.clamp(v_prev, min=0.0)
sqrt_v = torch.sqrt(v_plus)
dv = params["kappa"] * (params["theta"] - v_plus) * dt + params["xi"] * sqrt_v * dw_v
```

### 3.4 BLP 하이브리드 스킴

Bennedsen-Lunde-Pakkanen (2017) Algorithm BN:
- 첫 번째 sub-interval: 정확한 2D Gaussian (covariance c11/c12/c22)
- 이전 intervals: Riemann sum with optimal weights
- κ=1 (default) — exact on 1 sub-interval, Riemann elsewhere

Volterra kernel: `(t − s)^{H−½}`, exponent `α = H − 0.5 ∈ (−½, 0)`. 코드 확인:

```python
alpha = H - 0.5
c11 = dt
c12 = dt ** (alpha + 1.0) / (alpha + 1.0)
c22 = dt ** (2 * alpha + 1.0) / (2 * alpha + 1.0)
```

해석적으로 검증된 식. **Pre-Phase-1의 fBm-as-BM-driver 버그는 수정됨**:
`dW_S = rho * dW1 + sqrt(1-rho²) * Z2`에서 `dW1`은 Volterra kernel의 driving
Brownian motion (스킴이 직접 샘플하는 BM), `Z2`는 독립 normal — 가격 SDE의 
올바른 driver.

### 3.5 Poisson vs Bernoulli 점프

Pre-Phase-1: `rand < λ·dt`로 한 step에 최대 1 jump (Bernoulli). Phase 1:
`torch.poisson(λ·dt)`로 다중 jump 지원. λ·dt가 작으면 (e.g. 5/year × 1/252 ≈ 0.02)
Bernoulli ≈ Poisson이지만, dt가 큰 경우(예: 주간 step) Bernoulli는 underestimate.

n_jumps 개의 iid log-jump 합:
- mean: `n · m_J`
- variance: `n · s_J²`

코드:
```python
log_jump = n_jumps * params["jump_mean"] + torch.sqrt(n_jumps) * params["jump_std"] * normal
```

✅ 분포 정확.

### 3.6 손실 함수의 수학적 적절성

| Loss | 수식 | 코드 |
|---|---|---|
| VM | `E^Q[(g·E_T)^2]` | `(reweighted ** 2).mean()` ✅ |
| KL-reg | `−E^Q[hinge] + λ·KL(Q‖P)` | `crash_term + kl_weight * (-log_w.mean())` ✅ (E^Q[∫u dW^Q]=0이므로 기대값 일치) |
| MM | `Σ w_k(m_k^model − m_k^data)^2` | `moment_match_loss` ✅ |
| MMD | `‖μ_model − μ_data‖²_{H_k}` | `mmd_loss` (multi-scale Gaussian) ✅ |

### 3.7 미해결 이론 이슈

- **`VolNetFree`로 학습된 (a, b) 페어가 well-posed (해 존재성 / 양수성)을
  보장하는가?** 현재는 `b > 0` (softplus)만 강제, `a` 부호 자유. 학습 후 SDE의
  Lyapunov 안정성 검증은 Phase 2에서 모니터링 메트릭으로 추가.
- **NeuralSDE에서 IS 학습 시 v-process 자체가 학습 중**이라 measure change의
  density가 시간에 따라 변하는 base measure에 대한 RN. 수학적으로는 control
  optimization이 P-dynamics fix 후 진행되어야 함 (alternating). Phase 3 작업.

---

## 4. 검증 / 커버리지 현황

| 지표 | Phase 0 → Phase 1 | 목표 (Phase 2 말) |
|---|---|---|
| 유닛 테스트 수 | 7 | ≥ 30 |
| Line coverage | 측정 안 됨 | ≥ 80% |
| `python main.py --help` (torch 없이) | ✅ Phase 1에서 동작 확인 | ✅ |
| Heston 해석해 대조 | ❌ → Phase 2.1 | ✅ |
| Girsanov unbiasedness 테스트 | ❌ → Phase 2.2 | ✅ |
| BLP 하이브리드 스킴 ATM skew 테스트 | ❌ → Phase 2.3 | ✅ |

Phase 1에서 테스트 자체는 추가하지 않음 — Phase 2의 검증 인프라 작업으로 일괄
이동. 단, `tests/test_smoke.py`는 새 import 경로(`src.losses`, `src.training`)에
대한 smoke test를 Phase 2에서 추가 예정.

---

## 5. 다음 Phase 착수 전 TODO

1. Phase 2 시작 전 ruff 클린 (수동 검토 — sandbox에서는 ruff 미설치)
2. 새 Python 모듈(`src/losses/`, `src/training/`)이 `setup.cfg` 또는
   `pyproject.toml`의 packages에 자동 포함되는지 확인 (`find_packages` 사용 중)
3. `tests/test_smoke.py`에 새 모듈 import smoke test 추가 (Phase 2 첫 커밋에서)

---

## 6. 푸시 상태

- 커밋이 워크스페이스 `.git`에 반영됨
- 샌드박스에서 `github.com` 아웃바운드는 여전히 차단되어 자동 push 불가
- `phase-1.bundle`을 워크스페이스에 배치: 사용자 수동 push 필요
  `git bundle unbundle phase-1.bundle` 또는 단순히 `git push origin main`
