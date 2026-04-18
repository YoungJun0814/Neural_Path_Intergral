# DriftNet / Neural Path Integral — 개선 로드맵

> 목적: 이론적 정합성 확보 → 기술적 결함 제거 → 실무/연구 가치 제고
>
> 전체 기간 권장: 3–4 개월 (1인 풀타임 기준) / 6 개월 (파트타임)

---

## Phase 0. 긴급 수정 (1주차 — "브로큰" 상태 탈출)

이 단계의 목표는 **"다른 사람이 `git clone && python main.py` 했을 때 에러 없이 돌아가는"** 기본선 확보입니다.

### 0.1 실행 불가 이슈 제거
- `main.py`: 잘못된 import 수정
  - `from src.physics_engine import HestonSimulator` → `from src.physics_engine import MarketSimulator`
  - `from src.ai_calibrator import NeuralSDE` → `from src.ai_calibrator import NeuralCalibrator`
  - 실제 실행 가능한 3가지 모드(`simulate`, `calibrate`, `train_ipm`) 구현. 각 모드는 `src/cli/` 하위 모듈로 분리.
- `test_kurtosis_loss.py` line 13 Windows 경로 하드코딩 제거
  - `sys.path.insert(...)` 전체 삭제하고 `python -m pytest` 호환되게 `conftest.py` 사용
- `requirements.txt` 전체 재작성
  - 현재 사용 중인 패키지: `torch>=2.0`, `numpy>=1.23`, `scipy>=1.10`, `pandas>=2.0`, `matplotlib>=3.7`, `xgboost>=1.7`, `scikit-learn>=1.3`, `captum>=0.7` (XAI), `pyarrow`, `tqdm`
  - 개발 의존성은 `requirements-dev.txt`로 분리: `pytest`, `pytest-cov`, `ruff`, `mypy`, `jupyter`, `seaborn`

### 0.2 재현성 최소선 확립
- 모든 학습 스크립트 진입점에서 `utils.set_seed(seed)` 호출
- Hydra 또는 OmegaConf 도입 → `configs/` 하위에 모델/실험 설정을 YAML로 분리
- `config.yaml`에 `seed`, `device`, `dtype`, `git_commit_hash` 자동 로깅

### 0.3 저장소 위생
- `.gitignore`에 `checkpoints/*.pth`, `data/raw/*`, `__pycache__`, `*.ipynb_checkpoints`
- 대형 checkpoint는 Git LFS 또는 외부(예: HuggingFace Hub) 이동
- CI 설정: GitHub Actions에서 `pytest`, `ruff check`, `mypy src/` 자동 실행

---

## Phase 1. 수학적 정합성 확립 (2–4주차 — 가장 중요한 단계)

### 1.1 SDE 모형의 통일과 문서화
**문제**: README 수식과 `simulate_controlled` 구현이 다르고, Girsanov 해석도 명확하지 않음.

**할 일**:
1. `docs/formulation.md`를 작성해 **하나의 정확한 SDE 가족**을 명시.
   ```
   P-measure:
     dS_t = μ(S,v,t) S_t dt + σ(S,v,t) S_t dW_t^{P,S}
     dv_t = κ(θ − v_t) dt + ξ √v_t dW_t^{P,v}
     Corr(dW^{P,S}, dW^{P,v}) = ρ dt

   Controlled Q-measure (via u):
     dW_t^{P,S} = dW_t^{Q,S} + u_t dt           ← noise shift
     dS_t = (μ + σ u) S_t dt + σ S_t dW_t^{Q,S}
     dv_t = κ(θ − v_t) dt + ξ √v_t (ρ dW_t^{Q,S} + √(1−ρ²) dW_t^{Q,v⊥})
          = [κ(θ − v_t) + ρ ξ √v_t u] dt + ξ √v_t dW_t^{Q,v}

   Radon–Nikodym:
     dP/dQ = exp( −∫₀ᵀ u_t dW_t^{Q,S} − ½ ∫₀ᵀ u_t² dt )
   ```
2. README, 논문 draft, 코드 주석을 모두 이 하나의 수식과 일치시킴.
3. 단위 테스트: 제어된 시뮬레이션에서 얻은 `E^Q[f(S)·dP/dQ]`와 `E^P[f(S)]`가 95% CI 내 일치하는지 자동 검증(Phase 2.3 참조).

### 1.2 Girsanov의 v-process 교정 (핵심 버그)
**파일**: `src/physics_engine.py:MarketSimulator.simulate_controlled`, `src/neural_engine.py:NeuralSDESimulator.simulate_controlled`

**변경점**:
```python
# 현행 (rho ≠ 0일 때 biased):
dv = kappa*(theta - v) dt + xi*sqrt(v) * (rho*z1 + sqrt(1-rho**2)*z2) * sqrt_dt

# 수정 (explicit v drift correction 필요 없음: dW^{Q,v}로 두고 P로 환산 시 rho·u·sqrt(v)*ξ 항 반영):
z1_Q, z2_Q = torch.randn(...), torch.randn(...)
dW_S_Q = z1_Q * sqrt_dt
dW_v_Q = (rho * z1_Q + sqrt(1-rho**2) * z2_Q) * sqrt_dt
# v는 Q-dynamics 그대로 전개 (Q-measure 하에서 자기자신의 표현이므로 OK)
dv = kappa*(theta - v) dt + xi*sqrt(v) * dW_v_Q
# S에만 drift shift 적용
dS = (mu + sigma*u) * S * dt + sigma*S*dW_S_Q
# Girsanov weight는 dW^{Q,S}에 대해서만
log_weight -= u * z1_Q * sqrt_dt + 0.5 * u**2 * dt
```
**주의**: v가 Q-measure 하에서 같은 SDE 형태를 띠는지 여부는 control u가 v의 drift에 어떤 영향을 주느냐에 따라 결정됨. u가 S의 dW_S만 shift하는 측도 변환을 사용한다는 사실을 명시적으로 문서화.

### 1.3 Heston 이산화 스킴을 표준으로 교체
**파일**: `src/physics_engine.py`

**Option A (권장, 빠름)**: **Full Truncation Euler**
```python
v_plus = torch.clamp(v_prev, min=0.0)   # max(v,0) — 모든 곳에 동일 적용
dv = kappa*(theta - v_plus)*dt + xi*torch.sqrt(v_plus)*sqrt_dt*dw_v
v_next = v_prev + dv  # 음수 허용하지만 사용 시 v_plus
```

**Option B (정확, 느림)**: **Andersen QE (Quadratic-Exponential)**
- Andersen (2008) "Simple and efficient simulation of the Heston model"의 Algorithm 6
- `src/physics_engine.py`에 `simulate_heston_qe` 별도 메서드로 추가
- Feller 조건 위반 시(2κθ < ξ²)에도 수치 안정성 확보

**테스트**: Heston 해석해(characteristic function 기반 call price)와 `T=0.5, K=100` 콜옵션 가격이 10,000 paths에서 0.2% 내 일치하는지 검증.

### 1.4 rBergomi 하이브리드 스킴 구현 (치명 버그 수정)
**파일**: `src/physics_engine.py:RBergomiSimulator`

현행: `dW_S = ρ·dW_H + √(1-ρ²)·Z·√dt` — fBm 증분을 표준 Brownian으로 취급 (**잘못됨**).

**수정**: Bayer–Friz–Gatheral (2016)의 hybrid scheme 또는 Bennedsen–Lunde–Pakkanen (2017) Romano–Touzi 분해 사용.
```
1. Volterra process Y_t = ∫₀ᵗ (t−s)^{H−1/2} dW_s^{(1)} 를 hybrid scheme으로 근사
2. v_t = ξ · exp(η Y_t − ½η²·t^{2H})
3. 가격 Brownian은 dW^{(1)}, dW^{(2)} 두 개의 independent Brownian으로 구성:
     dW_t^S = ρ · dW_t^{(1)} + √(1-ρ²) · dW_t^{(2)}
   즉 fBm 생성에 쓴 underlying Brownian과 correlate.
```
참고 구현: McCrickerd & Pakkanen의 `rbergomi` Python 패키지.

### 1.5 Jump process를 정확히 Poisson으로 변경
현행 Bernoulli(λ·dt)를 Poisson(λ·dt)로 변경, multiple jumps per step 지원. `torch.distributions.Poisson` 사용.

### 1.6 VolNet의 과도한 구조적 제약 완화
`b = diff_net(S,v,t) · √v` → 두 버전 제공
- `VolNetCIR` (현재 디자인, Heston prior 강함)
- `VolNetFree` (Softplus만, `b`가 v와 무관하게 자유롭게 학습)
- ablation 실험에서 두 버전 성능 비교 보고

### 1.7 Kurtosis target 정정
- SPY daily raw 4th-moment kurtosis (`scipy.stats.kurtosis(fisher=False)`)를 실제 데이터에서 계산: 보통 15–30.
- `target_kurt`를 하드코딩 값 6.0 대신 **data-driven**으로 계산: `target = scipy_kurtosis(real_returns, fisher=False)`
- 더 나은 대안: moment matching 대신 **Maximum Mean Discrepancy (MMD)** 또는 **Wasserstein-1 distance** 손실 사용.
  - `src/losses/distribution_match.py` 신설
  - MMD with RBF kernel, Wasserstein via `torch.cdist` + POT 라이브러리
- Aggregational Gaussianity 편향 방지: 경로별 kurtosis를 계산 후 경로 간 평균 사용.

### 1.8 제어 학습 loss를 이론과 일치시킴
`notebooks/03_AI_Crash_Generator.ipynb`의 hinge loss는 buffer/heuristic. **이론적 정당성**을 위해:

**목적함수 A: 분산 최소화 IS (option pricing에 적합)**
$$\mathcal{L} = \mathbb{E}^Q\!\left[(g(S_T) \cdot L)^2\right] - \mathbb{E}^Q[g(S_T)\cdot L]^2$$

**목적함수 B: KL-regularized crash generation (stress testing에 적합)**
$$\mathcal{L} = -\mathbb{E}^Q[\log \mathbf{1}_{S_T < K}] + \lambda \cdot \mathrm{KL}(Q \| P)$$
여기서 $\mathrm{KL}(Q\|P) = \mathbb{E}^Q[\log dQ/dP] = \mathbb{E}^Q[\tfrac{1}{2}\int u^2 dt]$.

**목적함수 C: Cross-Entropy Method** (Rubinstein–Kroese)
- 희귀 이벤트 생성의 전통적 기법. 벤치마크로 반드시 포함.

세 목적함수 모두 `src/training/objectives.py`에 구현하고, 한 스크립트에서 `--objective A|B|C|hinge`로 전환 가능하게.

---

## Phase 2. 검증 인프라 구축 (4–6주차)

### 2.1 단위 테스트 스위트
`tests/` 디렉터리 신설, pytest 기반.

| 테스트 파일 | 검증 내용 |
|---|---|
| `test_heston_analytic.py` | Heston call price가 characteristic function 해석해와 일치 (차이 <0.5%) |
| `test_black_scholes.py` | σ → 0 극한에서 BS 해석해 수렴 |
| `test_girsanov_unbiased.py` | `E^Q[f·L] = E^P[f]` (bootstrap 95% CI overlap) |
| `test_fbm_covariance.py` | 생성된 fBm sample의 경험적 공분산이 이론과 일치 |
| `test_rbergomi_atm_skew.py` | ATM implied vol skew가 τ^{H-1/2}에 비례 (Alos–Leon–Vives) |
| `test_jumps_poisson.py` | 점프 개수가 Poisson(λT) 분포 따름 (KS test) |
| `test_simulator_gradient.py` | differentiable simulator에서 `u`에 대한 gradient 수치차분과 일치 |
| `test_config_roundtrip.py` | config YAML → dataclass → YAML이 동일 |

목표: `pytest --cov=src`에서 **coverage ≥ 80%**.

### 2.2 Unbiasedness 검증 프로토콜 (연구 핵심)
`experiments/unbiasedness_check.py`:
1. Barrier put option 가격을 (a) brute-force MC 10⁸ paths와 (b) controlled IS 10⁶ paths로 계산.
2. 두 추정값이 95% CI 내 겹치는지 확인.
3. **효율성 지표를 재정의**하여 보고:
   - Effective Sample Size: `ESS = (Σw)² / Σw²`
   - Variance Reduction Factor: `Var_MC / Var_IS` (동일 샘플 수)
   - Work-Normalized Variance: `Var × compute_time`
   - RMSE-to-fixed-accuracy 도달에 필요한 샘플 수
4. 결과를 LaTeX 표로 자동 출력.

### 2.3 Backtest 프레임워크
`src/evaluation/backtest.py`:
- SPY 2015–2023 rolling window.
- 매 월 t에 대해 (T_train, T_test) 분리, t에서 학습한 모델로 t+1 month VaR/ES 예측.
- **Kupiec POF test**, **Christoffersen independence test**, **conditional coverage** 통계 리포트.
- Basel III의 backtesting zone (green/yellow/red) 분류.

### 2.4 벤치마크 구축
단일 스크립트 `experiments/benchmark_rare_event.py`로 한 번에 비교:
| 방법 | 구현 |
|---|---|
| Plain MC | baseline |
| Stratified MC | `scipy` |
| Antithetic Variates | 직접 구현 |
| Control Variates (Geometric Asian for option bench) | 직접 구현 |
| Cross-Entropy Method | 직접 구현 |
| Adaptive Multilevel Splitting (AMS) | 직접 구현 또는 `feasst` |
| **DriftNet IS (본 방법)** | 본 저장소 |

모든 방법에 대해 동일한 target (e.g. `P(S_T < 80)`)을 RMSE 10⁻⁴로 맞추는 데 필요한 compute budget을 측정.

### 2.5 재현 가능한 실험 관리
- MLflow 또는 Weights & Biases 통합
- `experiments/{exp_name}/` 하위에 `config.yaml`, `metrics.json`, `checkpoint.pth`, `git_hash.txt`
- Docker 이미지 제공 (`Dockerfile` 추가, `nvidia/cuda:11.8`-base)

---

## Phase 3. 연구 가치 제고 (6–10주차)

### 3.1 이론적 기여 후보 (택 1–2)
1. **Finite-sample bound**: 학습된 control $\hat u$의 분산이 최적 control $u^*$ 분산 대비 얼마나 가까운지 bound 증명 (NN 표현력 + Rademacher complexity 활용).
2. **Convergence guarantee**: 학습 loss → 0이면 IS estimator의 variance → 0임을 보이는 부드러운 convex 문제로 재구성.
3. **Rough volatility + control**: rBergomi에서 Volterra 구조가 control policy 설계에 어떻게 영향을 미치는지 분석 (Bayer–Friz–Gulisashvili 계열 large deviation 활용).

결과물: 10–15 페이지 preprint (arXiv).

### 3.2 방법론 확장
- **Multi-asset**: 단일 자산 → 포트폴리오(S&P 500 50개 종목) correlated SDE 확장. `src/models/multi_asset_sde.py`.
- **Regime switching**: Hidden Markov Model로 regime label 추가, control이 regime-aware.
- **Jump-clustered dynamics**: Hawkes jump process 도입.
- **Transformer-based control**: 현재 MLP 기반 DriftNet을 sequential Transformer로 업그레이드하여 path-dependent control 실험.

### 3.3 XAI 엄밀화
현행 Integrated Gradients는 baseline 선택이 애매.
- **Path-specific attribution**: 각 crash path마다 어느 시점의 v_t, S_t가 결정적이었는지 분석.
- **Counterfactual**: "이 path에서 v_t를 10% 낮추면 crash 여부가 바뀌는가?"
- **SHAP + kernelSHAP**으로 crosscheck, baseline은 unconditional v, S 평균.
- **Shapley flow**: 시계열 입력에서의 temporal attribution.

### 3.4 벤치마크 데이터셋 공개
- SPY, VIX, 10년 국채 daily returns 2000–2024
- 합성 Heston / Bates / rBergomi 데이터
- Public release: Zenodo DOI 발급.

### 3.5 논문 draft 작성
| 챕터 | 내용 |
|---|---|
| 1 Introduction | rare event simulation 필요성, contributions |
| 2 Related Work | Neural SDE (Kidger 2021), IS (Glasserman), rough vol (Bayer), 이전 neural IS (Müller 2019) |
| 3 Methodology | Phase 1.1의 통일된 formulation, 학습 objective, 이론 |
| 4 Experiments | 벤치마크, unbiasedness, backtest |
| 5 Case Study | 2008, 2020, 2022 crash backtest |
| 6 Discussion | 한계, 측도 변환 해석, future work |

Target venue (순): *Quantitative Finance*, *Journal of Computational Finance*, *SIAM Journal on Financial Mathematics*, NeurIPS workshop on ML for Finance.

---

## Phase 4. 실무 배포 (8–12주차)

### 4.1 패키지화
- `setup.py` / `pyproject.toml` 작성, PyPI에 `driftnet-npi` 등록
- CLI: `driftnet calibrate --config configs/heston_spy.yaml`, `driftnet stress --portfolio my_pf.csv`
- 타입 힌트 전체 적용, mypy strict 통과

### 4.2 실전 사용 시나리오 3개 템플릿
`examples/` 디렉터리에:
1. **Counterparty Credit Risk (PFE)**: barrier option + exposure calculation
2. **Stress Testing (Basel III IMM)**: 1일, 10일, 1년 horizon 포트폴리오 loss distribution
3. **Derivatives Hedging PnL simulation**: Greeks 추정 + hedging error 분포

### 4.3 성능 최적화
- 큰 num_paths 처리: list accumulation → chunked checkpointing (`torch.utils.checkpoint`)으로 메모리 60% 감소
- `torch.compile` / TorchScript 적용
- CUDA 스트리밍으로 CPU↔GPU 전송 최소화
- Mixed precision (`torch.cuda.amp`)

### 4.4 API 서버 (선택)
- FastAPI 기반 `/price`, `/stress_test`, `/calibrate` 엔드포인트
- Docker image로 배포 가능

### 4.5 리스크 관리 가이드 문서
`docs/risk_usage_guide.md`:
- P-measure vs Q-measure **반드시 구분**, 내부 risk measurement는 P, 가격결정은 Q
- Controlled crash scenarios를 P-measure VaR로 **직접 쓰면 과대 추정** — Girsanov reweighting 방법 step-by-step
- 규제(CCAR, FRTB-IMA) 관점 제한사항 명시

---

## Phase 5. 문서화 & 커뮤니케이션 (10–12주차)

### 5.1 정확한 README
- 현행 "126× Speedup"을 "ESS ratio: 42×, Variance Reduction Factor: 126× (95% CI 110–145)"처럼 측정된 지표로 교체
- 수식을 Phase 1.1의 통일된 formulation으로
- 각 claim에 해당 notebook/experiment 링크

### 5.2 notebook 정리
- `notebooks/legacy/`, `notebooks/exotics/`는 "tutorial"로 재구성
- 핵심 4개(01_Data → 04_XAI)는 **처음부터 끝까지 Colab에서 1-클릭 실행** 보장
- 각 셀에 실행 시간 주석, GPU 요구사항 명시

### 5.3 블로그/프레젠테이션
- 3–5편 블로그(Medium, 개인 블로그): "Neural SDE 기본부터", "Girsanov의 실제 의미", "Crash 생성기의 unbiasedness"
- Jupyter Book 또는 mkdocs-material로 `docs/` 정적 사이트 빌드 → GitHub Pages 배포

### 5.4 OSS 커뮤니티
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- Good first issue 라벨링
- 한국 quant 커뮤니티(`quant-finance-kr`) 및 해외 `/r/quantfinance` 공유

---

## 우선순위 매트릭스 (임팩트 × 난이도)

| 작업 | 임팩트 | 난이도 | 우선 |
|---|---|---|---|
| 0.1 main.py 고치기 | 중 | 하 | **최우선** |
| 1.1 SDE 통일 문서 | 상 | 하 | **최우선** |
| 1.2 Girsanov v 교정 | 상 | 중 | 높음 |
| 1.4 rBergomi hybrid | 상 | 상 | 높음 |
| 2.1 단위 테스트 80% | 상 | 중 | 높음 |
| 2.2 Unbiasedness 검증 | 상 | 중 | **필수** |
| 2.4 벤치마크 | 상 | 상 | 높음 |
| 3.1 이론적 bound | 상 | 상 | 도전 |
| 3.2 Multi-asset | 중 | 상 | 선택 |
| 4.1 패키지화 | 중 | 중 | 실무용 |
| 4.5 리스크 가이드 | 상 | 하 | **필수** |

---

## 마일스톤

| 마일스톤 | 목표 시점 | 성공 기준 |
|---|---|---|
| M1. Green CI | 1주차 말 | main.py 실행, pytest 통과, CI green |
| M2. Math-consistent v1 | 4주차 말 | Phase 1.1–1.7 완료, README/코드 일치 |
| M3. Unbiasedness-verified | 6주차 말 | IS 추정과 brute-force MC가 95% CI 내 일치 공식 보고 |
| M4. Benchmark report | 10주차 말 | 7개 방법 비교 표·그림 publish |
| M5. Preprint v1 | 12주차 말 | arXiv 업로드, 코드 Zenodo DOI |
| M6. Journal submission | 16주차 | Target venue 제출 |

---

## 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| rBergomi hybrid 스킴 구현 난이도 | 일정 지연 | 참조 구현(`mccrickerd/rough_bergomi`)을 포팅 |
| Unbiasedness 검증에서 bias 발견 | 핵심 주장 흔들림 | Phase 1.2 Girsanov 교정이 효과 있음을 보이는 기회로 전환 |
| GPU 리소스 부족 | 대규모 실험 제약 | Lambda Labs, RunPod, Colab Pro+, 대학 클러스터 활용 |
| 선행연구 중복 | novelty 약함 | Phase 3.1 이론 기여 또는 Phase 3.2 멀티자산 확장으로 차별화 |

---

## 부록 A. 즉시 적용 가능한 구체 코드 패치 (Phase 0 일부)

### A.1 `main.py` 재작성 스켈레톤
```python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml

from src.physics_engine import MarketSimulator
from src.ai_calibrator import NeuralCalibrator
from src.neural_engine import NeuralSDESimulator

def cmd_simulate(cfg):
    sim = MarketSimulator(**cfg['heston'], device=cfg['device'])
    S, v = sim.simulate(**cfg['simulate'])
    print(f"S_T stats: mean={S[:,-1].mean():.4f}, std={S[:,-1].std():.4f}")

def cmd_calibrate(cfg):  ...   # TODO

def cmd_train_ipm(cfg):  ...   # TODO

CMDS = {'simulate': cmd_simulate, 'calibrate': cmd_calibrate, 'train_ipm': cmd_train_ipm}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=CMDS.keys())
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    CMDS[args.mode](cfg)

if __name__ == "__main__":
    main()
```

### A.2 `tests/test_girsanov_unbiased.py`
```python
import torch, numpy as np, pytest
from src.physics_engine import MarketSimulator

def test_girsanov_unbiased():
    torch.manual_seed(0)
    sim = MarketSimulator(mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, device='cpu')
    def const_control(t, S, v, A=None):
        return -0.3 * torch.ones_like(S)
    # Control 적용
    S_q, _, logw, _, _ = sim.simulate_controlled(
        S0=100, v0=0.04, T=0.5, dt=1/252, num_paths=200_000,
        control_fn=const_control
    )
    # P-measure 참값
    S_p, _ = sim.simulate(S0=100, v0=0.04, T=0.5, dt=1/252, num_paths=200_000)
    # 검증: E^Q[1{S<90} · dP/dQ] ≈ E^P[1{S<90}]
    est_q = (torch.exp(logw) * (S_q[:, -1] < 90).float()).mean().item()
    est_p = (S_p[:, -1] < 90).float().mean().item()
    se_q = (torch.exp(logw) * (S_q[:, -1] < 90).float()).std().item() / (200_000**0.5)
    assert abs(est_q - est_p) < 3 * se_q, f"Bias detected: {est_q} vs {est_p}"
```

---

## 부록 B. 체크리스트 (출판/배포 준비)

- [ ] README 수식 = 코드 구현 = 논문 수식 (일치)
- [ ] `pytest --cov` coverage ≥ 80%
- [ ] Girsanov unbiasedness 통계 검증 자동화
- [ ] Heston analytic benchmark 통과
- [ ] rBergomi hybrid scheme 구현 + ATM skew test 통과
- [ ] 7-method rare event benchmark 표 생성
- [ ] Backtest POF/Christoffersen test 리포트
- [ ] Docker / Colab 재현 가능
- [ ] Preprint arXiv 업로드
- [ ] Zenodo DOI 발급
- [ ] 실무 사용 가이드(docs/risk_usage_guide.md) 완성
- [ ] License 명시 (MIT 또는 Apache 2.0 권장)
