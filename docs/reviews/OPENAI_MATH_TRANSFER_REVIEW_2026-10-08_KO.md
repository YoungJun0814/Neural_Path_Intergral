# OpenAI math 저장소의 연구 전이 가능성 검토

작성일: 2026-10-08. 성격: 선행연구 조사·가정 점검·후속 연구 후보 제안. 구현 완료 또는 성능 입증 보고서가 아니다.

## 1. 결론

우리 연구에 반영할 만한 수학적 도구는 있다. 그러나 저장소의 정리를 가져오는 것만으로 현재 병목이나 독창성이 해결되지는 않는다. 가장 유망한 후보는 **Gaussian Stein 항등식을 이용한 저차원 Volterra 잔차 보정**이다. 이를 기존 정확한 중요도 샘플링과 결합하고, 독립 위험 검증 및 격자 검증 아래에서 평가하는 방향을 권한다.

추천을 두 순서로 구분한다.

1. 실행 우선순위: 독립 기준값·격자 검증 → 정적 가이드의 작은 개선 → 제한된 Stein 보정 → 필요할 때만 bank mutation 확장.
2. 새로운 연구 기여의 잠재력: Volterra 구조를 반영한 가중 Stein 잔차 이론 → 다중 시간척도 희귀 기여 커버리지 이론 → 조건부 보조변수 mixing 연구.

이는 성공 확률이나 상위 저널 게재 가능성을 수치로 예측한 순위가 아니다. 구현 난도, 현재 병목과의 관련성, 가정 충족 여부, 기존 연구와의 중복을 고려한 연구 판단이다.

특히 다음은 채택하면 안 된다.

- #139의 로그오목 샘플링 정리를 현재 사건/위험 분포의 고속 샘플링 보장으로 인용하는 것.
- #374의 Brenier-map 안정성을 Gaussian 중요도 샘플링의 분산 보장으로 바꾸는 것.
- 양자 다체계·PEPS 정리를 우리 비국소 Volterra 기억의 압축 또는 양자 가속 근거로 사용하는 것.
- Stein 제어변량, 혼합 가중치 최적화 또는 형식 검증 자체를 새 수학적 원리라고 주장하는 것.

## 2. 조사 범위와 신뢰도

외부 저장소는 다음 버전으로 고정했다.

`openai/math@adc7f1241b42e322a6451854ab7e4b4c146bf78a`

[고정 버전 README](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/README.md)는 수학 원고·증명 자료 모음이며 검증 단계가 서로 다르고, 미형식화 결과에 문제가 있을 수 있다고 명시한다. 목록에는 722개 원고·372개 family가 있다. 라이브러리나 금융 모델 패키지로 간주하면 안 된다.

전체 catalogue를 주제별로 선별하고 관련 원고의 TeX, 형식화 범위 설명, 일부 실제 Lean solution 및 Comparator 설정을 읽었다. catalogue에서 Volterra, Bergomi, Föllmer, Cameron, rare-event, importance sampling, Monte Carlo 직접 키워드의 적중은 없었다. 이것은 목록의 명시적 연결이 없다는 뜻이지, 722개 원고 전체의 모든 증명을 검사했다는 뜻은 아니다.

검토 수준을 구분한다.

| 수준 | 이번에 한 일 | 의미하지 않는 것 |
|---|---|---|
| 주제 선별 | catalogue 및 원고 설명 조사 | 전 원고 정밀 심사 |
| 적용 가정 검토 | 관련 원고·범위 설명과 우리 타깃 대조 | 우리 분포에 정리 적용 가능성 확정 |
| 증명 구조 조사 | #139, #374의 일부 원고 및 실제 Lean solution 확인 | 의존성 전체 커널 재검증 |
| 구현 가정 검사 | 실제 타깃 Hessian과 유한차분 교차 검사 | interval arithmetic으로 인증된 수학적 반례 |
| 방법 전이 제안 | 우리 estimator에 맞춘 항등식·설계 도출 | 새 알고리즘 구현·성능 개선 |

Lean/Comparator를 설치하거나 실행하지 않았다. `ComparatorChallenges`의 `sorry`는 비교용 정리 명세일 수 있으므로, 그 파일만 보고 실제 증명 부재라고 결론 내리지 않았다. #139는 `OAI.Probability.LogConcave.Main`, #374는 `OAI.Analysis.Brenier.Stability` 실제 solution 및 JSON 연결을 별도로 확인했다.

형식화는 명세·정의·허용 공리 아래의 증명을 검증한다. 그 명세가 금융 타깃의 의미를 정확히 표현하는지는 별도 심사 대상이다. [Comparator 공식 설명](https://github.com/leanprover/comparator), [저장소 Comparator 안내](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/ComparatorChallenges/README.md).

## 3. 현재 연구의 상태와 전이 기준

우리 저장소 HEAD: `aa9a21c2d9d5f16f1dc56798d476ee831f8556d0`.

기준 문서:

- [구조적 검토·개선계획](MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md)
- [보조 위험분포 전체 재학습 안정성](AUXILIARY_WHOLE_FIT_STABILITY_2026-10-07_KO.md)
- [기존 야간 실행계획](OVERNIGHT_GATED_EXECUTION_PLAN_2026-10-07_KO.md)

현재 구조는 finite-grid rBergomi의 Gaussian innovation 공간에서 독립 가격 잡음을 조건부로 적분하고, 남은 좌표를 정확한 defensive Gaussian mixture로 샘플링하는 것이다. N=32 개발 셀은 원래 3N 좌표에서 2N=64 좌표로 줄어든다. 두 local 좌표는 별도의 물리적 변동성 요인 두 개가 아니라 동일 Volterra/Brownian 구간의 Gaussian 기하를 나타낸다.

기호는 다음과 같이 고정한다.

\[
p(z)=\mathcal N(0,I_d),\quad 0\le g_N(z)\le1,\quad
\mu_N=\int g_Np,\quad q\ge\delta_qp.
\]

원래 사건확률 estimator와 보조 위험 estimator를 혼동하지 않는다.

\[
X\sim q:\quad W=\frac{g_N(X)p(X)}{q(X)},
\qquad M_2(q)=\int\frac{g_N^2p^2}{q}.
\]

\[
X\sim r:\quad Y=\frac{g_N(X)^2p(X)^2}{q(X)r(X)},
\qquad E_rY=M_2(q).
\]

최종 표본은 frozen proposal 아래의 새 IID 표본이며 self-normalization을 하지 않는다. q를 고정하고 r만 개선하면 M₂ 검증을 개선하는 것이지 원래 모델 q를 개선한 것이 아니다.

현재 v7 보조 검증은 10개 고정 q × 5개 전체 보조 재학습에서 개발용 안정성 조건을 모두 통과했다. 위험 RSE 중앙값은 canonical 약 2.57%, 높은 η 약 2.53%다. 다만 독립 참값 인증, 연속시간 정확성, q의 fixed-precision 총비용 우위는 미완료다.

같은 문서의 정적 가이드 대 학습 혼합 진단에서 관측된 N·RSE²는 canonical 117.89 대 173.67, 높은 η 112.31 대 168.14였다. 이 관측은 **학습 부분의 추가 가치가 아직 입증되지 않았음**을 보여준다. 확정된 모집단 분산 우위나 통계적 dominance 판정은 아니다.

큰 기여 경로의 단일 구간 집중은 격자 민감도 경고다. 이런 상황에서는 차원·operator를 추가하기 전에 다음을 확인해야 한다.

1. 독립 메커니즘의 μ/M₂ 기준값과 꼬리 기여 커버리지.
2. 같은 Gaussian 경로에 결합된 N=16/32/64 격자 차이.
3. 정적 가이드 대비 학습 비용을 지불할 이유.
4. 미사용 confirmation 셀에서 전체 학습 반복과 동일 정확도 총시간.

## 4. 관련 원고별 판단

| Family | 내용 | 우리에게 유용한 부분 | 직접 적용의 장애 | 판단 |
|---|---|---|---|---|
| #139 | 잘 조건화된 로그오목 샘플링 | Gaussian 보조변수, Stein/divergence 구조, exact marginal 사고 | 전역 Hessian 조건 불충족; oracle query와 wall-time 다름 | 방법 차용 1순위, 정리 직접 적용 금지 |
| #093 | subgaussian log-concave LSI | 가정·상수의 엄격한 분리 | 사건/위험 타깃 로그오목성 없음 | 이론 watchlist |
| #374 | Brenier map 1/3-Hölder 안정성 | bank 수송 안정성 진단, 작은 mode perturbation 경고 | compact uniform source와 Gaussian source 다름; M₂ 정리 아님 | 진단 도구 |
| #090 | circle-packing Fourier 수치 인증 | 유한 검증·tail proof·version 분리 | 원래 문제는 무관; 64차원 인증은 고비용 | 검증 프로토콜 차용 |
| #360 | weak-MTW transport regularity | 정리 범위의 명시 방식 | compact manifold, positive bounded density 등 불충족 | 현재 보류 |
| #221 | diluted spin-glass variational formula | 독립 bank 간 overlap 진단 발상 | spin system 가정 및 극한은 우리 모델과 다름 | 제한적 진단 발상 |
| #265 | gapped square-grid ground state PEPS | 구조적 압축에 필요한 가정 점검 | locality, spectral gap, finite local dimension 없음 | 양자 확장 근거로 부적합 |

### 4.1 #139: 가장 가까운 방법론, 그러나 가장 위험한 과잉 해석

[형식화 범위](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/139.md)는 전역적으로 I≤∇²V≤2I인 조건 아래 exact value/gradient query 수를 다룬다. TV 오차는 1/10이고 arithmetic/bit 비용은 제한하지 않는다. 따라서 실제 GPU·CPU 총시간 보장, 희귀확률 상대오차 보장, 일반 비로그오목 샘플링 결과가 아니다.

[Introduction](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Subpolynomial-query-complexity-for-well-conditioned-log-concave-sampling-September-26-2026/build/sections/introduction.tex), [sampling 구성](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Subpolynomial-query-complexity-for-well-conditioned-log-concave-sampling-September-26-2026/build/sections/sampling.tex), [centering 구성](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Subpolynomial-query-complexity-for-well-conditioned-log-concave-sampling-September-26-2026/build/sections/centering.tex)을 구분해 읽었다. 보조 Gaussian 변수와 divergence 항등식을 쓰는 증명 구조는 참고할 만하다. 이상적인 초기 law·정확한 평균을 사용하는 중간 flow 구성은 그대로 실행 가능한 샘플러가 아니다.

TV≤0.1이면 bounded payoff의 절대 기대오차는 제어할 수 있어도 극소 μ의 상대오차는 보장하지 못한다. 또한 샘플러에서 표본을 얻었다는 사실만으로 정규화된 q(x)를 평가할 수 있는 것은 아니다. 우리 최종 IS는 full density를 요구한다.

### 4.2 #093: 차원 무관 부등식의 의미

이 원고의 대상은 log-concave density이고 모든 단위 방향에 대해 통일된 subgaussian 조건이 필요하다. covariance가 유한하거나 Gaussian prior에서 출발한다는 사실만으로 조건을 만족하지 않는다. [원고 Introduction](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/A-dimension-free-logarithmic-Sobolev-inequality-for-subgaussian-log-concave-measures-September-23-2026/build/sections/01-introduction.tex).

Gaussian prior 자체의 Poincaré 부등식은 별도로 활용할 수 있다. 그러나 payoff gradient의 적분가능성 및 μ로 나눈 상대 위험 상수까지 분석해야 한다. 차원에 독립적인 절대오차 상수가 희귀도에 독립적인 상대오차 상수는 아니다. 이 snapshot의 catalogue/설명에서 #093에 연결된 Lean 범위 안내는 찾지 못했다.

### 4.3 #374: 수송은 안정적이어도 IS 위험은 불안정할 수 있다

compact convex body의 uniform source와 공통 compact target support에서 optimal quadratic maps의 L² 거리를 W₂의 1/3승으로 제어한다. 1/2승 일반 주장은 성립하지 않는다는 sharpness 결과도 포함한다. [형식화 범위](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/374.md).

우리 source는 unbounded Gaussian이고 fitted mixture가 Brenier 최적수송이라는 보장도 없다. 절단하면 원래 타깃과 달라지므로 누락 mass 및 M₂ tail을 따로 제어해야 한다. 상수가 dimension/grid에 균일하다고 해석해서도 안 된다.

### 4.4 #090, #360, #221, #265

#090은 작은 정확 대수 검증·수치 인증과 continuum/tail 논증을 분리하는 좋은 검증 설계 예시다. 수치 스크립트가 성공해도 논문의 모든 무한 영역 주장이 자동으로 검증되는 것은 아니다. [검증 README](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/A-sharp-Fourier-certificate-for-planar-circle-packing-September-23-2026/verification/README.md).

#360은 compact Riemannian setting 등 적용 조건이 강하고, 형식화 범위 자체에도 제한을 명시한다. 이를 unbounded rare-event density에 옮기지 않는다. [범위 설명](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/360.md).

#221의 여러 replica overlap 관점은 bank가 같은 작은 mode에 갇혔는지 보는 진단 발상으로 사용할 수 있다. 다만 overlap 구조를 관측했다고 spin-glass phase, replica symmetry breaking, ultrametricity를 주장할 수 없다. [원고 Introduction](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-Mezard-Parisi-formula-for-diluted-spin-glasses-September-23-2026/build/sections/introduction.tex).

#265의 PEPS 결과는 local Hamiltonian·unique ground state·uniform gap 등의 조건 아래 존재적 압축을 다룬다. rough Volterra의 nonlocal memory는 이 구조와 같지 않다. 존재 정리를 실용적 tensor algorithm이나 quantum speedup으로 해석하지 않는다. [원고 main.tex](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-PEPS-approximation-of-gapped-square-grid-ground-states-September-24-2026/build/main.tex).

## 5. 실제 타깃의 곡률 검사: 직접 적용을 막는 수치 증거

원고 가정이 추상적으로 까다롭다는 이유만으로 배제하지 않고, 실제 구현의 potential을 검사했다.

\[
V_{event}(z)=\tfrac12\|z\|^2-\log g_N(z),
\]

\[
V_{risk}(z)=\tfrac12\|z\|^2-2\log g_N(z)+\log(q(z)/p(z)).
\]

두 번째 식은 risk law ∝g²p²/q의 음의 로그이며, q/p 항의 부호가 중요하다. additive normalizer는 Hessian에 영향을 주지 않는다.

기존 artifact `results/post_audit/r2_family_diagnosis_five_fit_v1.json`의 각 셀 parent training replicate 0, `parent-as-is` q를 사용했다. artifact SHA256:

`b0f5dc0a4888aa17f5b0679e51eacfbd603e56771f41c9f66cc144dec99b19b0`

float64, CPU, torch thread 1, autograd Hessian의 대칭화 후 최소 고유값을 계산했다. 무작위 탐색이 아니라 사전 정의된 네 종류 좌표의 국소 검사다. guide 방향 u는 `volterra_monitoring_operator`의 row 15를 정규화하고 e는 local 좌표 index 32의 단위벡터로 설정했다. 두 개발 셀 모두 N=32다.

| 셀 | 검사점 | min eig(event) | min eig(risk) |
|---|---|---:|---:|
| canonical η=1.5 | z=0 | −198.480606 | −397.960634 |
| canonical | 첫 non-natural q mean | −10.353616 | −21.707137 |
| canonical | 4u+2e | −13.759939 | −27.852805 |
| canonical | 8u+2e | 약 1.000000 | 약 1.000000 |
| 높은 η=2 | z=0 | −637.416754 | −1275.811890 |
| 높은 η | 첫 non-natural q mean | −41.217349 | −83.434120 |
| 높은 η | 4u+2e | −3.303844 | −7.495488 |
| 높은 η | 8u+2e | 약 1.000000 | 약 1.000000 |

z=0에서 최소 고유벡터 방향의 central finite difference로 교차 검사했다. h=10⁻³과 10⁻⁴를 표에 싣는다.

| 셀·타깃 | autograd | FD h=10⁻³ | FD h=10⁻⁴ |
|---|---:|---:|---:|
| canonical event | −198.480606 | −198.480597 | −198.480643 |
| canonical risk | −397.960634 | −397.960614 | −397.960639 |
| 높은 η event | −637.416754 | −637.416701 | −637.416906 |
| 높은 η risk | −1275.811890 | −1275.811782 | −1275.812156 |

h=10⁻⁵에서도 모두 큰 음수지만 cancellation 오차가 더 커졌다. 위 일치는 구현된 potential이 전역 강한 로그오목성 조건을 만족한다고 볼 수 없다는 강한 수치 증거다. floating-point 검사이므로 엄밀한 interval-certified 반례라고 부르지는 않는다. 특정 점의 양의 Hessian은 전역 조건을 증명하지 않는다. 단순 affine whitening은 음의 곡률의 inertia를 제거하지 못한다.

비로그오목성 자체가 우리 모델의 수학적 오류라는 뜻은 아니다. 일반적인 nonlinear rare-event target에서 가능한 성질이다. 오류가 되는 것은 이 타깃에 로그오목성 전제의 정리를 적용하거나 그 정리로 mixing·효율성을 보증하는 주장이다.

### 5.1 핵심 검사 재현 코드

아래는 zero-point 및 FD 교차검사의 재현용 코드다. 기존 artifact를 읽을 뿐 원래 학습·최종 표본을 변경하지 않는다. 프로젝트 루트, 기존 Python 환경에서 실행한다.

```python
import json
from pathlib import Path
import torch
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

torch.set_num_threads(1)
a = json.loads(Path("results/post_audit/r2_family_diagnosis_five_fit_v1.json").read_text())
for row in a["records"]:
    if row["parent_training_rep"] != 0:
        continue
    item = next(c for c in row["candidates"] if c["method"] == "parent-as-is")
    q = proposal_from_parameters(item["proposal_parameters"])
    problem = _problem(a["config"]["model"], row["cell"], a["config"]["smc"]["steps"])
    z = torch.zeros(problem.local_dimension, dtype=torch.float64)
    for risk in (False, True):
        def V(x):
            batch = x.unsqueeze(0)
            log_g = _log_potential(problem, batch)[0]
            return .5*x.square().sum() - (2 if risk else 1)*log_g + (
                q.log_q_over_p(batch)[0] if risk else 0)
        H = torch.autograd.functional.hessian(V, z)
        vals, vecs = torch.linalg.eigh((H + H.T)/2)
        v = vecs[:, 0]
        fds = {h: float((V(z+h*v)-2*V(z)+V(z-h*v))/(h*h))
               for h in (1e-3, 1e-4, 1e-5)}
        print(row["cell"]["id"], risk, float(vals[0]), fds)
```

이 검사에서 확인한 것은 정리의 적용 가정이지, 새 모델의 성능이 아니다. 원래 μ/M₂를 새로 추정하거나 confirmation을 실행하지 않았다.

## 6. 후보 A: Gaussian Stein–Volterra 잔차 보정

### 6.1 핵심 아이디어

“희귀 경로를 더 자주 생성한다”는 q 개선에 더하여, **평균이 정확히 0인 보정 함수를 빼서 estimator의 변동을 줄인다**. 보정 항의 평균을 신경망으로 추정하지 않고 Gaussian 적분 항등식으로 정확히 고정한다.

p=N(0,I_d), 적절한 미분가능성·경계 소멸·적분가능성을 갖춘 vector field v에 대해

\[
C_v(z)=\nabla\!\cdot v(z)-z^Tv(z),\qquad E_pC_v=0.
\]

따라서 frozen v, β 및 q 아래 새 IID 표본 X~q에 대해

\[
\widehat\mu_{CV}=\frac1n\sum_i\frac{p(X_i)}{q(X_i)}
\{g_N(X_i)-\beta C_v(X_i)\}
\]

는 μ_N을 유지한다. 중요한 점은 Stein identity를 사건 law가 아니라 **알려진 Gaussian prior p**에 적용한다는 것이다. 사건/위험 타깃의 로그오목성은 필요하지 않는다.

이 식은 본 검토의 적용 설계이지 #139가 우리 금융 모델에 대해 증명한 정리가 아니다. Gaussian Stein 제어변량은 기존 방법이다. [Control functionals](https://arxiv.org/abs/1410.2392), [Neural Control Variates](https://arxiv.org/abs/1806.00159), [Neural Control Variates, 2020](https://arxiv.org/abs/2006.01524).

### 6.2 위험 estimator에 먼저 작은 검사를 할 수 있다

q를 고정하고 h_q=g_N²p/q라 두면 M₂(q)=E_p h_q이다. 독립 보조분포 r로부터

\[
\widehat M_{2,CV}=\frac1m\sum_i\frac{p(X_i)}{r(X_i)}
\{h_q(X_i)-\beta C_v(X_i)\},\quad X_i\sim r
\]

도 같은 M₂(q)를 추정한다. 이것은 원래 q의 성능 개선이 아니라 검증 estimator 개선이다. 이를 “우리 모델의 원래 분산이 감소했다”라고 쓰면 틀리다.

### 6.3 첫 구현은 bounded low-rank analytic field

처음부터 64차원 신경망을 쓰지 않는다. Volterra point/window feature의 orthonormal basis U∈R^{d×k}, k=2–8, UᵀU=I를 구성하고

\[
s=U^Tz,\quad v(z)=Ua(s),\quad C_v=\nabla_s\!\cdot a(s)-s^Ta(s)
\]

를 사용한다. divergence는 k차원에서 정확히 계산할 수 있다. polynomial×Gaussian window 또는 smooth compact-support feature field로 C_v를 bounded하게 만들 수 있다. feature 공간 compact support면 충분하며, 전체 d차원 compact support를 요구하지 않는다. Gaussian complement의 적분과 sᵀa(s) 구조로 경계 항을 확인해야 한다.

비정규 직교 feature matrix A를 쓴다면 위 단순 식을 그대로 쓰지 않고 Gram matrix/Jacobian을 반영하거나 먼저 whiten한다. rank deficiency는 SVD로 처리한다. 무작위 trace 추정은 첫 버전에서 사용하지 않는다.

### 6.4 학습 목적과 위험한 실수

q 아래 사건 estimator의 second moment를 줄이는 목적은

\[
J(v,\beta)=E_q[(p/q)^2(g_N-\beta C_v)^2]
\]

다. prior 아래 unweighted MSE만 줄여도 q 아래 분산이 감소한다는 보장은 없다. 위험 보정은 r 가중치와 h_q에 맞는 별도 목적을 사용한다.

- training, selection, final, independent reference 표본을 분리한다.
- 최종 표본으로 β를 선택하지 않는다. 처음에는 sample split을 사용한다.
- β=0 baseline을 항상 포함한다. estimator 평균을 target에 맞추는 보정은 금지한다.
- 보정 후 표본값은 음수일 수 있다. clipping 또는 0으로 잘라내면 일반적으로 편향이 생긴다.
- signed correction을 positive SMC potential로 사용하지 않는다. bank 생성과 final CV를 분리한다.
- 미관측 rare mode를 CV가 자동으로 발견하는 것은 아니다. 두 estimator가 같은 mode를 놓칠 수도 있다.
- 신경망의 activation·성장률·미분가능성·gradient/divergence 정확도를 따로 검사한다.

| 보정 항이 bounded일 때 | 유효한 절대 bound |
|---|---|
| 사건 CV, \|C\|∞≤B | \|W_CV\|≤(1+\|β\|B)/δ_q |
| 위험 CV, \|C\|∞≤B | \|Y_CV\|≤(1/δ_q+\|β\|B)/δ_r |

따라서 원래 nonnegative Y≤100 규칙을 보정 estimator에 그대로 적용하면 안 된다. signed bound와 새 variance/CI 절차를 써야 한다. bound가 유효해도 작은 상대오차가 자동 인증되지는 않는다.

### 6.5 논문 독창성이 생길 수 있는 부분

새로울 가능성이 있는 것은 다음의 **Volterra-specific 연결 정리**다.

\[
\text{kernel/time-scale/rarity}
\longrightarrow\text{weighted residual approximation}
\longrightarrow\text{relative estimator variance}
\longrightarrow\text{training-inclusive work}.
\]

필요한 증명에는 q/p 가중치, Gaussian Sobolev integrability, 희귀도에 따른 상수, grid refinement, feature rank가 모두 등장해야 한다. 단순 Stein unbiasedness나 학습 loss 감소는 이 기여를 대신하지 못한다. 기존 구조 계획의 conditional/Poincaré 이론과 결합할 수 있지만 상대 M₂에 대한 상수가 희귀도·격자에 따라 폭발하는지 숨기지 않는다.

2026년에도 conditional neural Stein CV 연구가 있다. [Conditional neural control variates](https://proceedings.mlr.press/v337/siahkoohi26a.html). 따라서 “조건부+neural+Stein” 조합만으로 신규성을 주장할 수 없다.

## 7. 후보 B: 다중 시간척도 excursion mixture의 위험 기반 보정

현재 정적 가이드는 시간별 한 점의 Volterra excursion과 가격 방향을 결합한다. 기존에 실제 성능 신호가 있는 이 구조부터 확장하는 것이 구현 위험이 낮다.

1. 단일 point 방향.
2. 짧은 window 평균/증가 방향.
3. 여러 구간에 분산된 변동성 방향.

이들을 simulator의 실제 linear operator B에서 구성한다. DCT나 무작위 feature가 물리적 기여 방향을 나타낸다고 가정하지 않는다. 단일 구간 spike가 coarse-grid artifact이면 그것을 더 잘 학습하는 것이 continuous-time 개선은 아닐 수 있다.

선형 제약 Am=b에서 b∈range(A)이면 최소 에너지 Gaussian mean은

\[
m_*=A^T(AA^T)^\dagger b.
\]

이는 제약 아래 최소 Cameron–Martin energy일 뿐, event-optimal shift가 아니다. joint constraint의 covariance·rank·불가능한 b를 검사해야 한다.

성분 density φ_j를 고정하고

\[
q_w=\delta p+(1-\delta)\sum_jw_j\phi_j,\qquad w\in\Delta
\]

로 설정하면 M₂(q_w)는 w에 대해 convex다. 정당화 가능한 미분/적분 교환 아래 Hessian은

\[
\partial^2_{jk}M_2=2(1-\delta)^2
\int\frac{g_N^2p^2\phi_j\phi_k}{q_w^3},
\]

이므로 PSD다. w_j≥ε>0 같은 조건은 derivative integrability를 확인하는 한 방법이다. 방어 floor만으로 모든 성분의 임의의 higher moment 미분이 자동 정당화된다고 단정하지 않는다.

이 convexity와 mixture/CV joint optimization은 선행연구다. [He–Owen, 2014](https://arxiv.org/abs/1411.3954). 현재 개선계획의 qα 보정과도 중복된다. 이번 조사가 이를 새 발명으로 제안하는 것은 아니다.

새 r 표본으로 ∫g²p²/q_w를 정확히 reweight해 학습하되, final sample과 분리한다. static-only, equal weights, risk-trained weights, 기존 learned mixture를 동등 정확도·총시간으로 비교한다. 원래 q를 바꾸는 실험과 보조 r만 바꾸는 실험을 분리해야 한다.

## 8. 후보 C: Gaussian 보조변수 기반 SMC mutation

#139의 발상을 참고해 bank target f_β(x)=p(x)h(x)^β에 대해

\[
\widetilde\pi_\beta(x,y)\propto f_\beta(x)
\mathcal N(y;x,\tau I)
\]

를 고려할 수 있다. y를 적분하면 x marginal이 원래 π_β다. y|x는 exact Gaussian이고 x|y 이동은 정확한 MH ratio 또는 별도 정당화된 conditional sampler를 사용한다.

이것은 payoff를 smoothing해서 다른 μ를 계산하는 것과 다르다. 최종 g_N·정확한 q density를 변경하지 않는다. bank mutation은 proposal fitting에만 사용하고 MCMC 표본을 IID final로 취급하지 않는다.

가장 중요한 한계는 “Gaussian proximal term을 추가했으므로 globally convex”라는 주장이 성립하지 않는다는 것이다. ∇²V의 전역 하한 −L이 알려져 있을 때만 1/τ>L 같은 논증이 가능하다. 우리 모델에서 그런 L은 아직 없다. τ가 작으면 local 안정성은 좋아져도 mode 사이 이동이 악화될 수 있다.

현재 pCN 및 정확한 independence MH와 동일 potential/시간 예산으로 비교한다. acceptance rate, ancestry, weighted ESS만으로 성공 판정하지 않고 whole-fit risk 안정성·tail mode 기여를 함께 사용한다. 이미 정적 가이드의 장점이 있으므로 이 확장은 기준값 검증 후 필요한 경우에만 진행한다.

## 9. 수송·KL 진단을 분산 보장으로 오해하지 않는 법

q*=g_Np/μ_N라 두면

\[
M_2(q)/\mu_N^2-1=\chi^2(q_*\|q).
\]

즉 원래 중요도 추정의 상대 분산을 지배하는 것은 이 방향의 χ²다. 작은 KL, TV 또는 W₂만으로 이를 제어할 수 없다.

자체 도출한 두 점 반례를 보자. 거리 1인 두 점 A,B에서 π(A)=ε, q(A)=ε³로 두면

\[
TV=\epsilon-\epsilon^3\to0,\qquad
W_2=\sqrt{\epsilon-\epsilon^3}\to0,
\]

\[
KL(\pi\|q)=2\epsilon\log(1/\epsilon)
+(1-\epsilon)\log\frac{1-\epsilon}{1-\epsilon^3}\to0,
\]

반면

\[
\chi^2(\pi\|q)=1/\epsilon
+\frac{(1-\epsilon)^2}{1-\epsilon^3}-1\to\infty.
\]

이는 일반 분포 간 거리의 반례이지, 현재 고정 δ와 μ의 Gaussian q에서 같은 발산을 직접 관측했다는 주장은 아니다. 우리 defensive floor는 M₂/μ²≤1/(δ_q μ)를 제공하지만 희귀도가 커질수록 이 bound는 약해진다.

따라서 bank 간 weighted W₂, overlap, clustering은 **coverage 진단**으로 사용한다. 실제 tail contribution 및 독립 M₂ 검증을 대체하지 않는다. empirical atomic transport를 density q로 그대로 쓰면 p/q는 정의되지 않으므로, ordinary IS에는 정규화된 연속 mixture 등 정확한 density가 필요하다.

## 10. 검증 방식에 즉시 차용할 수 있는 것

### 10.1 작은 수학적 계약

Comparator 구조를 참고하여 정리 문장·정의·가정·증명·실행 테스트를 분리한다.

- finite-grid Gaussian 좌표 변환과 조건부 적분의 동일 law.
- normalized q/r 및 defensive floor.
- ordinary IS 기대값 및 M₂ 항등식.
- bounded low-rank Stein field의 경계 항과 zero mean.
- adjacent-grid Gaussian coupling의 marginal covariance.

처음부터 외부 Lean library 전체를 가져오기보다는 작은 명세 2–3개를 별도 검증 후보로 삼는다. 이미 알려진 항등식의 Lean proof는 정확성 자산이지 새 이론 기여는 아니다. Python/CUDA 구현의 bug-free 보장도 아니다.

### 10.2 toy 문제의 interval-certified 기준값

#090처럼 유한 영역의 계산 검증과 tail 논증을 분리할 수 있다. 적은 차원의 toy 문제에서 independent quadrature/interval arithmetic을 사용하고 Gaussian tail은 해석적으로 처리한다.

영역 A 밖에 대해 g≤1, q≥δ_qp이면

\[
\int_{A^c}g p\le P_p(A^c),\quad
\int_{A^c}g^2p^2/q\le P_p(A^c)/\delta_q.
\]

이것은 valid absolute truncation bound지만 희귀 μ 또는 M₂에 대한 작은 상대오차를 얻으려면 매우 작은 tail mass가 필요하다. 64차원에서 동일 quadrature가 실용적이라고 가정하지 않는다. toy 인증 성공을 실제 개발 셀의 참값 인증으로 승격하지 않는다.

## 11. 제안 실행 순서와 통과·중단 기준

이 절은 후속 실행 제안이다. 이번 조사에서 실행하지 않았다. 기존 개선계획의 병목 우선 원칙을 유지하고 확장을 작게 제한한다.

### S0. 기준값과 격자 관문

- 현재 q·g·source artifact를 고정한다.
- 공통 learned/excursion guide에 의존하지 않는 별도 reference mechanism을 포함한다.
- M₂의 정확도와 원래 μ의 정확도를 각각 점검한다.
- 적은 차원 oracle 및 tail certificates로 reference 구현의 항등식을 검사한다.
- N=16/32/64에서 정확한 Gaussian marginal을 갖는 coupling으로 grid difference와 uncertainty를 기록한다.
- 실패 시 reference 또는 discretization 원인을 해결한다. CV·새 neural architecture로 넘어가지 않는다.

### S1. 최소 정적 가이드 대조

- point/window/distributed feature를 소수만 사전 고정한다.
- static-only 대비 같거나 적은 component 수에서 equal/convex-fitted weights를 비교한다.
- bank/selection/final은 새로 생성한다. 이전 final에 맞춘 template는 개발용으로 표시한다.
- q 변경과 r 변경은 서로 다른 experiment ID로 관리한다.
- 동일 정확도 자격 없이는 speedup을 쓰지 않는다.
- 학습 가중치가 총비용 이득이 없으면 equal/static weights를 유지한다.

### S2. bounded analytic Stein microstudy

- toy에서 exact derivatives·zero-mean identity·boundedness를 점검한다.
- U rank 2/4/8과 β=0를 사전 고정한다.
- 먼저 frozen q의 보조 M₂ evaluator에서 bounded analytic CV를 시험한다.
- original event-CV는 별도 결과로 평가한다.
- 성능 지표: 독립 reference와의 일치, empirical variance 및 whole-run 변동, derivative 포함 총시간, tail concentration, signed CI 적합성.
- 저차원 oracle에서 bias 또는 gradient/divergence 불일치가 나오면 즉시 중단한다.
- 개발 통과 기준 제안: 같은 reference qualification 아래 raw estimator 대비 variance×total-cost 개선의 독립 반복 CI가 개선 방향을 지지할 것. 필요한 최소 효과 크기·반복 수는 실행 전에 budget/power 분석으로 고정한다. 결과를 보고 기준을 낮추지 않는다.
- 새 analytic baseline보다 neural field의 추가 총비용 이득이 있을 때만 neural 확장을 고려한다.

### S3. bank mixing 확장은 잔여 coverage 문제에만

- 위 관문을 통과했는데도 bank 간 mode mass가 불안정하면 Gaussian augmentation을 검토한다.
- pCN, independence MH, augmented mutation을 같은 work 예산으로 비교한다.
- invariant kernel 증명과 finite-run mixing 증거를 구분한다.
- ancestry 개선만 있고 최종 위험 안정성 개선이 없으면 채택하지 않는다.

### S4. 논문 기여 고정과 confirmation

- 가장 작은 성공 구조를 고정한다. 여러 아이디어를 모두 넣지 않는다.
- 기존 구조 계획의 새 task·독립 전체 학습 반복·총시간 규칙을 유지한다.
- learned q, static guide, convex weights, standard CV/Stein CV, CE/SMC baseline을 포함한다.
- offline bank, fit, selection, derivative, reference preparation, inference 비용을 따로 기록한다.
- 재사용 학습 비용은 task 수에 따른 amortization까지 공개한다.
- 상대오차·grid·희귀도·rank를 잇는 정리가 실제로 증명되지 않으면 주장 수준을 수치 방법 연구로 낮춘다.

## 12. 상위 저널을 목표로 할 때의 판단

가능성 있는 작업 제목은 다음과 같다.

**Mesh-aware Volterra Stein Residual Correction with Independently Audited Rare-event Risk**

이는 확정 제목이나 신규성 선언이 아니다. 현재의 exact conditional residual path-space transport 틀에서 다음 세 가지를 함께 입증해야 한다.

1. 수학: Volterra 구조·희귀도·격자에 따른 weighted residual/relative-risk 정리. 조건과 상수가 명시되고 counterexample 범위까지 다룰 것.
2. 계산: 표준 IS·static guide·standard CV에 비해 의미 있는 fixed-precision total-work 개선. 적어도 두 방법이 동일 accuracy 자격을 갖춘 셀에서만 비교할 것.
3. 신뢰성: 독립 reference, 미사용 confirmation, 전체 학습 반복, continuous-time/finite-grid 구분, 외부 재현.

이 조건이 충족되면 상위 수준의 계산 확률·금융 수학 연구 주제로 발전할 여지가 있다. 저장소에서 정리를 차용하거나 network 이름을 바꾸는 것만으로 그 수준에 도달하지 않는다. 현재 상태에서 상위 저널 채택 가능성이 높다고 단정할 증거는 없다.

## 13. 선행연구 중복 및 추가 조사

다음은 반드시 차별화 표에 들어가야 한다.

| 선행연구 | 이미 존재하는 요소 | 우리가 새로 입증해야 하는 것 |
|---|---|---|
| [Oates–Girolami–Chopin](https://arxiv.org/abs/1410.2392) | Stein/control functional 기반 Monte Carlo 개선 | Volterra 희귀도·격자·가중 위험 특화 |
| [Wan 등](https://arxiv.org/abs/1806.00159) | neural CV와 augmented training | exact conditional 구조에서의 추가 이득 |
| [Müller 등](https://arxiv.org/abs/2006.01524) | neural CV/sampling 결합 | 최소 정적 가이드 대비 총비용 우위 |
| [He–Owen](https://arxiv.org/abs/1411.3954) | mixture weights/CV convex optimization | kernel-aware coverage와 relative-risk 정리 |
| [Rotskoff 등](https://proceedings.mlr.press/v145/rotskoff22a.html) | rare-event dominated objective의 active sampling | 독립 검증·finite-grid/continuous 구분 |
| [Siahkoohi–Oh](https://proceedings.mlr.press/v337/siahkoohi26a.html) | conditional neural Stein CV | 본 타깃·이론·cost에서의 차별화 |

추가로 [Unified Rough Volatility Framework…](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5912782)의 공개 검색 초록은 IS·learned low-rank CV·smoothing·kernel stability 조합을 언급한다. 전문 접근이 제한되어 초록 수준만 확인했다. 이 문헌과의 theorem/algorithm 비교 전에는 조합 자체의 신규성을 주장하지 않는다.

외부 원고 재사용 시 각 원고의 citation/BibTeX 및 실제 복사 대상의 license를 확인한다. 저장소 root의 Apache-2.0 표시만으로 제3자 자료 전체의 권리를 추정하지 않는다.

## 14. 이번에 수행한 것과 하지 않은 것

수행: 외부 원고 선별 조사, 관련 가정·형식화 범위 대조, 기존 모델/결과 확인, 16개 local Hessian 검사 및 zero-point FD 교차검사, 본 보고서 작성.

미수행: 새로운 모델·CV·SMC kernel 구현, Lean 재컴파일, reference 인증 완료, 성능 실험, 대규모 confirmation, 커밋·푸시·PR merge.

최종 권고: **기존 기준값/격자 관문을 먼저 끝내고, 다음 연구 후보는 bounded low-rank Gaussian Stein–Volterra 보정으로 제한한다.** 정적 다중척도 가이드를 강한 최소 baseline으로 유지하며, neural/augmentation 확장은 독립 위험과 총비용 이득이 있을 때만 채택한다.
