# G7 Controlled Multilevel Volterra 구현·사후 감사 보고서

Date: 2026-07-17

Protocol: `g7-controlled-multilevel-volterra-v1`
Decision: **finite-grid 이론 및 구현 정확성 통과, 저명 저널용 효율성 핵심 주장 기각**

## 1. 한 줄 결론

구현된 방법은 선언한 finite-grid BLP rBergomi 법칙에 대해 수학적으로 타당하고,
fine/coarse 보정항·Girsanov 우도·defensive mixture가 모두 정확하다. Correction-CEM은
기존 event-CEM보다 세 보정 레벨 모두에서 분산을 줄였다. 그러나 hard hit-plus-occupation
event의 인접 격자 보정분산이 격자 세분화와 함께 감소하지 않았고, 최적 MLMC 할당 후에도
finest-grid 단일 piecewise-CEM보다 약 `9.14x` 많은 online work가 필요했다. 따라서 이
조합을 현 상태로 논문의 주모델로 밀어서는 안 된다.

이 결론은 “코드가 틀려서 실패했다”는 뜻이 아니다. 오히려 exact-law gate를 통과한
구현으로 핵심 MLMC 가정이 이 문제의 hard functional에서 성립하지 않음을 falsify한
결과다.

## 2. 구현 범위

### 2.1 정확한 adjacent BLP coupling

- `src/path_integral/rbergomi_coupling.py`
- fine step `h`, coarse step `2h`를 공유하는 rBergomi BLP 경로 생성
- 첫 fine cell에서 다음 세 적분을 joint Gaussian으로 생성

$$
X_0=\int_0^h dW_s,\qquad
X_1=\int_0^h(h-s)^\alpha dW_s,\qquad
X_2=\int_0^h(2h-s)^\alpha dW_s,
$$

여기서 $\alpha=H-1/2$. 두 번째 fine cell의

$$
Y_0=\int_h^{2h}dW_s,\qquad
Y_1=\int_h^{2h}(2h-s)^\alpha dW_s
$$

와 결합해 coarse recent-cell pair를

$$
\Delta W^c=X_0+Y_0,\qquad I^c=X_2+Y_1
$$

로 구성한다. `X1 + Y1`은 첫 cell의 kernel endpoint가 틀리므로 사용하지 않는다.

### 2.2 하나의 fine-space change of measure

fine increment마다 pre-increment adapted control $u_i$를 계산하고

$$
\Delta W_i^P=\Delta W_i^Q+u_i h
$$

를 적용한다. 모든 fine/coarse kernel integral에도 같은 Cameron--Martin shift의 적분값을
더한다. 보정항 전체에 적용되는 우도는 오직

$$
L=\exp\left[-\sum_i u_i\cdot\Delta W_i^Q
-\frac12\sum_i\lVert u_i\rVert^2h\right]
$$

하나다. Fine payoff와 coarse payoff에 서로 다른 우도를 붙이지 않는다.

### 2.3 exact defensive mixture

- `src/path_integral/rbergomi_multilevel.py`
- natural component와 controlled component를 canonical fine target path에서 모두 replay
- stable log-sum-exp로 $dQ_{mix}/dP$ 계산
- 최종 기여도는 self-normalization 없이
  $(H_\ell-H_{\ell-1})dP/dQ_{mix}$ 사용

### 2.4 correction-focused CEM

- `src/training/rbergomi_multilevel_cem.py`
- 학습 target은 signed correction이 아니라
  $D_\ell=|H_\ell-H_{\ell-1}|$
- segment별 weighted MLE는

$$
\widehat u_{\ell,j}=
\frac{\sum_nD_\ell^{(n)}L^{(n)}
\sum_{i\in I_j}\Delta W_i^{P,(n)}}
{|I_j|h\sum_nD_\ell^{(n)}L^{(n)}}.
$$

- 불일치 표본이 최소 개수보다 적으면 soft proxy로 결과를 꾸미지 않고 명시적으로 중단
- 최종 추정 단계에서는 signed correction의 부호를 복원

### 2.5 MLMC 할당과 비용

- `src/evaluation/multilevel.py`
- 주어진 pilot variance $V_\ell$, cost $C_\ell$와 variance budget $\epsilon_v^2$에서

$$
N_\ell=\left\lceil
\frac{\sum_k\sqrt{V_kC_k}}{\epsilon_v^2}
\sqrt{\frac{V_\ell}{C_\ell}}
\right\rceil
$$

를 계산
- 0레벨은 불필요한 coarse path 없이 standalone G0 piecewise-CEM 사용
- 비영 보정 레벨에만 adjacent coupling과 defensive correction mixture 사용
- training-inclusive break-even을 별도 계산

## 3. 사전·사후 이론 감사

### 3.1 Proposition A: 두 BLP 주변분포의 정확성

첫 cell triple covariance와 두 번째 cell pair covariance에서

$$
\operatorname{Var}(\Delta W^c)=2h,
$$

$$
\operatorname{Cov}(\Delta W^c,I^c)
=\frac{(2h)^{\alpha+1}}{\alpha+1},
$$

$$
\operatorname{Var}(I^c)
=\frac{(2h)^{2\alpha+1}}{2\alpha+1}
$$

가 정확히 복원된다. 이전 coarse block의 Brownian increment는 coarse BLP historical
weight에 들어가고 현재 block의 $I^c$만 singular local term으로 들어가므로 전체 coarse
법칙도 standalone BLP 법칙과 같다. Fine marginal은 각 fine cell의 기존 2차원 BLP pair를
그대로 갖는다.

결론: **오류 없음.** 이 주장은 연속 rBergomi exact simulation 주장이 아니라 구현된
`kappa=1` finite-grid BLP scheme의 exact marginal coupling 주장이다.

### 3.2 Proposition B: controlled telescoping

bounded adapted fine control 아래 target augmented path의 임의 bounded functional $F$에
대해 discrete Cameron--Martin identity로

$$
E_Q[FL]=E_P[F]
$$

가 성립한다. 따라서

$$
E_Q[(H_\ell-H_{\ell-1}^{c})L]
=E_P[H_\ell]-E_P[H_{\ell-1}]
$$

이고 독립 level 평균을 합하면 $E_P[H_L]$로 telescope한다.

결론: **오류 없음.** Coarse 경로에 별도 우도를 계산하면 이 항등식이 깨질 수 있으므로
구현에서 금지했다.

### 3.3 Proposition C: mixture likelihood

각 component control을 동일한 fine target increments에서 causal replay하면

$$
\frac{dQ_{mix}}{dP}=\sum_k a_k\frac{dQ_k}{dP}
$$

를 경로별로 정확히 계산할 수 있다. 선택된 component의 직접 $dP/dQ_k$와 replay 결과가
기계 정밀도 수준에서 일치했다.

결론: **오류 없음.** Component별 weight를 평균하거나 선택 label의 weight만 쓰는 방식은
사용하지 않았다.

### 3.4 hard functional의 정리 경계

현재 payoff는 barrier hit와 occupation threshold를 동시에 포함하는 불연속 indicator다.
따라서 smooth/Lipschitz payoff용 표준 MLMC weak-rate를 그대로 적용할 수 없다. Digital과
barrier payoff에는 별도의 conditional smoothing 또는 path branching이 필요할 수 있다는
기존 분석과 일치한다. 관련 1차 문헌은
[Giles (2009)](https://doi.org/10.1007/s00780-009-0092-1)와
[Giles & Bernal (2022)](https://arxiv.org/abs/2209.03017)이다. BLP coupling 자체는
[Bennedsen, Lunde & Pakkanen](https://arxiv.org/abs/1507.03004)의 hybrid 구조에 맞췄다.

결론: **연속시간 수렴률 정리는 아직 없다.** 이번 구현이 증명한 것은 finite-grid
unbiased telescoping identity이며 continuous monitoring 또는 양의 MLMC rate가 아니다.

## 4. 기술 검증

검증 항목:

- triple covariance와 positive-definite Cholesky
- fine/coarse local BLP covariance
- coupled terminal marginal과 standalone fine/coarse simulator의 통계적 일치
- 모든 augmented kernel integral의 deterministic mean shift
- fine Brownian likelihood의 직접 재구성
- feedback control의 pre-increment adaptedness
- bounded payoff correction의 controlled/natural 일치
- hard-event finite-grid telescoping
- single-component mixture reduction
- multi-component likelihood normalization과 all-expert replay
- correction-CEM weighted MLE의 수식 재현
- 불충분 disagreement update의 fail-closed 동작
- MLMC allocation의 variance-budget 준수

최종 정적 검사:

- changed-file Ruff: pass
- changed-file mypy: pass
- full pytest: pass

Repository-wide 기존 정적 분석 문제는 이번 변경과 분리한다. 결과 파일과 사용자 소유
untracked artifact는 수정하거나 삭제하지 않았다.

## 5. 동결 5-seed 결과

설정:

- rBergomi: $H=0.1$, $\eta=1.9$, $\xi=0.04$, $\rho=-0.7$, $T=1$
- event: hit `30`, stress `60`, minimum occupation `0.30`
- levels: `16, 32, 64, 128`
- correction training seeds: `8001, 8002, 8003`
- untouched validation seeds: `8101`--`8105`
- validation paths: level/method/seed당 `20,000`
- final estimator: non-self-normalized

### 5.1 정확성

| 항목 | 결과 | Gate |
|---|---:|---:|
| natural MLMC mean | `0.003690` | reference |
| correction-CEM MLMC mean | `0.003718` | - |
| aggregate difference z | `0.0860` | `|z| <= 3`, pass |
| maximum likelihood-normalization z | `2.1540` | `<= 4`, pass |
| maximum selected replay error | `1.60e-14` | `<= 1e-10`, pass |

### 5.2 correction 분산

| Fine steps | Natural variance | Event-CEM variance | Correction-CEM variance | Event/Correction |
|---:|---:|---:|---:|---:|
| 32 | `2.200e-3` | `3.945e-4` | `2.815e-4` | `1.471x` |
| 64 | `2.260e-3` | `3.196e-4` | `3.069e-4` | `1.136x` |
| 128 | `2.490e-3` | `2.277e-4` | `1.999e-4` | `1.092x` |

Correction-CEM은 3/3 레벨에서 event-CEM을 이겼다. 이 부분은 양의 결과다. 하지만
자연측도 disagreement probability는 `0.00220, 0.00226, 0.00249`였고 correction
variance도 같은 수준으로 유지됐다. $h$에 대한 empirical variance log-slope는
`-0.0894`로, 세분화할수록 감소하는 양의 rate가 관찰되지 않았다. 이 값은 정리가 아니라
세 finite level의 진단값이다.

### 5.3 총-work

| 항목 | 결과 | Gate |
|---|---:|---:|
| geometric baseline/candidate work ratio | `0.1094` | `> 1.25`, fail |
| candidate slowdown | `9.14x` | fail |
| improving seeds | `0/5` | `>= 4/5`, fail |
| 10% relative-error baseline online work | `0.2585 s` | - |
| 10% relative-error candidate online work | `2.3159 s` | - |
| correction training | `25.21 s` | recorded |
| break-even | 없음 | candidate online work가 더 큼 |

5개 개별 seed의 work ratio는 모두 `1`보다 작았다. 따라서 timing noise나 한 seed의
outlier로 설명할 수 있는 실패가 아니다.

## 6. 객관적 판정

### 유지할 수 있는 주장

1. Exact adjacent-grid coupling of two finite-grid BLP marginals.
2. One fine-space causal change of measure for signed rough-path corrections.
3. Exact balance-mixture likelihood by all-expert target-path replay.
4. Correction-focused CEM이 shared event-CEM보다 보정항 분산을 줄였다는 동결 결과.
5. Hard path functional에서 정확성과 end-to-end efficiency가 다를 수 있음을 보인
   falsification framework.

### 해서는 안 되는 주장

1. 현재 controlled MLMC가 finest single-level CEM보다 빠르다는 주장.
2. 보정분산이 $O(h^\beta)$, $\beta>0$로 감소한다는 주장.
3. continuous-monitoring hard event에 대한 unbiased estimator라는 주장.
4. rBergomi 전 parameter 영역 또는 다른 payoff로 일반화된다는 주장.
5. 현재 결과만으로 저명 저널의 주기여가 완성됐다는 주장.

## 7. 실패 원인과 다음 연구 액션

현재 병목은 control 표현력이 아니라 payoff coupling이다. 새 fine monitoring points가
barrier hit 또는 occupation count를 바꾸는 확률이 이 rough regime의 관찰 범위에서
감소하지 않아, 여러 level을 합치는 비용을 상쇄할 correction decay가 없다.

따라서 다음 우선순위는 controller를 더 복잡하게 만드는 것이 아니다.

1. **Conditional-smoothed/path-branching correction**: barrier/occupation 경계 근처에서만
   fine continuation을 branch해 discontinuity를 완화한다. Digital payoff MLMC의
   [path branching 방법](https://arxiv.org/abs/2209.03017)을 rough-Volterra augmented
   state에 맞게 새로 정식화해야 한다.
2. **Payoff hierarchy 재설계**: 동일 hard event의 level difference 대신, conditional
   survival/occupation probability를 telescope하고 최종 level에서 hard target과 연결하는
   exact 또는 controlled-bias hierarchy를 연구한다.
3. 위 둘 중 하나가 먼저 자연 보정분산의 양의 decay rate를 보일 때만 correction-CEM을
   다시 결합한다.

이 순서를 지키지 않고 control 차원, neural memory, quantum-inspired 표현을 추가하면
현재 확인된 coupling 병목을 해결하지 못한 채 학습비용만 증가할 가능성이 높다.

## 8. 재현 명령

```bash
python -m experiments.g7_controlled_mlmc \
  --output results/g7_controlled_mlmc_frozen_2026-07-17.json
```

```bash
python -m pytest -q
```

Primary artifact:
`results/g7_controlled_mlmc_frozen_2026-07-17.json`
