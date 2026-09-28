# 감사 기반 R1: 성능 병목 분리 실험과 이론·기술 재검토

작성일: 2026-09-28

연결 계획: [감사 기반 실행계획 R1](../plans/POST_AUDIT_RESEARCH_EXECUTION_PLAN_2026-09-27_KO.md)

판정: **R1의 개발 진단을 실행했다. 탐색 bank의 퇴화는 확인했지만, rank·basis·목적함수 중 하나를 새 모델의 확정 해법으로 채택할 증거는 없다. R2 대형 모델 구현 gate는 통과하지 않았다.**

## 1. 무엇을 실행했는가

- `N=32`에서 canonical `H=.05,K=1`, `K=2`, regular `H=.12`, `η=2`, `ρ=-.9`, joint `H=.12,η=2,ρ=-.9` 6개 개발 셀을 사용했다. 각 셀에서 독립 CE 학습 seed 5개, 조건부 2N 학습 bank 512개, 방법별 독립 held-out 1,024개를 생성했다. 3개 대표 셀에서는 rank `8/16/24/48/64`, 고정 DCT·target-weighted PCA·FIS gradient matrix·실험적 risk-PCA를 같은 bank에서 맞췄다.
- `N=16` canonical에서 rank `8/16/24/32`와 5개 학습 seed를 별도 sanity check로 수행했다. `rank=32`는 해당 `2N` local 공간 전체이지 연속시간 경로공간의 full rank가 아니다.
- 첫 대조는 동일 제안분포와 local path에서 hard indicator에 독립 price noise를 뽑는 raw 추정, 그 price noise를 해석적으로 적분한 conditional 추정이다. proposal 자체의 차이와 conditioning 차이를 혼동하지 않도록 같은 local draw를 사용했다.
- `U=BᵀZ`와 reference complement `V=(I-BBᵀ)Z`를 분리하여 inner `16/64/256`, outer 32개×독립 반복 3개로 `m₁,m₂`와 plug-in floor를 조사했다. outer는 알려진 shifted Gaussian `G_U`에서 뽑고 `φ_U/G_U`를 정확히 반영했다.
- 독립 SMC bank 2개씩에 stage별 ESS, 최대 incremental weight 비중, resampling, 원래 조상 수, pCN acceptance, log-normalizer 경로를 저장했다. 원인 상호작용을 확인하기 위해 대표 3개 셀에 CE 12,288회와 SMC 12,544회로 **학습 payoff 평가 횟수**를 2.1% 이내로 맞춘 추가 실험을 했다. SMC pCN scale `.35/.75`, 각 방법의 독립 held-out 2,048개, 5개 학습 seed, 학습과 분리된 SMC 기준 추정 32회다.
- `defensive only / learned only / defensive+learned / defensive+safety+learned`를 첫 학습 seed에서 탐색적으로 분해했다. `learned only`는 이 유한차원 identity-covariance Gaussian shift에서는 전역 support와 유한 second moment를 갖지만 `p/q≤1/δ`의 방어 상한은 없다. safety 성분을 넣으면 다른 성분의 mixture mass가 바뀌므로 단일 효과에 관한 결론은 내리지 않았다.

재실행 결과: [N=32 6셀](../../results/post_audit/r1_diagnostics_v1.json), [N=16 sanity](../../results/post_audit/r1_n16_sanity_v1.json), [동일 payoff 예산 bank 대조](../../results/post_audit/r1_bank_followup_v1.json). 세 파일은 전부 `development_not_confirmation`으로 표시된다.

## 2. 주요 관측: bank와 최종 추정의 불안정

`N=32`의 512개 CE bank에서 `g p/q_bank`로 계산한 목표 ESS 평균은 셀별로 약 **1.4–4.0**이다. 이때 한두 경로가 weighted PCA, FIS target 가중치, M₂ empirical fit을 사실상 지배할 수 있다. 대표 셀에서 독립 SMC bank는 각 256개 초기 입자 중 최종 원래 조상이 보통 **3–8개**였다. incremental ESS가 약 `.88–.92`로 높더라도 48회 반복 resampling으로 계보가 축소된 것이다. 이것은 탐색 다양성 문제의 관측이지, 특정한 실제 tail mode를 놓쳤다는 증명은 아니다.

동일 payoff-budget 추가 대조의 중앙값은 아래와 같다. `log μ`는 확률 추정의 자연로그이고 RSE는 그 평가 실행 하나의 **표본 기반 상대 표준오차**다. `reference RSE=.21`인 canonical 기준은 정확도 판정에 충분하지 않다.

| 개발 셀 | 독립 SMC 기준 `log μ` (RSE) | CE 중앙 `log μ` / RSE | SMC `.35` 중앙 `log μ` / RSE | SMC `.75` 중앙 `log μ` / RSE |
|---|---:|---:|---:|---:|
| canonical | `-16.80` (`.21`) | `-17.06` / `.39` | `-17.57` / `.23` | `-17.64` / `.32` |
| 높은 η | `-12.03` (`.08`) | `-12.62` / `.39` | `-12.56` / `.32` | `-13.86` / `.30` |
| joint | `-10.33` (`.07`) | `-10.35` / `.16` | `-10.36` / `.11` | `-10.18` / `.22` |

5개 seed의 중앙값을 나열한 표이지, 중앙값끼리의 평균·표준오차를 계산한 최종 성능 순위가 아니다. 특히 높은 η의 개별 실행에는 기준과 여러 표준오차 이상 떨어지는 결과가 반복된다. canonical도 기준 RSE와 방법 RSE가 모두 커서 근소한 차이의 의미를 판단할 수 없다. joint의 SMC `.35`는 신호가 있지만 독립 confirmation, 강한 비교기 및 training-inclusive fixed-precision 검정을 통과한 결과가 아니다.

R0에서 고정한 공통 자격 규칙(방법 RSE≤`.25`, 독립 기준 RSE≤`.10`, `1.96`배 결합 SE를 포함한 차이 상한≤기준의 `.25`)을 각 학습 반복에 적용하면, 세 셀의 CE·SMC `.35`·SMC `.75` **모두 0/5회 qualified**다. canonical은 기준부터 정밀도 gate에 실패한다. 이 결과가 중앙값 표의 겉보기 승자를 공식적인 성능 우위로 승격시키지 않는 직접적 이유다.

CE 학습을 1,536회에서 12,288회로 늘리자 일부 셀의 추정이 개선됐다. 따라서 첫 rank sweep의 낮은 ESS가 **학습 예산·bank 품질과 얽혀** 있고, 단순히 basis만 확대해 해결됐다고 볼 수 없다. `N=16` full local rank 32조차 5개 seed의 held-out `log μ`가 약 `-14.40`부터 `-20.52`까지 흔들렸다. 두 SMC 개발 추정은 `-15.69`, `-17.42`였지만 기준 정밀도가 낮아 정확도 자격은 없다.

## 3. 가설별 R1 gate

| 가설 | R1 판정 | 근거와 제한 |
|---|---|---|
| price-noise conditioning이 유용하다 | **이론적으로 supported** | 같은 local proposal의 second moment 차이는 `E_q[(p/q)² g(1-g)]≥0`. raw/conditional을 같은 local draws에서 구현했고 기존 conditional-payoff oracle 및 새 gradient oracle을 통과했다. 작은 표본의 관측 M₂가 매번 이 순서를 따를 필요는 없다. |
| 고정 저차원 subspace 누락이 주요 병목이다 | **unresolved** | full-rank·PCA에서 held-out M₂가 낮아 보이는 seed가 있지만 추정 평균 자체가 독립 기준보다 여러 자릿수 작아지는 사례가 있다. 높은 η 첫 seed의 PCA `log μ≈-22.87`, 독립 기준 `≈-12.03`: 이를 분산 개선으로 간주하면 오류다. 낮은 training ESS와 미발견 tail 기여를 분리할 수 없다. |
| KL 대비 empirical M₂ 적합이 일관되게 낫다 | **not_supported in tested family/budget** | 같은 DCT·defensive identity-covariance family와 같은 bank에서 M₂ 적합이 KL보다 낮은 held-out log-M₂를 얻은 seed 수는 canonical `2/5`, K2 `2/5`, regular H `4/5`, 높은 η `0/5`, 강한 ρ `1/5`, joint `2/5`. 이는 보편적 M₂ 원리의 반증이 아니라 이 극도로 퇴화한 bank에서의 경험적 이득 부재다. |
| mode/rare-path 탐색 품질이 병목이다 | **operationally supported; mode omission itself unresolved** | CE 목표 ESS `1.4–4.0/512`, SMC 계보 `3–8/256`, scale `.35/.75`의 결과 변동이 확인됐다. 동일 payoff-budget에서 SMC `.35`의 held-out RSE가 joint에서 감소했으나 높은 η·canonical 정확성은 불안정하고 SMC 입자는 상관돼 있다. |
| 계산비용이 우위를 상쇄한다 | **unresolved** | 학습 payoff 수는 맞췄고 stage wall을 저장했다. 따뜻한 실행 기준 CE fit 대략 `0.19초`, SMC fit 대략 `0.27초`지만 첫 최적화 호출의 warmup, 노트북 전력 상태, 작은 측정 시간이 통제되지 않았다. 자격 있는 fixed-precision 방법이 없어 training-inclusive 속도비를 산출하지 않았다. |

정리하면 **병목 후보의 우선순위는 bank/rare-path 탐색 안정성 → 정확한 독립 기준과 held-out 정밀도 → 그 후 rank/basis/objective**다. R1 gate는 원인의 일부를 확인했지만 기하·목적함수의 독립 효과가 미해결이므로 R2의 큰 mixture/operator를 추가하는 판단은 보류한다.

## 4. 이론 점검

1. Frozen `N`의 rBergomi BLP local whitened 좌표는 `2N`, 독립 price 좌표는 `N`이다. 최종 payoff `g_N(z)=P(H=1|Z_local=z)`는 0–1 사이의 조건부 정규 CDF이며, 최종 추정은 항상 `n⁻¹Σg_N(z_i)p(z_i)/q(z_i)`다. 학습 가중치 정규화와 self-normalized 최종 IS를 혼동하지 않았다.
2. FIS 행렬은 `E_{p g/μ}[∇log g ∇log gᵀ]`이고 이산 Gaussian prior가 whitened이므로 ordinary eigenspace를 사용한다. [Uribe et al., 식 (3.7)](https://arxiv.org/pdf/2006.05496)의 smooth-indicator gradient second moment와 대응하지만, 본 코드는 **최종 conditional CDF의 FIS 행렬 진단**만 구현했다. 논문의 annealed iCEred 전체 알고리즘을 비교했다고 주장하지 않는다.
3. 같은 mixture family에서 KL fit은 `-E_{pg/μ} log(q_θ/p)`의 bank 근사, M₂ fit은 `E_{q_bank}[g²p²/(q_bank q_θ)]`의 bank 근사다. optimizer가 반환하는 마지막 제안분포의 objective를 다시 평가하도록 수정했다. 둘 다 bank 표본을 과적합할 수 있어서 새 최종 표본으로만 평가했다.
4. Nested floor `[(E_{φ_U}√m₂(U))]²`는 **reference complement를 그대로 둔** `q_U φ_V` family에만 해당한다. `√hat m₂`의 Jensen 편향, outer 제곱·비율 편향, 이중 Monte Carlo 오차가 있다. 실제 3개 대표 셀의 inner 16→256 추정 log-floor가 1–5 이상 움직이거나 outer 반복 간 spread가 약 3–13까지 나타났다. 따라서 lower certificate, V5 전체 mixture 하한 또는 성공한 new theorem으로 사용하지 않는다. noisy-IS 최적 proposal과의 관계는 [Llorente et al.](https://arxiv.org/pdf/2201.02432)와 대조해야 한다.
5. `q=δp+(1-δ)q_learned`의 모든 성분은 정규화된 밀도를 정확히 합산하며 `p/q≤1/δ`를 갖는다. learned-only identity Gaussian shift는 finite-grid에서 support 문제는 없고 `∫p²/q=exp(||m||²)<∞`이지만 균일 방어 상한이 없다. SMC의 높은 단계별 ESS나 final cluster 모양만으로 mode 완전성을 선언하지 않는다.

## 5. 기술 검토·재현성·남은 한계

- [진단 모듈](../../src/path_integral/r1_bottleneck_diagnostics.py)은 exact density, FIS matrix, KL/M₂ 동일 family fit, zero-hit을 `null`로 처리하는 log-domain 요약, `U/V` Gaussian decomposition을 구현한다. [SMC](../../src/path_integral/tempered_conditional_smc.py)는 원래 조상 수·최대 단계 가중치 비중·resampling·acceptance·log-normalizer 이력을 추가로 남긴다.
- [동일-bank 실행기](../../experiments/post_audit_r1_diagnostics.py), [학습-budget 추가 대조](../../experiments/post_audit_r1_bank_followup.py), [전면 replay 감사기](../../experiments/post_audit_r1_audit.py)는 서로 다른 용도다. 각 결과에는 config/source-tree digest, 역할별 seed ledger, 제안분포의 모든 Gaussian 파라미터·digest, 단계 비용과 독립 최종 요약이 포함된다. 결과를 그대로 전면 재실행해 timing을 제외한 수치·제안분포·seed가 일치하는지 검사한다. 이는 코드/결과 무결성이지 confidence interval의 진실성 보장이 아니다.
- Oracle 테스트는 mixture의 `log(q/p)`를 독립 dense 식과 대조하고, FIS와 PCA를 구분하고, KL/M₂ 기록 loss를 반환 proposal에서 재계산하고, log-payoff autodiff와 finite difference를 비교하며, 알려진 1방향 toy에서 complement variance가 0인 nested identity를 확인한다. 새 테스트와 SMC 회귀, Ruff 및 전체 회귀 검사를 분리해 기록한다.
- 학습 seed 5개는 실패율 인증에 부족하다. 같은 bank를 여러 fit에 공유한 표는 *기하 진단*이지 독립 end-to-end 비교가 아니다. SMC resampled particles는 IID가 아니며 최종 IS에 그대로 사용하지 않았다. 독립 최종 IS 추정의 심한 tail concentration 때문에 단일 held-out M₂의 낮은 값만으로 승자를 고르지 않는다.
- 여러 basis의 `rank=2N`은 이론상 같은 전체 평균-shift family를 표현하지만, 유한회 Adam은 회전좌표에 불변인 최적화기가 아니다. full-rank basis 사이의 유한반복 차이를 subspace 효과라고 해석하지 않는다. 본 R1의 rank/basis 결론을 보류하는 추가 이유이며, 후속 실험에는 수렴 기준·다중 초기값·동일한 좌표불변 최적화 점검이 필요하다.
- 모든 실험은 노트북 CPU 개발 실행이다. 상대 wall-time, power mode, warmup, thread interference가 정밀하게 고정되지 않았다. 학습·gradient·SMC 비용은 기록했지만 최종 저널급 비용 우위 주장은 없다. source manifest는 실행 당시 dirty tree를 기록한다. 저장된 code-tree digest와 최종 수정 소스의 재실행 일치 여부를 별도로 감사한다.

검증 기록: `ruff check src tests experiments main.py train_driftnet.py` 통과, `python -m pytest -q` **1,068개 통과**. R1/SMC 집중 테스트는 10개 통과했다. `git diff --check`도 통과했다. 이 로컬 Python 환경에는 `mypy` 모듈이 없어 정적 타입 검사는 실행하지 못했다. 이 사실을 타입 검사 통과로 바꾸어 쓰지 않는다. 최종 세 결과의 runtime source-tree digest는 동일하게 `5fd6da36760a0166ace65bd1abfd2e32e7598f25f385e59ee73f8ef24c7269ba`다. 세 결과 모두 전면 수치 replay에서 저장된 값·proposal·seed ledger가 일치해 **무결성 검사 통과**했다. 이는 통계적 유효성이나 모델 우월성 판정이 아니다.

## 6. 다음 연구 결정

R2 모델 복잡도를 늘리기 전에 작은 범위에서 다음 조건을 충족해야 한다.

1. 높은 η와 canonical에 독립 SMC 기준의 RSE를 충분히 낮추고, 다른 메커니즘의 기준 estimator와 상호 검증한다. SMC에서 3–8명 조상으로 수렴하는 현상을 완화할 bridge/resampling·mutation 대조를 사전 고정한다.
2. 동일 2N conditional target에서 CE·SMC bank의 training ESS/ancestry를 실질적으로 개선한 후, rank/basis/KL-M₂ ablation을 **새 bank·새 최종 표본**으로 재실행한다. 희귀 기여 모드 탐색이 실제로 개선되는지 별도 clustering과 tail-contribution 분석으로 점검한다.
3. 최소 두 방법이 동일 accuracy 자격을 얻은 셀에서만 fixed-precision total wall-time(offline, fit, selection, inference)을 비교한다. 저널용 데이터는 새로 봉인한 confirmation 셀과 더 많은 독립 학습 반복으로만 생산한다.

현재로서는 차원을 늘리거나 새로운 operator를 붙이는 것보다 **잘못 낮아 보이는 분산을 걸러내고 학습 bank를 신뢰할 수 있게 만드는 일**이 우선이다.

재현 명령:

```powershell
python -m experiments.post_audit_r1_diagnostics --config configs/post_audit/r1_n16_sanity_v1.yaml
python -m experiments.post_audit_r1_diagnostics --config configs/post_audit/r1_diagnostics_v1.yaml
python -m experiments.post_audit_r1_bank_followup --config configs/post_audit/r1_bank_followup_v1.yaml
python -m experiments.post_audit_r1_audit results/post_audit/r1_n16_sanity_v1.json results/post_audit/r1_diagnostics_v1.json results/post_audit/r1_bank_followup_v1.json
```

실행기는 기존 결과를 덮어쓰지 않는다. 동일 경로 재생성이 필요하면 연구자가 먼저 기존 파일을 보존 또는 명시적으로 정리한 뒤 실행해야 한다.
