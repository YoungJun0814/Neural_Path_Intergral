# 우선순위 실행 결과: 독립 coverage와 exact joint conditional

작성일 2026-10-09. 기준: [사전 고정 프로토콜](../plans/REFERENCE_COVERAGE_AND_JOINT_CONDITIONAL_PROTOCOL_2026-10-09_KO.md), [수학 검토](../theory/REFERENCE_COVERAGE_JOINT_CONDITIONAL_REVIEW_2026-10-09_KO.md).

## 1. 결론

**exact conditional 기반과 guide-independent 대조군을 구현하고 고정 실험을 완료했다. 그러나 실제 Volterra reference 병목은 해결하지 못했다.**

새 local variance-inflation mixture는 정답을 아는 shifted/bimodal toy의 rare contribution 탐색을 개선했다. 반면 실제 canonical·높은 η 셀에서는 기존 guide보다 RSE와 기여 집중도가 훨씬 나빴다. 이 방법을 production reference로 채택하지 않는다. 이 실패는 한 가지 고정 family의 결과이지 모든 covariance proposal 또는 연구 전체의 불가능성을 뜻하지 않는다.

현재 올바른 방향은 모델을 더 크게 만드는 것이 아니라 **여러 시간대에 걸친 collective contribution을 신뢰성 있게 탐색하고 평가하는 것**이다. 이 원인 설명은 이론과 관측이 일치하는 가설이며 지배 mode를 완전히 식별했다는 증명은 아니다.

## 2. 우선순위별 처리

| 이전 우선순위 | 이번 처리 | 상태 |
|---|---|---|
| 독립 rare contribution coverage 개선 | q/guide fitting을 읽지 않는 평균 0 local scale mixture 구현·실험 | toy 개선, 실제 실패 |
| cached evaluator·정확한 target 유지 | 기존 cache 유지, original simulator로 contribution 재검산 | 구현 검증 통과 |
| full-r static M₂ baseline 유지 | 고정 q의 기존 guide와 fresh IID 비교 | 유지, truth로 자기 인증 금지 |
| joint Gaussian conditional 도출 | arbitrary fixed orthogonal split의 exact posterior·밀도·inner sampler 구현 | 수학/단위 검사 완료, 실제 nested production 미배치 |
| 독립 대안 하나 사전 고정 | λ=25, δ=0.1, 모든 local block, 별도 seed | 선택·연장·sweep 없이 완료 |
| P1 이후 P2/confirmation | 정확도 관문 미충족 | 계속 잠금 |

## 3. 구현한 내용

- `local_variance_reference.py`: disjoint block 중 하나만 covariance λ배 하는 normalized defensive mixture. 모든 component의 밀도를 logsumexp로 합산한다. mean shift, fitted q, 과거 bank 또는 guide의 mode 정보는 사용하지 않는다.
- `joint_shift_conditional.py`: identity-covariance shift-mixture를 고정 orthogonal U에서 분할한다. α_j(A), r_A/p_A, r(B|A)/p_B, split/join 및 조건부 inner sampler를 제공한다. nonzero covariance rank와 표본별 회전을 허용하지 않는다.
- `post_audit_v2_reference_coverage_joint.py`: 고정 config, source ZIP, q/input hash, stream identity, append-only raw shard 및 정확도 gate를 결합한다. 원래 M₂의 분모는 q 그대로다.
- 테스트: SciPy logpdf·normalization·sampling moments·orthogonal factorization·conditional posterior·원래 risk integral·seed/config·비용 gate·감사 thread 설정을 검사한다.

이번 conditional core가 일반 Volterra H_q의 exact conditional integral을 계산한다는 뜻은 아니다. future path를 변경하는 joint block은 원래 비선형 Volterra 경로를 다시 계산해야 한다. 실제 nested sampler의 비용과 outer coverage는 다음 단계의 별도 연구다.

## 4. 실험 설계 및 자원

toy 3종은 natural IID vs local-scale, 실제 2셀은 guide IID vs local-scale로 비교했다. 방법별 8개 독립 반복 × 32,768 draw, batch 4,096이다. 방법·반복·batch의 stream은 분리했다. 같은 draw에서 μ와 M₂를 계산하므로 두 target의 오차는 **paired**이며 독립이라고 주장하지 않는다.

- 비교 단위 640개, 별도 preparation checkpoint 1개.
- toy potential calls: 1,572,864회.
- 실제 Volterra calls: 1,048,576 표본 + original oracle 64 + guide 준비 128 = **1,048,768회**.
- preparation·I/O·summary 포함 phase: **195.65초**, 900초 상한 이내.
- 입력 읽기·source archive 포함, 최종 result write 제외 시간: **201.34초**.
- 결과 디렉터리: 약 **109.91MB**.

전체 회귀 테스트가 별도 process에서 일부 기간 병행됐다. 난수 stream·추정값에는 공유 상태가 없지만, 계산 시간은 resource-isolated benchmark가 아니다. 정확도 자격을 얻은 방법 쌍이 없으므로 이 timing을 fixed-precision winner 선정에 사용하지 않았다. 학습·선택·inference를 포함한 total wall-time 우위는 주장하지 않는다.

원본 과학 run source ZIP SHA256은 `b2b1ecbf158be2158dc43cf52df0001b2561b9044591813379c4d4cffa0286bd`다. 이후 auditor 개선과 원본 run source는 구분한다. 표본은 다시 생성하거나 결과를 덮어쓰지 않았다.

## 5. 정확한 toy 결과

각 행은 μ 또는 q=p M₂의 oracle 상대 오차와 표본 RSE다.

| toy·target | natural 오차 | natural RSE | local-scale 오차 | local-scale RSE |
|---|---:|---:|---:|---:|
| centered μ | +0.058% | 0.113% | +0.301% | 0.352% |
| centered M₂ | +0.168% | 0.175% | +0.339% | 0.437% |
| shifted μ | −94.161% | 46.106% | +1.379% | 1.309% |
| shifted M₂ | −99.981% | 87.998% | +4.680% | 1.831% |
| bimodal μ | −31.210% | 32.903% | −0.626% | 0.932% |
| bimodal M₂ | −96.885% | 80.705% | −2.138% | 1.870% |

centered에서는 natural이 더 효율적이다. local-scale이 모든 문제에 유리한 것은 아니다. shifted/bimodal에서는 natural 표본이 중요한 기여를 충분히 보지 못한다. local-scale은 이를 크게 완화했지만 shifted M₂의 +4.68% 오차를 “오류 0”으로 표현하지 않는다.

추가 해석용으로 12개 endpoint의 Bonferroni normal approximation(z≈2.865)을 사용하면 local-scale 6개 interval은 알려진 truth를 포함했다. natural의 shifted μ/M₂와 bimodal M₂는 포함하지 않았다. 이는 이번 known-truth 실험의 관측이지 반복적인 frequentist coverage 검증이나 일반 tail certificate가 아니다.

고정 R={2.5≤z0≤7.5, |z1|≤2}에서 bimodal μ의 참 기여 질량은 이전 exact oracle 0.65459다. local-scale의 추정 normalized share는 0.65140, natural은 0.49858이었다. M₂ local-scale share는 0.99999750이었다. normalized share가 높아도 전체 normalizer를 과소평가할 수 있다. 실제로 natural bimodal M₂의 영역 share는 0.999923이지만 총 적분은 약 96.9% 낮다.

## 6. 실제 모델 결과

| 셀·target | guide 평균 | guide RSE | local-scale 평균 | local-scale RSE |
|---|---:|---:|---:|---:|
| canonical μ | 5.00645e−8 | 7.042% | 4.81772e−8 | 70.023% |
| canonical M₂ | 1.43363e−9 | 2.112% | 1.42131e−9 | 56.374% |
| 높은 η μ | 4.69496e−6 | 1.783% | 3.18022e−6 | 19.299% |
| 높은 η M₂ | 6.11184e−7 | 1.931% | 6.17982e−7 | 22.471% |

canonical 평균이 비슷해 보인다고 상호 검증이 성공한 것은 아니다. local-scale canonical μ는 단일 draw가 전체 기여의 **67.16%**, M₂는 **52.91%**를 차지했다. leave-one-whole-replicate 상대 변화는 각각 **70.20%/50.42%**였다. 높은 η의 단일 기여 비중도 **11.55%/17.50%**, whole-replicate 민감도는 **21.82%/27.22%**다.

RSE≤2%, 최대 단일 기여≤1%, whole-replicate 민감도≤10%, family-adjusted 10% mean equivalence라는 개발 gate에서 4개 비교 모두 자격을 얻지 못했다. delta-method equivalence 역시 실패했다. 높은 RSE에서 정규근사는 부정확할 수 있으므로 이 수치를 정확한 coverage 증거로 쓰지 않는다.

**비용 승자는 선택하지 않았다. production reference=false, P2=false다.** guide의 높은 η RSE가 낮다는 것만으로 independent truth로 인증하지 않는다. 이전 guide-free SMC 결과의 불안정성도 그대로 유효하다.

## 7. Geometry와 실패 원인

새 raw diagnostic은 stochastic left variance(v1,…,vN−1)의 최대 위치를 4개 시간 구간으로 나눴다. 초기 고정 v0는 제외했으므로 이전 16-mode partition과 동일한 diagnostic이 아니다. 이 정의의 sample-weighted μ/M₂ contribution share는 탐색 관측이지 unseen mode 부재 증명은 아니다.

canonical guide μ의 시간 구간 기여는 약 [1.49, 10.33, 33.79, 54.40]%, M₂는 [18.63, 56.04, 21.15, 4.18]%다. μ와 M₂에 중요한 시간대가 다르다. local-scale canonical M₂는 두 번째 구간에 86.02%가 몰렸으나 단일 draw가 절반 이상인 상황이므로 새로운 안정된 mode라고 해석하지 않는다.

합리적인 실패 가설은 다음과 같다.

1. toy의 중요한 방향은 한 coordinate에 집중되어 있어 한 block 확대가 충분하다.
2. Volterra 기억은 한 local innovation만 아니라 여러 시간대의 조합을 통해 variance·가격 drift·조건부 event probability에 영향을 준다.
3. local-scale은 흩어진 collective 경로에 낮은 density를 줄 수 있다. 수학 검토의 d=64 예시는 이 한계를 보여준다.
4. 단일 극단 draw가 지배하므로 추가 particle island나 동일 family의 단순 표본 확대만으로 해결됐다고 가정할 수 없다.

이는 원인을 뒷받침하는 정합적인 분석이지만 경로의 진짜 multimode 구조를 완전히 규명하지는 못했다. λ/rank를 결과에 맞춰 골라 다시 좋은 숫자만 보고하지 않았다.

## 8. 발견한 기술적 문제와 재검토

첫 저장 통계 재감사에서 exact dictionary equality가 실패했다. 실행기는 thread=1인데 감사기는 기본 thread 수로 reduction했다. 원본 logs는 변하지 않았고, auditor에 같은 thread 설정을 적용하자 동일 통계가 재현됐다. 이 수정은 계산·seed·표본을 바꾼 것이 아니라 감사 실행 환경을 맞춘 것이다.

이후 auditor를 강화해 원래 simulator와 원래 q 밀도로 2셀×2방법의 첫 16개 contribution을 재생성하도록 했다. sampler는 원래 전체 batch 크기로 재생성한 뒤 16개를 택해 label/path RNG consumption을 유지한다. 원본 archive는 보존한다. 이번 실행기는 preparation cache가 process-local이므로 중단 후 무조건 재개를 지원하지 않는다. 완료된 preparation을 재사용해 cache가 없는 경우 명시적으로 차단한다. resume-ready라고 과장하지 않는다.

저장 grid·source archive·q·seed·raw summary·work accounting 감사는 통과했다. 강화된 원래 simulator 감사는 64개 contribution과 guide 준비 128회를 추가 평가했고, 최대 log contribution 차이는 **7.1054e−15**였다. 이는 제한된 replay이며 전체 표본의 형식적 동일성 증명은 아니다.

- 신규 core 및 실행기 초기 검사: 13개 통과. 감사 thread 회귀 추가 후 runner 3개 통과.
- 첫 전체 pytest: 1,267개 통과, 275.37초. 감사 회귀를 포함한 **최종 전체 pytest는 1,268개 통과, 252.62초**다.
- Ruff 전체 범위 통과, CI mypy 183개 source file 통과, 새 experiment direct-target 타입 검사 통과.
- 실행 source와 최종 감사 source는 다르다. 감사 개선 후 runtime source tree digest는 `179cb3a0afb993db1cbfca6556fc42913144646e28b5d89f299c055bbedabaef`다. 원본 science snapshot SHA와 혼동하지 않는다.
- 기존 Requests dependency warning을 보존하고 환경 package를 변경하지 않았다.

## 9. 다음 방향 재설정

1. **single-local-block covariance family는 실제 production 후보에서 제외**한다. toy coverage regression 대조로 유지할 수 있다.
2. guide는 strong development baseline으로 남긴다. 그 평균을 truth로 고정하지 않는다.
3. 다음 후보는 model-law로 정한 **여러 시간대 collective direction의 covariance transport**다. q/bank fitting과 분리하고 원래 q·target을 유지해야 한다. Volterra monitor direction과 미래 가격 innovation의 causal orthogonality, covariance determinant 및 conditional law를 먼저 도출한다.
4. 새 후보는 단 하나를 사전 고정하고, 작은 independent diagnostic으로 검증한다. 현재 λ=25 결과를 보고 최적 λ를 골랐다고 포장하거나 여러 후보 중 성공만 보고하지 않는다.
5. joint conditional core는 여기에 사용할 수 있지만 outer missed mode를 inner 반복만으로 복구하지 못한다. exact law와 유용한 estimator는 서로 다른 완료 조건이다.
6. 서로 다른 메커니즘의 방법이 같은 accuracy gate를 통과한 셀에서만 fixed-precision 전체 비용을 비교한다. 그 전에는 모델 복잡도·새 operator·P2·top-journal novelty 주장을 확대하지 않는다.

현재는 이론적으로 타당한 계산 기반과 병목 반증 실험이 진전된 상태다. 실무 신뢰성 또는 박사/최상위 저널 성능 기여가 완성됐다는 판단은 여전히 불가하다.

## 10. 재현

```powershell
python -m experiments.post_audit_v2_reference_coverage_joint --audit results/post_audit/reference_coverage_joint_v1/session_20261009/result.json
python -m pytest -q
python -m ruff check src tests experiments main.py train_driftnet.py
python -m mypy src main.py train_driftnet.py
python -m mypy --follow-imports=silent experiments/post_audit_v2_reference_coverage_joint.py
```

현재 새 구현·결과·문서는 local 변경으로 유지한다. 이전 push `a0754dd` 및 원본 실패 자료를 변경하거나 main에 병합하지 않았다.
