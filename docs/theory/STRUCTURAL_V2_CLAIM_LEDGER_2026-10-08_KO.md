# Structural V2 이론·주장 계약

작성일: 2026-10-08. P0의 명세·검토 상태이며 새로운 proof certificate가 아니다.

기존 [claim ledger](POST_AUDIT_CLAIM_LEDGER_2026-09-28.md)를 대체하거나 미완료 정리를 완료로 바꾸지 않는다. [통합 계획](../plans/MODEL_STRUCTURAL_IMPROVEMENT_PLAN_V2_2026-10-08_KO.md)의 신규 후보만 별도로 관리한다.

| ID | 명제/주장 | 상태와 정확한 범위 | 실행 관문 |
|---|---|---|---|
| V2-T0 | 조건부 Gaussian terminal payoff | 기존 finite BLP grid·adapted left variance·비퇴화 조건 안에서 유지 | 기존 독립 oracle 회귀 |
| V2-T1 | frozen ordinary IS 평균 | normalized full mixture, fresh IID, integrability 아래 μ_N | role/density 계약 |
| V2-T2 | raw 위험 평가 항등식 | E_r[g²p²/(qr)]=M₂,raw(q); q/r natural floor | 기존 quadrature·constant oracle |
| V2-T3 | signed Gaussian Stein 평균 | 적절한 integration by parts, frozen field, fresh IID일 때만 conditional unbiasedness | P4 field·경계·signed 구현 검증 미완료 |
| V2-T4 | corrected 사건 분산 | Var(μhat_CV)=(M₂,CV−μ_N²)/n; raw M₂와 다름 | P5 independent corrected-risk 평가 미실행 |
| V2-T5 | bounded atom·signed bound | 계획의 Gaussian-window/orthonormal U 수식; full field 검증 전 사용 조건부 | metadata binding은 bound proof가 아님 |
| V2-T6 | convex fixed-mixture fitting | integrability·fixed component 아래 기존 원리 | auxiliary 목적 F=h_q와 사건 F=g 분리 |
| V2-T7 | exact adjacent marginal coupling | coarse-kernel 추가 Gaussian 관측·whitening·단일 joint likelihood | adapter·marginal oracle 미실행 |
| V2-T8 | 독립 raw μ/M₂ 기준값 | 아직 확보했다고 주장하지 않음 | P1 별도 메커니즘·정밀도·일치 |
| V2-T9 | Volterra weighted residual 효율 | 새 정리 후보, 미증명 | kernel·rank·rarity·grid 상수 필요 |
| V2-T10 | 학습의 추가 총비용 이득 | 미입증; static-only 필수 baseline | P3/P5/P7 이후 판단 |
| V2-T11 | continuum bias/efficiency | 기존 T16-5/9/11 review pending 유지 | P2 숫자만으로 증명하지 않음 |
| V2-T12 | task 일반화·상위 저널 준비 | 미입증 | 새 confirmation·novelty·외부 재현 |

schema의 estimand/estimator/SE 일치, source/proposal hash, 서로 다른 seed는 실행 선언의 정합성 검사다. 실행 순서, 실제 독립성, rare-tail coverage, field의 미분가능성이나 bound를 증명하지 않는다.

Stein 보정에서 negative sample을 clipping하면 평균 유지 항등식은 일반적으로 깨진다. signed residual을 positive SMC potential로 그대로 전달하지 않는다. q,r의 정확한 natural 질량과 normalized density 계약은 별도로 유지한다.

auxiliary second moment를 더 정확히 측정해도 original q의 second moment가 작아지는 것은 아니다. auxiliary CV와 event CV는 서로 다른 학습 목적·control binding을 가진다.

전이 조사에서 관측된 음의 Hessian은 #139의 global log-concavity 가정 직접 적용에 반대되는 수치 근거다. 원래 금융모형이 수학적으로 틀렸다는 결론은 아니다. #374의 compact Brenier theorem도 Gaussian whole-space M₂ bound로 쓰지 않는다.

P0 완료의 의미는 증거 inventory·명세·감사 기반을 구축했다는 것이다. P1/P2/P4/P5의 scientific 관문을 통과했다는 뜻이 아니다.
