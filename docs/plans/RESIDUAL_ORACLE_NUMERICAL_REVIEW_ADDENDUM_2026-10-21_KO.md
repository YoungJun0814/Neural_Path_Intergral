# Residual oracle — 수치 검증 오류에 대한 사후 부록

2026-10-21. 첫 production 결과를 관찰한 **뒤에** 작성한 구현 검토다. 사전등록 검정의 재통과로 해석하지 않는다.

## 보존하는 실패

`exact_residual_structure_v1/session_20261021/result.json`의 판정은
`INVALID_STRUCTURAL_ORACLE`다. production 66회와 원본 source/계획/결과를 수정하지 않는다.
원래 `--audit`는 canonical/discovery의 첫 probe에서
`saved scalar probe/density identity changed`로 중단되었다.

모든 mean의 61/63번 성분이 정확히 0인데, 같은 좌표를 바꾼 dense GEMM 결과에
`torch.equal`을 요구했다. 추가적인 **밀도 계산만** 수행한 조사에서 log-density 차이는
최대 3.553e-15였다. q/G 또는 payoff 모델의 좌표 의존이 발견된 것과 구별한다.

## 검토 방법

1. 원래 result/source/상위 결과 hash를 확인한다. 원래 INVALID와 실패를 새 receipt에 함께 보존한다.
2. 좌표 j를 제외한 context들이 정확히 같고 `means[:,j] == 0`인지 직접 확인한다.
   identity-covariance shift mixture의 log ratio는 `logsumexp(log(w)+z·m-||m||²/2)`이므로
   j와 무관하다는 대수적 항등식을 사용한다. 수치적으로 작은 차이만으로 이 항등식을 추론하지 않는다.
3. dense log-density 차이와 scale-aware float64 진단 허용폭을 함께 기록한다.
   허용폭은 `512*eps*(1+max Σ|z_i m_i|+.5 max ||m||²+max |log w|)`다.
   이는 BLAS 구현의 rigorous interval error bound라는 주장이 아니다.
4. 고정 여섯 context의 scalar식/모든 quadrature를 재계산하고, 양 셀의 zero context만
   전체 evaluator로 재생한다. 새로운 context/좌표/표본을 고르지 않는다.
5. 원래 판정은 INVALID로 유지한다. 별도 판정은
   `POSTHOC_STRUCTURAL_IDENTITY_VERIFIED_ORIGINAL_GATE_INVALID`로 한정한다.
   population M2, 새 proposal 채택, 성능 개선, P2/D3의 근거로 사용하지 않는다.

## 비용과 한계

- 최초 production 66회 + 중단된 audit 11회 + 본 부록 full replay 22회 = **99회**, 기존 cap100 이내.
- 중단된 audit의 11회와 scalar 수는 원래 코드의 실패 지점/control flow에서 **재구성**한 값이다.
  원래 audit가 완료 receipt를 남긴 것처럼 쓰지 않는다. 비용 확인을 위해 다시 full run하지 않는다.
- 본 검토 cap은 physical22, combined scalar50,000, 검토 계산 wall900초다.
  source 동결/입출력 전체 wall-time을 완전 계측했다는 주장은 하지 않는다.
- zero context 두 개의 작은 conditional moment는 absolute-only 수렴이다.
  그 context의 큰 conditional M2/mu² 수치를 상대정밀도가 검증된 효과크기로 사용하지 않는다.
- 나머지 네 context도 이미 관측한 경로다. population 대표성/독립 confirmation은 없다.

이 부록의 구현과 테스트를 먼저 고정하고 한 번 실행한다. 원래 코드를 고쳐 결과를 덮어쓰지 않는다.
