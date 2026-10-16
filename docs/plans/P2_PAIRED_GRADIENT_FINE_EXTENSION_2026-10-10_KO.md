# P2 paired-gradient 미세 격자 확장: 고정된 개발 진단

작성일: 2026-10-10. 아래 설계는 새 실험을 시작하기 전에 고정한다.

## 근거와 목적

첫 32 whole-run에서 canonical의 N512→1024 차이는 -4.167%, paired SE
1.546%, 4개 비교 동시 상한 8.029%였다. high-η는 -10.655%, SE 3.978%,
상한 20.591%로 실패했다. μ 자체의 RSE는 10–18%이므로 이것은 정확한
미세 격자 가격이나 연속시간 bias를 인증한 결과가 아니다.

이번에는 같은 수학적 방법을 N1024→2048 및 N2048→4096에 적용한다.
이전 IID envelope의 낮은 정밀도에 가려졌던 격자 차이를 다른 메커니즘으로
측정한다. 원래 N32 q를 미세 격자 q로 옮기거나 새 학습을 진행하지 않는다.

## 바꾸지 않는 수학적 계약

- joint x=(zf,ε)~N(0,I), dimension=5N/2. 정확한 adjacent Gaussian coupling.
- H=(gf+gc)/2. fixed-temperature SMC product normalizer와 terminal component
  ratios를 함께 사용한다. ratio만 사용하거나 self-normalization하지 않는다.
- gradient proposal의 실제 forward/reverse MH 밀도 보정. drift norm cap20은
  proposal만 제한하며 payoff, target, contribution은 clipping하지 않는다.
- 기존 BLP driver와 adapted Itô left-price law. coarse local L=C1+L2 유지.
- 독립 whole-run 두 출력의 covariance를 포함한 paired delta-method SE.

## 실행 설계와 자원

2 original cells × fine N2048/4096 × 8 whole = 32 whole. 모든 seed는
`paired-fine-gradient-v1` namespace로 새로 생성한다. 실패 seed 교체, prefix 판정,
이전 pilot와 pooling, 결과를 보고 whole 수를 연장하는 행위는 금지한다.

128 particles, 64 quadratic temperature points, stage별 4 moves, scale .35,
every4 unbiased stratified resampling, final mutation 없음. 외부 guide나 bank 없음.
whole당 joint forward/backward 31,872회, physical grid evaluation 63,744회.
총 physical 2,039,808 / joint backward 1,019,904 / time-step forward
4,699,717,632회. 준비 단계는 FFT kernel와 coupling map만 생성해 basis-path call 0회.

cap: physical 3m, joint backward 1.2m, phase wall 4,800초, whole 120초,
준비 30초, RSS 4 GiB. 사전/사후 RSS guard는 true peak-memory certificate가 아니다.
현재 노트북에서 다른 검증과 동시에 실행하므로 wall-time은 공정한 속도 비교가 아니다.

## 판정과 종료

32회 전체가 끝나야 4개 셀-격자의 μ RSE, signed relative difference, paired SE,
10% small-difference 상한, LOO, maximum whole share, ancestry를 보고한다.
성공/실패 모두 보존한다. 작은 인접 차이의 통과는 **개발 진단의 통과**일 뿐이다.
μ accuracy qualification, 미관측 mode coverage, 남은 continuum bias는 별개다.

source/anchor/model/seed/grid/work/unnormalized identity 감사와 각 셀-격자 rep0의
whole numerical replay를 수행한다. audit work는 본 실험과 별도로 보고한다.
P1/P2 전체 승인 및 large-training 승인 flag는 이번 결과만으로 true가 될 수 없다.

작은 차이와 안정성이 관측되면 새 precision allocation 및 독립 미세 격자 reference를
먼저 설계한다. 유의한 차이가 지속되면 capacity를 늘리기보다 grid law의 bias와
논문 주장 범위를 재검토한다. 이 연구의 극단 digital target에 적용 가능한 weak-error
정리와 상수를 확보하지 않은 상태에서 임의의 H 또는 2H rate를 넣지 않는다.
