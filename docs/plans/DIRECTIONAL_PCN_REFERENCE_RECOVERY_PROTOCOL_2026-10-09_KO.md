# Directional pCN reference recovery: 새 개발 pilot

앞선 두 물리 reference pilot과 별개로, natural/proposal-start SMC의 mutation만 개선한다.
정밀도 기준, 원본 q와 target은 유지한다. 기존 pilot 결과는 모두 남기고 새 namespace를 쓴다.

## 수학적 정당성

단위 Volterra 방향 u와 독립 next-price 방향 v에 대해 U=(u,v), U^T U=I이다.
z'=z+U[(sqrt(1-s²)-1)U^T z+s epsilon], epsilon~N(0,I2).
직교 여공간을 고정하고 두 차원만 pCN으로 이동하므로 표준 Gaussian p에 가역적이다.
방향을 현재 state와 무관하게 균등 선택한 mixture도 p에 가역적이다.
따라서 natural bridge MH ratio는 beta*(logH'-logH), proposal bridge는 이에
(1-beta)*(logD_r'-logD_r)를 더한다. 방향 선택을 state-dependent로 바꾸면 이 증명이
유효하지 않으므로 금지한다. global r independence move의 별도 ratio는 유지한다.

## 실행 고정

원본 2셀×parent0 q×mu/M2×natural/proposal-start, 각 8 whole SMC.
particle1024, quadratic64 temperatures, 4 mutations, pCN .35,
stratified resampling every4 stages, global every2 mutations, 마지막 단계 mutation 없음.
매 단계 mutation0=full pCN, 1=global, 2=rank-two directional pCN, 3=global.
정확한 조건부 excursion r(모든 time, thresholds4/5/6, next-price1, defense.1) 유지.
기존 IID reference와 합쳐 precision을 부풀리지 않기 위해 새 IID 표본도 같이 생성한다.
새 namespace `conditional-excursion-directional`. 동일 320단위, 17,367,296 potentials,
phase work<=40m, wall<=1800초, RSS<=4GiB.

pilot는 학습 bank나 최종 confirmation이 아니다. 8 whole-run에서 추정한 낮은 RSE만으로
success를 선언하지 않는다. 전체 2셀×5q production은 모든 원본 q의 fresh pilot 및
사전에 정한 고정 precision allocation/독립 검증 후에만 허용한다.
조건부 excursion은 fitted guide와 독립인 구조 기반 대조이지만 natural/proposal SMC가
같은 global proposal을 쓰므로 blind spot 가능성이 사라졌다는 주장은 금지한다.
