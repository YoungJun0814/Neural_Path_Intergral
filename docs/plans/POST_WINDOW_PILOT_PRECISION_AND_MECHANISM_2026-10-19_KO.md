# Window pilot 이후의 정밀도·메커니즘 우선순위

2026-10-19. [실제32block 결과](../reviews/BATTERY_INDEPENDENT_EXECUTION_2026-10-19_KO.md)를
본 뒤 작성한 다음 개발 단계다. 이전 pilot의 사전 계획이나 새 confirmation으로 소급하지 않는다.
여기의 다음 실험들은 아직 실행하지 않았다. 커밋·푸시하지 않는다.

## 1. 결과로 정해진 판단

G1의 fine RSE4.37%/3.05%, G0의48.29%/29.44%는 G1을 남길 충분한 **개발 신호**다.
그러나 G1도1.5% 미달이며, SMC fine RSE6.53–11.97% 때문에 모든 교차 구간이 넓다.
G1과 SMC 평균이 대략 가까운 것은 정확도 인증이 아니다. G0/G1은 vol template/시점과
price 방향을 동시에 바꿨으므로 단일 원인 설명도 아직 안 된다.

## 2. 첫 우선순위: 두 독립 메커니즘의 precision을 함께 설계

1. 현재 두 result와 두 replay receipt SHA를 다음 계획 config에 binding한다.
2. G1의 기존 worst(fine,coarse) RSE/safety3 규칙은 canonical3,338,752와
   high-η2,081,280 **새 draw**를 예측한다. 합계 fine5,420,032, paired physical10,840,064,
   time-step33,300,676,608이다. seed namespace는 기존pilot와 분리한다.
3. 새 IID32,768draw/block을 기본 단위로 한다면 끝의 remainder도 사전 고정한다.
   모든 draw를 동일 가중으로 합치고, unequal-size 마지막 block을 equal-block sensitivity에
   그대로 넣지 않는다. 또는 전체 count를 block 배수로 미리 올림하고 비용을 다시 고정한다.
4. G1만 정밀하게 해도 독립SMC와의 agreement는 해결되지 않는다. 기존 사전 선택 규칙의
   more-temperatures forecast는 새744/463whole, paired physical156,040,960이다.
   이는 노트북에서 큰 비용이며 독립SMC 정밀도 검증 없이 G1만 oracle로 승격하지 않는다.
5. SMC 비용을 감당하기 전에 reference 개선 후보는 최대2개로 제한한 **새 작은 allocation
   pilot**을 별도 봉인한다. 후보는 원래 full-reverse-MH corrected gradient kernel의
   고정 bridge/mutation 배분 변경과, 검증 가능한 조건부 Rao–Blackwell reference다.
   두 번째 후보는 dense all-driver conditioning을N4096에 무리하게 적용하지 않는다.
   새 빠른 solve는 covariance/conditional variance/안정성 및 독립dense 작은grid oracle를
   먼저 통과해야 하며, 새 방법의 장점을 가정한 채 비용 예산을 정하지 않는다.
6. 결과를 본 후 선택된 방법의 최종 count는 **새 confirmation 표본 생성 전에** 고정한다.
   정확성/precision/민감도/독립 agreement가 모두 충족되기 전 wall-speed 비교는 보류한다.

관측 RSE만 보고 종료하는 optional stopping을 사용하지 않는다. 기존 source에 새 알고리즘을
덮어써서 과거 receipt의 runtime identity를 깨지 않고 별도 module/version으로 구현한다.

## 3. 둘째 우선순위: 개선 원인의 matched ablation

기본3arms는 A=기존G0, B=G0의31point-volatility 시점과 방향을 그대로 두고 price만
T/8 future window(만기 절단)로 변경, C=기존G1이다. amplitude/373components/natural.1/
identity covariance/총draw/평가chunk를 맞춘다. A→B는 price 방향 효과, B→C는
volatility template/시점 변경의 결합 효과를 본다. B→C를 window-volatility 하나의
순수 효과라고 부르지 않는다. 모든 시점·sign·에너지·전체density를 표본 전에 고정한다.

작은 독립 dense driver operator, exact mixture sampling/density, unit norm/causality,
dyadic physical embedding을 먼저 검사한다. 새namespace와고정count를 사용하고 동일
development셀에서 튜닝한 것이므로 저널용미사용task confirmation과 구분한다.
이 ablation은 μ precision gate를 대신하지 않으며 G0를 수억draw까지 연장하지 않는다.

## 4. 셋째 우선순위: 기여 경로 기록

새 pilot에서만 mixture label, template/time/amplitude, log contribution, I/J,
최대volatility/시점 및 외부 geometry 거리의 bounded-memory sufficient summary를 저장한다.
label별 기여는 full-mixture importance weight를 유지한다. component density로 바꾸지 않는다.
mode clustering과 미관측tail 보증을 구분하고, threshold는 독립 calibration에서 고정한다.
현재 moments-only archive에서 존재하지 않는 path/label을 복원했다고 하지 않는다.
replay로 새 진단을 만들면 비용과 비독립성을 명시하고 새confirmation으로 세지 않는다.

## 5. 다음 gate와 금지된 도약

- 둘 이상의 독립 메커니즘에서 충분한 reference precision/감도/동등성.
- 그 뒤에도 continuum relative bias는 별도 문제다. adjacent차이와 느슨한 절대상한은
  상대bias certificate가 아니며, 고정N논문으로 scope를 좁히는 선택지도 남긴다.
- 원래q의 training bank, 새fit, μ/M2 및전체학습반복은 별도새표본으로검증한다.
- offline/fit/selection/inference 포함 같은accuracy 총비용은 위자격 이후에만 비교한다.
- 신경망/operator/차원추가나 top-journal 독창성 주장은 이gate를 통과시켜 주지 않는다.

기술 검증 통과와 “오류가 전혀 없음”은 동일하지 않다. 새단계마다 작은oracle→국소회귀→
소프트웨어전체검사→고정pilot→재현감사→독립확인 순서로 근거를 쌓는다.
