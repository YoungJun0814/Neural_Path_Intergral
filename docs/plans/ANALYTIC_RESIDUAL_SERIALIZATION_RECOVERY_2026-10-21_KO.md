# 해석적 residual 실행 저장 오류 복구 부록

2026-10-21. original 계산이 종료된 뒤 result JSON 저장에서 numpy.bool_ 직렬화 오류가 났다.
원래 run/source/protocol과 실패 receipt를 보존한다. 원래 계획이나 수치 core를 수정하지 않는다.
첫 결과의 숫자/개별 timing은 저장되지 않아 복구하지 못했다. 성능 실패·성공으로 단정하지 않는다.

## 변경 범위

- 새 wrapper에서 numpy scalar만 Python builtin scalar로 바꾼다. 수치값/threshold/order/순서는 동일하다.
- 새 known-answer/수치 결과 payload 전체의 canonical JSON 직렬화 테스트를 실행 전에 추가한다.
- 새 recovery 디렉터리의 source를 별도 동결한다. one-shot/실패비용 기록을 유지한다.
- 262 profile,3회AB/BA/AB의kernel측정은같이재실행한다. 첫측정값중좋은결과를고른것이아니라
  저장오류로모든첫결과가유실된것이다. 재실행자체가사후절차변경임을명시한다.

## 보수적 비용과 joint 범위 축소

첫 kernel의정확평가횟수는유실되어full cap12,000,000을사용예약으로보존한다.
recoverykernel은새cap12,000,000/wall600초, auditcap3,000,000/wall120초다.
이는원래총scalar예산이늘어난복구비용이며원래예산안에서완료한것처럼쓰지않는다.

첫 joint stage60physics/54scalar는저장함수까지도달한controlflow로재구성한다.
원래physical총cap100을넘지않도록recovery는**양셀zero의두context만**full검증한다(20physics).
감사도동일두context를재생한다(20physics). 총60+20+20=100이다.
원래예정한여섯context의완료자료가남았다고주장하지않는다.
삭제된first/lastcontext의성공여부를보고골라낸것이아니다. 데이터대표성주장은없다.

지금검증하는것은scalar수식/배치계산도구·joint법칙의제한된구현일치·finite위험분해다.
원래Q1통계실패/P2/D3보류는그대로다. 추가무작위표본·새G학습·모델성능채택은없다.
