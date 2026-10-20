# G0/G1 마지막 고정 블록 중단 복구

2026-10-19. 원래 battery-independent pilot는31complete와 마지막 high-eta200-k1/
mixed-window/rep7의 start-only reservation을 남기고 프로세스가 종료됐다.
종료 원인은 외부 중단과 시간적으로 겹치지만 OS/process 원인을 확정할 로그는 없다.
원래32개 전체 protocol pass/result는 없으며 그 상태를 보존한다.

완성된31개는 성능 수치로 선택하지 않고 원래 고정 membership/source/checksum/seed/
geometry/work 규칙으로 검증해 재사용한다. 마지막1개는 동일 namespace·seed·표본수·
chunk·dtype·8threads·likelihood·physics로 처음부터 재계산한다. 중간 기여를 사용하지 않는다.
새 directory는 physical_window_reference_recovery_v1/session_20261019다.
새 nonlinear physical32768/density16384/time-step100663296, prep124linear rows다.
원래 interrupted reservation physical32768/density16384/time-step100663296은 별도
conservative discarded cost로 기록한다. 원래 source/data/receipt는 변경하지 않는다.

새 phase cap1800초/work32768, prep60초/unit180초/RSS4GiB를 유지한다.
배터리 조회 없음, OS 설정 변경 없음, 커밋·푸시 없음. 완료 후 전체32개의 원래 요약과
각cell/guide rep0의4블록 replay(physical131072/density65536)를 수행한다.
이는 고정 grid의 수치 복구이지 새 confirmation/벽시계 benchmark/qualification이 아니다.
교차 comparator는 원래 미완성 result 대신 이 별도 복구 result와 replay receipt를 사용한다.
비교 family/alpha/margin/precision/training flags는 변경하지 않는다.
