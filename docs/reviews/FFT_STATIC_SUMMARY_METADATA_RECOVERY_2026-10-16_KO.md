# FFT static pilot: 완료 표본의 집계 메타데이터 복구

2026-10-16. 16개 inference blocks와 prep가 모두 complete 상태로 종료된 뒤 final
summarize에서 `KeyError: log_maximum`이 발생했다. recursive merge helper의 출력에
maximum_fraction은 있지만 log_maximum이 없어 두 번째 merge가 실패한 것이 원인이다.
이는 sampling/density/target 오류가 아니라 미검출 집계 인터페이스 오류다.

기존 source ZIP, manifest, 16 finish는 바꾸지 않는다. 원래 result.json은 없는 상태를
유지한다. 새 `recovered-result.json`과 별도 recovery source/config를 작성한다.
모든 raw finish의 SHA를 묶고 완료 전체를 사용한다. seed 교체/추가표본/pooling 없음.

각 block에 log_sum, log_sum_squares, count, log_mean, maximum_fraction가 남아 있다.
log_maximum=log_sum+log(maximum_fraction)으로 최대 기여 정보를 복구할 수 있다.
평균·분산·RSE를 결정하는 raw sufficient moments는 변하지 않는다.
log/exp round trip으로 최대값 metadata에 roundoff가 생길 수 있어 replay에서
그 추가 필드는1e-10 이내를 검사하고, 기존 저장 필드는 bitwise 동일 비교한다.

앞으로 producer는 원래 chunk maxima의 log_maximum을 명시적으로 보존한다.
범용 merge helper의 출력 형식을 바꾸지 않아 다른 frozen 실험의 schema를 깨지 않는다.
두 단계 합산과 직접 합산의 회귀 및 전체 complete-grid summarize 검사를 추가한다.

original phase cap7200초 안에서 recovery를 완료하는지 확인한다. postprocessing의
새 physical sample은0회다. 고정 첫 block2개의 재현 비용 physical131,072/density65,536회는
별도 감사 비용이다. 복구 직후가 아니라 이 감사까지 통과해야 수치 결과를 신뢰한다.
복구된 pilot라도 P1/P2/large-training 승인이나 continuum certificate는 아니다.
