# HMC 고정 실험의 운영 실패 보존과 전체 재생 복구

2026-10-28 America/New_York. 통계 가설·설정·seed·표본 수·통과 기준은 바꾸지 않는다.
원래 실행의 failure.json과 source는 보존하며 운영 성공으로 덮어쓰지 않는다.

## 1. 발견 사실

원래 `p2_hamiltonian_tail_v1/session_20261028`는 pilot20/32를 저장했고,
precision0개 상태에서 TimeoutError로 종료했다. failure의 phase wall31403.44초가
고정7200초를 넘었다. 완료20whole의 기록 wall 합계는 약1483초로 훨씬 짧다.
장시간 정지/대기 가능성이 있으나 OS suspend나 특정 외부 작업의 원인은 입증하지 않았다.
미완료21번째 whole의 부분 normalizer는 저장·사용하지 않았다.

## 2. 복구 계약

1. 별도 `p2_hamiltonian_operational_replay_v1/session_20261028`를 사용한다.
2. 원래 protocol/result가 없는 failure/완료20개/preflight4개와 원래 source ZIP을
   모두 hash anchor로 묶는다. 현재 original frozen source hash 일치를 먼저 검사한다.
3. **원래4개 preflight와32개 pilot 전부를 같은 seed로 재실행한다.**
   기존20개는 wall을 제외한 모든 값이 같아야 한다. 불일치면 운영 복구 실패로 중단한다.
   새 seed를 뽑거나 좋은20개를 고르지 않는다. 처음20개를 새독립20개로 더하지 않는다.
4. 이 실행에 별도 phase7200초/whole300초/RSS4GiB/physical1,050,000 상한을 둔다.
   원래 실패한 운영 상한을 통과한 것으로 소급 변경하지 않는다. 비용은 별도 총합 공개한다.
5. 32전체가 끝나면 원래의 pilot_decision을 그대로 실행한다. 원래 primary HMC,
   fresh allocation128whole/cell 및 forecast10800초 gate도 그대로다.
   조건 미달이면 precision을 실행하지 않는다. 통과하면 원래 예약 precision seed로
   새 전량을 수행한다. pilot을 final/confirmation으로 승격하지 않는다.
6. 결과 source/seed/whole/work/동일20개 재현/결정 binding을 감사한다.
   각cell×arm rep0와 precision이 있으면 각cell 첫whole도 재생한다.
   auditcap200,000/1800초. 전체 회귀와 기존 archive 무결성도 확인한다.

복구는 동일 통계 실험의 numerical replay이지 독립 confirmation이 아니다.
원래 운영 실패를 삭제하거나 성공 seed 찾기로 바꾸지 않는다. 이 복구도 실패하면
원래 기록과 함께 보존하며 자동으로 계속 새 운영 시도를 생성하지 않는다.
P2/production/large-training은 실제 두 의무 해결 전 false로 유지한다.
