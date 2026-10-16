# N4096 자원 대조의 운영 실패와 새 고정 실행

2026-10-16. v1 첫 whole-run이 240초 예약을 초과했다. 결과 normalizer를 활용하거나
성공 prefix로 세지 않는다. 실패 checkpoint/source/manifest를 보존한다.
경로: `results/post_audit/p2_fine_resource_v1/session_20261016`.
이 문서는 새 v2 실행 이전에 동결한다.

수학적 target, N4096 exact Gaussian pair, 두 설계(128/128 대256/64), 각8whole,
2셀 총32whole, 물리/gradient/timestep work와 accuracy 기준은 바꾸지 않는다.
v1에서 숫자 추정을 이용해 설계를 고른 것도 아니다. 바꾸는 것은 실행 시간 예산과
새 independent seed namespace, 별도의 저장 위치다.

- namespace `paired-fine-resources-v2`, directory `p2_fine_resource_v2/session_20261016`.
- whole cap600초, phase cap21,600초(6시간), prep30초, RSS4GiB.
- physical cap5m, joint backward cap2.5m 유지.
- canonical μ215회 검증이 끝난 뒤 실행한다. 다른 대형 로컬 실험과 동시 실행하지 않는다.
- v1 실패 finish의 SHA-256를 새 source/manifest에 묶는다. 실패 seed만 교체하지 않고
  전체32whole을 새 stream으로 생성한다. v1 normalizer pooling 없음.
- 완료 뒤 고정4whole replay 및 source/anchor/identity/cost 감사를 수행한다.

v1 중단 뒤 발견된 자료형 annotation 2곳을 명시적으로 보강했다. 이는 runtime 계산,
무작위수, target, MH 식을 바꾸지 않으며 v1 archive는 덮어쓰지 않는다.
정밀도 forecast와 작은 paired difference는 여전히 P2/continuum/large-training 승인이 아니다.

## 시작 전 CPU backend 선택 추가

별도로 사전 고정한 [backend 교정](FINE_BACKEND_THREAD_CALIBRATION_2026-10-16_KO.md)을
완료했다. 동일 N4096 callback/input의 values·gradient 수치 일치를 확인했고,
thread8의 관측 median callback time이 thread1보다3.495배 빨라 선택됐다.
이는 full SMC/추정기 speedup 주장이 아니다. v2는8threads를 시작 전에 고정하고
calibration result/source hash를 manifest에 묶는다. replay에도 같은 backend를 쓴다.
physical/backward count·MH 식·표본 수·통계 기준·wall cap은 그대로다.
