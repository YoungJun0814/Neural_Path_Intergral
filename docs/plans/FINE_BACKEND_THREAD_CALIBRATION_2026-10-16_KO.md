# N4096 CPU backend 교정: 수치 일치 후 thread 선택

2026-10-16. 결과 생성 전 고정. 물리16코어 노트북에서 threads1 고정이 큰 callback을
불필요하게 느리게 하는지 검사한다. 통계적 target·seed 설계·표본 수 선택은 하지 않는다.

원래 high-η fine4096/coarse2048 paired callback, 동일128 Gaussian inputs 한 묶음,
thread1/2/4/8 각각 warmup1회+timed5회. 총3072joint forward/actual backward,
physical6144grid evaluations. 준비 basis-path simulation0회. 각 arm의 log components,
family log potential 및 gradient를 thread1과 비교한다(rtol5e-10,atol1e-8).
proposal drift cap20과 CPUfloat64를 유지한다. 추정 μ/RSE/성능 우위는 보고하지 않는다.

최소 median time 설정이 thread1 대비20% 이상 빠르고 모든 numerical checks를
통과한 경우에만 그 thread count를 후속 v2에 고정한다. 그렇지 않으면1을 유지한다.
wall cap180초,RSS4GiB. 단일 callback 교정은 full SMC/논문 speedup benchmark가 아니다.
동시 static 진단의 영향을 포함한 로컬 운영 선택일 뿐이다.

calibration source/result를 봉인하고 v2 manifest에 hash와 선택한 thread count를 묶는다.
v2가 시작한 뒤 thread 수를 바꾸지 않는다. 기존 v1 실패와 active static 실험은 그대로다.
