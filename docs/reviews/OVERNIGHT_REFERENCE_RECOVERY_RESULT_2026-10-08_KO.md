# 야간 reference 복구 실행 결과와 연구 판단

작성일: 2026-10-08. 기준: [야간 계획](../plans/OVERNIGHT_REFERENCE_RECOVERY_AND_RESEARCH_DECISION_PLAN_2026-10-08_KO.md).

## 결론

N0–N7의 고정 개발 실험·구현·감사를 수행했다. 696개 기본 checkpoint 단위가 완료됐다. Gaussian oracle과 원래 simulator 경로 검사는 통과했고 evaluator는 빨라졌다. 그러나 독립 reference 정밀도와 marginal-reference의 일관된 효율 개선은 입증되지 않았다. 조건부 추가 반복, P2, 생산용 성능 주장과 confirmation은 조건 미충족으로 실행하지 않았다.

N4는 저장 I/O를 포함한 시간 상한을 넘었다. 원본 결과를 보존하고 운영 실패를 추가 기록했다. 소프트웨어 테스트 통과와 희귀사건 정확도 인증은 별개다.

## 수행 범위

- N0: target, q, seed namespace, 입력/source hash를 고정했다.
- N1: checksum checkpoint, 작업 예약, 실패 시 차단 및 재개를 구현했다. 의도적인 1단위 중단 후 동일 실험을 재개했다.
- N2: centered·shifted·bimodal Gaussian bump의 정확한 μ 및 q=p의 M₂ oracle을 구현하고 whole-run 결과와 비교했다.
- N3: CPU float64 Volterra terminal evaluator와 rank-zero shift density를 캐시하고 원래 구현과 비교했다.
- N4: 2셀 × 2target × 2방법 × 2독립 반복, 512 IID batch, 총 4,194,304개 독립 draw를 실행했다.
- N5: population 512와 4×128 islands를 같은 초기 총 particle 수로 비교했다. aggregate SMC whole-run은 64개다.
- N6: 조건부 기대값, 밀도비, 비용 및 향후 exact joint conditional law의 범위를 검토했다.
- N7: source/q/seed, 저장 통계, 원래 simulator 재계산 및 전체 회귀 검사를 수행했다.

## 수학적 계약

p=N(0,I), g는 독립 price-driver를 조건화한 payoff, q는 원래 고정 proposal이다. μ=E_p[g], M₂(q)=E_p[g²/(q/p)]다. auxiliary reference r은 q를 대체하지 않는다.

z=(A,B)의 마지막 두 Gaussian 좌표 B를 적분한 μ estimator는 full-r estimator의 조건부 기대값이므로 Rao–Blackwell 분산 비증가가 성립한다. 비용 개선 보장은 없다. M₂ marginal L=1은 reference 변경이며 full-r 대비 분산 비증가 보장이 아니다. g의 조건부 평균을 제곱하거나 q의 마지막 좌표·전체 norm을 제거하지 않았다.

Nested 분산 Vout+Vin/L, 비용 cout+L cin에서 양의 분산·비용 조건의 연속 최적점은 sqrt(Vin cout/(Vout cin))다. 음수 잡음 보정 분산을 잘라 적용하지 않는다. 이는 알려진 conditional Monte Carlo 원리이며 새 정리로 주장하지 않는다. 자세한 가정은 [수학 검토](../theory/OVERNIGHT_REFERENCE_RECOVERY_MATHEMATICAL_REVIEW_2026-10-08_KO.md)에 있다.

## evaluator 결과

고정 두 셀·7개 batch 크기에서 cached/original median 시간 비율은 약 0.25–0.38이었다. evaluator 구간의 약 62–75% 감소이며 학습·선택·저장을 포함한 fixed-precision 전체 효율 개선은 아니다.

원래 simulator로 16개 첫 shard, 524,288 FFT+CDF 단위를 재계산했다. 최대 log 차이는 1.1369e−13, 계산 시간은 7.94초였다. 전체 표본 재계산이 아닌 제한된 첫 batch 경로 감사다.

## N4 결과

비율은 marginal/full의 분산×단위비용이다. 작을수록 좋다. 구간은 반복별 4개 비교에 대한 simultaneous block bootstrap이며 tail 누락 인증이 아니다. 각 비교 방법은 262,144개 독립 단위를 사용했다.

| 셀·target | 반복 0 비율 [구간] | 반복 1 비율 [구간] | 평균 동등성 통과 반복 |
|---|---|---|---|
| canonical μ | 0.918 [0.263, 2.902] | 0.625 [0.195, 2.418] | 없음 |
| canonical M₂ | 16.371 [0.838, 63.809] | 0.537 [0.290, 0.898] | 1 |
| 높은 η μ | 0.803 [0.559, 1.156] | 0.463 [0.296, 0.746] | 0 |
| 높은 η M₂ | 0.782 [0.645, 0.947] | 0.957 [0.684, 1.505] | 0 |

canonical M₂ marginal 반복 0의 평균은 1.6520e−9, RSE 8.407%, 최대 단일 기여 비중 8.016%였다. 반복 1은 평균 1.4757e−9, RSE 1.971%였다. full-r 평균은 각각 1.4129e−9, 1.4796e−9이며 이것도 독립 truth 인증은 아니다. 한 반복의 낮은 분산만으로 채택하면 안 된다. 모든 셀·반복에서 안정적인 20% 개선을 얻지 못해 추가 parent 반복은 실행하지 않았다.

## N5와 toy의 한계

| 셀·target | population RSE | islands RSE | 최대 whole-run 기여 비중 population/islands |
|---|---:|---:|---:|
| canonical μ | 22.43% | 25.56% | 27.11% / 28.73% |
| canonical M₂ | 66.84% | 59.47% | 69.30% / 62.92% |
| 높은 η μ | 12.67% | 10.49% | 18.61% / 21.09% |
| 높은 η M₂ | 38.39% | 24.35% | 44.22% / 27.63% |

각 설정은 8개 독립 aggregate whole-run이다. canonical M₂ 평균은 population 3.4296e−9, islands 3.1305e−10이다. 민감도가 크므로 평균내 truth로 만들지 않는다. terminal particle을 IID 표본으로 취급하지 않았다.

centered toy의 μ·M₂ 상대 오차는 약 0.4% 이내였지만 bimodal μ의 population/islands 오차는 −37.28%/−43.92%, M₂는 −16.24%/+156.76%였다. 정확한 oracle이 있어도 missed contribution 문제가 드러났다.

고정 영역 2.5≤z0≤7.5, |z1|≤2에서 bimodal μ의 참 기여 질량은 0.65459인데 추정 영역 적분/참 영역 적분은 0.42303/0.32680이었다. 낮은 총 RSE만으로 중요한 영역 탐색 성공을 주장할 수 없다. 영역 membership과 mixture label은 같지 않으며 fractional β oracle은 주장하지 않는다.

## 발견한 운영 오류 및 수정

기존 checkpoint 예산은 callback 시간만 합산해 큰 JSON의 반복 읽기·검증 I/O가 빠져 있었다. N4 callback 합계 56.12초와 달리 file timestamp 재구성 phase는 1,809.77966초로 1,800초 상한을 9.77966초 넘었다. 원본 complete_development는 단위 완료를 뜻할 뿐 운영 gate 통과가 아니다. 실패는 path_audit.json에 별도 기록했다. seed 변경이나 재실험으로 실패를 지우지 않았다.

현재 코드는 작은 검증 budget metadata만 size/mtime 기준 캐시하고, I/O 포함 phase 시계·예약·작업 후 점검·저장 후 초과 기록·초과 prefix 재사용 차단을 추가했다. sample payload를 변경 가능한 공유 cache로 취급하지 않는다. 신뢰할 수 있는 단일 writer 환경의 최적화이지 적대적 filesystem의 보안 증명이 아니다.

원본 source ZIP과 수정된 현재 코드를 구분한다. 현재 source로 원본 manifest를 resume하지 않는다. 원본 timing을 수정 코드의 성능으로 제시하지 않는다.

## 최종 검증과 자원

- 전체 pytest: **1,254개 통과, 215.28초**.
- Ruff 전체 범위 통과. Mypy CI 대상 181개 source file 통과. 새 두 experiment direct-target는 follow-imports=silent 검사 통과. 모든 기존 experiment가 타입 검증됐다는 뜻은 아니다.
- 저장 통계/source/q/seed/island 평균과 696 expected units 감사 통과. 기존 micro/kernel/block 저장 결과 감사도 통과했지만 과학 gate 통과나 tail 인증은 아니다.
- oracle, observer RNG 불변, manifest 변조, shard 손상, 미완료 예약, 실패 단위 교체 금지, checkpoint I/O 초과 차단 등을 검사했다.
- 기존 Requests dependency warning은 보존했으며 package를 임의 업데이트하지 않았다.

FFT+CDF는 N3 367,100, N4 8,388,608, N5 4,194,304, 감사 524,288로 총 **13,474,300**이다. toy 3,096,576회는 별도 단위다. 테스트 내부의 미계측 호출을 0으로 보고하지 않는다.

관측 I/O 포함 phase는 N2 58.71초, N3 10.48초, N4 1,809.78초, N5 92.49초다. file timestamp 재구성이므로 중단을 포함한 엄밀한 전체 monotonic wall과 다르다. N4 도중 peak working set 792,969,216 bytes를 관측했으나 최종 전체 peak RSS라는 주장은 하지 않는다. 결과 디렉터리는 약 132.6MB로 2GiB 아래다.

## 다음 우선순위

1. model 확대보다 guide-independent reference의 rare contribution coverage를 개선한다.
2. cache·analytic μ·수정 checkpoint를 유지하되 속도를 정밀도 증거로 대체하지 않는다.
3. full-r static M₂를 개발 baseline으로 유지하지만 자기 자신을 truth로 인증하지 않는다.
4. 고정 joint Gaussian 방향의 exact conditional law와 outer coverage를 먼저 도출한다. 새 joint sampler는 이번에 구현하지 않았다.
5. blind spot을 줄이는 대안 하나를 사전 고정한다. 무제한 kernel/β/island 탐색은 하지 않는다.
6. 관련 fixed q 전체에서 P1을 통과한 후 P2·학습 포함 fixed-precision 비교·봉인 confirmation에 진입한다.

현재는 검증 기반과 실패 원인 정량화 단계이며 최상위 저널 제출 수준의 성능·이론·독창성 완성 단계는 아니다.

## 재현과 Git

결과: results/post_audit/overnight_reference_recovery_v1/session_20261008/. 원본 source ZIP SHA256: eb2db73bd7c573367e0a17409eb1cf9ed16098856a43832e8c84f6d49954b8ba. 기존 결과는 덮어쓰지 않는다.

```powershell
python -m experiments.post_audit_v2_overnight_reference_recovery --audit results/post_audit/overnight_reference_recovery_v1/session_20261008/result.json
python -m pytest -q
python -m ruff check src tests experiments main.py train_driftnet.py
python -m mypy src main.py train_driftnet.py
python -m mypy --follow-imports=silent experiments/post_audit_v2_overnight_reference_recovery.py experiments/post_audit_v2_overnight_path_audit.py
```

경로 감사 CLI는 기존 output을 덮어쓰지 않는다. 새 suffix 재감사는 별도 output 설정이 필요하다. Git 범위는 현재 연구 branch에 한 번 commit/push이며 main merge·force push는 포함하지 않는다.
