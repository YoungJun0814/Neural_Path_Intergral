# Structural V2 P0 실행·검토 보고서

작성일: 2026-10-08. 대상: [통합 개선계획 V2](../plans/MODEL_STRUCTURAL_IMPROVEMENT_PLAN_V2_2026-10-08_KO.md)의 P0만.

## 1. 결론과 범위

P0의 evidence inventory, additive measurement/role schema, claim ledger 및 감사 기반을 구현했다. 새로운 금융 모델·Stein field·독립 reference·mesh 생산 실험은 실행하지 않았다. 기존 raw evidence는 변경하지 않았으며 커밋·푸시·PR 병합도 하지 않았다.

P0의 통과는 증거·역할·추정 대상의 명세와 binding 검사가 갖춰졌다는 뜻이다. 독립 참값, 미관측 tail coverage, 실제 성능 우위, 연속시간 정확성, 신규성 또는 무오류를 인증하지 않는다.

## 2. 구현한 내용

### V2 계약은 기존 v1을 보존한다

[새 계약 모듈](../../src/path_integral/structural_v2_contract.py)에 `npi.structural-v2.measurement.v1`을 추가했다. 기존 historical result schema와 JSON은 그대로 남겼다.

구분하는 항목:

- event probability, raw event second moment, corrected event second moment, auxiliary estimator second moment, adjacent probability difference.
- ordinary IS, event Stein CV, auxiliary IS/CV, raw SMC normalizer, corrected-risk IS, coupled IS.
- IID draw, whole SMC run, IID coupled draw라는 SE 단위.
- q/r/target/grid/control/basis digest 및 control bound의 선언 방식.
- parent/auxiliary/CV fit, selection/final/reference 반복 ID와 mesh pair.

잘못된 조합, raw estimator에 control 부착, corrected estimator에 control 누락, SMC particle을 IID 단위로 표시, self-normalization, final identity 누락 등을 거부한다. control parameter/basis hash를 재검증하지만 parameter shape·field의 수학적 bound 전체를 인증하는 것은 아니다. 해당 구현은 P4 관문이다.

### 표본 역할과 paired 사용

`audit_sample_uses`는 별도 선언한 use ID grid, 전체 seed 사용, consumer binding, 역할 일치를 검사한다. training/calibration/selection stream 재사용을 final pairing으로 위장하지 못하게 한다.

paired 사용은 distinct consumer의 동일 target/grid/estimand/q/r/final identity에서 명시한 final IID stream에만 허용한다. 다른 q의 공통 base noise coupling과 복잡한 cross-fitting은 첫 schema 범위 밖이다. 필요한 경우 별도 계약을 만든다.

다른 seed와 올바른 선언이 실제 독립성이나 실행 순서를 증명하는 것은 아니다. 실제 runner는 이 계약을 호출하고 draw 생성부터 역할을 기록해야 한다. P1/P4 이후 runner 구현에서 해당 통합을 수행한다.

### seed·snapshot 보강

- 기존 SeedKey에 boolean/float index 및 non-string role을 거부하는 검사 추가.
- 기존 유효한 키의 seed derivation 및 serialization은 변경하지 않음.
- 기존 `freeze_source`에 optional `extra_snapshot_paths` 추가. workspace 안의 기존 파일만 허용하여 V2 계획·claim ledger를 새 snapshot에 포함.
- historical archive를 재작성하지 않음. 새 결과에 새 ZIP을 생성.

### bounded-memory inventory

[inventory 모듈](../../src/path_integral/structural_v2_inventory.py)은 gzip/plain JSON을 기록 단위로 읽고 대용량 records 배열 전체를 메모리에 적재하지 않는다. duplicate fields, malformed/truncated JSON, oversize value를 거부한다.

감사 범위: artifact SHA/크기, gzip 해제 bytes 및 로컬 raw와의 일치, archived source/config hash와 entry completeness, saved q/r parameter digest, exact-zero natural mixture floor, seed ledger, parent 및 auxiliary grid, row work 합계, reference/guide dependency binding.

모든 역사적 estimator의 batch moment 재계산이나 simulator replay를 다시 한 것은 아니다. 큰 archive의 파일별 hash를 검사했지만 그 source code의 수학적 정합성을 커널 검증한 것도 아니다.

## 3. 실제 evidence inventory 결과

[실행 artifact](../../results/post_audit/structural_v2_p0_inventory_v1.json), [봉인 source ZIP](../../results/post_audit/structural_v2_p0_inventory_v1.source.zip), [실행 config](../../configs/post_audit/structural_v2_p0_v1.yaml).

| historical evidence | 기록 수 | seed ledger 항목 수 | binding |
|---|---:|---:|---|
| family diagnosis pilot | 2 | 158 | 통과 |
| family diagnosis five-fit | 10 | 1,086 | 통과 |
| risk geometry stability | 10 | 1,000 | 통과 |
| auxiliary risk crosscheck | 10 | 1,040 | 통과 |
| auxiliary whole-fit v1 | 50 | 5,200 | 통과 |
| auxiliary bank mixing v2 | 30 | 3,120 | 통과 |
| auxiliary covariance v3 | 30 | 3,120 | 통과 |
| auxiliary global refresh v4 | 30 | 3,120 | 통과 |
| auxiliary island ensemble v5 | 50 | 16,400 | 통과 |
| static Volterra pilot v6 | 50 | 3,200 | 통과 |
| auxiliary Volterra safeguard v7 | 50 | 8,400 | 통과 |
| 합계 | 322 | 45,844 | 11/11 |

합계는 각 artifact ledger의 항목 합이다. 45,844개 전부가 서로 독립인 추론 표본이라는 뜻이나 과거 run 전체의 global independence 인증이 아니다.

v5/v6/v7 gzip과 로컬 raw의 byte-identical 비교를 수행했다. 기존 파일의 내용은 수정하지 않았다. 각 단계의 기록 수는 parent 전체 학습, 고정-q auxiliary fit, static-only final 반복이라는 서로 다른 단위를 섞으므로 322회를 독립 original-model 학습 횟수로 세지 않는다.

현재 runtime source digest와 historical digest는 **11/11 불일치**다. 이는 새 코드가 추가됐기 때문에 예상되는 상태이며, historical source ZIP·config binding은 각각 통과했다. digest를 현재 값으로 덮어쓰지 않았다.

inventory 자체의 elapsed time 약 **59.95초**, sampled peak process RSS **612,892,672 bytes(약 584.5 MiB)**. 노트북에서 읽기 전용 확인한 시작 가용 RAM은 약 1.48GB였다. memory 상한 1GiB·wall 상한 600초 안에서 실행했다. 이는 algorithm speed benchmark가 아니다.

새 scientific target sample, potential evaluation, model training run은 모두 **0**이다. 테스트·감사 내부 호출은 그 0에 합쳐 scientific production처럼 표시하지 않으며 별도 미계측으로 기록했다.

## 4. 이론적 재검토

[별도 V2 claim ledger](../theory/STRUCTURAL_V2_CLAIM_LEDGER_2026-10-08_KO.md)를 작성했다.

1. E_q[gp/q]=μ_N과 E_r[g²p²/(qr)]=M₂,raw(q)를 서로 다른 estimand로 유지.
2. 사건 Stein CV의 분산에는 M₂,CV=∫(g−Cθ)²p²/q가 필요. raw M₂로 대체하지 않음.
3. auxiliary risk CV는 원래 M₂,raw를 평가하는 estimator만 바꿈. q의 위험 감소로 해석하지 않음.
4. Gaussian Stein identity는 적절한 경계·적분가능성 및 frozen field 조건부 명제. schema pass만으로 성립하지 않음.
5. signed 보정에 positive log-summary/SMC potential 계약을 적용하지 않음.
6. exact natural floor는 정확한 zero mean·identity covariance만으로 계산. near-zero diagnostic을 floor로 사용하지 않음.
7. finite-grid correctness와 continuum bias를 분리. 기존 T16 미완료 정리는 그대로 유지.
8. 낮은 RSE, ancestry, KL/W₂ 일치, metadata binding을 rare-tail coverage 인증으로 쓰지 않음.

P0에서 새로 증명한 Volterra 특화 효율 정리는 없다. corrected field/estimator는 아직 구현하지 않았으며 P4/P5에서 검증해야 한다.

## 5. 기술 검증 상태

초기 targeted 계약·reader·seed 55개 통과. 이후 runner tamper 검사를 추가한 새 두 테스트 파일은 49개 통과. 기존 constant/density/raw-risk oracle 등을 포함한 관련 회귀는 변경 중간 시점 80개 통과했다. 이들은 겹치는 실행이므로 서로 더해 unique test 수로 쓰지 않는다.

전체 Ruff 통과, mypy 172개 소스 파일 통과. 테스트 시작 시 설치 환경의 requests/urllib3 관련 compatibility warning은 관측됐으며 패키지 설치·변경은 하지 않았다.

최종 fresh inventory artifact의 재감사도 11/11 통과했다. 선택한 historical 산술 재감사는 family five-fit(10 whole-parent, 30 candidate, 7 final summary)와 auxiliary initial crosscheck(10 parent, 10 whole auxiliary job)에서 통과했다. 같은 key의 source·proposal·seed·moment/cost 결합을 확인했으며, 전체 simulator 재실행이나 독립 참값 검증은 아니다.

전체 pytest 첫 실행은 기본 다중 스레드의 높은 CPU 비용 때문에 24% 진행 시 운영상 중단했다. 이 partial 실행은 통과로 세지 않는다. 테스트 프로세스에만 `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`을 지정해 새 전체 실행을 시작했다. OS 전원 설정·설치 패키지를 변경하지 않았다.

최종 전체 pytest는 **1,174개 모두 통과**, 실행 시간 **195.98초**였다. 이 결과에는 새 계약·inventory 테스트 49개가 포함된다. 테스트 통과는 검사한 조건에 대한 회귀 결과이며 미실행 scientific gate 또는 모든 이론적 오류의 부재를 인증하지 않는다.

## 6. 실행·재현 명령

```powershell
python -m experiments.post_audit_v2_p0_inventory --audit results/post_audit/structural_v2_p0_inventory_v1.json
python -m pytest tests/test_structural_v2_contract.py tests/test_structural_v2_inventory.py tests/test_seed_ledger.py -q
python -m ruff check src tests experiments main.py train_driftnet.py
python -m mypy src main.py train_driftnet.py
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest -q
```

inventory 최초 실행 명령은 `python -m experiments.post_audit_v2_p0_inventory`다. 이미 있는 output/ZIP을 덮어쓰지 않는다. 다시 실행하려면 새 output/config suffix를 사용한다. source/runtime가 바뀌었을 때는 기존 artifact가 아닌 새 snapshot 아래에서 실행한다.

## 7. 다음 단계

P1의 독립 raw μ/M₂ reference runner·pilot부터 진행할 수 있는 코드·명세 기반을 마련했다. P1 기준값, P2 mesh, P4 Stein correctness, P5 효율, P8 confirmation은 이번 P0에서 통과하지 않았다. 실제 rare-event 성능 comparison은 해당 관문 이후에만 열린다.

원래 V2 계획 파일은 P0 snapshot의 evidence document이므로 결과 기록을 위해 수정하지 않았다. 이 보고서가 P0의 완료 상태를 별도로 갱신한다. 이렇게 해야 봉인 계획의 hash가 결과 작성 때문에 깨지지 않는다.
