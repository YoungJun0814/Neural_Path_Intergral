# q 동결 위험 경로 진단 결과와 채택/중단 판단

작성일: 2026-10-21. N32의 두 개발 셀에 한정한다.

## 1. 결론

다음 우선순위였던 **기존 모델을 바꾸지 않은 원인 진단**을 실행했다.
결과는 `STOP_FEATURE_NOT_INDEPENDENTLY_CONFIRMED`다.

발견 자료에서는 늦은 시간에 변동성 질량이 몰리는 특징이 선택됐다. 독립 확인 자료에서도
양의 점추정 신호는 남았으나, 두 셀 모두 반복성 기준은 **4/8블록**으로 요구6/8에 미달했다.
M₂ RSE도48–63%로 정밀도 기준20%를 넘었다. 이 특징을 확정 원인이나 새로운 모델의
구성 요소로 채택하지 않았다. 모델 q, 평가분포 G, 가중치, geometry는 모두 그대로다.

이 판정은 “늦은 변동성이 무관하다”는 증명도, 기존 q가 근본적으로 틀렸다는 증명도 아니다.
한 가지 단순한 특징으로 위험을 안정적으로 설명한다는 가설이 이번 사전 기준을 통과하지
못했다는 뜻이다. 성공할 때까지 표본/seed/특징을 교체하지 않고 이 사이클을 종료한다.

계획: [q 동결 위험 경로 진단](../plans/FROZEN_Q_RISK_PATH_ATTRIBUTION_2026-10-21_KO.md).

## 2. 무엇을 진행했는가

1. 기존 D2 저장 통계만으로 전체 μ+11개 q의 성분별 기여와 성분 내부 집중도를 재집계했다.
2. 기존 두 셀의 층화8블록을 전부 재생해 경로 특징을 복원했다. **476,160건의 새로운
   physics 비용**이며, 이미 본 discovery 자료의 재사용이지 독립 확인이 아니다.
3. 사전 지정6특징×low/high의12후보 중, 양 셀의 가장 약한 U enrichment가 최대인 한 가지를
   선택했다. q는 U와 기존 event-kl-0만 진단 endpoint로 사용하고, 기존 성능으로 best q를 새로 고르지 않았다.
4. 특징·방향·셀별 수치 cut을 봉인한 뒤, 새 namespace/role의 층화 표본 **476,160건**으로 확인했다.
5. 저장 배열 전수 통계 재집계, 기존 결과 결합 검사 및 지정한 대표 블록의 physics 재현 감사를 진행했다.

q의 성능을 측정하는 G가 안정적이지 못한 문제와 q 자체의 구조 한계는 구분했다.
이번에는 q나 G를 바꾸지 않았으므로 분산 감소나 성능 개선 성공을 주장하지 않는다.

## 3. 성분별 재집계에서 확인한 집중

| 셀 / q | 가장 큰 sampling label | 해당 label의 관측 위험 비중 | 그 label 내부 최대 단일 표본 비중 |
|---|---:|---:|---:|
| canonical / U | 354 | 47.05% | 99.8315% |
| canonical / event-kl-0 | 354 | 38.03% | 99.8712% |
| high-eta / U | 292 | 89.22% | 99.9964% |
| high-eta / event-kl-0 | 292 | 87.98% | 99.9966% |

label354는 window template29, vol amplitude4/price amplitude4다.
label292는 window template24, vol amplitude4/price amplitude0다.
이는 **표본을 뽑은 혼합 성분의 ID**이지 경로가 속한 진짜 mode나 원인의 분류가 아니다.
일부 성분을 더 자주 뽑으면 해결된다는 결론도 아니다. 해당 성분 안에서도 거의 한 표본이
기여를 지배했으므로 label-count 균형만으로 남은 집중 현상을 설명하기 어렵다.

## 4. 발견 자료에서 봉인한 특징

선택된 것은 `variance_time_center`의 높은 쪽이다.

```
C_V = sum_i ((i/32) * V_i) / sum_i V_i,  i=0,...,31
```

V_i는 실제 terminal 계산에 사용되는 각 구간 시작 시점의 모형 분산이다.
계산에 쓰이지 않는 마지막 V_32는 제외했다. C_V는 **분산 질량의 평균 시간**이며,
최대 분산의 발생 시각이나 “마지막 몇% 구간만 분산이 존재한다”는 조건이 아니다.

공통 특징/방향 하나에 셀별 discovery90% quantile을 사용했다.

| 셀 | 고정 cut(C_V≥) | G 표본 비중 | U 위험 비중 | 사건 기여 비중 | U enrichment |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7753010213 | 10.00% | 68.41% | 33.84% | 6.84 |
| high-eta | 0.8014578864 | 10.00% | 89.94% | 37.23% | 8.99 |

12후보 중2개가 discovery eligibility 조건을 만족했다. 위 규칙 하나만 선택했으며,
confirmation 결과를 보고 다른 후보로 바꾸지 않았다. 위 숫자는 discovery 설명량이다.

## 5. 독립 확인 결과

각 셀238,080개의 새 층화 표본을 사용했다. 같은 rule을 모든8개 완전 블록에 적용했다.

| 셀 / q | G 비중 | 위험 비중 τ | 사건 비중 e | enrichment | M₂ RSE | 최대 단일 기여 | 반복 조건 /8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical / U | 9.98% | 74.56% | 34.19% | 7.47 | 48.16% | 40.47% | 4 |
| canonical / event-kl-0 | 9.98% | 79.23% | 34.19% | 7.94 | 51.41% | 37.86% | 4 |
| high-eta / U | 9.96% | 50.62% | 30.51% | 5.08 | 54.10% | 44.11% | 4 |
| high-eta / event-kl-0 | 9.96% | 58.91% | 30.51% | 5.92 | 62.75% | 54.76% | 4 |

블록별 반복 조건은 τ≥.2, τ/Gmass≥2, τ>e의 동시 성립이다.
사전 요구6/8에 비해 전체4rows 모두4/8이었다. 높은 pooled 비중이 모든 독립 블록에서
같은 기여 집중이 반복됨을 의미하지는 않는다.

| 셀 / q | raw의 3대조 lower 최솟값 | block의 3대조 lower 최솟값 | M₂ block RSE | LOO 최대 변화 | 확인 |
|---|---:|---:|---:|---:|---|
| canonical / U | +0.0876 | +0.1044 | 44.81% | 35.26% | fail |
| canonical / event-kl-0 | +0.1835 | +0.2057 | 48.04% | 31.98% | fail |
| high-eta / U | −0.5230 | −0.5797 | 45.26% | 36.39% | fail |
| high-eta / event-kl-0 | −0.4981 | −0.5595 | 55.83% | 48.41% | fail |

3대조는 τ−.2, τ−2Gmass, τ−e다. 24개 양측 normal/delta 구간에 family alpha=.05를 적용했다.
canonical에서 근사 lower가 양수여도 M₂ 정밀도·집중도·반복성의 실패를 덮어쓰지 않는다.
이는 점근적 개발 진단으로, 미관측 tail 또는 유한표본 coverage의 증명이 아니다.

## 6. 이론·기술 검토

- 고정q의 위험 integrand는 H_q=g²/(D_GD_q)다. 확률 기여 Y=g/D_G와 혼동하지 않았다.
- 특징 영역A의 위험 비중은 τ=E_G[H_q1_A]/E_G[H_q], 사건 비중은 e=E_G[Y1_A]/E_G[Y]다.
  이 ratio 점추정을 유한표본 unbiased estimator로 부르지 않는다.
- delta gradient는 dτ=da_q/M_q−a_q dM_q/M_q²다. e도 동일하게 계산하고 같은 경로의
  cross-covariance를 포함한 ∇fᵀCov∇f를 썼다. 성분별 SE와8balanced-block SE는 따로 계산했다.
- indicator의0은 정확한0이다. log0은 JSON null로 가역 부호화했으며, epsilon으로 가짜 기여를 넣지 않았다.
- 평균·공분산은 알려진 질량π_j에 따른 Σπ_j mean_j, Σπ_j² sample_cov_j/n_j다.
  unequal allocation의 수기 계산과도 대조했다. 현재 정확 비례 allocation에서 행 질량은1/N이다.
- g와 특징은 같은 한 번의 cached quantities 계산에서 얻었다. 특징 때문에 g를 두 번 평가하지 않았다.
- discovery16blocks 전부 기존 D2의12target 전성분 모멘트·seed·sample digest와 일치를 요구했다.
- single-template residual은 직교 방향 한 쌍으로의 projection 잔차다. 전체 dictionary span 밖의 성분이 아니다.
- 선택한 late-time rule은 경로의 설명이지 정규화된 proposal이 아니다. G를 이 비선형 조건으로
  절단하려면 정규화 상수를 다뤄야 하며, 그 확률을 아는 것처럼 밀도 보정에 사용해서는 안 된다.

검사 범위에서 이번 추정치를 무효로 만드는 수치·이론 불일치는 발견하지 못했다.
다만 근사 구간·유한 반복·미관측 기여의 한계는 남으며, 새 원리나 geometry 한계의 증명이 아니다.

### 해석상 주의

- raw/block 구간이 양수인 것과 원인이 확인됐다는 것은 다르다. 이번에는 별도 정밀도·반복성 gate가 실패했다.
- g≤1과 방어적 밀도 때문에 integrand는 유계다. 실제 무한분산 또는 heavy-tailed law를 발견했다고 표현하지 않는다.
- 표본에서 발견한 상관관계이며, late-time 특징이 위험의 인과적 원인임을 입증한 실험이 아니다.
- 원자료 전체의 모멘트를 재집계했지만, 모든 경로의 특징을 독립 evaluator로 전수 재계산하지는 않았다.
  특징 계산의 검사 범위는 수기/단위 대조와 사전 지정한 대표 블록 수치 replay다.

## 7. 저장·감사·회귀

새 코어 `risk_path_attribution.py`, runner `post_audit_frozen_q_path_attribution.py`, 전용 test와
사전 계획을 source freeze에 포함했다. 모든 표본의6특징·3integrand·label을 압축 저장했다.
블록별 각q 상위8개 슬롯에는 좌표와variance 경로도 저장했으며, q 사이에 같은 경로가 중복될 수 있다.
array checksum, start/finish ledger, selection cut, input/source hash를 결합했다.

감사는 **통과**했다. 32개 블록의 array hash·성분 수·event/risk 모멘트를 전수 대조하고,
저장 array에서 discovery 선택과 confirmation 통계·판정을 재집계했다.
사전 지정한4개 대표 블록(discovery/confirmation 각각 두 셀의block0)을 재생해
**119,040건**의 physics 계산으로 특징·integrand·label·상위 경로 기록이 일치함을 확인했다.
이는 모든32개 블록의 특징을 다시 physics 계산한 것과는 다르다.

실행 전 관련 테스트42개, 전용16개 및 새2파일 Mypy가 통과했다.
실험·대표 replay가 끝난 뒤 실행한 최종 로컬 검증 결과는 다음과 같다.

- 전체 pytest **1,662개 통과 / 626.98초**, source coverage **85%**(22,168 statements, 3,401 missed).
- 새 통계 코어 `risk_path_attribution.py`의 coverage **94%**(109 statements, 7 missed).
- CI와 동일 범위의 Ruff 통과, Mypy **212 source files 통과**.
- 새 runner의 별도 Mypy **1 source file 통과**.
- Python3.10.11 / Windows 로컬 검사다. 원격 GitHub CI나 Python3.11 실행을 이번에 확인한 것은 아니다.

기존 RequestsDependencyWarning은 별도 환경 경고로 남겨뒀으며 연구 중 의존성 업그레이드는 하지 않았다.
코드·회귀 검사의 성공과 과학적 특징 확인 실패는 별개의 결과다.

### Seed 검사 범위와 절차적 편차

sampling 전에는 이번 discovery/confirmation 예정 role끼리의 seed uniqueness를 검사했다.
이후 기존 D2의 IID·preflight까지 포함한 더 넓은 범위로 **13,198개 role 전부 충돌 없음**을 확인했다.
discovery가 기존16개 층화 block의 seed를 의도적으로 재사용한 것은 새 확인 자료로 세지 않았다.
넓은 범위의 기존 IID/preflight 대조가 sampling 이후였다는 순서상의 한계는 기록한다.
확인 결과에 따라 seed를 변경하거나 다시 실행하지 않았다.

### 결과 anchor

- 부모 D2 result SHA256: `323b00268d797b6e70222a748ce500ee4c5f7f1ff9da1e6724acd750cb3ee1bb`
- 이번 result SHA256: `eff4fd1406719163bdc333c62f79484377ba155ef3a6d58910c3739ba4dbd9f5`
- source snapshot SHA256: `d8f1e1f96d77618a502b7c113d4774f06da7514453c018d296eab0a48dc38040`
- selection SHA256: `b6e802c31593f6e01bfd7b231a4624da41c94f564b00c0dd7aaf10fed05feb53`
- 결과 디렉터리: `results/post_audit/frozen_q_path_attribution_v1/session_20261021/`
- `selection.json`, `component_description.json`, `audit.json`, 압축 trace32개 및 journal을 보존했다.

별도 post-run integrity 검사에서 source에 봉인된 수치 파일472개의 hash 불변,
32개 완료 unit, 단일 unit wall 최댓값22.03초와 violation 없음도 확인했다.
confirmation phase clock은 selection 파일 저장 이후 시작됐다. 이는 source·실행 순서의
검사이지, filesystem timestamp만으로 외부 사전등록을 인증했다는 뜻은 아니다.

## 8. 비용과 작업 경계

| 비용 | Volterra calls |
|---|---:|
| 기존 기록의 성분별 분석 | 0 |
| discovery 기존 경로 재생 | 476,160 |
| 독립 confirmation | 476,160 |
| 지정 representative audit | 119,040 |
| **실제 합계** | **1,071,360** |

production wall은521.20초였다. CPU float64 1thread로 실행했고, 실험 중 전체pytest를 병행하지 않았다.
경쟁 모델과의 속도 benchmark가 아니며, 회귀테스트 내부 physics까지 계측한 전체 작업 합계도 아니다.
원본/binary artifact와 기존 dirty work를 보존했다. 봉인 후 수치 source를 변경하지 않았다.
커밋·푸시는 하지 않았다.

## 9. 연구 방향에 반영할 판단

채택: q와G의 역할을 분리하고, 성분 내부 집중과 독립 표본의 반복성을 진단하는 체계.

보류/불채택: 단일 late-time 특징을 안정적인 원인으로 인정하는 것, 이를 이유로 새q/G를
자동 채택하는 것, weight-only/D3/대규모 학습을 재개하는 것.

이번 관측은 “늦은 분산 집중 특징에 강한 pooled 관측 연관성이 있지만, 이것만으로 독립 블록의
위험 기여를 사전 기준에 맞게 안정적으로 설명하지 못했다”는 것이다. 복수 기여 메커니즘,
상호작용 또는 표본 부족은 여전히 가설이며 원인을 확정하지 않았다.

본 사이클은 여기서 종료한다. 같은 불안정한 risk 비율에 의존한 특징 찾기를 무제한 반복하지 않는다.
다음 별도 연구에 투자하려면 정답이 알려진 문제의 구조적 반례/조건, 또는 현재의 불안정한
M₂ 점추정에만 의존하지 않는 원인 식별 방법을 먼저 구체화해야 한다.
이 보고서만으로 최상위 저널에 필요한 독창성·성능·실용성을 확보한 것은 아니다.
