# q 동결 위험 경로 진단: 한 번의 discovery/confirmation 계획

작성일 2026-10-21. 사용자 요청에 따른 다음 우선순위의 별도 제한 실험이다.
기존 D2 중단 결과를 재시도해 통과시키거나 미사용 D3 예산을 전용하지 않는다.

## 목적과 범위

질문: 사건 모델 q와 평가용 G를 모두 고정했을 때, 위험 기여와 연관된 **설명 가능한 한 가지
경로 특징**이 두 N32 셀의 독립 확인 표본에서도 재현되는가?
확인되더라도 geometry 부족의 증명, 모델 개선, 새 평가분포의 성능 또는 P2 통과가 아니다.
G 또는 q 변경·학습·D3 실행·차원 확장·커밋·푸시는 이 계획에 없다.

## 동결 대상

- 기존 D2 result SHA256 `323b00268d797b6e70222a748ce500ee4c5f7f1ff9da1e6724acd750cb3ee1bb` 및 audit receipt.
- 두 셀 canonical-h005-k1 / high-eta200-k1, N32, 기존 373성분 mixed-window G.
- q는 U(=G)를 primary, 기존 event-kl-0를 secondary로 고정. 0은 기존 성능을 보고 고른 best가 아니라
  열거의 첫 반복 ID이며, 이미 관측된 자료에서 정한 대조군이다. q의 독립 성능 확인으로 부르지 않는다.
- 기존 원본·수치 코드는 수정하지 않는다. 새 source snapshot에 본 계획·실행기·코어·테스트를 봉인한다.

## 단계 A: 추가 경로 없는 성분별 재집계

기존 층화 16개 블록의 모든 11개 q와 μ에 대해 성분별 추정 기여, 최대 단일 표본 비중,
해당 성분 내부 최대 표본 비중과 분산 기여를 산출한다. guide label을 실제 경로 mode로 부르지 않는다.
현재 checkpoint는 경로 좌표/특징을 저장하지 않았으므로 특징 복원은 새로운 physics 비용이다.

## 단계 B: discovery 재생과 특징 선택

두 셀 × 기존 D2 층화 8블록 ×29,760 = **476,160건**을 재생한다.
seed·경로 digest·전체 12개 integrand의 성분별 cross-moments를 기존 D2와 대조한다.
이것은 새 독립 자료가 아니라 이미 결과를 본 discovery 자료다.

한 번의 cached terminal quantities 계산으로 g와 특징을 같이 얻는다. 특징은 다음 6개로 고정한다.

1. log integrated variance (최종 사용 구간의 variance만 포함).
2. log(구간 최대 variance / initial variance).
3. 최대 variance 구간의 시작 시간/T (동률은 첫 구간).
4. variance 시간 질량의 HHI = Σv²/(Σv)².
5. variance 시간 질량의 중심 시간/T.
6. single-template residual fraction = 1−max_j[(z·u_j)²+(z·v_j)²]/||z||².

마지막 값은 orthogonal past-vol/future-price 한 쌍의 projection 잔차이며,
전체 dictionary span 밖의 에너지나 모델 표현력 하한이 아니다. label을 빼는 임의 centering은 하지 않는다.

각 특징의 낮은 10%/높은 10% 후보 총12개. cut은 각 셀의 pooled discovery G 표본 quantile(.1/.9),
비교는 <=/>=로 고정한다. 정확 비례층화라 각 행의 적분 질량이1/N이다.
동률 때문에 실제 G 비중이 커질 수 있으므로, 두 셀 모두 관측 G 비중[.02,.30], U의 위험 비중≥.20,
위험 비중/G 비중≥2인 후보만 eligible이다.
두 셀의 U enrichment 중 작은 값이 가장 큰 후보 **하나**를 선택한다. 동률은 특징 순서/low 우선.
선택 후 각 셀의 숫자 cut을 한 번만 봉인한다. eligible 후보가 없으면 confirmation 없이 종료한다.

저장: 모든 표본의 특징·log event/risk integrand·label을 압축 array로 저장하고 SHA를 연결한다.
블록별 U 및 secondary 위험 상위 각8개의 좌표와 variance 경로도 저장한다.
후보별 point contrast는 discovery 설명량이며 검정 결과가 아니다.

## 단계 C: 독립 confirmation

새 namespace/role의 두 셀×8×29,760 = **476,160건**, 같은 층화 G를 한 번만 실행한다.
선택/cut을 수정하지 않는다. 기존 또는 discovery seed와 충돌하지 않음을 생성 전에 검사한다.
confirmation은 feature 확인용이지 D2 통과를 다시 시도하는 실험이 아니다.

Y=g/D_G, H_U=g²/D_G², H_E=g²/(D_G D_E), rule indicator I를 정의한다.
성분별 (Y,H_U,H_E,I,YI,H_UI,H_EI)의 충분통계와 cross-moments를 보존한다.
indicator 때문에 정확0이 생기므로 log0을 임의 epsilon으로 바꾸지 않는다.
분산은 알려진 질량 π와 성분 수 n_j로 Σπ_j² sample_cov_j/n_j를 쓴다.

각 q의 위험 비중 τ=E[HI]/E[H], proposal 비중 b=E[I], 사건 비중 e=E[YI]/E[Y].
독립 확인에서 τ−.2, τ−2b, τ−e의 delta lower를 raw stratum 및8개 완전 balanced-block 단위로 계산한다.
2셀×2q×2SE단위×3대조=24개 양측 normal 구간, family alpha=.05.
모든 구간은 점근적 개발 진단이며 유한표본 tail/coverage 인증이 아니다.

확인 판정은 두 셀 모두 U에서 다음을 동시에 충족할 때만 허용한다.

- H의 raw RSE≤.20, block RSE≤2raw, block LOO≤.15, 최대 단일 가중 기여≤.10.
- raw/block 각각 세 adjusted lower 모두>0.
- 같은 rule의 블록별 τ≥.20, enrichment≥2, τ>e가8개 중6개 이상.

secondary q는 동일 기준으로 별도 표시한다. primary가 실패하면 secondary 성공으로 구제하지 않는다.
실패/집중/미정밀은 `STOP_FEATURE_NOT_INDEPENDENTLY_CONFIRMED`이며 rule/seed/표본을 바꾸지 않는다.
성공해도 `FEATURE_CONFIRMED_DEVELOPMENT_ONLY`로만 표기, 새 G 변경은 별도 계획의 대상이다.

## 예산·자원·감사

- discovery 재생476,160 + fresh confirmation476,160 = production 최대952,320.
- 각 단계/cell block0을 대표 재생:4×29,760=119,040, 전체 최대 **1,071,360 Volterra calls**.
- confirmation 미실행이면 discovery block0 두 개만 감사59,520, 미사용 예산 전용 없음.
- 각 단계wall900초, 블록90초, audit900초, 전체3600초, CPUfloat64 1thread, RSS4GiB.
- 전용 성능 실험과 전체 pytest를 동시에 실행하지 않는다. 고정 예산 이후 automatic extension 없음.
- append-only start/finish·source/input/array SHA·실제 calls·I/O wall 보존. nonempty directory 재실행/복구 금지.
- 감사는 저장 array 전체 통계 재집계·선택/cut/판정/seed/count 검증과 지정한 대표 physics 재생을 구분한다.
- 오류가 있으면 기존 기록을 보존하고 INVALID로 보고. 통과 seed 대체나 사후 기준 변경 금지.

## 구현·검토

새 nonnegative moments/ratio-delta 코어에서 exact zero, unequal stratum covariance,
수기 gradient, 상수·빈 tail, quantile tie, 선택의 양 셀 최약 기준과 no eligible 분기를 테스트한다.
trace의 g/특징을 기존 evaluator 및 손계산과 대조하고, 과거 12개 target과의 일치를 전수 확인한다.
코드·전용 테스트 통과 후 첫 discovery draw 전 source/policy를 봉인한다.
실험 완료 후 Ruff/Mypy/전체pytest·source 불변성·대표재생을 검사하고 한 개의 결과 보고서를 작성한다.
