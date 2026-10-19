# 373성분 G0/G1: fine-grid reference의 새 고정 pilot

2026-10-16. [geometry oracle 계획](PHYSICAL_WINDOW_GUIDE_ORACLE_2026-10-16_KO.md)의
수정된 geometry oracle와 runner 검사를 통과한 후, 새 표본 생성 전에 동결한다. resource-v2의
[절전 복구 계획](RESOURCE_V2_SLEEP_RECOVERY_2026-10-16_KO.md)에 따른 전체32개 고정 seed 데이터 복구·4whole 감사
이후에만 실행하고 그 장시간 실험과 겹치지 않는다. 원래 v2 운영 실패는 보존한다.
아직 실제 G0/G1 표본 생성 전 선행 anchor만 변경하며 본 설정은 v2의 prefix 수치를
보고 바꾸지 않는다. 아래는 실제 sampling 실험이지만 원래 q의 학습/성능 인증은 아니다.

## 질문과 동일 조건

N4096에서 한-cell price shift가 약해지는 구조적 한계를 fixed physical window로
바꾸면 reference variance가 줄어드는지 본다. G0:31physical point templates.
G1:15point-volatility+16causal-window-volatility templates이며, **price 방향은31개 전부
고정 물리 window**다. 만기 절단을 포함한 세부사항은 oracle 계획을 따른다.
초기 geometry-only prototype에서 남았던 한-cell price 방향을 실제 표본 생성 전에 제거했다.
양쪽373Gaussian components, natural.1,
amplitudes2/4/6/8,price0/2/4, 실제 component energy 상한40를 유지한다.
전부 identity covariance이고 전체 정규화된 mixture density를 평가한다.
window의 길이T/8, 시점과 geometry는 oracle 계획대로 고정하며 data에 맞추지 않는다.

## 고정 grid와 추정량

원래2cells ×2guides ×8independent blocks=32blocks. 각block16384fine samples,
chunk512. fine4096/coarse2048의 exact Gaussian coupling, independent augmentation,
fine proposal density 하나를 사용한 ordinary paired means. 이전all-monitor49141성분
pilot와 pooling하거나 같은 복잡도 비교로 섞지 않는다.

namespace `physical-window-reference-v1`, TorchCPUfloat64/8threads 고정.
total fine524288draw, physical1048576g calls, density524288calls,
time-step physical3,221,225,472. 준비는124개의 linear adjoint geometry rows이며
basis-path/nonlinear-payoff simulation0회다. 별도의 실제 준비 wall을 기록한다.

caps: physical1.2m,density.6m,phase7200초,whole180초,prep60초,RSS4GiB.
완료 prefix 선택/실패 seed 교체/추가 표본 없음. 원래 법칙과 payoff/compensator를 바꾸지 않는다.

## 평가·감사·다음 분기

4개cell-guide의 μf/μc RSE, raw maximum share, equal-block LOO/RSE,
4comparison paired-difference upper를 보고한다. 같은cell의G0/G1 fine μ는
2comparison10% equivalence로 대조한다. 두 guide는 관련 geometry를 공유하므로
이를 guide-independent blind-spot exclusion으로 부르지 않는다.

RSE1.5%의 safety3 fresh-draw forecast를512 단위로 올림하며 minimum262144를 둔다.
이는 precision 달성이나 final 실행의 승인이 아니다. 최종 qualified reference에는
guide-free SMC와 새 fixed-count 검증이 필요하다. continuum accuracy도 별도다.

source/resource-v2/backend/geometry/seed/moment/cost를 묶고 각cell-guide rep0의
4block numerical replay를 수행한다. 감사physical131072/density65536회는 별도다.
P1/P2/large-training flag는false로 유지한다. G2 weights/bank/NN 학습은 미실행이다.
