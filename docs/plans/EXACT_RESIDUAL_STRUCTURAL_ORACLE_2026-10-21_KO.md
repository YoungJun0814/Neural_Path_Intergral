# 정답 기반 residual 구조 진단 — 다음 우선순위 실행계획

작성일2026-10-21. 기존 특징 확인 실패를 재시도하지 않는 별도 이론/수치 진단이다.
사용자 요청 범위는 이 단계 완료·검토 후 누적 연구 내용을 GitHub main에 커밋·푸시하는 것이다.

## 질문과 제한

1. 좋은 사건 proposal q도 나쁜 평가분포 G 때문에 위험 평가가 실패할 수 있음을 정답이 알려진 예로 구분한다.
2. 평균이동이 다루지 않는 Gaussian residual에 g가 의존할 때 weight-only 수정의 한계를 수식으로 도출한다.
3. 실제 N32 dictionary에서 exact-zero 이동 좌표와 payoff에 쓰이지 않는 좌표를 구분한다.
4. 중요하게 보인 한 좌표의 조건부 payoff를 기존 전체 evaluator와 별도 scalar식으로 대조한다.

q/G/373성분·가중치·D3 gate를 바꾸지 않는다. 새 희귀사건 MC bank, 대규모 학습, 새 모델 채택은 없다.
일반적인 Gaussian 적분·Cauchy–Schwarz·Rao–Blackwell 원리를 새 정리라고 주장하지 않는다.

## 단계1 — 식과 exact toy

p=N(0,I_d), g(z)=exp(-||z-c||²/(2s²)), q_m=N(m,I_d)를 사용한다.
정확 μ와M₂를 completing-square로 계산한다. m*=c/(1+s²)라 두면

```
log(M2(q_m)/mu²)
 = d*log((s²+1)/sqrt(s²*(s²+2))) + (s²+1)/(s²+2)*||m-m*||².
```

고정q 위험의 정규화 target r_q=g²p²/(qM₂)는 Gaussian이며
mean=(2c-s²m)/(s²+2), variance=s²/(s²+2)다.
G=N(b,wI), r=N(a,vI)에 대해

```
log integral(r²/G)=d*log(w/sqrt(v*(2w-v)))+||a-b||²/(2w-v),  2w>v.
```

event RSE와 risk-evaluation RSE를 정확식으로 분리한다.
고정2D c=(3,0),s=1,m=m*, Gbad=N((0,4),I), Gmatched=r_q를 비교한다.
이는 production defensive mixture를 그대로 모사한 모델이 아니므로 toy의 수치 크기를 실제 셀로 이전하지 않는다.
추가 residual-only 1D bump c=4,s=.75의 floor와, 영향 없는 residual g=1의 floor1을 대조한다.

## 단계2 — 일반 subspace 식과 적용 조건

Z=(X,Y), p=p_Xp_Y, q=q_Xp_Y이고 A(X)=E[g²|X]라면

```
M2(q)=E_pX[A(X)/(q_X/p_X)] >= (E_pX[sqrt(A(X))])².
```

unrestricted q_X에서 equality density는 p_X sqrt(A)/E sqrt(A)다.
μ=E E[g|X]와의 gap은 조건부 변동과 연결되지만, 실제 Volterra의 양적 floor는 아직 구하지 않는다.
평균이동 mixture의 모든 means가 S에 있으면 S⊥에서 base conditional Gaussian을 유지한다.
defensive floor가 있어도 이 하한은 relaxed family에 대한 하한으로 유효하지만 equality는 일반적으로 불가하다.

## 단계3 — 실제 구조의 결정론적 검사

부모 D2 result SHA `323b00268d797b6e70222a748ce500ee4c5f7f1ff9da1e6724acd750cb3ee1bb`,
특징 진단 result SHA `eff4fd1406719163bdc333c62f79484377ba155ef3a6d58910c3739ba4dbd9f5`를 연결한다.
source를 읽은 사전 점검에서 exact-zero mean 좌표가[57,61,63], 수치 rank가49인 것을 관측했다.
rank는 tolerance에 따른 수치 진단이며 정리의 가정이나 양적 성능 하한으로 쓰지 않는다.

61=2N−3은 V_(N−1)에 들어가는 마지막 local integral residual이다. q/G 밀도비는 이 좌표와 무관하다.
63=2N−1은 미사용 마지막V_N에만 들어가므로 terminal payoff와 무관해야 한다.
좌표57은 이동되지 않는 사실만 기록하며 이번에 별도 선택/보강하지 않는다.

고정 context는 두 셀 각각 zero, 기존 discovery block0/U top0, 기존 confirmation block0/U top0의3개다.
뒤의 두 개는 이미 본 경로의 기술적 대조로서 population 대표표본이나 독립 통계검정이 아니다.
각context에서 좌표61과63을 각각[-4,-2,0,2,4]로 바꿔 full evaluator와 별도 조건부식을 대조한다.

U=z61의 조건부식은 V31(U)=V31(0)exp(beta U), beta=eta*sqrt(2H)*L11이며,
A0/I0는 마지막 사용 구간 이전 terminal/int variance, dW_last는 마지막 가격 증분 driver다.

```
g(U)=Phi((log K-A0+.5*dt*V31(U)-rho*sqrt(V31(U))*dW_last)
         /sqrt((1-rho²)*(I0+dt*V31(U)))).
```

과거 evaluator·density를 고치지 않고 새 module로 구현한다. Gaussian quadrature48/96/192와
adaptive scalar quadrature[-12,12]에서 E[g],E[g²]를 비교한다. absolute omitted tail≤2Phi(-12).
상대수렴≤1e-5를 요구하되, 매우 작은 값은 absolute≤1e-12로 통과할 수 있으며 그때 relative resolved로 부르지 않는다.
kernel/numerical overflow/불일치는 INVALID, 수렴 실패 context도 지우지 않고 UNRESOLVED로 보존한다.

## 예산과 판정

- 실제 cached quantities call: context마다 baseline1 + probe10 =11, 두셀×3contexts=66.
- 대표 audit: 두셀×zero context의 baseline/probe를재생22. full Volterra cap **100**.
- synthetic full quadrature/조건부 scalar evaluator/독립 감사 cap **50,000**, wall900초, CPUfloat64 1thread.
- toy normal integral은 adaptive scalar 검증, randomness/새 학습 seed 없음.
- production/audit 분리계측, 실패비용 보존, nonempty result 디렉터리 덮어쓰기 금지.
- code/계획/anchors를 첫probe 전에동결. 판정 후 source를 결과에 맞춰 수정하지 않음.
- q/G invariance와g의해당좌표 의존이 확인되면 STRUCTURAL_BLIND_COORDINATE_CONFIRMED로만표시.
  포화된context는g변화가안보일수있다. 각셀에서한context이상의의존검출을요구하며,
  모든context에서동일한영향크기를요구하지않는다. 식의불일치와finiteprobe의무변화를구분한다.
  global M₂ 지배·실용 비용개선·P2통과·논문독창성은확인하지않음.
- conditional integration은 진단용으로만 수행. 생산 Rao–Blackwell estimator의 근사 bias/비용을 검증한것이아님.

## 검토·푸시

analytic 식·별도수치적분·수기조건부식·변조/무관좌표·기존회귀테스트를검증한다.
전체로컬pytest, Ruff/Mypy와원본anchorhash를검사하고 README/결과/이론문서를작성한다.
누적미커밋연구파일의size/민감정보/임시파일을확인한다. 기존연구증거를보존하고정상fast-forward만사용한다.
커밋작성자는기존저장소설정을유지하고불필요한기여자표기를추가하지않는다.
push후로컬HEAD와원격main SHA를대조한다. force-push는하지않는다.
