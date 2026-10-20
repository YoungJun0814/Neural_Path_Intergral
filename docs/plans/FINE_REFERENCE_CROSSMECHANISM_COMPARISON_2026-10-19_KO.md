# Fine-grid reference의 서로 다른 메커니즘 교차 검증

2026-10-19. complete resource32whole 및 G0/G1 실제 표본 결과를 읽기 전에 고정한다.
새 physical simulation0회인 postprocessing이며, 양쪽의 numerical replay receipt가 필요하다.

2cells×2SMCdesigns×2IIDguides의8개 fineN4096 μ 비교를 전부 산출한다. 잘 맞는
SMC design만 골라 비교하지 않는다. 서로 다른 namespace의 SMC whole와 IID guide는
독립 stream으로 취급하고 기존 symmetric-relative delta-method endpoint를 사용한다.
8comparison Bonferroni,margin10%,alpha.05. 이는 이8개 개발 비교만의 조정이며
전체 연구의 여러 trial을 포함한 simultaneous coverage 또는 distribution-free 보증이 아니다.
특히8SMC whole의 normal approximation과 미관측 tail 한계는 남는다.

관측 RSE1.5% 여부와 agreement uncertainty를 구분해 기록한다. 둘 다 작게 보인다는
사실을 독립 accuracy qualification이나 wall-time 비교의 자동 승인으로 쓰지 않는다.
source/result/receipt의 SHA와 archive bindings를 검사하고 모든8개를 한 번에 보고한다.

분기: disagreement/불확실성이 남으면 reference를 먼저 검토한다. agreement는 맞지만
precision이 부족하면 새 fixed-count precision study를 사전 고정한다. 둘 다 관측 기준을
충족해도 별도 새 confirmation/tail-coverage gate를 준비할 뿐 large-training은 열지 않는다.
G0/G1끼리의 agreement만으로 guide-independent blind spot을 배제하지 않는다.
