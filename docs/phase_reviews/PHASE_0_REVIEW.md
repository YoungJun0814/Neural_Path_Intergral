# Phase 0 자체 검토

**커밋**: `9e4aa5c chore(phase-0): make project runnable and CI-ready`
**범위**: 실행 불가 상태 탈출, CI 파이프라인, 테스트 스캐폴드, 설정 관리

---

## 1. 구현 내용 요약
- `main.py`를 서브커맨드 기반 CLI로 완전 재작성, 잘못된 import 제거
- `src/utils.py`에 `pick_device`, `git_hash` 추가, 시드 고정 강화
- `pyproject.toml` + `requirements-dev.txt`로 빌드/테스트/린트 도구 체계 확립
- `configs/default.yaml`로 파라미터 외부화
- `tests/` 디렉터리 + `conftest.py`로 pytest 스위트 시작 (7개 스모크 테스트)
- `.github/workflows/ci.yml`에서 ruff + pytest (py 3.10/3.11) 자동화
- `.gitattributes`(LF 강제), `.gitignore`(pyc/checkpoint/data raw 차단)

---

## 2. 기술적 오류 검토

### 2.1 남아있는 이슈
1. **`main.py` 최상단 `import torch`**: `python main.py --help`조차 torch 설치를 요구. Phase 1에서 torch import를 subcommand 내부로 내리고 `--help`는 의존성 없이 동작하도록 개선 필요.
2. **`cmd_calibrate`가 플레이스홀더**: "Phase 3에서 완성" 주석만 남김. 스모크 테스트가 이 경로를 실질적으로 검증하지 못함. Phase 3 전에 최소한 `pytest -k calibrate`가 통과하도록 mock/synthetic 파이프라인 필요.
3. **`tests/test_smoke.py::test_market_simulator_heston_shapes`** 는 num_paths=64로 매우 작아 통계적 특성은 검증 못 함. 본격 Heston 검증은 Phase 2.1에서 해석해 대조 테스트로 보강 예정. ✅ (계획된 진행)
4. **`configs/default.yaml`의 `target_kurtosis: null`** 주석은 "data-driven"이지만, `train_ipm` 실행 경로에서 이 값을 읽고 자동 계산하는 로직은 아직 없음. Phase 1.7에서 구현 필요.
5. **`.github/workflows/ci.yml`의 pip 캐시 키가 python 버전만 포함**: 매트릭스에 OS 추가 시 캐시 충돌 가능. 현재는 ubuntu-only이므로 문제 없음. ✅
6. **`conftest.py` sys.path 조작은 표준 관행에 약간 벗어남** (editable install이 더 청결). Phase 4에서 `pip install -e .` 체계로 이행 예정. ✅

### 2.2 잠재적 회귀 위험
- `src/utils.py`의 기존 퍼블릭 심볼(`set_seed`, `to_numpy`, `HestonDataset`, `plot_weighted_paths`)은 시그니처 유지. 기존 노트북 호환성 OK.
- `test_kurtosis_loss.py`의 기존 `if __name__ == "__main__"` 경로는 `tests/` 내부 함수를 재사용하므로 동작은 유지되지만 import 경로 변경됨. 외부 사용자가 직접 이 스크립트를 호출하는 경우는 없으므로 영향 낮음.
- `requirements.txt`에서 `torchvision`, `torchaudio` 제거. 프로젝트 내부에서 이 두 패키지를 사용한 흔적은 없어 안전.

---

## 3. 이론적 오류 검토

Phase 0은 수학적 모델을 건드리지 않으므로 해당 없음. 단:

- `IMPROVEMENT_PLAN.md`에 명시된 Phase 1.1–1.8의 모든 수학 정정은 Phase 1에서 일괄 처리.
- Phase 0에서 확정한 CLI와 config는 Phase 1의 구현을 받아낼 수 있는 형태로 설계되었는지 점검:
  - `configs/default.yaml`은 `heston.jump_lambda` 등 jump 파라미터를 이미 수용. ✅
  - rBergomi 전용 파라미터 (`H`, `eta`, `xi`, `rho`)는 현재 config에 없음. Phase 1.4에서 `rbergomi:` 섹션 추가 필요. ⚠️
  - Girsanov control 관련 파라미터 (`u_bound`, `kl_weight`)는 Phase 1.8 구현 시 새 섹션 필요. ⚠️

---

## 4. 커버리지 / 검증 현황

| 지표 | 현재 | 목표 (Phase 2 말) |
|---|---|---|
| 유닛 테스트 수 | 7 | ≥ 30 |
| Line coverage | 측정 안 됨 | ≥ 80% |
| CI 통과 여부 | 푸시 후 확인 | Green |
| Import/Smoke | ✅ 통과 확인 (로컬 AST 파싱) | ✅ |
| Heston 해석해 대조 | ❌ 없음 | ✅ (Phase 2.1) |
| Girsanov unbiasedness | ❌ 없음 | ✅ (Phase 2.2) |

---

## 5. 다음 Phase 착수 전 TODO

1. **Phase 0 문제점 수정**은 Phase 1 커밋에 포함:
   - `main.py`의 top-level torch import를 함수 내부로 이동
   - `cmd_calibrate` 플레이스홀더 최소 동작 보강
   - `configs/default.yaml`에 `rbergomi:`, `girsanov:` 섹션 선제 추가
2. Phase 1 시작 전에 `docs/formulation.md` 초안 작성 (수학적 통일이 가장 먼저)
3. 린트/포맷 오류 없이 통과하는지 ruff 드라이런 (sandbox에선 실제 ruff 실행 불가, 코드 리뷰 기준으로 semantic clean)

---

## 6. 푸시 상태

- 커밋 `9e4aa5c`가 워크스페이스 `.git`에 반영됨
- 샌드박스에서 `github.com`으로의 아웃바운드가 프록시로 차단되어 자동 `git push` 불가
- `phase-0.bundle` 파일을 워크스페이스에 배치 (`git bundle unbundle phase-0.bundle` 대안 경로 제공)
- **사용자 수동 조치 필요**: 로컬 터미널에서 `cd` 후 `git push origin main`
