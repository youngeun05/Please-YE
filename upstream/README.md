# upstream/

다른 개발 PC에서 만들어져 GitHub에 **zip 형태로** 올라온 코드를 풀어서 보관하는 곳입니다.
`git log`로 diff가 남지 않는 zip 대신, 파일 단위로 비교·검토할 수 있게 여기에 둡니다.

| 폴더 | 출처 | 상태 |
|---|---|---|
| `baseline-v1/` | youngeunlee, 2026-09-17 17:18 `Please-YE-baseline-v1.zip` (커밋 f5fb41e) | GPU 검증 안 됨 (동봉 `SHARING.txt` 참조) |

## baseline-v1 → 본 레포 반영 내역 (2026-09-22)

| 파일 | 처리 |
|---|---|
| `scripts/baseline_common.py` | `scripts/`에 채택 (공용 helper: id 읽기, 누수 guard, KITTI 16열 행 생성) |
| `scripts/benchmark_model.py` | `scripts/`에 채택 (파라미터 수 · FLOPs 추정 · latency). **정사각형 imgsz만 지원** → 1248×384 직사각형 옵션 추가 필요 |
| `scripts/train_baseline.py` | **채택 안 함**. PC00에서 실제 baseline(AP40 95.59)을 낸 `scripts/train_baseline.py`가 이미 있고 API가 다름. 여기 보관본과 비교만 가능 |
| `scripts/predict_kitti.py` | **채택 안 함**. 같은 이유. PC00 버전은 `--imgsz 1248x384` 직사각형 예측을 지원 |
| `scripts/test_baseline.py` | 채택 안 함. 보관본 `train_baseline.validate_data`에 의존하므로 본 레포 버전과 맞지 않음 |
| `README.md` | 보관본의 "Baseline v1" 절은 정사각형 640 기준. 본 레포는 공지 제약(가로 ≤1280, 직사각형 권장)에 따라 1248×384를 사용 |
| `splits/*`, `eval_val.txt`, 기존 8개 스크립트 | 본 레포와 **내용 동일** (줄바꿈만 CRLF) — 분할 SHA256 일치 확인 |

zip 원본은 `git show f5fb41e:Please-YE-baseline-v1.zip`으로 언제든 복원할 수 있습니다.
