# 이 PC(PC00) 세팅 가이드

원본 `README.md`의 경로는 다른 PC(`C:\Users\pro\...`) 기준입니다.
이 문서는 **이 PC에서 그대로 따라 할 수 있는 순서**만 정리했습니다.

- 레포 위치: `C:\Users\PC00\Desktop\이영은\이영은 모커톤` (C: = SSD)
- **데이터셋 위치: `D:\datasets\KITTI`** (D: = HDD, 여유 1.3TB)
- **ZIP 위치: `D:\downloads`**

> **왜 D: 인가**: C:(SSD) 여유가 20.7GB뿐이라 ZIP 11.7GB + 압축 해제 12GB를 담을 수 없습니다.
> D:는 HDD지만 이 PC RAM이 31.7GB이므로 첫 epoch 이후 12GB 이미지가 페이지 캐시에
> 대부분 올라가 학습 속도 손실은 크지 않습니다. C:는 venv와 `runs\` 체크포인트용으로 남깁니다.
> 그래서 아래 모든 명령에 `-DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads`가 붙어 있습니다.

---

## 0. PowerShell 열기 + venv 활성화

레포 폴더에서 **PowerShell**을 엽니다. (탐색기 주소창에 `powershell` 입력)
아래 명령은 전부 레포 루트에서, **venv를 활성화한 상태로** 실행합니다.

```powershell
Set-Location "C:\Users\PC00\Desktop\이영은\이영은 모커톤"; .\.venv\Scripts\Activate.ps1
```

프롬프트 앞에 `(.venv)`가 붙으면 정상입니다. 활성화가 거부되면 한 번만:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

> **중요**: 활성화하지 않으면 PATH의 `python`이 MSYS2 빌드
> (`C:\msys64\ucrt64\bin\python.exe`)로 잡히고, 여기에는 PyTorch가 설치되지 않습니다
> (MinGW 휠 태그라 `download.pytorch.org`의 win_amd64 휠을 받지 못함).
> `run_prepare.ps1`도 PATH의 `python`을 쓰므로 반드시 venv가 먼저 활성화돼 있어야 합니다.

## 1. 환경 점검

```powershell
.\scripts\check_env.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads
```

Python 버전, PyTorch/CUDA, GPU, 디스크 여유 공간, KITTI 존재 여부를 한 번에 출력합니다.
확인 기준:

- `python path`가 `...\.venv\Scripts\python.exe` (MSYS2 경로면 venv 활성화 안 된 것)
- `cuda avail : True`
- D: 여유 **30GB 이상**

## 2. 파이썬 패키지 설치 (이미 완료됨 — 재구축이 필요할 때만)

venv는 이미 만들어져 있습니다. 날려먹었을 때 아래 순서로 똑같이 복구합니다.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
$env:PIP_CACHE_DIR = "D:\pipcache"
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r .\requirements-tested.txt
```

- **Python 3.13.2**를 씁니다. PATH의 MSYS2 python(3.11.9)과 anaconda base(3.8.18)로는 안 됩니다.
  Python 3.14도 설치돼 있지만 ultralytics/torch 휠이 아직 얇아 3.13을 골랐습니다.
- **cu124 / torch 2.6.0을 고정합니다.** 원본 `README.md`가 검증 환경으로 명시한 버전이고,
  cu124 채널에 Python 3.13 휠이 있어 이 PC에서도 그대로 쓸 수 있습니다
  (`cuda avail: True`, `get_arch_list()`에 `sm_86` 포함, GPU matmul 통과).
  더 새 버전(2.14.0+cu126)도 동작하지만 **검증 스택에서 벗어나므로 쓰지 않습니다.**
- `requirements-tested.txt`에 없는 패키지는 넣지 않습니다. 특히 **numba를 설치하지 마세요**:
  `evaluate_kitti.py`에 의미를 보존하는 폴백이 있어 없이도 동작하며, 계획의 검증 대상이 아닙니다.
- `PIP_CACHE_DIR`을 D:로 돌리는 이유는 torch 휠 2.6GB가 C: 여유를 두 번 먹지 않게 하려는 것입니다.

## 3. KITTI 2D 객체 검출 데이터 내려받기

필요한 파일은 두 개입니다.

| 파일 | 크기 (미러 실측) | 내용 |
|---|---:|---|
| `data_object_image_2.zip` | 11.7GB (12,569,945,557 B) | left color images (training 7,481 + testing) |
| `data_object_label_2.zip` | 5.3MB (5,601,213 B) | training labels |

```powershell
.\scripts\download_kitti.ps1 -ZipDir D:\downloads
```

중간에 끊기면 **같은 명령을 다시 실행**하면 이어받습니다.
미러가 거부하면 KITTI 공식 페이지에서 가입 후 같은 두 파일을 `D:\downloads`에 직접 받으세요.
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d>

## 4. 데이터 준비 (압축 해제 → 감사 → 변환)

```powershell
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads
```

내부에서 순서대로 실행됩니다.

1. `prepare_kitti.py` — ZIP CRC 전수 검사 후 `training\image_2`, `training\label_2`만 안전하게 추출
2. `audit_kitti.py` — 7,481장 이미지/라벨 일치, PNG 포맷, bbox 범위, 클래스 분포 검사 → `runs\audit_report.local.json`
3. (선택) `make_splits.py` — 분할 재생성 후 커밋된 분할과 SHA256 비교 (아래 주의 참고)
4. `data\kitti_yolo_exact3\images\all`, `data\kitti_yolo_neighbors3\images\all` **junction 생성** (이미지 복사 안 함)
5. `convert_kitti_to_yolo.py` — exact3(3클래스만) / neighbors3(Van→Car, Person_sitting→Pedestrian) 두 버전 생성

유용한 옵션:

```powershell
# 분할 재현성까지 확인 (시간이 더 걸림)
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads -VerifySplits

# ZIP CRC 전수 검사 생략 (빠르지만 덜 엄격)
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads -SkipCrc

# 이미 압축을 풀었으면 추출 단계만 건너뛰기
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads -SkipPrepare

# 다른 드라이브 사용
.\scripts\run_prepare.ps1 -DatasetRoot E:\datasets\KITTI -ZipDir E:\downloads
```

## 5. 산출물 검증

```powershell
python .\scripts\verify_prepared.py --kitti-root D:\datasets\KITTI
```

확인 항목:

- 분할 개수(5,481 / 1,000 / 1,024 / 1,000), 중복 없음, **세 집합 교집합 0**, calibration ⊂ train
- 추출된 KITTI 7,481장 이미지/라벨 id 일치
- YOLO 라벨 파일 7,481개, 5필드, class id ∈ {0,1,2}, 정규화 좌표 ∈ [0,1]
- `lists\*.txt`의 이미지 경로 실재 + 라벨 경로 매핑(`images`→`labels`) 성립
- `kitti.local.yaml`, `calibration.local.yaml` 생성

데이터 없이 분할만 확인하려면:

```powershell
python .\scripts\verify_prepared.py --splits-only
```

## 6. CUDA 스모크 테스트

```powershell
python .\scripts\smoke_train.py --device 0 --fraction 0.01
```

55장으로 1 epoch만 돌려 이미지·라벨·손실·역전파·CUDA 경로만 확인합니다.
**성능 수치로 쓰지 않습니다.**

## 7. 평가기 설치 및 검증 (선택, 지금 단계에서는 데이터 준비 후)

```powershell
.\scripts\setup_evaluator.ps1
python .\scripts\verify_evaluator.py --kitti-root D:\datasets\KITTI --work-dir .\runs\evaluator_sanity
```

정답 박스를 그대로 예측하면 세 클래스 100.0, 빈 예측은 0.0이어야 정상입니다.

---

## 절대 지켜야 할 기준 (원본 README와 동일)

- `eval_val.txt`의 1,000장은 학습·하이퍼파라미터 선택·증강 탐색에 **사용 금지**
- 최종 비교 지표는 Car/Pedestrian/Cyclist의 **KITTI 2D Moderate AP40 평균**
- 이미지·생성 라벨·체크포인트·변환 산출물은 git에 올리지 않음 (`.gitignore`)
- `해체분석.md`는 공지 전 초안이므로 실험 기준으로 쓰지 않음

## 이 PC에서 추가된 파일

| 파일 | 용도 |
|---|---|
| `scripts\check_env.ps1` | Python/CUDA/GPU/디스크/데이터 존재 여부 점검 |
| `scripts\download_kitti.ps1` | KITTI 두 ZIP 이어받기 다운로드 |
| `scripts\run_prepare.ps1` | 추출→감사→(분할검증)→junction→변환 일괄 실행 |
| `scripts\verify_prepared.py` | 분할 무결성 + YOLO 산출물 형식 검증 |
| `SETUP_LOCAL.md` | 이 문서 |

원본 스크립트(`prepare_kitti.py`, `audit_kitti.py`, `make_splits.py`, `convert_kitti_to_yolo.py`, `evaluate_kitti.py`, `verify_evaluator.py`, `smoke_train.py`)는 **수정하지 않았습니다.**

---

## 공지에서 온 하드 제약 (설계 결정의 근거)

`트랙2 지정주제 벤치마크 공지.hwpx`와 `2026_모바일_하반기공모전_트랙2_최종안내.docx`에서
모델·해상도 결정에 직접 걸리는 항목입니다. **이 문서를 먼저 읽지 않고 설계하지 마세요.**

- **입력 해상도**: 가로 **1280 이하로 제한**, 양변 32의 배수, **직사각형 강력 권장**.
  정사각형(640×640)은 letterbox 여백에 NPU 연산을 낭비한다고 명시.
  권장 프리셋은 **1280×384**(정밀도) / **640×192**(초경량).
  이 PC에서 실측한 KITTI 원본 크기는 4종(1242×375 6057장, 1224×370 770장, 1238×374 358장,
  1241×376 296장)으로 비율이 모두 약 3.31이며, 두 프리셋(3.33)과 거의 일치합니다.
- **아키텍처**: 팀 자유 선정이나 Hailo Model Zoo에서 호환 검증된 **Conv 기반(YOLOv8/v10/v11) 권장**.
  Hailo DFC 미지원 커스텀 레이어나 복잡한 어텐션은 Pi 5 **CPU Fallback**을 유발.
- **평가 메트릭**: KITTI 표준 **Moderate AP40**, Car/Pedestrian/Cyclist 평균.
- **누수**: `eval_val.txt` 1,000장을 학습에 포함하면 **즉각 실격**.
- **예선 평가는 정확도 경쟁이 아닙니다**: 양자화 전후 Moderate mAP 유지율(Drop Rate),
  파라미터 수, FLOPs, 메모리, 데스크탑 Latency를 종합. 배점은 최적화 기법 25 + 성능 개선도 25
  vs 정확도 유지 15. baseline 모델과 양자화 모델을 **둘 다** 제출하며 학술부가 직접 재현 검증.
- **일정**: 예선 보고서 마감 **11월 1일 23:59**. 본선은 `.hef` 컴파일 필수(Hailo-10H 전용, 칩 간 비호환).

## 이 GPU에서 실측한 배치 상한 (RTX 3080 10GB)

`scripts\find_batch_size.py`로 실제 forward → detection loss → backward → AdamW step을
돌려 측정한 peak allocated 기준입니다. 데스크톱이 약 0.9GB를 쓰므로 가용 VRAM은 8.88GB이고,
검증·플로팅 여유를 위해 **peak 7GB 이하**를 권장값으로 잡았습니다.

| 모델 | 1280×384 | 640×192 |
|---|---|---|
| yolo11n | **48** (6.82GB) — 56까지 가능(7.93GB) | **128** (4.49GB) |
| yolo11s | **24** (6.34GB) — 28까지 가능(7.34GB) | **96** (6.24GB) |
| yolo11m | **12** (6.63GB) — 14까지 가능(7.64GB) | **48** (6.57GB) |

fp16 AMP 기준입니다. AMP를 끄면 위 값을 대략 절반으로 줄여야 합니다.

GPU 기준 epoch 시간은 전부 0.3~1.4분이라 **연산이 병목이 아닙니다.** 이미지가 D:(HDD)에 있어
첫 epoch은 디스크가 병목입니다. RAM이 31.7GB이므로 1280×384 기준 5,481장 캐시가 약 8GB로
`cache='ram'`이 들어갑니다 — 본 학습 때 검토하세요.

## 아직 레포에 없는 것 (예선 제출에 필요)

1. **양자화 파이프라인** — 예선 배점의 절반이 여기에 걸려 있는데 코드가 없습니다.
   `splits/calibration.txt`(1,024장)와 `calibration.local.yaml`은 준비돼 있습니다.
2. **본 학습 스크립트** — `smoke_train.py`는 스모크 전용이며 "성능 수치로 쓰지 않는다"고 명시.
3. **측정 스크립트** — 파라미터 수, FLOPs, 메모리, Latency는 예선 평가 항목인데 측정 도구가 없습니다.
4. **ONNX/HEF export 경로** — Hailo DFC 미지원 레이어 사전 확인 포함.
5. **exact3 vs neighbors3 결정** — README가 "내부 AP40 결과로 결정, 아직 가정하지 않음"으로 미뤄둠.

## 원본 README와 의도적으로 다른 점

아래 세 가지 외에는 `README.md`의 계획을 그대로 따릅니다. 새로 추가할 때도 이 목록에 근거를 남기세요.

**1. 분할 재생성 출력 경로 — 계획 문구와 다름 (의도적)**

README는 `make_splits.py --output-dir .\splits`로 적혀 있는데, 이는 **커밋된 분할을 덮어씁니다.**
커밋된 분할은 이 프로젝트의 계약이므로 덮어쓰지 않습니다. `run_prepare.ps1 -VerifySplits`는
`runs\splits_check`에 생성한 뒤 `train/internal_val/calibration` 세 파일의 SHA256을 커밋본과
비교하고, 하나라도 다르면 즉시 중단합니다. 재현성 확인이라는 목적은 같고 계약만 보호합니다.

**2. 하드웨어 제약에서 온 차이**

| 항목 | README | 이 PC | 이유 |
|---|---|---|---|
| GPU | RTX A6000 48GB ×2 | RTX 3080 10GB ×1 | 이 PC 사양. 배치 크기는 `find_batch_size.py`로 재측정 |
| 데이터 위치 | `C:\...\Documents\Codex\datasets\KITTI` | `D:\datasets\KITTI` | C: 여유 20.7GB < 필요 24GB |
| Python | (명시 없음) | 3.13.2 + `.venv` | PATH의 python이 MSYS2 빌드라 torch 휠 설치 불가 |

**3. 버전은 계획을 그대로 따릅니다 (제약 아님)**

torch는 README 검증 버전인 **2.6.0+cu124**를 씁니다. cu124 채널에 Python 3.13 휠이 있어
GPU가 달라도 버전을 낮출 이유가 없습니다. `requirements-tested.txt`의 네 패키지도 핀 그대로입니다.
검증 목록에 없는 패키지(numba 등)는 추가하지 않습니다.

---

## 이 PC에서 실측·검증된 환경 (2026-09-17)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 10GB (sm_86), 드라이버 595.79 |
| RAM | 31.7 GB |
| C: (SSD, Samsung MZVL21T0HCLR) | 941GB 중 여유 20.7GB — venv + `runs\` 용 |
| D: (HDD, ST2000DM008) | 1,863GB 중 여유 1,315GB — 데이터셋 + ZIP + pip 캐시 용 |
| Python | 3.13.2 (`.venv`) |
| PyTorch | 2.6.0+cu124 (README 검증 버전), `cuda avail: True`, `sm_86`, GPU matmul 통과 |
| ultralytics / numpy / Pillow / PyYAML | 8.4.108 / 2.4.6 / 11.0.0 / 6.0.2 (`requirements-tested.txt` 그대로) |
| 분할 무결성 | `verify_prepared.py --splits-only` 21개 항목 ALL CHECKS PASSED |
| KITTI 미러 | `s3.eu-central-1.amazonaws.com/avg-kitti` HTTP 200, `Accept-Ranges: bytes` (이어받기 가능) |

아직 남은 단계: 3(다운로드) → 4(준비) → 5(검증) → 6(스모크) → 7(평가기).
