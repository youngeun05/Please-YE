# Please-YE — KITTI 2D 객체 검출 INT8 최적화 (트랙 2 지정주제)

YOLO11n을 KITTI로 학습하고, ONNX Runtime 정적 INT8 양자화(PTQ)로 최적화한 결과를 재현하는 레포입니다.
**GPU 없이 CPU만으로** 아래 명령을 따라 하면 보고서의 정확도와 속도 숫자가 다시 나옵니다.

| | Baseline | 최적화 모델 |
|---|---|---|
| 모델 | YOLO11n FP32 (ONNX) | YOLO11n INT8 (ONNX QDQ, percentile 보정) |
| 파일 | `models/yolo11n_kitti_1248x384_fp32.onnx` | `models/yolo11n_kitti_1248x384_int8.onnx` |
| 입력 | 1×3×384×1248 (고정) | 1×3×384×1248 (고정) |

보고서 수치는 `eval_val.txt` 1,000장에서 KITTI 2D **Moderate AP40**(Car·Pedestrian·Cyclist 평균)으로 측정했습니다.
개발 중 수치(internal_val 기준)는 [`실험결과1013.md`](실험결과1013.md)에 있습니다.

---

## 재현 절차

필요한 것: Python 3.10~3.13, 인터넷, 디스크 약 **30 GB** (KITTI ZIP 12 GB + 압축 해제 12 GB + 패키지). GPU는 필요 없습니다.
명령은 레포 루트에서 실행합니다. Windows는 PowerShell, Linux·macOS는 bash 기준입니다.

### 1. 환경 구성

```powershell
# Windows (PowerShell)
git clone https://github.com/youngeun05/Please-YE.git
cd Please-YE
py -3.12 -m venv .venv            # 3.10~3.13 중 설치된 버전
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt -r requirements-onnx.txt
```

```bash
# Linux / macOS (bash)
git clone https://github.com/youngeun05/Please-YE.git
cd Please-YE
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt -r requirements-onnx.txt
```

PowerShell에서 스크립트 실행이 막히면 먼저 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`를 실행합니다.

### 2. KITTI 데이터 (공식 벤치마크 지정 파일 2개)

`data_object_image_2.zip`(약 12 GB)과 `data_object_label_2.zip`(약 5 MB)을 받습니다.
[KITTI 공식 페이지](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)에서 받거나, 아래 미러를 씁니다.

```powershell
# Windows: 끊기면 같은 명령을 다시 실행하면 이어받습니다
.\scripts\download_kitti.ps1 -ZipDir D:\downloads
python scripts\prepare_kitti.py --image-zip D:\downloads\data_object_image_2.zip --label-zip D:\downloads\data_object_label_2.zip --eval-split eval_val.txt --output-root D:\datasets\KITTI
```

```bash
# Linux / macOS
mkdir -p ~/downloads && cd ~/downloads
curl -L -C - -O https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
curl -L -C - -O https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
cd -
python scripts/prepare_kitti.py --image-zip ~/downloads/data_object_image_2.zip --label-zip ~/downloads/data_object_label_2.zip --eval-split eval_val.txt --output-root ~/datasets/KITTI
```

`prepare_kitti.py`는 ZIP 전체의 CRC를 검사한 뒤 `training/image_2`와 `training/label_2`(각 7,481개)만 추출합니다. 수 분이 걸립니다.

### 3. 평가기 (MMDetection3D 커밋 `fe25f7a`에 고정)

```powershell
.\scripts\setup_evaluator.ps1          # Windows
```

```bash
bash scripts/setup_evaluator.sh        # Linux / macOS
```

### 4. 모델 파일

GitHub Release에서 `please-ye-models.zip`을 받아 **레포 루트에서** 압축을 풉니다. `models/` 폴더가 생깁니다.

- 다운로드: <https://github.com/youngeun05/Please-YE/releases/download/prelim-v1/please-ye-models.zip>

```powershell
Invoke-WebRequest https://github.com/youngeun05/Please-YE/releases/download/prelim-v1/please-ye-models.zip -OutFile please-ye-models.zip
Expand-Archive please-ye-models.zip -DestinationPath . -Force
```

```bash
curl -L -O https://github.com/youngeun05/Please-YE/releases/download/prelim-v1/please-ye-models.zip
unzip -o please-ye-models.zip
```

파일 목록·SHA-256·포맷·버전은 압축 안의 `models/MODELS.md`에 있습니다.
원본 PyTorch 체크포인트(`yolo11n_kitti_fp32.pt`)도 함께 들어 있습니다.

### 5. 벤치마크 실행 (명령 하나)

```powershell
# Windows
python scripts\run_benchmark.py --kitti-root D:\datasets\KITTI --split eval_val --confirm-final
```

```bash
# Linux / macOS
python scripts/run_benchmark.py --kitti-root ~/datasets/KITTI --split eval_val --confirm-final
```

이 명령이 하는 일:

1. Baseline과 최적화 모델로 eval_val 1,000장을 예측합니다 (CPU, 배치 1, 입력 1248×384). 이미지마다 KITTI 형식 파일을 하나씩 씁니다.
2. 고정 평가기로 Car·Pedestrian·Cyclist Moderate AP40과 평균을 계산합니다.
3. `benchmark_onnx.py`로 모델 크기, 파라미터 수, FLOPs, Latency(p50·p95), Throughput, 피크 메모리를 잽니다 (4스레드, warmup 30회, 반복 200회).
4. 결과를 보고서 표 형식으로 `runs/benchmark/eval_val/summary.md`에 씁니다 (2장 요약, 6.3 정확도, 6.4 속도).

`--split eval_val`에는 `--confirm-final`이 필요합니다. eval_val로 모델을 고르는 일을 막기 위한 장치로,
첫 실행 때 모델 해시를 `runs/benchmark/eval_val/FINAL_LOCK.json`에 기록하고 다른 모델로는 다시 채점하지 않습니다.
같은 모델로 다시 실행하는 것은 허용됩니다.

### 6. 기대 출력

실행이 끝나면 `summary.md` 맨 아래 「재현 확인」에 **재현 성공 / 재현 실패**가 자동으로 표시됩니다.
스크립트가 먼저 이 CPU의 INT8 계산 방식을 검사하고(`scripts/int8_probe.py`), 그 CPU 종류의 기대값과 0.01점 이내인지 비교합니다.

#### CPU에 따른 INT8 결과 (중요)

**같은 INT8 모델 파일이라도 CPU에 VNNI 명령어가 있느냐에 따라 정확도가 다르게 나옵니다.** FP32 baseline은 어떤 CPU에서나 같습니다.

| CPU 종류 | 예 | Baseline (FP32) | 최적화 (INT8) | Car / Pedestrian / Cyclist (INT8) |
|---|---|---|---|---|
| **VNNI 있음** (보고서 기준) | Intel 12세대 이상(AVX-VNNI), Xeon Cascade Lake 이상·AMD Zen 4(AVX512-VNNI) | 94.08 | **93.71** | 97.95 / 90.86 / 92.31 |
| VNNI 없음 | AVX2만 있는 CPU (Broadwell·Skylake Xeon, AMD Zen 2/3 등) | 94.08 | 93.05 | 98.03 / 90.03 / 91.09 |

- **이유**: ONNX Runtime은 uint8 활성값 × int8 가중치를 VNNI가 있으면 32비트로 정확히 더하고,
  없으면 VPMADDUBSW 명령으로 두 곱을 16비트에 더하면서 값이 32,767에서 잘립니다(포화).
  `python scripts/int8_probe.py`가 255 × 127을 64번 곱해 더해 보는데, VNNI CPU는 정확한 2,072,640,
  VNNI 없는 CPU는 1,048,544(= 32쌍 × 32,767)를 냅니다.
- 두 기대값은 같은 모델 파일(SHA-256 일치)로 각각 PC00(i9-12900)과 Google Colab(Xeon @ 2.20GHz, AVX2)에서 얻었습니다:
  [`results/eval_val_expected_by_cpu.json`](results/eval_val_expected_by_cpu.json), [`results/colab_avx2/`](results/colab_avx2/).
- 보고서의 INT8 수치는 정확한 계산 경로인 **VNNI 기준**입니다. 본선 장비(Hailo-10H)는 자체 INT8 하드웨어로 실행하므로 이 차이와 무관합니다.
- VNNI 없는 CPU에서는 INT8 속도 이득도 작게 나옵니다 (예: Colab 2코어에서 1.20배).

6.4의 속도와 메모리는 **측정하는 PC에 따라 달라집니다.** 보고서 수치는 아래 환경에서 측정했습니다.

- Intel i9-12900 (P코어 8 + E코어 8), Windows 11, ONNX Runtime 1.24.4 CPU provider, 4스레드, **P코어 고정**, 2회 측정 평균
- P코어와 E코어가 섞인 인텔 12세대 이상 CPU에서는 P코어에 고정해야 측정이 흔들리지 않습니다.
  Windows 예: `cmd /c 'start "" /affinity FFFF /b /wait .venv\Scripts\python.exe scripts\run_benchmark.py ...'`
  (`FFFF` = 논리 CPU 0~15, CPU에 맞게 조정. 자세한 내용은 [`ONNX_INT8.md`](ONNX_INT8.md))

### 재현 시 유의사항

- 처음 실행할 때 걸리는 시간: KITTI 다운로드(회선에 따라 수십 분), 압축 해제와 CRC 검사(수 분), 예측(모델당 수 분), 평가(모델당 수 분).
- `numba`는 설치하지 않아도 됩니다. 평가기가 결과가 같은 순수 Python 경로로 대신 실행합니다 (속도만 느려짐).
- ONNX Runtime의 INT8 이득은 CPU provider에서 나타납니다. CUDA provider는 INT8 연산 대부분을 float로 되돌려 실행합니다.
- 내부 검증셋으로만 확인하려면 `--split internal_val`을 씁니다 (`--confirm-final` 불필요).
- 측정 없이 정확도만 보려면 `--no-speed`를 붙입니다.
- 내 CPU가 어느 INT8 경로를 쓰는지만 확인하려면 `python scripts/int8_probe.py`를 실행합니다.

---

## 처음부터 다시 만들기 (학습 → ONNX → INT8)

| 단계 | 문서 / 스크립트 |
|---|---|
| 데이터 준비, 분할, YOLO 변환, 학습 (Windows·CUDA) | [`SETUP_LOCAL.md`](SETUP_LOCAL.md), `scripts/train_baseline.py` |
| ONNX export → INT8 PTQ → 측정 | [`ONNX_INT8.md`](ONNX_INT8.md), `scripts/export_onnx.py`, `scripts/quantize_ort.py`, `scripts/benchmark_onnx.py` |
| 제출 모델 패키징 | `scripts/package_models.py` |
| 실험 기록과 해석 | [`실험결과1013.md`](실험결과1013.md), [`계획점검0922.md`](계획점검0922.md), [`작업정리0922.md`](작업정리0922.md) |

## 데이터 분할 규칙

| 용도 | 수량 | 파일 |
|---|---:|---|
| 학습 | 5,481 | `splits/train.txt` |
| 내부 검증 (모델·설정 선택) | 1,000 | `splits/internal_val.txt` |
| INT8 보정 (calibration, 학습 집합의 부분집합) | 1,024 | `splits/calibration.txt` |
| **공식 평가 (학습·선택에 사용 금지)** | 1,000 | `eval_val.txt` |

세 독립 집합(train, internal_val, eval_val)의 교집합은 0이며, 학습·export·양자화 스크립트가 실행할 때마다 이를 검사합니다.
분할 알고리즘과 클래스·난이도 분포는 `splits/split_report.json`에 있습니다.
KITTI 이미지, 생성 라벨, 체크포인트, ONNX 파일은 git에 올리지 않습니다 (`.gitignore`).
