# Please-YE

> 로컬 복제 버전: `Please-YE-baseline-v1`, 브랜치 `baseline-v1`.
> 원격 원본에는 변경을 게시하지 않았습니다. 아래 스크립트는 이 복제본에서 실행합니다.

KITTI 2D 객체 검출을 Hailo-10H에 배포하기 위한 재현 가능한 실험 레포입니다. 현재 1단계는 **데이터 무결성, 누수 방지 분할, KITTI→YOLO 변환, AP40 평가기, CUDA 학습 경로**까지 검증되었습니다.

## 절대 지켜야 할 기준

- `eval_val.txt`의 1,000장은 학습·하이퍼파라미터 선택·증강 탐색에 사용하지 않습니다.
- 최종 정확도는 Car/Pedestrian/Cyclist의 **KITTI 2D Moderate AP40 평균**으로 비교합니다.
- KITTI 이미지, 생성 라벨, 체크포인트와 변환 산출물은 `.gitignore` 대상입니다.
- `해체분석.md`는 공지 전 초안이므로 실험 기준으로 사용하지 않습니다.

## 이 PC에서 확인한 경로

```text
repository  C:\Users\pro\Documents\ChatGPT\YE-please
KITTI       C:\Users\pro\Documents\Codex\datasets\KITTI
images      C:\Users\pro\Documents\Codex\datasets\KITTI\training\image_2
labels      C:\Users\pro\Documents\Codex\datasets\KITTI\training\label_2
source ZIP  C:\Users\pro\Downloads\data_object_image_2.zip
source ZIP  C:\Users\pro\Downloads\data_object_label_2.zip
```

레포의 `data/*/images/all`은 원본 이미지 폴더를 가리키는 Windows directory junction입니다. 이미지를 복사하지 않습니다.

## 고정 분할

| 용도 | 수량 | 파일 |
|---|---:|---|
| 학습 | 5,481 | `splits/train.txt` |
| 내부 검증 | 1,000 | `splits/internal_val.txt` |
| INT8 calibration | 1,024 | `splits/calibration.txt` (학습 집합의 부분집합) |
| 공식 평가 | 1,000 | `eval_val.txt` |

세 독립 집합의 교집합은 모두 0입니다. 분할 생성 알고리즘과 클래스/난이도 분포는 `splits/split_report.json`에 기록했습니다.

## 재현 순서

PowerShell에서 레포 루트를 기준으로 실행합니다.

```powershell
python scripts\prepare_kitti.py --image-zip C:\Users\pro\Downloads\data_object_image_2.zip --label-zip C:\Users\pro\Downloads\data_object_label_2.zip --output-root C:\Users\pro\Documents\Codex\datasets\KITTI --eval-split .\eval_val.txt

python scripts\audit_kitti.py --kitti-root C:\Users\pro\Documents\Codex\datasets\KITTI --eval-split .\eval_val.txt --report .\runs\audit_report.local.json

python scripts\make_splits.py --kitti-root C:\Users\pro\Documents\Codex\datasets\KITTI --eval-split .\eval_val.txt --output-dir .\splits
```

그다음 각 출력 루트에 `images\all` junction을 한 번 만든 뒤 변환합니다.

```powershell
New-Item -ItemType Junction -Path .\data\kitti_yolo_exact3\images\all -Target C:\Users\pro\Documents\Codex\datasets\KITTI\training\image_2
New-Item -ItemType Junction -Path .\data\kitti_yolo_neighbors3\images\all -Target C:\Users\pro\Documents\Codex\datasets\KITTI\training\image_2

python scripts\convert_kitti_to_yolo.py --kitti-root C:\Users\pro\Documents\Codex\datasets\KITTI --splits-dir .\splits --official-eval .\eval_val.txt --output-root .\data\kitti_yolo_exact3 --neighbor-policy ignore
python scripts\convert_kitti_to_yolo.py --kitti-root C:\Users\pro\Documents\Codex\datasets\KITTI --splits-dir .\splits --official-eval .\eval_val.txt --output-root .\data\kitti_yolo_neighbors3 --neighbor-policy map
```

`exact3`는 세 평가 클래스만 학습합니다. `neighbors3`는 KITTI 평가에서 ignore 이웃 클래스로 취급되는 Van→Car, Person_sitting→Pedestrian 매핑을 비교하기 위한 실험군입니다. 어느 쪽이 우수한지는 내부 AP40 결과로 결정하며 아직 가정하지 않습니다.

## 평가기 검증

평가 코드는 OpenMMLab MMDetection3D의 커밋 `fe25f7a51d36e3702f961e198894580d83c4387b`에 고정했습니다.

```powershell
.\scripts\setup_evaluator.ps1
python scripts\verify_evaluator.py --kitti-root C:\Users\pro\Documents\Codex\datasets\KITTI --work-dir .\runs\evaluator_sanity
```

정답 박스를 그대로 예측하면 세 클래스 모두 100.0, 빈 예측은 모두 0.0이어야 합니다. 실제 예측 폴더 평가는 다음과 같습니다.

```powershell
python scripts\evaluate_kitti.py --kitti-root C:\Users\pro\Documents\Codex\datasets\KITTI --split .\splits\internal_val.txt --predictions .\runs\predictions --output .\runs\metrics.json
```

예측 파일은 이미지마다 하나씩 존재해야 하며 KITTI 16열 형식이어야 합니다. 검출이 없으면 빈 파일을 둡니다. 현재 기준 구현은 널리 쓰이는 R40 참조이며, 주최 측이 별도 평가 코드를 배포하면 동일 예측으로 수치 일치를 다시 확인해야 합니다.

## CUDA 스모크 테스트

현재 확인 환경은 RTX A6000 48GB ×2, PyTorch 2.6.0+cu124, Ultralytics 8.4.108입니다.

```powershell
python scripts\smoke_train.py --device 0 --fraction 0.01
```

이 테스트는 55장으로 1 epoch를 돌려 이미지·라벨·손실·역전파·CUDA 경로만 확인합니다. 성능 수치로 사용하지 않습니다. 본 학습 전에 입력 크기, 사전학습 가중치, Hailo 변환 가능 모델을 별도 확정합니다.

## Baseline v1: 정식 학습 → KITTI 예측 → AP40 → 벤치마크

위 데이터 준비·변환을 **현재 PC의 경로로 이 복제본 안에서** 먼저 수행합니다.
README의 `C:\Users\pro` 경로는 이전 PC의 기록이며 자동으로 존재하는 경로가 아닙니다.
Git 복제에는 데이터, junction, `kitti.local.yaml`, 체크포인트가 포함되지 않습니다.
기존 데이터 폴더를 사용하려면 `--data`에 그 폴더의 YAML을 지정할 수 있습니다.
이 경우에도 YAML의 실제 train/val 목록이 이 복제본의 고정 분할과 일치해야 합니다.

CUDA용 PyTorch를 먼저 설치하고 `python -m pip install -r requirements-tested.txt`를 실행합니다.
FLOPs용 `thop`은 Ultralytics 의존성으로 설치되는 `ultralytics-thop`에서 제공합니다.
아래 명령은 복제본 루트의 PowerShell에서 실행합니다. `$KittiRoot`를 실제 데이터 경로로 바꿉니다.

```powershell
$KittiRoot = 'D:\datasets\KITTI'

# 실제 이미지/라벨 존재 여부와 YAML 목록의 누수 검사만 수행
python scripts\train_baseline.py --check-only

# COCO pretrained yolo11n.pt fine-tuning; 최초 가중치 다운로드 시 인터넷 필요
python scripts\train_baseline.py --device 0 --batch 32

# 출력 폴더가 이미 있으면 학습 도구가 새 이름을 붙이므로 마지막 Checkpoint 경로를 확인
$Weights = '.\runs\baseline\yolo11n_640_exact3\weights\best.pt'
python scripts\predict_kitti.py --weights $Weights --kitti-root $KittiRoot --output .\runs\predictions\yolo11n_640_exact3

# 기존 고정 평가기 사용 (처음에는 scripts/setup_evaluator.ps1 실행)
python scripts\evaluate_kitti.py --kitti-root $KittiRoot --split .\splits\internal_val.txt --predictions .\runs\predictions\yolo11n_640_exact3 --output .\runs\baseline\yolo11n_640_exact3\kitti_ap40.json

python scripts\benchmark_model.py --weights $Weights --device 0 --imgsz 640 --batch 1 --warmup 50 --repeats 200 --output .\runs\baseline\yolo11n_640_exact3\benchmark.json
```

### 기본값과 보호 장치

- 학습: `yolo11n.pt`, `imgsz=640`, `epochs=100`, `batch=32`, `device=0`,
  `seed=20260911`, `workers=0`, AMP 사용, 조기 종료 비활성화, 전체 train 사용.
  `--epochs`, `--imgsz`, `--batch`, `--workers`, `--name`, `--project`로 조정합니다.
  메모리 부족 시 batch를 낮추고, 제한된 Windows의 cache scan 오류에는 `--serial-cache`를 사용합니다.
  AMP를 끄려면 `--no-amp`를 사용합니다. CPU는 `--device cpu`로 명시할 수 있습니다.
- 학습 guard: 고정 분할 간 교집합, calibration 부분집합, 실제 YAML train/val 이미지 목록과
  고정 분할의 완전 일치, 중복, 클래스 순서, exact3 변환 보고서, 이미지/라벨 존재 여부를 검사합니다.
  원본 `image_2`로 junction을 풀어 쓰면 라벨 연결이 깨지므로 `images/all` 경로를 유지합니다.
- 추론: 기본 split은 `splits/internal_val.txt`, `conf=0.001`, `iou=0.7`, `max_det=300`입니다.
  낮은 confidence는 AP 계산 시 recall 손실을 줄이기 위한 시작값입니다. 튜닝은 내부 검증에만 수행합니다.
  `--split`, `--conf`, `--iou`, `--max-det`, `--batch`를 지정할 수 있습니다.
  원본 픽셀 좌표, Car/Pedestrian/Cyclist 순서, confidence를 포함한 16열 txt를 생성하고
  검출이 없으면 빈 txt를 만듭니다. 3D·방향 필드는 미측정 placeholder이며 2D bbox 평가 전용입니다.
  정사각형 letterbox(`rect=False`)와 FP32 추론을 사용합니다.
  재실행은 새 `--output` 폴더를 지정합니다. 일부 실패한 결과의 재사용을 막기 위해 비어 있지 않은 폴더는 거부합니다.
  공식 eval ID는 기본 차단하며, 최종 평가에만 `--allow-official-eval --split .\eval_val.txt`를 명시합니다.
- 벤치마크: 실제 가중치의 파라미터 수와 체크포인트 파일 크기, THOP MACs × 2 기반 FLOPs **추정치**를 기록합니다.
  FLOPs는 입력 크기를 그대로 적용한 batch 1 계산이며 지원되지 않는 연산은 포함되지 않을 수 있습니다.
  실패하면 0 대신 `null`과 원인을 기록합니다. latency는 warmup 후 CUDA 동기화를 포함한
  eager PyTorch forward 시간이며 이미지 읽기·전송·전처리·NMS를 제외합니다.
  JSON에는 평균/중앙값/p95, 원시 반복 시간, batch, 정밀도, 장치, 라이브러리 버전을 기록합니다.
  `--half`는 CUDA에서만 허용합니다. Hailo 성능 또는 실제 서비스 전체 지연시간으로 해석하면 안 됩니다.
- `best.pt`는 Ultralytics 내부 검증 fitness로 선택됩니다. 최종 보고 지표는 반드시 기존 KITTI
  Moderate AP40으로 다시 계산합니다. AP40 최적 epoch를 직접 선택하는 기능은 이 버전에 없습니다.

### exact3, 640과 학습 시간

`exact3`는 Car/Pedestrian/Cyclist만 학습 라벨에 넣는 정책입니다. Van·Person_sitting은
각각 Car·Pedestrian으로 합치지 않습니다. 이는 별도의 ignore-region loss를 추가했다는 뜻은 아닙니다.
`640`은 letterbox 기준 입력 크기입니다. 원본 사진을 가로세로 640으로 찌그러뜨리는 것이 아니라
비율을 유지해 크기를 조절하고 여백을 채웁니다. 학습은 기본 640×640 입력을 사용합니다.

현재 복제 작업 환경에서는 GPU 학습 시간을 측정하지 않았습니다. A6000이라는 이름만으로
100 epoch 소요시간을 확정할 수 없으므로 같은 설정으로 먼저 짧게 측정하세요.

```powershell
python scripts\train_baseline.py --epochs 5 --name yolo11n_640_exact3_timing
```

`results.csv`의 누적 time 차이 또는 학습 로그에서 초기 준비 이후 epoch 시간을 확인합니다.
예를 들어 안정된 한 epoch가 60초면 100 epoch는 약 100분, 120초면 약 200분에
초기 다운로드·데이터 스캔 시간을 더합니다. 이는 계산 예시이며 이 PC의 측정치가 아닙니다.
정식 학습은 위 기본 명령으로 별도 실행합니다.

### 로컬 검증

```powershell
python -m compileall -q scripts
python scripts\test_baseline.py
python scripts\train_baseline.py --help
python scripts\predict_kitti.py --help
python scripts\benchmark_model.py --help
```

테스트는 저장된 실제 분할, manifest 누수/중복 차단, 클래스 순서, 기존 평가기의 16열 및
빈 파일 파싱 호환성을 확인합니다. 실제 학습·추론·GPU 성능 검증을 대체하지 않습니다.
모든 데이터와 가중치, runs 산출물은 기존 `.gitignore` 정책대로 제외합니다.

API 확인 자료: [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/),
[Ultralytics torch utilities](https://docs.ultralytics.com/reference/utils/torch_utils/).
