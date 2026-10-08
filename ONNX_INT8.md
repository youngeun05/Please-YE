# ONNX FP32 → INT8 (ONNX Runtime) 실행 순서

예선 비교표의 baseline(FP32 ONNX)과 최적화 모델(INT8 ONNX)을 만들고 같은 조건에서 측정합니다.
모든 명령은 레포 루트, venv 활성화 상태 기준입니다. 경로는 PC00 기준이니 자기 PC에 맞게 바꾸세요.

```powershell
python -m pip install -r .\requirements-onnx.txt
$W = "runs\baseline\yolo11n_exact3_1248\weights"
```

## 1. FP32 ONNX export + 정합성 검사

```powershell
python scripts\export_onnx.py --weights $W\best.pt --kitti-root D:\datasets\KITTI
```

입력이 `[1, 3, 384, 1248]`로 고정되고, calibration 이미지 1장으로 PyTorch와 출력 차이를 검사합니다.
결과: `$W\best_1248x384_fp32.onnx` + 같은 이름의 `.json` 기록.

## 2. FP32 ONNX의 AP40 확인 (95.59와 거의 같아야 정상)

```powershell
python scripts\predict_kitti.py --weights $W\best_1248x384_fp32.onnx --output runs\predictions\onnx_fp32_internal_val --device cpu --batch 1
python scripts\evaluate_kitti.py --kitti-root D:\datasets\KITTI --split splits\internal_val.txt --predictions runs\predictions\onnx_fp32_internal_val --output runs\metrics_onnx_fp32.json
```

## 3. INT8 PTQ (calibration 1,024장)

```powershell
python scripts\quantize_ort.py --model $W\best_1248x384_fp32.onnx --kitti-root D:\datasets\KITTI
python scripts\quantize_ort.py --model $W\best_1248x384_fp32.onnx --kitti-root D:\datasets\KITTI --keep-head-fp32
```

먼저 `--limit 64`로 빠르게 돌려 파이프라인을 확인한 뒤 전체로 돌리면 시간을 아낄 수 있습니다.
비교 실험 축: `--method minmax|entropy|percentile`, `--per-tensor`, `--keep-head-fp32`.

## 4. INT8의 AP40과 Drop Rate

2번과 같은 명령에서 `--weights`만 INT8 파일로 바꿉니다.
Drop Rate는 절대값(baseline − INT8)과 상대값(÷ baseline)을 둘 다 기록합니다.

## 5. 같은 조건에서 크기·파라미터·FLOPs·메모리·Latency 측정

```powershell
python scripts\benchmark_onnx.py --models $W\best_1248x384_fp32.onnx $W\best_1248x384_int8_minmax.onnx $W\best_1248x384_int8_minmax_headfp32.onnx --threads 4 --output runs\benchmark_onnx.json
```

첫 번째 모델이 기준입니다. 측정 중에는 다른 프로그램을 끄고, `--threads` 값을 보고서에 적으세요.
CPU provider로 재는 이유: ONNX Runtime은 QDQ INT8을 CPU에서 실제 INT8 커널로 실행하지만, CUDA provider에서는 대부분 float로 되돌려 실행합니다.

## 주의

- 모든 단계가 `eval_val.txt`를 읽지 않습니다. 최종 1회 측정은 모든 선택이 끝난 뒤에만 `--split eval_val.txt`로 합니다.
- FLOPs와 파라미터 수는 INT8에서 그대로입니다(정상). 이 둘을 줄이려면 640×192 학습 같은 별도 축이 필요합니다.
