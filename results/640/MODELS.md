# Please-YE 제출 모델 파일

| 파일 | 역할 | 포맷 | 입력 | 크기 (MiB) | SHA-256 |
|---|---|---|---|---|---|
| `yolo11n_kitti_fp32.pt` | baseline (original) | PyTorch checkpoint, Ultralytics | 1x3x192x640 | 5.22 | `6b0ec8ca2fe38bb7b8101976eff2c8df0ba6fce323424fe1f6c5d88a47d39819` |
| `yolo11n_kitti_640x192_fp32.onnx` | baseline (measured) | ONNX FP32, opset 17 | 1x3x192x640 | 10.00 | `c449b45ffe9a8ec767cf3f339413b45970f2c4718f15852c89ee2c5af25bfae0` |
| `yolo11n_kitti_640x192_int8.onnx` | optimized | ONNX INT8 QDQ (ONNX Runtime static PTQ) | 1x3x192x640 | 2.97 | `28a72863c82a2a6eb6543877f1544e3a8023c7bddc94bddbb92ef315a6a83c51` |

실행 환경 (export 기록): python 3.13.2, torch 2.6.0+cu124, ultralytics 8.4.108, onnx 1.23.2, onnxruntime 1.24.4

INT8 설정: {"method": "percentile", "percentile": 99.999, "weights": "int8 symmetric per-channel", "activations": "uint8 asymmetric", "format": "QDQ", "head_decode": "float", "excluded_nodes": 24, "calibration_images": 1024, "calibration_ids_sha256": "ad3c6cbf292709e2c8716085c1e10ebc92bcf7c70787473ff8b3913b14ad351c", "calibration_chunk": 16, "quant_pre_process": true}

검증: `python scripts/run_benchmark.py --kitti-root <KITTI>` (README.md 참고).
파일이 바뀌지 않았는지는 SHA-256으로 확인합니다 (Windows: `Get-FileHash <파일>`, Linux: `sha256sum <파일>`).
