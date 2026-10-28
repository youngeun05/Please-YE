# Please-YE 제출 모델 파일

| 파일 | 역할 | 포맷 | 입력 | 크기 (MiB) | SHA-256 |
|---|---|---|---|---|---|
| `yolo11n_kitti_fp32.pt` | baseline (original) | PyTorch checkpoint, Ultralytics | 1x3x384x1248 | 5.26 | `5b7bc89be9bebebbada2e2b8813eec8194f85e145dcc14798dc2a18abd689d87` |
| `yolo11n_kitti_1248x384_fp32.onnx` | baseline (measured) | ONNX FP32, opset 17 | 1x3x384x1248 | 10.14 | `f7c13c8484c69965c2698ccacaabe7160b27d1cc68df78b5aa17a6a0184f47b6` |
| `yolo11n_kitti_1248x384_int8.onnx` | optimized | ONNX INT8 QDQ (ONNX Runtime static PTQ) | 1x3x384x1248 | 3.11 | `2d5477348402719ca713860b9d3bf317d8dbe9f88c2d38fb816286c571565d27` |

실행 환경 (export 기록): python 3.13.2, torch 2.6.0+cu124, ultralytics 8.4.108, onnx 1.23.2, onnxruntime 1.24.4

INT8 설정: {"method": "percentile", "percentile": 99.999, "weights": "int8 symmetric per-channel", "activations": "uint8 asymmetric", "format": "QDQ", "head_decode": "float", "excluded_nodes": 24, "calibration_images": 1024, "calibration_ids_sha256": "ad3c6cbf292709e2c8716085c1e10ebc92bcf7c70787473ff8b3913b14ad351c", "calibration_chunk": 16, "quant_pre_process": true}

검증: `python scripts/run_benchmark.py --kitti-root <KITTI>` (README.md 참고).
파일이 바뀌지 않았는지는 SHA-256으로 확인합니다 (Windows: `Get-FileHash <파일>`, Linux: `sha256sum <파일>`).
