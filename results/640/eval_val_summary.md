# 벤치마크 결과 — eval_val (1000장)

생성: 2026-10-28 18:53 대한민국 표준시  
정확도: KITTI 2D Moderate AP40, strict IoU (Car 0.7 / Ped·Cyc 0.5), MMDetection3D `fe25f7a`  
모델 크기 MiB = 2^20 바이트. Latency는 중앙값(p50), Throughput = 1000 / Latency (배치 1).

- baseline: `yolo11n_kitti_1248x384_fp32.onnx` sha256 `f7c13c8484c69965…`
- optimized: `best_640x192_int8_percentile.onnx` sha256 `28a72863c82a2a6e…`
- y640_fp32: `best_640x192_fp32.onnx` sha256 `c449b45ffe9a8ec7…`
- y1248_int8: `yolo11n_kitti_1248x384_int8.onnx` sha256 `2d5477348402719c…`

## 2. 결과 요약

| 항목 | Baseline | 최적화 후 | 변화율 |
|---|---|---|---|
| 모델 크기 (MiB) | 10.14 | 2.97 | -70.7% |
| 파라미터 수 (M) | 2.583 | 2.583 | +0.0% |
| 연산량 (GFLOPs) | 7.47 | 1.90 | -74.6% |
| 추론 Latency (ms) | 26.7 | 5.1 | -80.9% |
| Throughput (FPS) | 37.5 | 196.3 | +424.2% |
| 피크 메모리 사용량 (MiB) | 173 | 90 | -47.8% |
| 정확도 (Moderate AP40 평균) | 94.08 | 88.72 | -5.36%p |

## 6.3 정확도 — Moderate AP40

| 클래스 | Baseline | 최적화 후 | Drop (%p) | y640_fp32 | y640_fp32 Drop (%p) | y1248_int8 | y1248_int8 Drop (%p) |
|---|---|---|---|---|---|---|---|
| Car | 98.08 | 97.43 | 0.64 | 98.00 | 0.07 | 97.95 | 0.12 |
| Pedestrian | 91.44 | 80.99 | 10.44 | 82.07 | 9.37 | 90.86 | 0.57 |
| Cyclist | 92.72 | 87.74 | 4.98 | 87.96 | 4.76 | 92.31 | 0.41 |
| 평균 (mAP) | 94.08 | 88.72 | 5.36 | 89.35 | 4.73 | 93.71 | 0.37 |

평균 Drop: 5.36%p (상대 5.69%, 유지율 94.31%)

## 6.4 속도 및 리소스

| 항목 | Baseline | 최적화 후 | 개선율 (%) |
|---|---|---|---|
| 추론 Latency (ms, p50) | 26.7 | 5.1 | 80.9 |
| Throughput (FPS) | 37.5 | 196.3 | 424.2 |
| 모델 크기 (MiB) | 10.14 | 2.97 | 70.7 |
| 피크 메모리 (MiB) | 173 | 90 | 47.8 |

속도 향상: 5.24배 (p50 기준). p95: 32.1 → 5.9 ms. 측정 2회 평균, 4스레드, warmup 30 / 반복 200.
측정 환경: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel, Windows-11-10.0.26200-SP0, ONNX Runtime 1.24.4, CPU provider.
