# 벤치마크 결과 — eval_val (1000장)

생성: 2026-10-28 13:48 대한민국 표준시  
정확도: KITTI 2D Moderate AP40, strict IoU (Car 0.7 / Ped·Cyc 0.5), MMDetection3D `fe25f7a`  
모델 크기 MiB = 2^20 바이트. Latency는 중앙값(p50), Throughput = 1000 / Latency (배치 1).

- baseline: `yolo11n_kitti_1248x384_fp32.onnx` sha256 `f7c13c8484c69965…`
- optimized: `yolo11n_kitti_1248x384_int8.onnx` sha256 `2d5477348402719c…`
- y11n_int8_minmax: `best_1248x384_int8_minmax.onnx` sha256 `80d9b93fb0421afc…`
- y11n_int8_entropy: `best_1248x384_int8_entropy.onnx` sha256 `b3c27068338e2e6a…`
- y11n_pt: `best.pt` sha256 `5b7bc89be9bebebb…`
- y8n_fp32: `best_1248x384_fp32.onnx` sha256 `c7810bf3a1511c73…`
- y8n_int8_minmax: `best_1248x384_int8_minmax.onnx` sha256 `7aea164036d0d5c6…`
- y8n_int8_percentile: `best_1248x384_int8_percentile.onnx` sha256 `4b8eb48c041bff24…`
- y8n_pt: `best.pt` sha256 `ce90d2b609ebebae…`

## 2. 결과 요약

| 항목 | Baseline | 최적화 후 | 변화율 |
|---|---|---|---|
| 모델 크기 (MiB) | 10.14 | 3.11 | -69.3% |
| 파라미터 수 (M) | 2.583 | 2.583 | +0.0% |
| 연산량 (GFLOPs) | 7.47 | 7.47 | +0.0% |
| 추론 Latency (ms) | 25.5 | 16.1 | -37.1% |
| Throughput (FPS) | 39.1 | 62.3 | +59.1% |
| 피크 메모리 사용량 (MiB) | 172 | 111 | -35.6% |
| 정확도 (Moderate AP40 평균) | 94.08 | 93.71 | -0.37%p |

## 6.3 정확도 — Moderate AP40

| 클래스 | Baseline | 최적화 후 | Drop (%p) | y11n_int8_minmax | y11n_int8_minmax Drop (%p) | y11n_int8_entropy | y11n_int8_entropy Drop (%p) | y11n_pt | y11n_pt Drop (%p) | y8n_fp32 | y8n_fp32 Drop (%p) | y8n_int8_minmax | y8n_int8_minmax Drop (%p) | y8n_int8_percentile | y8n_int8_percentile Drop (%p) | y8n_pt | y8n_pt Drop (%p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Car | 98.08 | 97.95 | 0.12 | 97.97 | 0.11 | 97.95 | 0.13 | 98.08 | 0.00 | 98.16 | -0.08 | 98.76 | -0.68 | 99.20 | -1.13 | 98.16 | -0.08 |
| Pedestrian | 91.44 | 90.86 | 0.57 | 90.40 | 1.04 | 90.98 | 0.46 | 91.44 | 0.00 | 91.55 | -0.11 | 90.48 | 0.96 | 90.81 | 0.63 | 91.55 | -0.11 |
| Cyclist | 92.72 | 92.31 | 0.41 | 92.03 | 0.69 | 91.90 | 0.82 | 92.72 | 0.00 | 95.57 | -2.85 | 94.75 | -2.03 | 93.28 | -0.56 | 95.57 | -2.85 |
| 평균 (mAP) | 94.08 | 93.71 | 0.37 | 93.47 | 0.61 | 93.61 | 0.47 | 94.08 | 0.00 | 95.09 | -1.02 | 94.66 | -0.59 | 94.43 | -0.35 | 95.09 | -1.02 |

평균 Drop: 0.37%p (상대 0.39%, 유지율 99.61%)

## 6.4 속도 및 리소스

| 항목 | Baseline | 최적화 후 | 개선율 (%) |
|---|---|---|---|
| 추론 Latency (ms, p50) | 25.5 | 16.1 | 37.1 |
| Throughput (FPS) | 39.1 | 62.3 | 59.1 |
| 모델 크기 (MiB) | 10.14 | 3.11 | 69.3 |
| 피크 메모리 (MiB) | 172 | 111 | 35.6 |

속도 향상: 1.59배 (p50 기준). p95: 30.7 → 18.7 ms. 측정 2회 평균, 4스레드, warmup 30 / 반복 200.
측정 환경: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel, Windows-11-10.0.26200-SP0, ONNX Runtime 1.24.4, CPU provider.
