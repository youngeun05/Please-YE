# 벤치마크 결과 — eval_val (1000장)

생성: 2026-10-29 02:02 UTC  
정확도: KITTI 2D Moderate AP40, strict IoU (Car 0.7 / Ped·Cyc 0.5), MMDetection3D `fe25f7a`  
모델 크기 MiB = 2^20 바이트. Latency는 중앙값(p50), Throughput = 1000 / Latency (배치 1).

- baseline: `yolo11n_kitti_1248x384_fp32.onnx` sha256 `f7c13c8484c69965…`
- optimized: `yolo11n_kitti_1248x384_int8.onnx` sha256 `2d5477348402719c…`

## 2. 결과 요약

| 항목 | Baseline | 최적화 후 | 변화율 |
|---|---|---|---|
| 모델 크기 (MiB) | 10.14 | 3.11 | -69.3% |
| 파라미터 수 (M) | 2.583 | 2.583 | +0.0% |
| 연산량 (GFLOPs) | 7.47 | 7.47 | +0.0% |
| 추론 Latency (ms) | 376.7 | 313.3 | -16.8% |
| Throughput (FPS) | 2.7 | 3.2 | +20.2% |
| 피크 메모리 사용량 (MiB) | 180 | 119 | -34.1% |
| 정확도 (Moderate AP40 평균) | 94.08 | 93.05 | -1.03%p |

## 6.3 정확도 — Moderate AP40

| 클래스 | Baseline | 최적화 후 | Drop (%p) |
|---|---|---|---|
| Car | 98.08 | 98.03 | 0.05 |
| Pedestrian | 91.44 | 90.03 | 1.40 |
| Cyclist | 92.72 | 91.09 | 1.63 |
| 평균 (mAP) | 94.08 | 93.05 | 1.03 |

평균 Drop: 1.03%p (상대 1.09%, 유지율 98.91%)

## 6.4 속도 및 리소스

| 항목 | Baseline | 최적화 후 | 개선율 (%) |
|---|---|---|---|
| 추론 Latency (ms, p50) | 376.7 | 313.3 | 16.8 |
| Throughput (FPS) | 2.7 | 3.2 | 20.2 |
| 모델 크기 (MiB) | 10.14 | 3.11 | 69.3 |
| 피크 메모리 (MiB) | 180 | 119 | 34.1 |

속도 향상: 1.20배 (p50 기준). p95: 808.7 → 641.6 ms. 측정 1회 평균, 4스레드, warmup 30 / 반복 200.
측정 환경: Intel(R) Xeon(R) CPU @ 2.20GHz, Linux-6.6.159+-x86_64-with-glibc2.39, ONNX Runtime 1.24.4, CPU provider.
