# INT8 손실 지점 분석 — 층별 SQNR (보고서 6.5절 "손실 지점", 그림 2)

제출 INT8 모델(`yolo11n_kitti_1248x384_int8.onnx`)의 각 중간값을 FP32 모델(`yolo11n_kitti_1248x384_fp32.onnx`)과 비교해,
INT8 오차가 어디서 생기는지 SQNR(dB, 높을수록 FP32와 가까움)로 잰 결과입니다.

- 데이터: `splits/internal_val.txt` 앞 16장 (eval_val은 쓰지 않음), 전처리는 `scripts/quantize_ort.py`의 보정 리더와 같음
- 방법: ONNX Runtime 1.24.4 `onnxruntime.quantization.qdq_loss_debug`
  (`modify_model_output_intermediate_tensors` → `collect_activations` → `create_activation_matching` → `compute_activation_error`,
  가중치는 `create_weight_matching` → `compute_weight_error`). 메모리 때문에 4장씩 나눠 모은 뒤 제곱합으로 합쳤고,
  첫 묶음에서 공식 함수 값과 비교해 최대 차이 0.0025 dB를 확인했습니다 (`run_log.txt`).
- `qdq_err`: 그 텐서 자체의 양자화 오차, `xmodel_err`: INT8 모델과 FP32 모델의 같은 텐서 비교(앞 층에서 쌓인 오차 포함)

| 파일 | 내용 |
|---|---|
| `block_sqnr.csv` | Ultralytics 블록(model.N)별 출력의 누적 SQNR과 입력 대비 변화(Δ) — 그림 2 |
| `per_tensor_sqnr.csv` | 텐서 330개 전체의 qdq_err / xmodel_err |
| `weights_sqnr.csv` | 가중치 175개의 SQNR |
| `summary.json`, `run_log.txt` | 요약과 실행 기록 |
| `layer_loss.py` | 분석 스크립트 (PC00 경로가 코드에 고정돼 있음: `REPO`, `IMAGE_DIR`. 모델 사본과 결과를 스크립트 옆에 쓰므로 빈 폴더에 복사해 실행) |

요약:

- 자체 양자화 오차가 가장 큰 텐서는 C2PSA(model.10) 어텐션의 Softmax 출력(qdq 16.9 dB, 누적 13.6 dB).
  토큰 468개(39×12)의 평균 어텐션 가중치 1/468이 UINT8 한 칸(약 1/255)보다 작습니다.
- 누적 오차가 가장 크게 늘어나는 블록: model.2 C3k2(−6.0 dB), model.1(−5.5), model.17(−4.2), model.10 C2PSA(−4.0).
- head(model.23)에서는 분류 분기(cv3)의 depthwise 합성곱 출력이 14.5~16.5 dB로 가장 낮고, 디코딩 후 클래스 확률 20.5 dB·박스 거리 20.1 dB.
- 가중치 SQNR은 최저 36.4 dB(박스 분기 cv2), 중앙값 41.8 dB → 오차의 주원인은 가중치가 아니라 활성값(UINT8) 양자화.
- 한계: 16장·에너지 기반 지표라 AP40 하락으로 바로 환산되지 않습니다. 층별 값은 그래프 최적화를 끈 QDQ 시뮬레이션이며,
  실제 융합 INT8 커널(VNNI)과의 출력 차이는 박스 43.8 dB·클래스 24.5 dB 수준입니다. VNNI가 없는 CPU의 포화는 포함하지 않습니다.
