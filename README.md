# Please-YE

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
