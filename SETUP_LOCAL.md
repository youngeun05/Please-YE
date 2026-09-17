# 이 PC(PC00) 세팅 가이드

원본 `README.md`의 경로는 다른 PC(`C:\Users\pro\...`) 기준입니다.
이 문서는 **이 PC에서 그대로 따라 할 수 있는 순서**만 정리했습니다.

- 레포 위치: `C:\Users\PC00\Desktop\이영은\이영은 모커톤` (C: = SSD)
- **데이터셋 위치: `D:\datasets\KITTI`** (D: = HDD, 여유 1.3TB)
- **ZIP 위치: `D:\downloads`**

> **왜 D: 인가**: C:(SSD) 여유가 20.7GB뿐이라 ZIP 11.7GB + 압축 해제 12GB를 담을 수 없습니다.
> D:는 HDD지만 이 PC RAM이 31.7GB이므로 첫 epoch 이후 12GB 이미지가 페이지 캐시에
> 대부분 올라가 학습 속도 손실은 크지 않습니다. C:는 venv와 `runs\` 체크포인트용으로 남깁니다.
> 그래서 아래 모든 명령에 `-DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads`가 붙어 있습니다.

---

## 0. PowerShell 열기 + venv 활성화

레포 폴더에서 **PowerShell**을 엽니다. (탐색기 주소창에 `powershell` 입력)
아래 명령은 전부 레포 루트에서, **venv를 활성화한 상태로** 실행합니다.

```powershell
Set-Location "C:\Users\PC00\Desktop\이영은\이영은 모커톤"; .\.venv\Scripts\Activate.ps1
```

프롬프트 앞에 `(.venv)`가 붙으면 정상입니다. 활성화가 거부되면 한 번만:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

> **중요**: 활성화하지 않으면 PATH의 `python`이 MSYS2 빌드
> (`C:\msys64\ucrt64\bin\python.exe`)로 잡히고, 여기에는 PyTorch가 설치되지 않습니다
> (MinGW 휠 태그라 `download.pytorch.org`의 win_amd64 휠을 받지 못함).
> `run_prepare.ps1`도 PATH의 `python`을 쓰므로 반드시 venv가 먼저 활성화돼 있어야 합니다.

## 1. 환경 점검

```powershell
.\scripts\check_env.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads
```

Python 버전, PyTorch/CUDA, GPU, 디스크 여유 공간, KITTI 존재 여부를 한 번에 출력합니다.
확인 기준:

- `python path`가 `...\.venv\Scripts\python.exe` (MSYS2 경로면 venv 활성화 안 된 것)
- `cuda avail : True`
- D: 여유 **30GB 이상**

## 2. 파이썬 패키지 설치 (이미 완료됨 — 재구축이 필요할 때만)

venv는 이미 만들어져 있습니다. 날려먹었을 때 아래 순서로 똑같이 복구합니다.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
$env:PIP_CACHE_DIR = "D:\pipcache"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r .\requirements-tested.txt
```

- **Python 3.13.2**를 씁니다. PATH의 MSYS2 python(3.11.9)과 anaconda base(3.8.18)로는 안 됩니다.
  Python 3.14도 설치돼 있지만 ultralytics/torch 휠이 아직 얇아 3.13을 골랐습니다.
- **cu126 / torch 2.14.0**: 드라이버 595.79 + RTX 3080(sm_86) 조합에서 검증했습니다
  (`torch.cuda.get_arch_list()`에 `sm_86` 포함). 원본 README의 cu124는 다른 PC 기준입니다.
- `PIP_CACHE_DIR`을 D:로 돌리는 이유는 torch 휠 2.6GB가 C: 여유를 두 번 먹지 않게 하려는 것입니다.

## 3. KITTI 2D 객체 검출 데이터 내려받기

필요한 파일은 두 개입니다.

| 파일 | 크기 (미러 실측) | 내용 |
|---|---:|---|
| `data_object_image_2.zip` | 11.7GB (12,569,945,557 B) | left color images (training 7,481 + testing) |
| `data_object_label_2.zip` | 5.3MB (5,601,213 B) | training labels |

```powershell
.\scripts\download_kitti.ps1 -ZipDir D:\downloads
```

중간에 끊기면 **같은 명령을 다시 실행**하면 이어받습니다.
미러가 거부하면 KITTI 공식 페이지에서 가입 후 같은 두 파일을 `D:\downloads`에 직접 받으세요.
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d>

## 4. 데이터 준비 (압축 해제 → 감사 → 변환)

```powershell
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads
```

내부에서 순서대로 실행됩니다.

1. `prepare_kitti.py` — ZIP CRC 전수 검사 후 `training\image_2`, `training\label_2`만 안전하게 추출
2. `audit_kitti.py` — 7,481장 이미지/라벨 일치, PNG 포맷, bbox 범위, 클래스 분포 검사 → `runs\audit_report.local.json`
3. (선택) `make_splits.py` — 분할 재생성 후 커밋된 분할과 SHA256 비교
4. `data\kitti_yolo_exact3\images\all`, `data\kitti_yolo_neighbors3\images\all` **junction 생성** (이미지 복사 안 함)
5. `convert_kitti_to_yolo.py` — exact3(3클래스만) / neighbors3(Van→Car, Person_sitting→Pedestrian) 두 버전 생성

유용한 옵션:

```powershell
# 분할 재현성까지 확인 (시간이 더 걸림)
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads -VerifySplits

# ZIP CRC 전수 검사 생략 (빠르지만 덜 엄격)
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads -SkipCrc

# 이미 압축을 풀었으면 추출 단계만 건너뛰기
.\scripts\run_prepare.ps1 -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads -SkipPrepare

# 다른 드라이브 사용
.\scripts\run_prepare.ps1 -DatasetRoot E:\datasets\KITTI -ZipDir E:\downloads
```

## 5. 산출물 검증

```powershell
python .\scripts\verify_prepared.py --kitti-root D:\datasets\KITTI
```

확인 항목:

- 분할 개수(5,481 / 1,000 / 1,024 / 1,000), 중복 없음, **세 집합 교집합 0**, calibration ⊂ train
- 추출된 KITTI 7,481장 이미지/라벨 id 일치
- YOLO 라벨 파일 7,481개, 5필드, class id ∈ {0,1,2}, 정규화 좌표 ∈ [0,1]
- `lists\*.txt`의 이미지 경로 실재 + 라벨 경로 매핑(`images`→`labels`) 성립
- `kitti.local.yaml`, `calibration.local.yaml` 생성

데이터 없이 분할만 확인하려면:

```powershell
python .\scripts\verify_prepared.py --splits-only
```

## 6. CUDA 스모크 테스트

```powershell
python .\scripts\smoke_train.py --device 0 --fraction 0.01
```

55장으로 1 epoch만 돌려 이미지·라벨·손실·역전파·CUDA 경로만 확인합니다.
**성능 수치로 쓰지 않습니다.**

## 7. 평가기 설치 및 검증 (선택, 지금 단계에서는 데이터 준비 후)

```powershell
.\scripts\setup_evaluator.ps1
python .\scripts\verify_evaluator.py --kitti-root D:\datasets\KITTI --work-dir .\runs\evaluator_sanity
```

정답 박스를 그대로 예측하면 세 클래스 100.0, 빈 예측은 0.0이어야 정상입니다.

---

## 절대 지켜야 할 기준 (원본 README와 동일)

- `eval_val.txt`의 1,000장은 학습·하이퍼파라미터 선택·증강 탐색에 **사용 금지**
- 최종 비교 지표는 Car/Pedestrian/Cyclist의 **KITTI 2D Moderate AP40 평균**
- 이미지·생성 라벨·체크포인트·변환 산출물은 git에 올리지 않음 (`.gitignore`)
- `해체분석.md`는 공지 전 초안이므로 실험 기준으로 쓰지 않음

## 이 PC에서 추가된 파일

| 파일 | 용도 |
|---|---|
| `scripts\check_env.ps1` | Python/CUDA/GPU/디스크/데이터 존재 여부 점검 |
| `scripts\download_kitti.ps1` | KITTI 두 ZIP 이어받기 다운로드 |
| `scripts\run_prepare.ps1` | 추출→감사→(분할검증)→junction→변환 일괄 실행 |
| `scripts\verify_prepared.py` | 분할 무결성 + YOLO 산출물 형식 검증 |
| `SETUP_LOCAL.md` | 이 문서 |

원본 스크립트(`prepare_kitti.py`, `audit_kitti.py`, `make_splits.py`, `convert_kitti_to_yolo.py`, `evaluate_kitti.py`, `verify_evaluator.py`, `smoke_train.py`)는 **수정하지 않았습니다.**

---

## 이 PC에서 실측·검증된 환경 (2026-09-17)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 10GB (sm_86), 드라이버 595.79 |
| RAM | 31.7 GB |
| C: (SSD, Samsung MZVL21T0HCLR) | 941GB 중 여유 20.7GB — venv + `runs\` 용 |
| D: (HDD, ST2000DM008) | 1,863GB 중 여유 1,315GB — 데이터셋 + ZIP + pip 캐시 용 |
| Python | 3.13.2 (`.venv`) |
| PyTorch | 2.14.0+cu126, `cuda avail: True`, GPU matmul 통과 |
| ultralytics / numpy / Pillow / PyYAML | 8.4.108 / 2.4.6 / 11.0.0 / 6.0.2 (`requirements-tested.txt` 그대로) |
| 분할 무결성 | `verify_prepared.py --splits-only` 21개 항목 ALL CHECKS PASSED |
| KITTI 미러 | `s3.eu-central-1.amazonaws.com/avg-kitti` HTTP 200, `Accept-Ranges: bytes` (이어받기 가능) |

아직 남은 단계: 3(다운로드) → 4(준비) → 5(검증) → 6(스모크) → 7(평가기).
