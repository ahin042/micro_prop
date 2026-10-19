# 미세구조 → 경도 예측

언어: [English](README.md) | 한국어 | [日本語](README.ja.md)

DACON/DAKER "합성 미세구조 기반 재료 물성 예측 AI 경진대회"(2026년 9월 29일 ~ 10월 19일) 코드입니다.
과제: 합성 재료 구조를 찍은 256×256 흑백 이미지 한 장을 보고 경도 값을 예측합니다.
학습 이미지 500장(정답 있음), 테스트 이미지 1000장이 있습니다. 점수는 RMSE이고 낮을수록 좋습니다.

대회 데이터는 이 저장소에 **들어 있지 않습니다**. 직접 `data/` 폴더에 넣어야 합니다(아래 참고).

## 결과

| 파일 | 내용 | CV | public LB |
|---|---|---|---|
| `004_…csv` | ResNet18 CNN + 직접 만든 특징에 Ridge | 13.529 | 12.373 |
| `005_…csv` | 004 + DINOv2-S 특징에 Ridge (DINOv2는 학습 안 함) | 13.113 | 11.932 |
| `006_…csv` | 005와 같지만 CNN이 이미지를 2배 크기로 봄 | 12.988 | 12.002 |
| `007_…csv` | 005에서 CNN 대신 미세조정한 DINOv2-S 사용 | 12.790 | 11.677 |
| `008_…csv` | 007 + 미세조정한 DINOv2-B (4개를 똑같은 비율로 평균) | **12.604** | **11.497** |

- 모든 파일은 구성 모델의 단순 평균입니다. 섞는 비율을 따로 학습하지 않았습니다.
- CV = 학습 이미지로 15번(5-fold × 3회 반복, seed 42) 나눠 잰 RMSE의 평균입니다. 모든 모델이 같은 fold를 씁니다.
- Public LB = 테스트 이미지 30%에 대한 점수입니다. 모든 파일에서 public LB가 CV보다 약 1.0~1.2 낮았습니다.
- 파일 전체 이름, SHA-256, 메모: [reports/final_candidates.md](reports/final_candidates.md), `reports/lb_tracking.json`.
- 전체 실험 기록: [experiments.md](experiments.md) (요약: [experiments.ko.md](experiments.ko.md)).

## 실행 방법

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # 정확한 버전; 전체 목록은 requirements-lock.txt
brew install libomp                               # macOS만 (LightGBM에 필요)
.venv/bin/python src/run.py --mode fast           # 저장된 모델 예측으로 CSV 다시 만들기 (약 15분)
.venv/bin/python src/run.py --mode full           # 신경망까지 전부 다시 학습 (여러 시간)
```

- 데이터: `data/train/*.png`, `data/test/*.png`, `data/train.csv`, `data/sample_submission.csv`.
- 가중치: `weights/<이름>.safetensors` (아래 표 참고). 모든 스크립트가 파일의 SHA-256을 확인합니다. 인터넷은 필요 없습니다.
- `--target 004|005|006|007|008|all` 로 만들 파일을 고릅니다 (기본값: all).
- `HARDNESS_OUT=<폴더>` 를 주면 결과를 저장소 대신 새 폴더에 씁니다.
- Apple M5(16 GB)에서 full 모드 시간: 004~006 약 3시간 15분, 007은 약 2시간 추가, 008은 약 4.5시간 추가.

## 사전학습 가중치

| 가중치 | 사용한 파일 | 라이선스 |
|---|---|---|
| `resnet18.a1_in1k` | 004~008 (CNN, 그리고 특징 계산용 보조 신경망) | Apache-2.0 |
| `vit_small_patch14_dinov2.lvd142m` (DINOv2-S) | 005~008 (005/006은 고정, 007/008은 미세조정) | Apache-2.0 |
| `vit_base_patch14_dinov2.lvd142m` (DINOv2-B) | 008 (미세조정) | Apache-2.0 |

모두 대회 시작 훨씬 전에 공개된 가중치입니다. 다운로드 주소, 커밋, SHA-256은 `src/weights_manifest.json`에 있습니다.
다른 가중치(DINOv2-L, ConvNeXt, EfficientNet, ResNet50, MicroNet)는 시험만 해 봤고 어떤 제출 파일에도 쓰지 않았습니다.

## 지킨 대회 규칙

- 학습에는 학습 이미지와 `train.csv`만 썼습니다. 테스트 이미지는 예측할 때만 썼습니다.
- 외부 데이터, 직접 만든 추가 이미지, 온라인 API는 쓰지 않았습니다. 전부 오프라인에서 돌아갑니다.
- 사전학습 가중치: 공개 라이선스이고 2026-09-29 이전에 공개된 것만 썼습니다. 금지된 `DriftFox/voronoi-microstructure-resnet` 모델은 쓰지 않았습니다.
- Python만 사용, 상대 경로, UTF-8, 고정 seed.

## 재현성

- seed를 고정했고 PyTorch를 결정적(deterministic) 모드로 돌립니다.
- 기준 Mac에서 빈 폴더에 다시 만들면 CSV가 바이트 단위로 똑같이 나옵니다 (004~006은 fast/full 모드 모두, 007과 008은 fast 모드).
- 007의 DINOv2-S 미세조정은 세 번 돌렸고 매번 숫자가 완전히 같았습니다.
- 다른 하드웨어(다른 CPU, CUDA)에서는 seed를 바꾼 것처럼 숫자가 조금 다를 수 있습니다. 파일은 SHA-256 대신 값으로 비교하세요.

## 알게 된 점

1. 이미지마다 흐림, 노이즈, 대비가 크게 다르고, 이것은 경도와 관계가 없습니다. 기존의 결정립 크기 측정 방법은 재료보다 흐림을 주로 잽니다.
2. 직접 만든 특징 중 가장 좋은 것은 어두운 상의 양을, 각 이미지의 주된 회색 값을 기준으로 잰 것입니다.
3. 학습 이미지를 일부러 망가뜨린 복사본으로 학습한 작은 신경망이, 흐리거나 노이즈가 많은 이미지에서 구조를 더 안정적으로 잽니다.
4. 사전학습된 DINOv2 특징이 직접 만든 모든 특징보다 좋았습니다. DINOv2-S와 DINOv2-B를 미세조정한 것이 가장 크게 좋아졌습니다 (007, 008).
5. 서로 다른 종류의 모델을 평균하면 매번 좋아졌습니다.
6. 006은 CV로는 005보다 좋았지만 public LB에서는 더 나빴습니다. 이유는 찾지 못했고, public 테스트 이미지 300장에서 생긴 우연일 수 있습니다.

## 코드 (`src/`)

| 파일 | 역할 |
|---|---|
| `run.py` | 004~008을 한 번에 다시 만드는 명령 |
| `common.py` | 경로, fold, Ridge/LightGBM 도우미, CSV 작성 |
| `stage0_recon.py`, `features.py`, `p12_*.py`, `p13_*.py` | 데이터 점검과 직접 만든 이미지 특징 |
| `p13_latnet.py` | 망가진 이미지에서도 구조를 읽는 보조 신경망 |
| `cnn.py`, `cnn2.py` | ResNet18 CNN (004~006) |
| `cnn3.py`, `cnn4.py` | DINOv2 미세조정 (007, 008) |
| `p17_dino_probe.py`, `p13_build_004.py`, `p17_build.py` | 고정 DINOv2 특징, 모델 평균, CSV 파일 |
| `writeup.py` | 솔루션 PDF (`reports/solution.pdf`) |
| 나머지 `p1*`/`p2*` 스크립트 | experiments.md에 적힌 실험들 (CSV를 다시 만드는 데는 필요 없음) |
