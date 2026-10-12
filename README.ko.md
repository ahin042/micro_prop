# 미세구조 → 경도 회귀

언어: [English](README.md) | 한국어 | [日本語](README.ja.md)

이미지 회귀 문제입니다(256×256 그레이스케일 미세구조 → 합성 경도 대리값, 평가지표 RMSE).
전체 실험 로그는 [experiments.ko.md](experiments.ko.md)에 있으며([English](experiments.md), [日本語](experiments.ja.md)),
기계가 읽을 수 있는 결과는 `reports/*.json`에 있습니다.

## 재현 방법(단일 명령 흐름)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp            # macOS 전용: LightGBM의 OpenMP 런타임
.venv/bin/python src/run.py --mode full     # 전체 실행(CNN 학습 포함, Apple M5에서 약 1.5시간)
.venv/bin/python src/run.py --mode fast     # 동일하되 캐시된 CNN OOF/테스트 예측 재사용
```

데이터는 `./data/{train,test}/*.png`, `./data/train.csv`, `./data/sample_submission.csv` 위치에 두어야 합니다.
모든 경로는 저장소 루트를 기준으로 합니다. 출력물은 `submissions/*.csv`, `cache/`(피처, OOF), `reports/`에 생성됩니다.

## 실행 환경

| 항목 | 버전 |
|---|---|
| OS | macOS 26.3.1 (Darwin 25.3.0), Apple M5, 10 cores, 16 GB, PyTorch MPS backend |
| Python | 3.13.12 |
| core | numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26.0, joblib 1.6.0, pillow 12.3.0, pyarrow 26.0.0 |
| models | lightgbm 4.7.0 (libomp 23.1.3), torch 2.14.1, torchvision 0.29.1, timm 1.0.30 |

모든 패키지는 `requirements.txt`에 고정되어 있습니다(`pip freeze`).

## 사전학습 가중치와 라이선스

| 가중치 | 출처 | 라이선스 |
|---|---|---|
| `timm/resnet18.a1_in1k` | `timm.create_model(..., pretrained=True)`를 통한 Hugging Face Hub | Apache-2.0(모델 카드) |
| `timm/efficientnet_b0.ra_in1k`(ablation only) | timm을 통한 Hugging Face Hub | Apache-2.0(모델 카드) |
| timm library | PyPI | Apache-2.0 |

그 외 사전학습 가중치, 외부 데이터, 원격 API는 사용하지 않았습니다. 테스트 이미지는 학습에 절대 사용하지 않습니다.
이미지별 피처는 독립적으로 계산되며, 학습된 모든 통계량(스케일러, Box-Cox λ, CNN 입력 정규화,
타깃 표준화, 블렌드 가중치)은 학습 행만으로 적합됩니다.

## 재현성

- 모든 시드는 고정되어 있습니다(`SEED = 42`). CV 프로토콜은 `RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)`이며,
  모든 모델이 동일한 이미지 폴드를 사용하므로 OOF 예측을 폴드별로 블렌딩하고 비교할 수 있습니다.
- Stage 0 출력(`reports/stage0.json`)은 `n_jobs`와 캐시 설정이 달라도 비트 단위로 동일합니다. LightGBM은
  `deterministic=True`와 고정된 스레드 수로 실행됩니다.
- 일부 PyTorch MPS 커널은 비트 단위 결정론적이지 않으므로, CNN OOF는 실행마다 마지막 자릿수가 달라질 수 있습니다.

## 파이프라인(`src/`)

| 스크립트 | 목적 |
|---|---|
| `stage0_recon.py` | 라벨 구조 분석, 양성 대조군과 견고한 귀무분포를 사용한 중복 검색, PNG 포렌식, 기준선 앵커 |
| `baseline_cheap.py` | 앵커 테이블(const / ridge / kNN / LightGBM), CV 기반 타깃 변환 선택, submission 001 |
| `features.py` | 이미지별 피처 174개: nuisance(`q_`), 원시/정규화 추정량(`r_`/`n_`), MTF 보정(`m_`), 디컨볼루션(`d_`), 스케일 스페이스(`x_`) |
| `stage1_report.py` | 추정량 타당성(y와의 rho vs blur와의 rho), Stage 1 CV, submission 002 |
| `stage2_formula.py` | 멱법칙 검정, 열거식 공식 탐색, 잔차 진단 |
| `ceiling.py` | 피처 공간 쌍둥이 테스트, 잔차 구조 탐색 |
| `partb_report.py` | 블러 보정 평가와 CV |
| `stationarity.py` | 사분면 ICC: crop들이 같은 라벨을 공유할 수 있는가? |
| `cnn.py` | 전체 이미지에 대한 timm CNN, D4 + nuisance 증강, 동일 폴드 사용 |
| `blend.py` | 동일 가중치 vs cross-fitted NNLS 블렌드, paired fold test, 제출 파일 |
| `common.py` | 경로, CV 프로토콜, 모델, 제출 파일 작성기/검증기 |

## 핵심 결론: 모델 용량이 아니라 측정 편향 문제

이미지에는 라벨과 독립적인 nuisance가 강하게 존재합니다. 대표적으로 블러, 노이즈(σ 4–20), 대비(span 46–131)가 있습니다.
고전적인 미세구조 추정량은 재료 자체보다 이러한 nuisance를 측정합니다. 구조 피처 114개 중 84개가 경도보다 블러 또는 노이즈와 더 강하게 상관됩니다.
ASTM E112 평균 선형 절편은 y와의 rho가 −0.03이지만 노이즈와의 rho는 0.76입니다.
Hall-Petch 멱법칙을 적합하면 입자 크기 지수가 −0.5가 아니라 +0.02로 나타납니다.
분석적 블러 보정(스펙트럼의 MTF 디컨볼루션, 세그멘테이션 전 Richardson-Lucy, 스케일 스페이스 외삽)은
어떤 입자 크기 추정량에서도 지배적인 영향을 깨지 못했습니다. 자세한 내용은 `experiments.ko.md`와 PDF 보고서를 참고하세요.
