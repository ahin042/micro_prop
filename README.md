# Microstructure → hardness

Language: English | [한국어](README.ko.md) | [日本語](README.ja.md)

합성 미세구조 기반 재료 물성 예측 AI 경진대회 (Sep 29 – Oct 19, 2026).
The task: look at one 256×256 grayscale image of a synthetic material structure and predict its hardness value.
There are 500 training images (with answers) and 1000 test images. The score is RMSE (lower is better).

The competition data is **not** in this repository. Put it in `data/` yourself (see below).

## Results

| file | what it is | CV | public LB |
|---|---|---|---|
| `004_…csv` | ResNet18 CNN + Ridge on hand-made features | 13.529 | 12.373 |
| `005_…csv` | 004 + Ridge on DINOv2-S features (DINOv2 not trained) | 13.113 | 11.932 |
| `006_…csv` | 005, but the CNN sees the image at 2× size | 12.988 | 12.002 |
| `007_…csv` | 005, but the CNN is replaced by a fine-tuned DINOv2-S | 12.790 | 11.677 |
| `008_…csv` | 007 + a fine-tuned DINOv2-B (4 parts, equal weights) | **12.604** | **11.497** |

- Every file is a plain average of its parts. No blend weights were fitted.
- CV = average RMSE over 15 folds (5 folds × 3 repeats, seed 42) on the training images. Every model uses the same folds.
- Public LB = score on 30% of the test set. The public LB was about 1.0–1.2 lower than CV for every file.
- Full file names, SHA-256 and notes: [reports/final_candidates.md](reports/final_candidates.md) and `reports/lb_tracking.json`.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # exact versions; full list: requirements-lock.txt
brew install libomp                               # macOS only (needed by LightGBM)
.venv/bin/python src/run.py --mode fast           # rebuild the CSVs from saved model predictions (~15 min)
.venv/bin/python src/run.py --mode full           # also retrain every neural network (many hours)
```

- Data: `data/train/*.png`, `data/test/*.png`, `data/train.csv`, `data/sample_submission.csv`.
- Weights: `weights/<name>.safetensors` (see the table below). Every script checks the SHA-256 of the file. No internet is needed.
- `--target 004|005|006|007|008|all` picks which files to build (default: all).
- `HARDNESS_OUT=<folder>` writes all outputs to a new folder instead of the repository.
- Full mode time on an Apple M5 (16 GB): 004–006 about 3 h 15 min, 007 adds about 2 h, 008 adds about 4.5 h.

## Pretrained weights

| weights | used in | license |
|---|---|---|
| `resnet18.a1_in1k` | 004–008 (CNN and the helper network for hand-made features) | Apache-2.0 |
| `vit_small_patch14_dinov2.lvd142m` (DINOv2-S) | 005–008 (frozen in 005/006, fine-tuned in 007/008) | Apache-2.0 |
| `vit_base_patch14_dinov2.lvd142m` (DINOv2-B) | 008 (fine-tuned) | Apache-2.0 |

## Code (`src/`)

| file | role |
|---|---|
| `run.py` | one command to rebuild 004–008 |
| `common.py` | paths, folds, Ridge/LightGBM helpers, CSV writer |
| `stage0_recon.py`, `features.py`, `p12_*.py`, `p13_*.py` | data checks and hand-made image features |
| `p13_latnet.py` | helper network that reads structure from damaged images |
| `cnn.py`, `cnn2.py` | ResNet18 CNN (004–006) |
| `cnn3.py`, `cnn4.py` | DINOv2 fine-tuning (007, 008) |
| `p17_dino_probe.py`, `p13_build_004.py`, `p17_build.py` | frozen DINOv2 features, blends and CSV files |
| `writeup.py` | solution PDF (`reports/solution.pdf`) |
| other `p1*`/`p2*` scripts | other experiments (not needed to rebuild the CSVs) |
